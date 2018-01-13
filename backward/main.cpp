
#include <dolfin.h>
#include "wave.h"
#include "fdm.h"

using namespace dolfin;
#define Freq 15.0
#define C0 4.0


// Define the analytical solution  
class AnalyticalSolutionS : public Expression
{
public:

  // Constructor
  AnalyticalSolutionS(double _t) : t(_t) {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { 
      //double tmp[3]= {1/sqrt(3), 1/sqrt(3), 1/sqrt(3)};
      //values[0] = cos(2*DOLFIN_PI*4.0*(t - (x[0]*tmp[0] +x[1]* tmp[1] + x[2]*tmp[2])/2.5));
      //double R =sqrt( (x[0]-2.0) * (x[0]-2.0) +
      //                (x[1]-2.0) * (x[1]-2.0) +
      //                (x[2]-2.0) * (x[2]-2.0));
      double R =sqrt( (x[0]-0.5) * (x[0]-0.5) +
                      (x[1]-0.5) * (x[1]-0.5) +
                      (x[2]-0.5) * (x[2]-0.5));
      double tmp = t-R/0.7;
      if(tmp < DOLFIN_EPS)
        {values[0] = 0;}
      else
      {
          if(R < DOLFIN_EPS)
              R = 1.0e-6;
          values[0]= sin(2*DOLFIN_PI *15.0*tmp) *
                 exp(-DOLFIN_PI*DOLFIN_PI *225.0*tmp*tmp/4)
                 /(0.49*4*DOLFIN_PI*R);
      }
      //cout<<values[0]<<endl;
  }

  // Current time
  double t;
};

// Define the analytical solution  
class AnalyticalSolution : public Expression
{
public:

  // Constructor
  AnalyticalSolution(double _t) : t(_t) {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { 
      double tmp[3]= {1/sqrt(3), 1/sqrt(3), 1/sqrt(3)};
      values[0] = cos(2*DOLFIN_PI*Freq*(t - (x[0]*tmp[0] +x[1]* tmp[1] + x[2]*tmp[2])/C0));
  }

  // Current time
  double t;
};

// Define initial u_t  
class InitUt : public Expression
{
public:

  // Constructor
  InitUt()  {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { 
      double tmp[3]= {1/sqrt(3), 1/sqrt(3), 1/sqrt(3)};
      values[0] = -2*DOLFIN_PI*Freq*
          sin(2*DOLFIN_PI*Freq*( - (x[0]*tmp[0] +x[1]* tmp[1] + x[2]*tmp[2])/C0));
  }

  // Current time
};

// Define initial u_t  
class Coeff : public Expression
{
public:

  // Constructor
  Coeff()  {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { 
      if(x[2] > 0.6)
      {
          values[0] = 0.25;
      }
      else
      {
          values[0] = 0.49;
      }
  }

  // Current time
};

// Define Dirichlet boundary 
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary;
  }
};

int save_real_soltion();
int main()
{
  // Print log messages only from the root process in parallel
  //parameters["std_out_all_processes"] = false;
  //save_real_soltion();
  //return 0;
  // mesh 
  double DomainSize =1.0, dx =0.02;
  //double DomainSize =4.0, dx =0.05;
  std::size_t nx = DomainSize/dx;
  BoxMesh mesh(0.0,0.0,0.0, DomainSize,DomainSize,DomainSize, nx, nx, nx);

  // Create FunctionSpace
  wave::FunctionSpace V(mesh);
  wave::CoefficientSpace_c DG(mesh);

  
  // Set parameter values
  double dt = 0.010;
  double T = 0.40;

  // Define values for boundary conditions
  //AnalyticalSolution real_u(0.0);
  //InitUt ut0;
  Constant zero(0);

  // Define subdomains for boundary conditions
  DirichletBoundary boundary;


  // Create functions
  Function u(V);
  Function u1(V);
  Function u2(V);
  Function c(DG);
  Coeff coeff;
  c.interpolate(coeff);
  //u1.interpolate(real_u);
  u1.interpolate(zero);
  //real_u.t = -dt;
  //u2.interpolate(real_u);
  u2.interpolate(zero);

  // Create coefficients
  Constant k(dt);
  Constant f(0);
  Constant g(0);

  // Create forms
  wave::BilinearForm a(V,V);
  wave::LinearForm L(V);

  fdm::LinearForm Lfdm(V);
  AnalyticalSolutionS fdm(0.0);
  Function c_fdm(DG);
  c_fdm.interpolate(Constant(0.49));
  c_fdm = c_fdm-c;
  c_fdm = c_fdm*(-1.0); 
  Function  rcv_u(V);
  
  // Set coefficients
  a.k = k; a.c = c;
  L.k = k; L.u1 = u1; L.u2 = u2; L.f = f; L.g = g;
  Lfdm.c=c_fdm; Lfdm.k =k; Lfdm.uf=fdm;

  // Assemble matrices
  Matrix A;
  assemble(A, a);

  // Create vectors
  Vector b,bfdm;

  // Use amg preconditioner if available
  //list_krylov_solver_preconditioners();
  const std::string prec(has_krylov_solver_preconditioner("ml_amg") ? "ml_amg" : "default");
  cout<< "Using Preconditioner: "<< prec<<endl;

  // Create files for storing solution
  File ufile("results/uh.pvd");
  ufile <<u1;

  // Time-stepping
  double t = 0;
  while (t < T + DOLFIN_EPS)
  {
    // Update boundary condition
    t += dt;
    //real_u.t = t;

    // Define boundary conditions
    //DirichletBC bc(V, real_u, boundary);
    DirichletBC bc(V, zero, boundary);

    // Compute step
    begin("Step forward");
    assemble(b, L);

    fdm.t = t;
    assemble(bfdm, Lfdm);
    b +=  bfdm;
    
    // impulse source
    //Point p(2.0,2.0,2.0);
    //Point p(0.5,0.5,0.5);
    //double magnitude = dt*dt*sin(2*DOLFIN_PI*15.0*(t))* 
    //                     exp(-DOLFIN_PI*DOLFIN_PI*225.0*(t)*(t)/4);
    //cout<< "magnitude : "<< magnitude<<endl;
    //PointSource ps(V, p, magnitude);
    //ps.apply(b);
    bc.apply(A, b);

    Timer tsolver("solver");
    double tmp;
    tsolver.start();
    solve(A, *u.vector(), b, "cg", prec);
    tmp= tsolver.stop();
    cout<< "solver time: "<<tmp<<endl;
    //solve(A, *u.vector(), b);
    end();


    // Save to file
    fdm.t = t;
    rcv_u.interpolate(fdm);
    rcv_u  =rcv_u + u;
    ufile << rcv_u;
    //ufile << u;

    // Move to next time step
    u2 = u1;
    u1 = u;
    cout << "t = " << t << endl;
  }

  // Plot solution
  plot(u, "Displacement");
  interactive();

  return 0;
}

int save_real_soltion()
{
  // mesh 
  double DomainSize = 1.0, dx =0.02;
  std::size_t nx = DomainSize/dx;
  BoxMesh mesh(0.0,0.0,0.0, DomainSize,DomainSize,DomainSize, nx, nx, nx);

  // Create function spaces
  wave::FunctionSpace V(mesh);
  Function uh(V);
  AnalyticalSolution u(0.0);
  double t = 0, T = 1.0, dt = 0.05;
  
  // Create files for storing solution
  File ufile("results/u.pvd");
  t = -dt;
  while (t < T + DOLFIN_EPS)
  {
    // Update real solution
    t += dt;
    u.t = t;
    uh.interpolate(u);

    // Save to file
    ufile << uh;
    cout << "t = "<< t <<endl;
  }

  return 0;
}


