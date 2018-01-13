
#include <dolfin.h>
#include "wave.h"
#include "fdm.h"

using namespace dolfin;


// Define the analytical solution  
class AnalyticalSolution : public Expression
{
public:

  // Constructor
  AnalyticalSolution(double _t) : t(_t) {}

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

int main()
{
  // Print log messages only from the root process in parallel
  //parameters["std_out_all_processes"] = false;

  // mesh 
  //double DomainSize = 1.0, dx =0.02;
  double DomainSize = 4.0, dx =0.005;
  std::size_t nx = DomainSize/dx;
  cout << " Number of grid: "<<nx<<endl;
  BoxMesh mesh(0.0,0.0,0.0, DomainSize,DomainSize,DomainSize, nx, nx, nx);

  // Create FunctionSpace
  wave::FunctionSpace V(mesh);
  wave::CoefficientSpace_c DG(mesh);
  cout << " Dim of FunctionSpace: "<<V.dim()<<endl;

  
  // Set parameter values
  double dt = 0.0004;
  //double dt = 0.0002;
  double T = 0.40;



  // Define values for boundary conditions
  Constant zero(0.0);

  // Define subdomains for boundary conditions
  DirichletBoundary boundary;


  // Create functions
  Function u(V);
  Function u1(V);
  Function u2(V);
  Function c(DG);
  Coeff coeff;
  //c.interpolate(coeff);
  c.interpolate(Constant(16.0));
  u1.interpolate(zero);
  u2.interpolate(zero);;

  // Create coefficients
  Constant k(dt);
  Constant f(0);
  Constant g(0);

  // Create forms
  wave::BilinearForm a(V,V);
  wave::LinearForm L(V);
  
  //fdm::LinearForm Lfdm(V);
  //AnalyticalSolution fdm(0.0);
  //Function c_fdm(DG);
  //c_fdm.interpolate(Constant(0.49));
  //c_fdm = c_fdm-c;
  //c_fdm = c_fdm*(-1.0); 
  //Function  rcv_u(V);

  // Set coefficients
  L.c = c;
  L.k = k; L.u1 = u1; L.u2 = u2; L.f = f; L.g = g;
  //Lfdm.c=c_fdm; Lfdm.k =k; Lfdm.uf=fdm;

  // Assemble matrices
  Matrix A;
  assemble(A, a);

  // Create vectors
  Vector b,bfdm;

  // Use amg preconditioner if available
  list_krylov_solver_preconditioners();
  //const std::string prec(has_krylov_solver_preconditioner("ml_amg") ? "ml_amg" : "default");
  const std::string prec("jacobi");
  cout<< "Using Preconditioner: "<< prec<<endl;

  // Create files for storing solution
  //BoxMesh meshshow(0.0,0.0,0.0, DomainSize,DomainSize,DomainSize, 20, 20, 20);
  //wave::FunctionSpace Vshow(meshshow);
  //Function ushow(Vshow);
  //ushow.interpolate(u1);
  //ufile <<ushow;
  //save solution
  //AnalyticalSolution anlt_u(T);
  //anlt_u.t=T+dt;
  File ufile("./results/uh.pvd");
  ufile << u1;

  // Time-stepping
  double t = 0;
  while (t < T + DOLFIN_EPS)
  {
    // Update boundary condition
    t += dt;

    // Define boundary conditions
    DirichletBC bc(V, zero, boundary);

    // Compute step
    begin("Step forward");
    assemble(b, L);
    
    //fdm.t = t-dt;
    //assemble(bfdm, Lfdm);
    //b +=  bfdm;

    // impulse source
    Point p(2.0,2.0,2.0);
    //Point p(0.5,0.5,0.5);
    double magnitude = dt*dt*sin(2*DOLFIN_PI*15.0*(t-dt))* 
                         exp(-DOLFIN_PI*DOLFIN_PI*225.0*(t-dt)*(t-dt)/4);
    cout<< "magnitude : "<< magnitude<<endl;
    PointSource ps(V, p, magnitude);
    ps.apply(b);

    bc.apply(A, b);

    Timer tsolver("solver");
    tsolver.start();
    solve(A, *u.vector(), b, "cg", prec);
    double tmp = tsolver.stop();
    cout<<"solver time: "<<tmp<<endl;

    //solve(A, *u.vector(), b);
    end();


    // Save to file
    //ushow.interpolate(u);
    //ufile << ushow;
    
    if(int(t/dt)%20 ==0)
    {
       // fdm.t = t;
       // rcv_u.interpolate(fdm);
       // rcv_u  =rcv_u + u;
       // ufile << rcv_u;
        ufile << u;
        cout<<"save time: "<<t<<endl;
    }


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

