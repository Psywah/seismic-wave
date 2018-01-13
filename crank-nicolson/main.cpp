
#include <dolfin.h>
#include "CNu.h"
#include "CNv.h"
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
  InitUt(double _t) :t(_t)  {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { 
      double tmp[3]= {1/sqrt(3), 1/sqrt(3), 1/sqrt(3)};
      values[0] = -2*DOLFIN_PI*Freq*
          sin(2*DOLFIN_PI*Freq*(t - (x[0]*tmp[0] +x[1]* tmp[1] + x[2]*tmp[2])/C0));
  }

  // Current time
  double t;
};

// Define coeff  
class Coeff : public Expression
{
public:

  // Constructor
  Coeff()  {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { 
      if(x[2]>0.6){ 
        values[0] = 0.25;
      } else{
        values[0] = 0.49;      
      }
  }
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
  //double DomainSize =4.0, dx =0.05;
  double DomainSize =1.0, dx =0.02;
  std::size_t nx = DomainSize/dx;
  BoxMesh mesh(0.0,0.0,0.0, DomainSize,DomainSize,DomainSize, nx, nx, nx);

  // Create FunctionSpace
  CNu::FunctionSpace V(mesh);
  CNu::CoefficientSpace_c DG(mesh);

  
  // Set parameter values
  double dt = 0.01;
  double T = 0.40;

  // Define values for boundary conditions
  //AnalyticalSolution real_u(0.0);
  //InitUt real_ut(0.0);
  Constant zero(0);

  // Define subdomains for boundary conditions
  DirichletBoundary boundary;

  // Create functions
  Function u(V);
  Function v(V);
  Function u1(V);
  Function v1(V);
  Function c(DG);
  Coeff coeff;
  c.interpolate(coeff);
  //plot(c);
  //interactive();
  //u1.interpolate(real_u);
  u1.interpolate(zero);
  //v1.interpolate(real_ut);
  v1.interpolate(zero);

  // Create coefficients
  Constant k(dt);
  Constant eta(0.5);
  Constant f1(0);
  Constant f2(0);

  // Create forms
  CNu::BilinearForm au(V,V);
  CNu::LinearForm Lu(V);
  CNv::BilinearForm av(V,V);
  CNv::LinearForm Lv(V);

  fdm::LinearForm Lfdm(V);
  AnalyticalSolutionS fdm(0.0);
  Function c_fdm(DG);
  c_fdm.interpolate(Constant(0.49));
  c_fdm = c_fdm-c;
  c_fdm = c_fdm*(-1.0); 
  Function  rcv_u(V);
  
  // Set coefficients
  Lv.k = k; Lv.eta = eta; Lv.u1 = u1; Lv.v1= v1; Lv.u = u; Lv.f1 = f1; Lv.f2= f2; Lv.c =c;
  Lu.k = k; Lu.eta = eta; Lu.u1 = u1; Lu.v1= v1; Lu.f1 = f1; Lu.f2= f2; Lu.c=c;
  au.k = k; au.eta = eta; au.c = c;
  Lfdm.c=c_fdm;  Lfdm.uf=fdm;


  // Assemble matrices
  Matrix Au, Av;
  assemble(Au, au);
  assemble(Av, av);

  // Create vectors
  Vector bu, bv, bfdm;

  // Use amg preconditioner if available
  list_krylov_solver_preconditioners();
  //const std::string prec(has_krylov_solver_preconditioner("ml_amg") ? "ml_amg" : "default");
  const std::string prec("jacobi");
  cout<< "Using Preconditioner: "<< prec<<endl;

  // Create files for storing solution
  File ufile("results/uh.pvd");
  ufile <<u1;

  // Time-stepping
  double t = 0;
  while (t < T + DOLFIN_EPS)
  {
    t += dt;
    // Update boundary condition
    //real_u.t = t;
    //real_ut.t = t;

    // Define boundary conditions
    //DirichletBC bcu(V, real_u, boundary);
    DirichletBC bcu(V, zero, boundary);
    //DirichletBC bcv(V, real_ut, boundary);

    // Compute step
    begin("Step forward");
    assemble(bu, Lu);

    fdm.t = t-dt/2;
    assemble(bfdm, Lfdm);
    bfdm *= (dt*dt/4);

    bu +=  bfdm;
    bcu.apply(Au, bu);

    // impulse source
    //Point p(2.0,2.0,2.0);
    //Point p(0.5,0.5,0.5);
    //double magu = dt*dt/4* (
    //               sin(2*DOLFIN_PI*15.0*(t-dt))* 
    //                exp(-DOLFIN_PI*DOLFIN_PI*225.0*(t-dt)*(t-dt)/4)
    //              + 
    //                sin(2*DOLFIN_PI*15.0*(t))* 
    //                exp(-DOLFIN_PI*DOLFIN_PI*225.0*(t)*(t)/4));
    //cout<< "magnitude u: "<< magu<<endl;
    //PointSource psu(V, p, magu);
    //psu.apply(bu);
    
    Timer timer_u("usolver");
    timer_u.start();
    solve(Au, *u.vector(), bu, "cg", std::string("ml_amg"));
    double tu=timer_u.stop();


    bfdm *=  (2.0/dt);
    assemble(bv, Lv);
    bv+=bfdm;
    //bcv.apply(Av, bv);
    
    /*double magv = dt/2* (
                    sin(2*DOLFIN_PI*15.0*(t-dt))* 
                    exp(-DOLFIN_PI*DOLFIN_PI*225.0*(t-dt)*(t-dt)/4)
                  + 
                    sin(2*DOLFIN_PI*15.0*(t))* 
                    exp(-DOLFIN_PI*DOLFIN_PI*225.0*(t)*(t)/4));
    cout<< "magnitude v: "<< magv<<endl;
    PointSource psv(V, p, magu);
    psv.apply(bv);
    */

    Timer timer_v("vsolver");
    timer_v.start();
    solve(Av, *v.vector(),bv,"cg", std::string("jacobi"));
    double tv=timer_v.stop();

    cout<< "u solver time: " <<tu<<"     v solver time"<<tv<<endl;

    end();


    // Save to file
    fdm.t = t;
    rcv_u.interpolate(fdm);
    rcv_u  =rcv_u + u;
    ufile << rcv_u;
    //ufile << u;

    // Move to next time step
    u1 = u;
    v1 = v;
    cout << "t = " << t << endl;
  }

  // Plot solution
  plot(u, "Displacement");
  interactive();

  return 0;
}

