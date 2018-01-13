
#include <dolfin.h>
#include "wave.h"

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
      double tmp[3]= {1/sqrt(3), 1/sqrt(3), 1/sqrt(3)};
      values[0] = cos(2*DOLFIN_PI*4.0*(t - (x[0]*tmp[0] +x[1]* tmp[1] + x[2]*tmp[2])/2.5));
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
      values[0] = -2*DOLFIN_PI*4.0*
          sin(2*DOLFIN_PI*4.0*( - (x[0]*tmp[0] +x[1]* tmp[1] + x[2]*tmp[2])/2.5));
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

double _magnitude(double _t)
{
#define COEF_a (DOLFIN_PI*DOLFIN_PI*225.0/4.0)
#define COEF_b 100.0
    return (_t<=0)? 0 : (exp(-COEF_a *_t*_t))*_t*COEF_b;
}
int main()
{
  // Print log messages only from the root process in parallel
  //parameters["std_out_all_processes"] = false;

  // mesh 
  double DomainSize =2.0, dx =1.0/50.0;
  std::size_t nx = DomainSize/dx;
  RectangleMesh mesh(0.0,0.0, DomainSize,DomainSize, nx,nx);

  // Create FunctionSpace
  wave::FunctionSpace V(mesh);
  wave::CoefficientSpace_c DG(mesh);

  
  // Set parameter values
  double dt = dx/40;
  double T = 0.70;

  // Define values for boundary conditions
  Constant zero(0);


  // Create functions
  Function u(V);
  Function u1(V);
  Function u2(V);
  Function c(DG);
  c.interpolate(Constant(1.0));
  u1.interpolate(zero);
  u2.interpolate(zero);;

  // Create coefficients
  Constant k(dt);
  Constant f(0);
  Constant g(0);

  // Create forms
  wave::BilinearForm a(V,V);
  wave::LinearForm L(V);

  // Set coefficients
  L.c = c;
  L.k = k; L.u1 = u1; L.u2 = u2; L.f = f; L.g = g;

  // Assemble matrices
  Matrix A;
  assemble(A, a);

  // Create vectors
  Vector b;

  // Use amg preconditioner if available
  list_krylov_solver_preconditioners();
  //const std::string prec(has_krylov_solver_preconditioner("ml_amg") ? "ml_amg" : "default");
  const std::string prec("jacobi");
  cout<< "Using Preconditioner: "<< prec<<endl;

  // Create files for storing solution
  File ufile("results/uh.pvd");
  ufile <<u1;

  // Time-stepping
  double t = dt;
  Point _p(DomainSize/2,DomainSize/2);
  while (t < T + DOLFIN_EPS*10)
  {

    // Compute step
    begin("Step forward");
    assemble(b, L);

    PointSource ps(V,_p,dt*dt*_magnitude(t-dt));
    ps.apply(b);
    solve(A, *u.vector(), b, "cg", prec);
    //solve(A, *u.vector(), b);
    end();


    // Save to file
    if(int(t/dt)%int(T/dt/10)==0 || t> T-DOLFIN_EPS*10)
    {
        ufile << u;
    }

    // Move to next time step
    u2 = u1;
    u1 = u;
    cout << "t = " << t << endl;
    t+=dt;
  }

  // Plot solution
  plot(u, "Displacement");
  interactive();

  return 0;
}

