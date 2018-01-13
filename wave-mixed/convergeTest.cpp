
#include <dolfin.h>
#include "Mass.h"
#include "Stiff.h"
#include "ErrorNorm.h"

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
      values[0] = t*t*sin(DOLFIN_PI*x[0])*sin(DOLFIN_PI*x[1]);
  }

  // Current time
  double t;
};

// Define the analytical solution of Sigma 
class AnalyticalSolutionSgm : public Expression
{
public:

  // Constructor
  AnalyticalSolutionSgm(double _t) :Expression(2), t(_t) {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { 
      {values[0] = 1.0/3.0*pow(t,3)*DOLFIN_PI*cos(DOLFIN_PI*x[0])*sin(DOLFIN_PI*x[1]);}
      {values[1] = 1.0/3.0*pow(t,3)*DOLFIN_PI*sin(DOLFIN_PI*x[0])*cos(DOLFIN_PI*x[1]);}
  }

  // Current time
  double t;
  std::size_t idx;
};

// Define Coefficient c
class Coeff : public Expression
{
public:

  // Constructor
  Coeff()  {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { 
      values[0] = 1.0;
      /*if(x[2] > 0.6)
      {
          values[0] = 0.25;
      }
      else
      {
          values[0] = 0.49;
      }
      */
  }

  // Current time
};

// Define Source term 
class Source : public Expression
{
public:

  // Constructor
  Source(double _t) : t(_t) {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { 
      values[0] = (2*t +2.0/3.0*DOLFIN_PI*DOLFIN_PI*pow(t,3))
                    *sin(DOLFIN_PI*x[0])*sin(DOLFIN_PI*x[1]);
  }

  // Current time
  double t;
};

// Define div sigma 
class DivSgm : public Expression
{
public:

  // Constructor
  DivSgm(double _t) : t(_t) {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { 
      values[0] = (-2.0/3.0*DOLFIN_PI*DOLFIN_PI*pow(t,3))
                    *sin(DOLFIN_PI*x[0])*sin(DOLFIN_PI*x[1]);
  }

  // Current time
  double t;
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
  parameters["std_out_all_processes"] = true;

  // mesh 
  double DomainSize = 1.0, dx =1.0/16.0;
  std::size_t nx = DomainSize/dx;
  cout << " Number of point per direction: "<<nx<<endl;
  //BoxMesh mesh(0.0,0.0,0.0, DomainSize,DomainSize,DomainSize, nx, nx, nx);
  RectangleMesh mesh(0.0,0.0, DomainSize,DomainSize, nx,nx);

  // Set parameter values
  double dt = dx*dx/15;
  double T = 1.00;
  
  // Create FunctionSpace
  Mass::FunctionSpace W(mesh);
  Mass::CoefficientSpace_c V(mesh);
  cout << " Dim of FunctionSpace: "<< W.dim()<<endl;


  // Define values for boundary conditions
  Constant zero(0.0);
  Constant vector_zero(0.0,0.0);

  // Define subdomains for boundary conditions
  DirichletBoundary boundary;

  // Create functions
  Function w(W);
  Function w0(W);
  Function c(V);
  Function f(V);

  Coeff coeff;
  Source src(0.0);
  c.interpolate(coeff);
  f.interpolate(src);
  //w0[0].interpolate(vector_zero);
  //w0[1].interpolate(zero);

  // Create coefficients
  Constant k(dt);

  // Create forms
  Mass::BilinearForm m(W,W);
  Stiff::BilinearForm a(W,W);
  Stiff::LinearForm L(W);

  // Set coefficients
  L.f = f;
  m.c = c;


  // Assemble matrices
  std::shared_ptr<Matrix> _M(new Matrix);
  Matrix A;
  Timer Tasmb("assemble");
  cout<< "Assemble stiff matrix..." <<endl;
  Tasmb.start();
  assemble(A, a);
  cout << " Assemble time: "<< Tasmb.stop()<<endl;
  Tasmb.start();
  cout<< "Assemble mass matrix..." <<endl;
  assemble(*_M, m);
  cout << " Assemble time: "<< Tasmb.stop()<<endl;

  LUSolver solver(_M);

  // Create files for storing solution
  //File ufile("./results/uh.pvd");
  //ufile << w0[1];

  // Use amg preconditioner if available
  list_krylov_solver_preconditioners();
  //const std::string prec(has_krylov_solver_preconditioner("ml_amg") ? "ml_amg" : "default");
  const std::string prec("jacobi");
  cout<< "Using Preconditioner: "<< prec<<endl;

  // Time-stepping
  double t = dt;

  // Create vectors
  Vector F;
  Vector midw[4],b[4];

  while (t <= T + DOLFIN_EPS*10)
  {
    // Compute step
    begin("Step forward");
    Timer Tsolver("solver");
    
    // forward Euler 
    /*src.t = t-dt;
    f.interpolate(src);
    assemble(F, L);
    A.mult(*w0.vector(),b[0]);
    b[0] += F;

    Tsolver.start();
    solver.solve(*w.vector(),b[0]);
    //solve(M, *w.vector(), b[0], "cg", prec);
    *w.vector() *=dt;
    *w.vector() +=*w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;
    */
    
    
    // Explicit midpoint method
    src.t = t-dt;
    f.interpolate(src);
    assemble(F, L);
    A.mult(*w0.vector(),b[0]);
    b[0] += F;

    Tsolver.start();
    //solver.solve(midw[1],b[0]);
    solve(*_M, midw[1], b[0],"cg",prec);
    midw[1]*=dt/2;
    midw[1] += *w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;

    src.t = t -dt/2;
    f.interpolate(src);
    assemble(F, L);
    A.mult(midw[1],b[1]);
    b[1] += F;
    

    Tsolver.start();
    //solver.solve(*w.vector(),b[1]);
    solve(*_M, *w.vector(), b[1],"cg",prec);
    *w.vector() *=dt;
    *w.vector() +=*w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;
    
    
    
    // Heun's method
   /* 
    src.t = t-dt;
    f.interpolate(src);
    assemble(F, L);
    A.mult(*w0.vector(),b[0]);
    b[0] += F;

    Tsolver.start();
    //solver.solve(midw[1],b[0]);
    solve(*_M, midw[1], b[0], "cg", prec);
    midw[1]*=dt;
    midw[1] += *w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;

    src.t = t;
    f.interpolate(src);
    assemble(F, L);
    A.mult(midw[1],b[1]);
    b[1] += F;
    b[1] +=b[0];

    Tsolver.start();
    solver.solve(*w.vector(),b[1]);
    //solve(*_M, *w.vector(), b[1], "cg", prec);
    *w.vector() *=dt/2;
    *w.vector() +=*w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;
    */
    

    // classic fourth-order method
    /*src.t = t-dt;
    f.interpolate(src);
    assemble(F, L);
    A.mult(*w0.vector(),b[0]);
    b[0] += F;

    Tsolver.start();
    //solver.solve(midw[1],b[0]);
    solve(*_M, midw[1], b[0], "cg", prec);
    midw[1]*=dt/2;
    midw[1] += *w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;

    src.t = t-dt/2;
    f.interpolate(src);
    assemble(F, L);
    A.mult(midw[1],b[1]);
    b[1] += F;

    Tsolver.start();
    //solver.solve(midw[2],b[1]);
    solve(*_M, midw[2], b[1]);
    midw[2] *=dt/2;
    midw[2] +=*w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;
   
    src.t = t-dt/2;
    f.interpolate(src);
    assemble(F, L);
    A.mult(midw[2],b[2]);
    b[2] += F;

    Tsolver.start();
    //solver.solve(midw[3],b[2]);
    solve(*_M, midw[3], b[2], "cg", prec);
    midw[3] *=dt;
    midw[3] +=*w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;

    src.t = t;
    f.interpolate(src);
    assemble(F, L);
    A.mult(midw[3],b[3]);
    b[3] += F;
    b[3] *= 1.0/6; b[2] *= 1.0/3; b[1] *= 1.0/3; b[0] *= 1.0/6;
    b[3] += b[2]; b[3] += b[1]; b[3] += b[0]; 

    Tsolver.start();
    solver.solve(*w.vector(),b[3]);
    //solve(*_M, *w.vector(), b[3], "cg", prec);
    *w.vector() *=dt;
    *w.vector() +=*w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;
    */
    
    
    
    end();

    
    /*if(int(t/dt)%20 ==0)
    {
        ufile << w[1];
        cout<<"save time: "<< t <<endl;
    }
    */


    // Move to next time step
    w0 = w;
    cout << "t = " << t << "  T =  "<<  T+DOLFIN_EPS  <<endl;
    t +=dt;
  }

  // Error
  AnalyticalSolution anlt_u(t-dt);
  cout<< "h = " << dx<<"  k = "<<dt<<"  errU : "<< errU(anlt_u, w[1], mesh)<<endl;
  AnalyticalSolutionSgm anlt_sgm(t-dt);
  cout<< "h = " << dx<<"  k = "<<dt<<"  errSgm : "<< errSgm(anlt_sgm,w[0],mesh)<<endl;
  DivSgm divsgm(t-dt);
  cout<< "h = " << dx<<"  k = "<<dt<<"  errDivSgm : "<< errDivSgm(divsgm, w[0], mesh)<<endl;

  // Plot solution
  plot(w[1], "Displacement");
  interactive();

  return 0;
}

