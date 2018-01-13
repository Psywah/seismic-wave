
#include <dolfin.h>
#include "Mass.h"
#include "Stiff.h"
//#include "ErrorNorm.h"

using namespace dolfin;

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

double _magnitude(double _t)
{
#define COEF_a (DOLFIN_PI*DOLFIN_PI*225.0/4.0)
#define COEF_b 100.0
    //return (_t<=0)? 0 : (exp(-COEF_b *_t ) *
    //                     (-COEF_b*sin(COEF_a*_t) - COEF_a* cos(COEF_a*_t)) +COEF_a) 
    //                    /(COEF_a*COEF_a + COEF_b*COEF_b);
    return (_t<=0)? 0 : (1-exp(-COEF_a *_t*_t))/(2*COEF_a)*COEF_b;
}



int main()
{
  // Print log messages only from the root process in parallel
  parameters["std_out_all_processes"] = true;

  // mesh 
  double DomainSize = 2, dx =1.0/64.0;
  std::size_t nx = DomainSize/dx;
  cout << " Number of point per direction: "<<nx<<endl;
  //BoxMesh mesh(0.0,0.0,0.0, DomainSize,DomainSize,DomainSize, nx, nx, nx);
  RectangleMesh mesh(0.0,0.0, DomainSize,DomainSize, nx,nx);

  // Set parameter values
  double dt = dx*dx/15;
  double T = 0.7;
  
  // Create FunctionSpace
  Mass::FunctionSpace W(mesh);
  Mass::CoefficientSpace_c V(mesh);
  cout << " Dim of FunctionSpace: "<< W.dim()<<endl;



  // Create functions
  Function w(W);
  Function w0(W);
  Function c(V);
  Function f(V);

  Coeff coeff;
  Source src(0.0);
  c.interpolate(coeff);
  f.interpolate(src);


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
  Parameters para("lu_solver");
  para.add("symmetric", true);
  para.add("reuse_factorization", true);
  solver.update_parameters(para);

  // Create files for storing solution
  File ufile("./results/uh.pvd");
  ufile << w0[1];

  // Use amg preconditioner if available
  list_krylov_solver_preconditioners();
  //const std::string prec(has_krylov_solver_preconditioner("ml_amg") ? "ml_amg" : "default");
  const std::string prec("jacobi");
  cout<< "Using Preconditioner: "<< prec<<endl;

  // Time-stepping
  double t = dt;

  // Create vectors
  Vector F;
  assemble(F,L);
  Vector midw[4],b[4];

  Point _p(DomainSize/2+1e-6,DomainSize/2+1e-6*2, DomainSize/2+1e-6*3);

  while (t <= T + DOLFIN_EPS*10)
  {
    // Compute step
    begin("Step forward");
    Timer Tsolver("solver");
    
    // forward Euler 
    /*
    //src.t = t-dt;
    //f.interpolate(src);
    //assemble(F, L);
    PointSource ps(W[1],_p,_magnitude(t-dt));
    F.zero();
    ps.apply(F);
    A.mult(*w0.vector(),b[0]);
    b[0] += F;

    Tsolver.start();
    //solver.solve(*w.vector(),b[0]);
    solve(*_M, *w.vector(), b[0], "cg", prec);
    *w.vector() *=dt;
    *w.vector() +=*w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;
    */
    
    
    
    // Explicit midpoint method
    //src.t = t-dt;
    //f.interpolate(src);
    //assemble(F, L);
    PointSource ps(W[1],_p,_magnitude(t-dt));
    F.zero();
    ps.apply(F);
    A.mult(*w0.vector(),b[0]);
    b[0] += F;

    Tsolver.start();
    solver.solve(midw[1],b[0]);
    //solve(*_M, midw[1], b[0],"cg",prec);
    midw[1]*=dt/2;
    midw[1] += *w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;

    //src.t = t -dt/2;
    //f.interpolate(src);
    //assemble(F, L);
    PointSource ps1(W[1],_p,_magnitude(t-dt/2));
    F.zero();
    ps1.apply(F);
    A.mult(midw[1],b[1]);
    b[1] += F;
    

    Tsolver.start();
    solver.solve(*w.vector(),b[1]);
    //solve(*_M, *w.vector(), b[1],"cg",prec);
    *w.vector() *=dt;
    *w.vector() +=*w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;
    
    
    
    
    // Heun's method
    /* 
    //src.t = t-dt;
    //f.interpolate(src);
    //assemble(F, L);
    PointSource ps(W[1],_p,_magnitude(t-dt));
    F.zero();
    ps.apply(F);
    A.mult(*w0.vector(),b[0]);
    b[0] += F;

    Tsolver.start();
    //solver.solve(midw[1],b[0]);
    solve(*_M, midw[1], b[0], "cg", prec);
    midw[1]*=dt;
    midw[1] += *w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;

    //src.t = t;
    //f.interpolate(src);
    //assemble(F, L);
    PointSource ps1(W[1],_p,_magnitude(t));
    F.zero();
    ps1.apply(F);
    A.mult(midw[1],b[1]);
    b[1] += F;
    b[1] +=b[0];

    Tsolver.start();
    //solver.solve(*w.vector(),b[1]);
    solve(*_M, *w.vector(), b[1], "cg", prec);
    *w.vector() *=dt/2;
    *w.vector() +=*w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;
    */
    
    

    // classic fourth-order method
    /*//src.t = t-dt;
    //f.interpolate(src);
    //assemble(F, L);
    PointSource ps1(W[1],_p,_magnitude(t-dt));
    F.zero();
    ps.apply(F);
    A.mult(*w0.vector(),b[0]);
    b[0] += F;

    Tsolver.start();
    //solver.solve(midw[1],b[0]);
    solve(*_M, midw[1], b[0], "cg", prec);
    midw[1]*=dt/2;
    midw[1] += *w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;

    //src.t = t-dt/2;
    //f.interpolate(src);
    //assemble(F, L);
    PointSource ps1(W[1],_p,_magnitude(t-dt/2));
    F.zero();
    ps1.apply(F);
    A.mult(midw[1],b[1]);
    b[1] += F;

    Tsolver.start();
    //solver.solve(midw[2],b[1]);
    solve(*_M, midw[2], b[1]);
    midw[2] *=dt/2;
    midw[2] +=*w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;
   
    //src.t = t-dt/2;
    //f.interpolate(src);
    //assemble(F, L);
    PointSource ps2(W[1],_p,_magnitude(t-dt/2));
    F.zero();
    ps2.apply(F);
    A.mult(midw[2],b[2]);
    b[2] += F;

    Tsolver.start();
    //solver.solve(midw[3],b[2]);
    solve(*_M, midw[3], b[2], "cg", prec);
    midw[3] *=dt;
    midw[3] +=*w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;

    //src.t = t;
    //f.interpolate(src);
    //assemble(F, L);
    PointSource ps3(W[1],_p,_magnitude(t));
    F.zero();
    ps3.apply(F);
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

    if(int(t/dt)% int(T/dt/10) || t+dt>T-DOLFIN_EPS*10)
    {
        ufile << w[1];
        cout<<"save time: "<< t <<endl;
    }

    // Move to next time step
    w0 = w;
    cout << "t = " << t << "  T =  "<<  T+DOLFIN_EPS  <<endl;
    t +=dt;
  }


  // Plot solution
  plot(w[1], "Displacement");
    
  interactive();
  

  return 0;
}

