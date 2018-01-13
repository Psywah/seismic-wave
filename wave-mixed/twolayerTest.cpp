
#include <dolfin.h>
#include "Mass.h"
#include "Stiff.h"
//include "ErrorNorm.h"
#include "fdm.h"

using namespace dolfin;

#define COEF_a (DOLFIN_PI*DOLFIN_PI*225.0/4.0)
#define COEF_b 100.0
#define DomainSize 1.2 

// Define the analytical solution  
class AnalyticalSolution : public Expression
{
public:

  // Constructor
  AnalyticalSolution(double _t) : t(_t) {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { 
      double R = sqrt( (x[0] - DomainSize/2-1e-6) *(x[0] - DomainSize/2-1e-6) +
                       (x[1] - DomainSize/2-1e-6*2) *(x[1] - DomainSize/2-1e-6*2) +
                       (x[2] - DomainSize/2-1e-6*3) *(x[2] - DomainSize/2-1e-6*3)); 
      values[0] = exp(- COEF_a *(t-R)*(t-R))*(t-R)/(4*DOLFIN_PI*R)*COEF_b;
  }

  // Current time
  double t;
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
      //values[0] = 1.0;
      if(x[2] > DomainSize/2+0.2)
      {
          values[0] = 0.8;
      }
      else
      {
          values[0] = 1.0;
      }
      
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
      double R = sqrt( (x[0] - DomainSize/2-1e-6) *(x[0] - DomainSize/2-1e-6) +
                       (x[1] - DomainSize/2-1e-6*2) *(x[1] - DomainSize/2-1e-6*2) +
                       (x[2] - DomainSize/2-1e-6*3) *(x[2] - DomainSize/2-1e-6*3)); 
      if (t<R){
            values[0] = 0; 
      } else
      {
          values[0] = (1 - exp(- COEF_a *(t-R)*(t-R)))/(4*DOLFIN_PI*R)*COEF_b/(2*COEF_a);
      }
  }

  // Current time
  double t;
};

double _magnitude(double _t)
{
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
  double dx =1.0/16.0;
  std::size_t nx = DomainSize/dx;
  cout << " Number of point per direction: "<<nx<<endl;
  BoxMesh mesh(0.0,0.0,0.0, DomainSize,DomainSize,DomainSize, nx, nx, nx);
  //RectangleMesh mesh(0.0,0.0, DomainSize,DomainSize, nx,nx);

  // Set parameter values
  double dt = dx*dx/10;
  double T = 0.4;
  
  // Create FunctionSpace
  Mass::FunctionSpace W(mesh);
  Mass::CoefficientSpace_c V(mesh);
  cout << " Dim of FunctionSpace: "<< W.dim()<<endl;


  // Create functions
  Function w(W);
  Function w0(W);
  Function c(V);

  
  Coeff coeff;
  Source src(0.0);
  c.interpolate(coeff);

  // Create forms
  Mass::BilinearForm m(W,W);
  Stiff::BilinearForm a(W,W);
  Stiff::LinearForm L(W);

  fdm::LinearForm Lfdm(W);
  AnalyticalSolution fdm(0.0);
  
  Function c_fdm(V);
  c_fdm.interpolate(Constant(1.0));
  c_fdm = c_fdm-c;
  c_fdm = c_fdm*(-1.0); 
  Function  rcv_u(V);

  // Set coefficients
  m.c = c;
  Lfdm.c = c_fdm;
  Lfdm.f = src;


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
  assemble(F,Lfdm);
  Vector midw[4],b[4];

  //Point _p(0.5+dx/3*2,0.5+dx/2);
  Point _p(DomainSize/2+1e-6,DomainSize/2+1e-6*2, DomainSize/2+1e-6*3);

  while (t <= T + DOLFIN_EPS*10)
  {
    // Compute step
    begin("Step forward");
    Timer Tsolver("solver");
    
    // forward Euler 
    //src.t = t-dt;
    //assemble(F, Lfdm);
    
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
    
    
    
    
    // Explicit midpoint method
    /*src.t = t-dt;
    assemble(F, Lfdm);
    //PointSource ps(W[1],_p,_magnitude(t-dt));
    //F.zero();
    //ps.apply(F);
    A.mult(*w0.vector(),b[0]);
    b[0] += F;

    Tsolver.start();
    //solver.solve(midw[1],b[0]);
    solve(*_M, midw[1], b[0],"cg",prec);
    midw[1]*=dt/2;
    midw[1] += *w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;

    src.t = t -dt/2;
    assemble(F, Lfdm);
    //PointSource ps1(W[1],_p,_magnitude(t-dt/2));
    //F.zero();
    //ps1.apply(F);
    A.mult(midw[1],b[1]);
    b[1] += F;
    

    Tsolver.start();
    //solver.solve(*w.vector(),b[1]);
    solve(*_M, *w.vector(), b[1],"cg",prec);
    *w.vector() *=dt;
    *w.vector() +=*w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;
    */
    
    
    
    
    // Heun's method
    /* 
    src.t = t-dt;
    assemble(F, Lfdm);
    //PointSource ps(W[1],_p,_magnitude(t-dt));
    //F.zero();
    //ps.apply(F);
    A.mult(*w0.vector(),b[0]);
    b[0] += F;

    Tsolver.start();
    //solver.solve(midw[1],b[0]);
    solve(*_M, midw[1], b[0], "cg", prec);
    midw[1]*=dt;
    midw[1] += *w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;

    src.t = t;
    assemble(F, Lfdm);
    //PointSource ps1(W[1],_p,_magnitude(t));
    //F.zero();
    //ps1.apply(F);
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
    /*src.t = t-dt;
    assemble(F, Lfdm);
    //PointSource ps1(W[1],_p,_magnitude(t-dt));
    //F.zero();
    //ps.apply(F);
    A.mult(*w0.vector(),b[0]);
    b[0] += F;

    Tsolver.start();
    //solver.solve(midw[1],b[0]);
    solve(*_M, midw[1], b[0], "cg", prec);
    midw[1]*=dt/2;
    midw[1] += *w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;

    src.t = t-dt/2;
    assemble(F, Lfdm);
    //PointSource ps1(W[1],_p,_magnitude(t-dt/2));
    //F.zero();
    //ps1.apply(F);
    A.mult(midw[1],b[1]);
    b[1] += F;

    Tsolver.start();
    //solver.solve(midw[2],b[1]);
    solve(*_M, midw[2], b[1]);
    midw[2] *=dt/2;
    midw[2] +=*w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;
   
    src.t = t-dt/2;
    assemble(F, Lfdm);
    //PointSource ps2(W[1],_p,_magnitude(t-dt/2));
    //F.zero();
    //ps2.apply(F);
    A.mult(midw[2],b[2]);
    b[2] += F;

    Tsolver.start();
    //solver.solve(midw[3],b[2]);
    solve(*_M, midw[3], b[2], "cg", prec);
    midw[3] *=dt;
    midw[3] +=*w0.vector();
    cout<<"solver time: "<<Tsolver.stop()<<endl;

    src.t = t;
    assemble(F, Lfdm);
    //PointSource ps3(W[1],_p,_magnitude(t));
    //F.zero();
    //ps3.apply(F);
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


    // Move to next time step
    w0 = w;
    cout << "t = " << t << "  T =  "<<  T+DOLFIN_EPS  <<endl;

    if(int(t/dt)%32 ==0 || t+dt>T-DOLFIN_EPS*10)
    {
        fdm.t =t;
        rcv_u.interpolate(fdm);
        assign(reference_to_no_delete_pointer(w[1]), reference_to_no_delete_pointer(rcv_u));
        w = w +w0;
        ufile <<w[1];
        cout<<"save time: "<< t <<endl;
    }

    t +=dt;
  }

  // Error
  /*AnalyticalSolution anlt_u(t-dt);
  cout<< "h = " << dx<<"  k = "<<dt<<"  errU : "<< errU(anlt_u, w[1], mesh)<<endl;
  AnalyticalSolutionSgm anlt_sgm(t-dt);
  cout<< "h = " << dx<<"  k = "<<dt<<"  errSgm : "<< errSgm(anlt_sgm,w[0],mesh)<<endl;
  DivSgm divsgm(t-dt);
  cout<< "h = " << dx<<"  k = "<<dt<<"  errDivSgm : "<< errDivSgm(divsgm, w[0], mesh)<<endl;
  */

  // Plot solution
  plot(w[1], "Displacement");
    
  interactive();
  

  return 0;
}

