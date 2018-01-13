#include <dolfin.h>
#include "Mass.h"
#include "Stiff.h"
#include "Penlty.h"
#include "MassSgm.h"
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
          //values[0] = (1 - exp(- COEF_a *(t-R)*(t-R)))/(4*DOLFIN_PI*R)*COEF_b/(2*COEF_a);
          values[0] = (COEF_b - 2*COEF_a*COEF_b*(t-R)*(t-R) )*
                        exp(-COEF_a*(t-R)*(t-R)) /(4*DOLFIN_PI*R);

      }
  }

  // Current time
  double t;
};

double _magnitude(double _t)
{
    return (_t<=0)? 0 : (1-exp(-COEF_a *_t*_t))/(2*COEF_a)*COEF_b;
}


int main()
{
  // Print log messages only from the root process in parallel
  parameters["std_out_all_processes"] = true;

  // mesh 
  double  dx =1.0/8.0;
  std::size_t nx = DomainSize/dx;
  cout << " Number of point per direction: "<<nx<<endl;
  BoxMesh mesh(0.0,0.0,0.0, DomainSize,DomainSize,DomainSize, nx, nx, nx);

  // Set parameter values
  double dt = dx/40;
  double T = 0.4;
  Point _p(DomainSize/2+1e-6,DomainSize/2+1e-6*2, DomainSize/2+1e-6*3);
  
  // Create FunctionSpace
  Mass::FunctionSpace W(mesh);
  Penlty::FunctionSpace Q(mesh);
  Mass::CoefficientSpace_c V(mesh);
  cout << " Dim of FunctionSpace (Mixed): "<< W.dim()<<endl;
  cout << " Dim of FunctionSpace (q): "<< Q.dim()<<endl;
  cout << " Dim of FunctionSpace (u): "<< V.dim()<<endl;


  // Define values for boundary conditions
  Constant zero(0.0);
  Constant vector_zero(0.0,0.0,0.0);


  // Create functions
  Function w(W);
  Function w0(W);
  Function sgm(Q);
  Function sgm0(Q);
  Function u(V);
  Function c(V);

  Coeff coeff;
  Source src(0.0);
  c.interpolate(coeff);
  sgm0.interpolate(vector_zero);
  assign(reference_to_no_delete_pointer(w0[0]), reference_to_no_delete_pointer(sgm0));
  u.interpolate(zero);
  assign(reference_to_no_delete_pointer(w0[1]), reference_to_no_delete_pointer(u));
  

  // Create forms
  Mass::BilinearForm m(W,W);
  Stiff::BilinearForm a(W,W);
  Penlty::BilinearForm pnl(Q,Q);
  MassSgm::BilinearForm mSgm(Q,Q);


  fdm::LinearForm Lfdm(W);
  Function c0(V);
  c0.interpolate(Constant(1.0));
  AnalyticalSolution fdm(0.0);
  Function  rcv_u(V);

  // Set coefficients
  Lfdm.f = src; Lfdm.c = c; Lfdm.c0=c0;
  m.c = c;
  mSgm.c =c;


  // Assemble matrices
  std::shared_ptr<Matrix> _M(new Matrix), _A(new Matrix), _PNL(new Matrix), _Msgm(new Matrix);
  Timer Tasmb("assemble");
  cout<< "Assemble stiff matrix..." <<endl;
  Tasmb.start();
  assemble(*_A, a);
  cout << " Assemble time: "<< Tasmb.stop()<<endl;
  Tasmb.start();
  cout<< "Assemble mass matrix..." <<endl;
  assemble(*_M, m);
  cout << " Assemble time: "<< Tasmb.stop()<<endl;
  Tasmb.start();
  cout<< "Assemble penlty matrix..." <<endl;
  assemble(*_PNL, pnl);
  cout << " Assemble time: "<< Tasmb.stop()<<endl;
  Tasmb.start();
  cout<< "Assemble mass matrix (Sigma)..." <<endl;
  assemble(*_Msgm, mSgm);
  cout << " Assemble time: "<< Tasmb.stop()<<endl;

  // LU solver for block diagonal matrix
  LUSolver Msolver(_M), MsgmSolver(_Msgm);
  Parameters para("lu_solver");
  para.add("symmetric", true);
  para.add("reuse_factorization", true);
  Msolver.update_parameters(para);
  MsgmSolver.update_parameters(para);


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
  Vector vec_f;
  _M->init_vector(vec_f,0);
  Vector midw[4],b[4];
  Matrix MPP; // mass plus penlty
  Vector bsgm;

  LUSolver MPPsolver;

  while (t <= T + DOLFIN_EPS*10)
  {
    // Compute step
    begin("Step forward");
    Timer SolverTimer("solver");
    
    // 1st order 
    // split 1: forward Euler
    /*if(t==dt) {
     MPP= *_PNL; MPP *= dt; MPP += *_Msgm;
     MPPsolver.set_operator(reference_to_no_delete_pointer(MPP));
     MPPsolver.update_parameters(para);
     }
    _Msgm->mult(*sgm0.vector(),bsgm);

    SolverTimer.start();
    MPPsolver.solve(*sgm.vector(),bsgm);
    //solve(MPP, *sgm.vector(), bsgm, "cg", "jacobi");
    cout<<"solver time: "<<SolverTimer.stop()<<endl;
    assign(reference_to_no_delete_pointer(w0[0]), reference_to_no_delete_pointer(sgm));

    // split 2: forward Euler
    //src.t = t-dt;
    //f.interpolate(src);
    //assemble(vec_f, L);
    PointSource ps(W[1],_p,_magnitude(t-dt));
    vec_f.zero();
    ps.apply(vec_f);

    _A->mult(*w0.vector(),b[0]);
    b[0] += vec_f;

    SolverTimer.start();
    Msolver.solve(*w.vector(),b[0]);
    //solve(*_M, *w.vector(), b[0], "cg", prec);
    *w.vector() *=dt;
    *w.vector() +=*w0.vector();
    cout<<"solver time: "<<SolverTimer.stop()<<endl;
    */
    

    // 2nd order
    // split 1: implicit midpoint 
    if(t==dt) {
     MPP= *_PNL; MPP *= dt/4; MPP += *_Msgm;
     MPPsolver.set_operator(reference_to_no_delete_pointer(MPP));
     MPPsolver.update_parameters(para);
     }
    _Msgm->mult(*sgm0.vector(),bsgm);

    SolverTimer.start();
    //MPPsolver.solve(*sgm.vector(), bsgm);
    solve(MPP, *sgm.vector(), bsgm, "cg", "jacobi");
    cout<<"solver time: "<<SolverTimer.stop()<<endl;
    _PNL->mult(*sgm.vector(),bsgm);
    bsgm *= -dt/2;
    MsgmSolver.solve(*sgm.vector(),bsgm);
    *sgm.vector() += *sgm0.vector();
    assign(reference_to_no_delete_pointer(w0[0]), reference_to_no_delete_pointer(sgm));

    // split 2: Explicit midpoint 
    src.t = t-dt;
    assemble(vec_f, Lfdm);
    //PointSource ps(W[1],_p,_magnitude(t-dt));
    //vec_f.zero();
    //ps.apply(vec_f);
    _A->mult(*w0.vector(),b[0]);
    b[0] += vec_f;

    SolverTimer.start();
    Msolver.solve(midw[1],b[0]);
    //solve(*_M, midw[1], b[0],"cg",prec);
    midw[1]*=dt/2;
    midw[1] += *w0.vector();
    cout<<"solver time: "<<SolverTimer.stop()<<endl;

    src.t = t -dt/2;
    assemble(vec_f, Lfdm);
    //PointSource ps1(W[1],_p,_magnitude(t-dt/2));
    //vec_f.zero();
    //ps1.apply(vec_f);
    _A->mult(midw[1],b[1]);
    b[1] += vec_f;
    

    SolverTimer.start();
    Msolver.solve(*w.vector(),b[1]);
    //solve(*_M, *w.vector(), b[1],"cg",prec);
    *w.vector() *=dt;
    *w.vector() +=*w0.vector();
    cout<<"solver time: "<<SolverTimer.stop()<<endl;

    // split 3: implicit midpoint
    assign(reference_to_no_delete_pointer(sgm0), reference_to_no_delete_pointer(w[0]));
    _Msgm->mult(*sgm0.vector(),bsgm);

    SolverTimer.start();
    //MPPsolver.solve(*sgm.vector(), bsgm);
    solve(MPP, *sgm.vector(), bsgm, "cg", "jacobi");
    cout<<"solver time: "<<SolverTimer.stop()<<endl;
    _PNL->mult(*sgm.vector(),bsgm);
    bsgm *= -dt/2;
    MsgmSolver.solve(*sgm.vector(),bsgm);
    *sgm.vector() += *sgm0.vector();
    assign(reference_to_no_delete_pointer(w[0]), reference_to_no_delete_pointer(sgm));
    
    
    
    
    end();

    
    if(int(t/dt) % (int)(T/dt/10) ==0 || t> T-DOLFIN_EPS*10)
    {
        //fdm.t =t;
        //rcv_u.interpolate(fdm);
        //assign(reference_to_no_delete_pointer(u), reference_to_no_delete_pointer(w[1]));
        //u = u + rcv_u;
        //ufile << u;
        ufile << w[1];
        cout<<"save time: "<< t <<endl;
    }
    
    


    // Move to next time step
    assign(reference_to_no_delete_pointer(sgm0), reference_to_no_delete_pointer(w[0]));
    w0 = w;
    cout << "t = " << t << "  T =  "<<  T+DOLFIN_EPS  <<endl;
    t +=dt;
  }


  // Plot solution
  plot(w[1], "Displacement");
  interactive();

  return 0;
}

