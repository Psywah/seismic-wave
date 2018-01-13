
#include <dolfin/log/LogManager.h>
#include <dolfin/log/log.h>
#include <dolfin.h>
#include "Mass2d.h"
#include "Stiff2d.h"

using namespace dolfin;

#define Xsize 8.0
#define Ysize 8.0
#define Zsize 8.0
#define Hstep 20.e-3
#define Tstep 1.7e-3;
#define EndTime 1.0
#define F0 16




// Define Coefficient c
class Coeff_lam : public Expression
{
public:

  // Constructor
  Coeff_lam()  {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { 
      values[0] = 1.0;
      if(x[1] < 3.2)
      {
          values[0] = 11;
      }else
      {
          values[0] = 1.5;
      }
  }

  // Current time
};
// Define Coefficient c
class Coeff_mu : public Expression
{
public:

  // Constructor
  Coeff_mu()  {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { 
      values[0] = 1.0;
      if(x[1] < 3.2)
      {
          values[0] = 15.0;
      }else
      {
          values[0] = 2.5;
      }
  }

  // Current time
};
// Define Coefficient c
class Coeff_rho : public Expression
{
public:

  // Constructor
  Coeff_rho()  {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { 
      values[0] = 1.0;
      if(x[1] < 3.2)
      {
          values[0] = 2.0;
      }else
      {
          values[0] = 1.5;
      }
  }

  // Current time
};


// Define Source term 
class Source : public Expression
{
public:

  // Constructor
  Source(double _t) : Expression(2),t(_t) {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { 
      values[0] = 0.0;
      values[1] = 0.0;
  }

  // Current time
  double t;
};

double wavelet(double t)
{
    double tmp = (0.6*F0*t-1)*(0.6*F0*t-1);
    return (t<0)? 0: -5.76*F0*F0*(1-16*tmp)*exp(-8*tmp)/1000;
}

double GaussQuad1d(double a, double b)
{
    double x[5]= {0.0000000000000000,
                 -0.5384693101056831,
                 0.5384693101056831,
                 -0.9061798459386640,
                 0.9061798459386640
    };
    double w[5] = {0.5688888888888889,
                   0.4786286704993665,
                   0.4786286704993665,
                   0.2369268850561891,
                   0.2369268850561891
    };
    double tmp_sum=0.0, size =(b-a)/2., original = (b+a)/2.;
    for(int i=0; i < 5; ++i)
        tmp_sum +=w[i]*wavelet(x[i]*size+original);
    return tmp_sum * size;
}



int main()
{
  // Print log messages only from the root process in parallel
  parameters["std_out_all_processes"] = true;
  parameters["linear_algebra_backend"] ="PETSc";
  parameters["ghost_mode"] = "shared_facet";
  LogManager::logger().set_log_active(dolfin::MPI::rank(MPI_COMM_WORLD) == 0);

  // mesh 
  double DomainSize = Xsize, dx =Hstep;
  std::size_t nx = Xsize/Hstep;
  info(" Number of point per direction: %d. ", nx);
  //BoxMesh mesh(0.0,0.0,0.0, DomainSize,DomainSize,DomainSize, nx, nx, nx);
  Point p1(0.,0.), p2(DomainSize, DomainSize);
  RectangleMesh mesh(p1, p2, nx,nx);

  // Set parameter values
  double dt = Tstep;
  double T = EndTime;
  
  // Create FunctionSpace
  Mass2d::FunctionSpace W(reference_to_no_delete_pointer(mesh));
  Mass2d::CoefficientSpace_lam V_coeff(reference_to_no_delete_pointer(mesh));
  info(" Dim of FunctionSpace: %d. ", W.dim());



  // Create functions
  Function w(reference_to_no_delete_pointer(W));
  Function w0(reference_to_no_delete_pointer(W));

  Coeff_lam coeff_lam;
  Coeff_mu coeff_mu;
  Coeff_rho coeff_rho;
  Source src(0.0);


  // Create forms
  Mass2d::BilinearForm m(reference_to_no_delete_pointer(W),reference_to_no_delete_pointer(W));
  Stiff2d::BilinearForm a(reference_to_no_delete_pointer(W),reference_to_no_delete_pointer(W));
  Stiff2d::LinearForm L(reference_to_no_delete_pointer(W));

  // Set coefficients
  L.f = reference_to_no_delete_pointer(src);
  m.lam = reference_to_no_delete_pointer(coeff_lam);
  m.mu = reference_to_no_delete_pointer(coeff_mu);
  m.rho = reference_to_no_delete_pointer(coeff_rho);


  // Assemble matrices
  std::shared_ptr<Matrix> _M(new Matrix);
  Matrix A;
  Timer Tasmb("assemble");
  info("Assemble stiff matrix...");
  Tasmb.start();
  assemble(A, a);
  info(" Assemble time: %.2f s", Tasmb.stop());
  Tasmb.start();
  info("Assemble mass matrix...");
  assemble(*_M, m);
  info(" Assemble time: %.2f s", Tasmb.stop());

  LUSolver solver(_M,"superlu_dist");
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
  info("Using Preconditioner: %s", prec.c_str());

  // Time-stepping
  double t = dt;

  // Create vectors
  Vector F;
  assemble(F,L);
  Vector midw[4],b[4];

  Point _p(4.+1e-6,2.88+1e-6*2, DomainSize/2+1e-6*3);
  double _magnitude=0.0;

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
    PointSource ps(W[1],_p,_magnitude);
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
    PointSource ps1(W[1],_p,_magnitude+GaussQuad1d(t-dt, t-dt/2));
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

    if((int(t/dt)% int(T/dt/10))==0 || t+dt>T-DOLFIN_EPS*10)
    {
        ufile << w[1];
        info("save time: %f",t);
    }

    // Move to next time step
    w0 = w;
    info("t = %f s, T = %f s. Norm u+p %f", t, T+DOLFIN_EPS, w.vector()->norm("l2"));
    _magnitude+= GaussQuad1d(t-dt,t);
    t +=dt;
  }

  info("solved!");

  // Plot solution
  //plot(w[1], "Displacement");
    
  //interactive();
  

  return 0;
}

