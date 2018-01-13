
#include <dolfin/log/LogManager.h>
#include <dolfin/log/log.h>
#include <dolfin.h>
#include "Mass2d.h"
#include "Stiff2d.h"

using namespace dolfin;

#define Xsize 5.4
#define Ysize 5.4
#define Zsize 5.4
#define PMLsize 0.2
#define Hstep 25.e-3
#define Tstep (Hstep)/400;
#define EndTime 1.5
#define F0 25


// Define PML domain
class PMLDomain : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[0]<DOLFIN_EPS || x[1]<DOLFIN_EPS || x[0]>Xsize - DOLFIN_EPS || x[1]> Ysize -DOLFIN_EPS;
  }
};

// Define PML NORMAL
class PMLNormal : public Expression
{
public:

  // Constructor
  PMLNormal():Expression(2) {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { 
      double dx = x[0] - 2.7;
      double dy = x[1] - 2.7;
      double r = sqrt(dx*dx+dy*dy);
      values[0] = dx/r;
      values[1] = dy/r;

      /*
      if(x[1] < DOLFIN_EPS)
      {
          values[0] = 0.0;
          values[1] = -1.0;
      }else if(x[1]>Ysize-DOLFIN_EPS)
      {
          values[0] = 0.0;
          values[1] = 1.0;
      }else if(x[0] < DOLFIN_EPS)
      {
          values[0] = -1.0;
          values[1] = 0.0;
      }else if(x[0]>Xsize-DOLFIN_EPS)
      {
          values[0] = 1.0;
          values[1] = 0.0;
      }else
      {
          values[0] = 0.0;
          values[1] = 0.0;
      }
      */
  }

};

// Define PML profile
class PMLProfile : public Expression
{
public:

  // Constructor
  PMLProfile() {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { 
      double d=0.0;
      if(x[1] < DOLFIN_EPS)
      {
          d = -x[1];
      }else if(x[1]>Ysize-DOLFIN_EPS)
      {
          d = x[1]-Ysize;
      }else if(x[0] < DOLFIN_EPS)
      {
          d = -x[0];
      }else if(x[0]>Xsize-DOLFIN_EPS)
      {
          d = x[0]-Xsize;
      }else d=0.0;
#define P_VELOCITY 4.0
#define REFLECTION_COEFF  1.e-3
      values[0]= -log(REFLECTION_COEFF)/log(2.0)*1.5*P_VELOCITY/PMLsize*d*d/PMLsize/PMLsize;
  }

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
      if(x[1] < 1.08)
      {
          values[0] = 2.0;
      }
      else if(x[0]>2.7 || x[1] <4.32)
      {
          values[0] = 3;
      }else
      {
          values[0] = 4;
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
      values[0] = 0.0;
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
  double pmlsize = PMLsize;
  std::size_t nx = (2*PMLsize+Xsize)/Hstep;
  info(" Number of point per direction: %d. ", nx);
  //BoxMesh mesh(0.0,0.0,0.0, DomainSize,DomainSize,DomainSize, nx, nx, nx);
  Point p1(-pmlsize,-pmlsize), p2(DomainSize+pmlsize, DomainSize+pmlsize);
  RectangleMesh mesh(p1, p2, nx,nx);
  info("p1 (%f,%f), p2 (%f,%f)", p1.x(),p1.y(),p2.x(),p2.y());

  // Set parameter values
  double dt = Tstep;
  double T = EndTime;
  
  // Create FunctionSpace
  Mass2d::FunctionSpace W(reference_to_no_delete_pointer(mesh));
  Mass2d::CoefficientSpace_c V_coeff(reference_to_no_delete_pointer(mesh));
  info(" Dim of FunctionSpace: %d. ", W.dim());



  // Create functions
  Function w(reference_to_no_delete_pointer(W));
  Function w0(reference_to_no_delete_pointer(W));

  Coeff coeff;
  Source src(0.0);
  PMLProfile pml_profile;
  PMLNormal pml_normal;
  PMLDomain pml_domain;


  // Create forms
  Mass2d::BilinearForm m(reference_to_no_delete_pointer(W),reference_to_no_delete_pointer(W));
  Stiff2d::BilinearForm a(reference_to_no_delete_pointer(W),reference_to_no_delete_pointer(W));
  Stiff2d::LinearForm L(reference_to_no_delete_pointer(W));

  // Set coefficients
  L.f = reference_to_no_delete_pointer(src);
  m.c = reference_to_no_delete_pointer(coeff);
  a.c = reference_to_no_delete_pointer(coeff);
  a.d = reference_to_no_delete_pointer(pml_profile);
  a.n_pml = reference_to_no_delete_pointer(pml_normal);

  MeshFunction<std::size_t> domain_marker(reference_to_no_delete_pointer(mesh), mesh.topology().dim(),1);
  info("mark pml layer");
  pml_domain.mark(domain_marker, 2, false);
  info("mark pml layer: done");
  m.dx = reference_to_no_delete_pointer(domain_marker);
  a.dx = reference_to_no_delete_pointer(domain_marker);

  std::vector<dolfin::la_index> dirichlet_dofs;
  std::vector<double> zeros;

  info("find dirichlet dofs");
  std::shared_ptr<const GenericDofMap> dofmap1 = W[2]->dofmap();
  std::shared_ptr<const GenericDofMap> dofmap2 = W[3]->dofmap();
  for(CellIterator cell(mesh);!cell.end();++cell)
  {
      if(domain_marker[cell->index()] == 1)
      {
        const ArrayView<const dolfin::la_index> cell_dofs1 = dofmap1->cell_dofs(cell->index());
        const ArrayView<const dolfin::la_index> cell_dofs2 = dofmap2->cell_dofs(cell->index());
        for(std::size_t j =0; j <cell_dofs1.size();++j)
        {
            dirichlet_dofs.push_back(cell_dofs1[j]);
        }
        for(std::size_t j =0; j <cell_dofs2.size();++j)
        {
            dirichlet_dofs.push_back(cell_dofs2[j]);
        }
      }
  }
  zeros.resize(dirichlet_dofs.size(),0.0);
  info("find dirichlet dofs: done");
  //Constant zero(0.0);
  //Constant zeros(0.0,0.0);


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
  _M->ident_local(dirichlet_dofs.size(),dirichlet_dofs.data());
  _M->apply("insert");
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

  Point _p(DomainSize/2+1e-6,DomainSize/2+1e-6*2, DomainSize/2+1e-6*3);
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
    b[0].set_local(zeros.data(),zeros.size(),dirichlet_dofs.data());
    b[0].apply("insert");
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
    b[1].set_local(zeros.data(),zeros.size(),dirichlet_dofs.data());
    b[1].apply("insert");
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

    if((int(t/dt)% int(T/dt/15))==0 || t+dt>T-DOLFIN_EPS*10)
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

