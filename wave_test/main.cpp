// Copyright (C) 2010-2011 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2010-08-30
// Last changed: 2013-03-21
//
// wave equation, see https://www.dealii.org/developer/doxygen/deal.II/step_23.html
// Begin demo

#include <dolfin.h>
#include "wave.h"

using namespace dolfin;

// Define clamp domain
class ClampDomain : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return (on_boundary &&
            (x[0] > 1.0- DOLFIN_EPS || x[1] <  -1.0 + DOLFIN_EPS || x[1] > 1.0 - DOLFIN_EPS || 
             (x[0] < -1.0 +  DOLFIN_EPS && x[1] > 0.333 - DOLFIN_EPS) ||
             (x[0] < -1.0 +  DOLFIN_EPS && x[1] < -0.333 + DOLFIN_EPS) 
             ));
  }
};

// Define wave domain
class WaveDomain : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  { 
      return (on_boundary && x[0] < -1.0 + DOLFIN_EPS &&
             x[1] > -0.333 - DOLFIN_EPS && 
             x[1] < 0.333 + DOLFIN_EPS ); 
  }
};
// Define wave domain
class Boundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  { 
      return on_boundary;
  }
};


// Define value at wave boundary  
class WaveFun : public Expression
{
public:

  // Constructor
  WaveFun() : t(0) {}

  // Evaluate displacement at wave boundary 
  void eval(Array<double>& values, const Array<double>& x) const
  { values[0] = sin(4.0*t*DOLFIN_PI); }

  // Current time
  double t;
};

int main()
{
  // Print log messages only from the root process in parallel
  parameters["std_out_all_processes"] = false;

  // mesh 
  RectangleMesh mesh(-1.0, -1.0, 1.0, 1.0, 50, 50);

  // Create function spaces
  wave::FunctionSpace V(mesh);
  wave::CoefficientSpace_c DG(mesh);

  // Set parameter values
  double dt = 0.05;
  double T = 10;

  // Define values for boundary conditions
  WaveFun wavefun;
  Constant zero(0);

  // Define subdomains for boundary conditions
  //ClampDomain clamp_domain;
  //WaveDomain wave_domain;
  Boundary boundary;

  // Define boundary conditions
  //DirichletBC clampBC(V, zero, clamp_domain);
  //DirichletBC waveBC(V, wavefun, wave_domain);
  DirichletBC bc(V, zero, boundary);
  std::vector<DirichletBC*> bcs;
  //bcs.push_back(&clampBC);
  //bcs.push_back(&waveBC);
  bcs.push_back(&bc);

  // Create functions
  Function u(V);
  Function u1(V);
  Function u2(V);
  Function c(DG);
  c.interpolate(Constant(1.0));
  u1.interpolate(Constant(0));
  u2.interpolate(Constant(0));

  // Create coefficients
  Constant k(dt);
  Constant f(0);
  Constant g(0);

  // Create forms
  wave::BilinearForm a(V,V);
  wave::LinearForm L(V);

  // Set coefficients
  a.k = k; a.c = c;
  L.k = k; L.u1 = u1; L.u2 = u2; L.f = f; L.g = g;

  // Assemble matrices
  Matrix A;
  assemble(A, a);

  // Create vectors
  Vector b;

  // Use amg preconditioner if available
  const std::string prec(has_krylov_solver_preconditioner("amg") ? "amg" : "default");

  // Create files for storing solution
  File ufile("results/u.pvd");

  // Time-stepping
  double t = dt;
  while (t < T + DOLFIN_EPS)
  {
    // Update boundary condition
    wavefun.t = t;

    // Compute step
    begin("Step forward");
    assemble(b, L);
    for (std::size_t i = 0; i < bcs.size(); i++)
      bcs[i]->apply(A, b);


    // SourcePoint
    Point p(0.5, 0.5);
    double f0 =15.0;
    double magnitude = dt*dt*sin(2*DOLFIN_PI*f0*t)* exp(-DOLFIN_PI*DOLFIN_PI*f0*f0*t*t/4);
    cout<< "magnitude : "<< magnitude<<endl;
    PointSource ps(V, p, magnitude);
    ps.apply(b);
    //solve(A, *u.vector(), b, "cg", "prec");
    solve(A, *u.vector(), b);
    end();


    // Save to file
    ufile << u;

    // Move to next time step
    u2 = u1;
    u1 = u;
    t += dt;
    cout << "t = " << t << endl;
  }

  // Plot solution
  plot(u, "Velocity");
  interactive();

  return 0;
}
