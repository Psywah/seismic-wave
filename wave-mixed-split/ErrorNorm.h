
#ifndef __ERRORNORM_H
#define __ERRORNORM_H

#include <dolfin.h>
#include "ErrorNormU.h"
#include "ErrorNormSgm.h"
#include "ErrorNormDivSgm.h"

using namespace dolfin;
double errU(Expression& u, Function& uh, Mesh& mesh)
{

  ErrorNormU::CoefficientSpace_e U(mesh);
  Function Iu(U);
  Function Iuh(U);
  Function eu(U);
  Iu.interpolate(u);
  Iuh.interpolate(uh);
  eu = Iu - Iuh;
  ErrorNormU::Functional Err(mesh);
  Err.e =eu;
  return sqrt(assemble( Err));
  //cout<< "h = " << dx<<"  k = "<<dt<<"  err : "<<err<<endl;
}


double errDivSgm(Expression& divsgm, Function& sgm, Mesh& mesh)
{

  ErrorNormDivSgm::CoefficientSpace_sgm W(mesh);
  ErrorNormDivSgm::CoefficientSpace_divsgm V(mesh);
  Function Iu(V);
  Function Iuh(W);
  Iu.interpolate(divsgm);
  Iuh.interpolate(sgm);
  ErrorNormDivSgm::Functional Err(mesh);
  Err.sgm =Iuh;
  Err.divsgm =Iu;
  return sqrt(assemble( Err));
  //cout<< "h = " << dx<<"  k = "<<dt<<"  err : "<<err<<endl;
}



double errSgm(Expression& u, Function& uh, Mesh& mesh)
{

  ErrorNormSgm::CoefficientSpace_e U(mesh);
  Function Iu(U);
  Function Iuh(U);
  Function eu(U);
  Iu.interpolate(u);
  Iuh.interpolate(uh);
  eu = Iu - Iuh;
  ErrorNormSgm::Functional Err(mesh);
  Err.e =eu;
  return sqrt(assemble( Err));
  //cout<< "h = " << dx<<"  k = "<<dt<<"  err : "<<err<<endl;
}


#endif
