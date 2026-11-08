// Lagged nonlinear heat, following MFEM Example 16, in backward Euler state form.
#include "mfem.hpp"
#include "native_default.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace mfem;
static double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static double initial(const Vector &x){return x.Norml2()<.5?2.:1.;}
static double residual(const SparseMatrix&A,const Vector&b,const Vector&x){Vector r(b.Size());A.Mult(x,r);r-=b;return r.Norml2()/b.Norml2();}
int main(int argc,char**argv){
  if(argc<4)return 1;
  Mesh mesh(argv[1],1,1);
  mesh.UniformRefinement();mesh.UniformRefinement();
  H1_FECollection fec(2,3);FiniteElementSpace fes(&mesh,&fec);
  int n=fes.GetTrueVSize();std::cout<<"Heat unknowns "<<n<<" tetrahedra "<<mesh.GetNE()<<std::endl;
  int cases=std::atoi(argv[3]);if(!cases)return 0;
  int cap=argc>4?std::atoi(argv[4]):50;
  GridFunction u(&fes),conductivity(&fes);FunctionCoefficient ic(initial);u.ProjectCoefficient(ic);
  BilinearForm mass(&fes);mass.AddDomainIntegrator(new MassIntegrator());mass.Assemble();mass.Finalize();
  SparseMatrix &M=mass.SpMat();
  double dt=.1,alpha=.01,kappa=.5;
  Vector b(n),x(n),ref(n);
  std::ofstream f(argv[2]);f<<std::setprecision(17);
  f<<"case,n,nnz,assembly_sec,import_sec,setup_sec,solve_sec,iterations,true_residual,converged,reference_setup_sec,reference_solve_sec,reference_residual,temperature_min,temperature_max,mass_integral\n";
  Vector ones(n);ones=1.;Vector massones(n);M.Mult(ones,massones);
  for(int i=1;i<=cases;i++){
    double start=now();
    for(int j=0;j<n;j++)conductivity[j]=kappa+alpha*u[j];
    GridFunctionCoefficient coefficient(&conductivity);
    BilinearForm stiffness(&fes);stiffness.AddDomainIntegrator(new DiffusionIntegrator(coefficient));
    stiffness.Assemble();stiffness.Finalize();
    SparseMatrix *A=Add(1.,M,dt,stiffness.SpMat());
    M.Mult(u,b);x=u;ref=u;
    double assembly=now()-start;
    double s[6],r[6];
    int code=pilot_default(n,A->GetI(),A->GetJ(),A->GetData(),b.GetData(),x.GetData(),1e-6,cap,0,s);
    double actual=residual(*A,b,x);
    int rc=pilot_default(n,A->GetI(),A->GetJ(),A->GetData(),b.GetData(),ref.GetData(),1e-10,1000,1,r);
    double rr=residual(*A,b,ref);
    if(rc || !std::isfinite(rr) || rr>1e-10){std::cerr<<"Reference failed "<<rc<<" "<<rr<<std::endl;return 2;}
    u=ref;
    f<<i<<','<<n<<','<<A->NumNonZeroElems()<<','<<assembly<<','<<s[0]<<','<<s[1]<<','<<s[2]<<','<<s[4]<<','<<actual<<','<<(code==0&&std::isfinite(actual)&&actual<=1e-6)<<','<<r[1]<<','<<r[2]<<','<<rr<<','<<u.Min()<<','<<u.Max()<<','<<(massones*u)<<'\n';f.flush();
    std::cout<<"Heat "<<i<<"/"<<cases<<" setup "<<s[1]<<" solve "<<s[2]<<" residual "<<actual<<std::endl;
    delete A;
  }
}
