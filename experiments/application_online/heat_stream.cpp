// MFEM Example 16 equation, lagged backward Euler; identical pilot mesh/physics.
// Python supplies a separately verified reference solution after all methods.
#include "mfem.hpp"
#include <chrono>
#include <thread>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <cmath>
#include <vector>
using namespace mfem;
static double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static double hot_amplitude=1.;
static double initial(const Vector &x){return x.Norml2()<.5?1.+hot_amplitude:1.;}
template<class T> void write(std::ofstream &f,const T *p,size_t n){f.write(reinterpret_cast<const char*>(p),sizeof(T)*n);}
int main(int argc,char**argv){
  if(argc!=6)return 1;
  Mesh mesh(argv[1],1,1); mesh.UniformRefinement(); mesh.UniformRefinement();
  H1_FECollection fec(2,3); FiniteElementSpace fes(&mesh,&fec);
  int n=fes.GetTrueVSize(),cases=std::atoi(argv[3]),steps=std::atoi(argv[4]); std::string dir=argv[2];
  struct Scenario {double kappa,alpha,dt,amplitude;};
  std::vector<Scenario> campaign;Scenario s;std::ifstream schedule(argv[5]);
  while(schedule>>s.kappa>>s.alpha>>s.dt>>s.amplitude)campaign.push_back(s);
  if(steps<1||cases<1||cases>steps*int(campaign.size()))return 1;
  GridFunction u(&fes),conductivity(&fes); FunctionCoefficient ic(initial); u.ProjectCoefficient(ic);
  BilinearForm mass(&fes); mass.AddDomainIntegrator(new MassIntegrator()); mass.Assemble(); mass.Finalize();
  SparseMatrix &M=mass.SpMat(); double dt=.1,alpha=.01,kappa=.5;
  Vector b(n),ones(n),massones(n);ones=1.;M.Mult(ones,massones);
  std::ofstream app(dir+"/application.csv");app<<std::setprecision(17);
  for(int i=1;i<=cases;i++){
    int scenario=(i-1)/steps,local_step=(i-1)%steps+1;
    if(local_step==1){
      s=campaign[scenario];kappa=s.kappa;alpha=s.alpha;dt=s.dt;
      hot_amplitude=s.amplitude;u.ProjectCoefficient(ic);
    }
    double start=now(),mean=u.Sum()/n,variance=0.;
    for(int j=0;j<n;j++)variance+=(u[j]-mean)*(u[j]-mean);
    double c[]={1.,std::log(dt),std::log(kappa),std::log(alpha),mean,std::sqrt(variance/n),u.Min(),u.Max()};
    double context_time=now()-start;start=now();
    for(int j=0;j<n;j++)conductivity[j]=kappa+alpha*u[j];
    GridFunctionCoefficient coefficient(&conductivity);
    BilinearForm stiffness(&fes);stiffness.AddDomainIntegrator(new DiffusionIntegrator(coefficient));
    stiffness.Assemble();stiffness.Finalize();
    SparseMatrix *A=Add(1.,M,dt,stiffness.SpMat());M.Mult(u,b);
    double assembly=now()-start;
    std::ofstream context(dir+"/context.json"); context<<std::setprecision(17)<<"{\"context\":[";
    for(int j=0;j<8;j++)context<<(j?",":"")<<c[j];
    context<<"],\"context_sec\":"<<context_time<<",\"assembly_sec\":"<<assembly
           <<",\"physical_time\":"<<(local_step-1)*dt<<",\"scenario\":"<<scenario+1
           <<",\"scenario_step\":"<<local_step<<"}";context.close();
    std::ofstream request(dir+"/request.bin",std::ios::binary);int nnz=A->NumNonZeroElems();
    write(request,&n,1);write(request,&nnz,1);write(request,A->GetI(),n+1);write(request,A->GetJ(),nnz);
    write(request,A->GetData(),nnz);write(request,b.GetData(),n);write(request,u.GetData(),n);request.close();
    std::ofstream ready(dir+"/ready");ready<<i;ready.close();
    while(!std::ifstream(dir+"/response.ready").good())std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::ifstream response(dir+"/response.bin",std::ios::binary);
    response.read(reinterpret_cast<char*>(u.GetData()),n*sizeof(double));
    if(!response)return 2;response.close();std::remove((dir+"/response.ready").c_str());
    Vector residual(n);A->Mult(u,residual);residual-=b;
    double rr=residual.Norml2()/b.Norml2();if(!std::isfinite(rr)||rr>1e-9)return 3;
    app<<i<<','<<assembly<<','<<u.Min()<<','<<u.Max()<<','<<(massones*u)<<','<<rr<<','<<scenario+1<<','<<local_step<<'\n';app.flush();
    delete A;
  }
}
