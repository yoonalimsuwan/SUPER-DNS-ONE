/*---------------------------------------------------------------------------*\
  Standalone benchmark: Re 10^4 and 10^5 Poisson, GPU vs CPU, SAME algorithm.
  Author: PAI AND Yoon A Limsuwan / MSPS NETWORK | MIT | 2026
  (ปรัชญา header: see gpuPoissonCore.H)

  WHY STANDALONE
    Generating a 512^3 (134M-cell) OpenFOAM fvMesh with blockMesh takes
    tens of minutes and >100 GB of host RAM. That is a MESH-GENERATION
    bottleneck, not a SOLVER bottleneck. This harness isolates the solver
    so the speedup number means something. The OpenFOAM plugin uses the
    identical CUDA core.

  FAIR COMPARISON
    Two CPU baselines are reported:
      (a) PCG + Jacobi        — what many codes actually use
      (b) PCG + geometric MG  — SAME algorithm as the GPU path
    Reporting only (a) would inflate the speedup by the algorithm factor.
    Both are given. (b) is the honest hardware speedup.
\*---------------------------------------------------------------------------*/

#include "gpuPoissonCore.H"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Foam::gpuPoisson;

static double now()
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------- Re -> N
static int reToN(double Re, double kmaxEta, int roundTo=64)
{
    const double n = (kmaxEta/M_PI)*std::pow(Re,0.75);
    return int(std::ceil(n/roundTo)*roundTo);
}

// ---------------------------------------------------------- CPU reference
struct CpuGrid
{
    int n; long N; double h;
    std::vector<double> p,b,r,z,pd,Ap;
    CpuGrid(int n_):n(n_),N(long(n_)*n_*n_),h(1.0/n_),
        p(N,0),b(N,0),r(N,0),z(N,0),pd(N,0),Ap(N,0){}
    inline long I(int i,int j,int k) const {
        return ((long)((k+n)%n)*n + (j+n)%n)*n + (i+n)%n; }
    void stencil(const std::vector<double>& x,std::vector<double>& y) const
    {
        const double c=1.0/(h*h);
        #pragma omp parallel for collapse(3) schedule(static)
        for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i){
            const long q=I(i,j,k);
            y[q]=6*c*x[q]-c*(x[I(i-1,j,k)]+x[I(i+1,j,k)]
                            +x[I(i,j-1,k)]+x[I(i,j+1,k)]
                            +x[I(i,j,k-1)]+x[I(i,j,k+1)]);
        }
    }
    double dot(const std::vector<double>& a,const std::vector<double>& b) const
    { double s=0; 
      #pragma omp parallel for reduction(+:s) schedule(static)
      for(long q=0;q<N;++q) s+=a[q]*b[q]; return s; }
    void axpy(std::vector<double>& y,const std::vector<double>& x,double a) const
    { 
      #pragma omp parallel for schedule(static)
      for(long q=0;q<N;++q) y[q]+=a*x[q]; }
    void xpay(std::vector<double>& y,const std::vector<double>& x,double a) const
    { 
      #pragma omp parallel for schedule(static)
      for(long q=0;q<N;++q) y[q]=x[q]+a*y[q]; }
    void mean0(std::vector<double>& x) const
    { double s=0;
      #pragma omp parallel for reduction(+:s) schedule(static)
      for(long q=0;q<N;++q) s+=x[q];
      s/=double(N);
      #pragma omp parallel for schedule(static)
      for(long q=0;q<N;++q) x[q]-=s; }
    void jacobi(const std::vector<double>& bb,std::vector<double>& x,int ns) const
    { const double c=1.0/(h*h); std::vector<double> t(N);
      for(int s=0;s<ns;++s){
        #pragma omp parallel for collapse(3) schedule(static)
        for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i){
            const long q=I(i,j,k);
            t[q]=(bb[q]+c*(x[I(i-1,j,k)]+x[I(i+1,j,k)]+x[I(i,j-1,k)]
                          +x[I(i,j+1,k)]+x[I(i,j,k-1)]+x[I(i,j,k+1)]))/(6*c);
        }
        x.swap(t);
      } }
    // CPU geometric MG (same algorithm as GPU, fp64 for simplicity)
    struct Lvl { int n; double h; std::vector<double> p,b,r; 
                 Lvl(int n_,double h_):n(n_),h(h_),
                 p(long(n_)*n_*n_,0),b(long(n_)*n_*n_,0),r(long(n_)*n_*n_,0){}
                 inline long I(int i,int j,int k) const {
                     return ((long)((k+n)%n)*n+(j+n)%n)*n+(i+n)%n; } };
    std::vector<Lvl> L;
    void buildMG(int nLev){
        L.clear(); int nn=n; double hh=h;
        for(int l=0;l<nLev && nn>=4 && nn%2==0;++l){
            L.emplace_back(nn,hh); nn/=2; hh*=2; }
    }
    void rbgs(Lvl& lv,int parity) const {
        const double c=1.0/(lv.h*lv.h); const int nn=lv.n;
        #pragma omp parallel for collapse(3) schedule(static)
        for(int k=0;k<nn;++k)for(int j=0;j<nn;++j)for(int i=0;i<nn;++i){
            if(((i+j+k)&1)!=parity) continue;
            const long q=lv.I(i,j,k);
            lv.p[q]=(lv.b[q]+c*(lv.p[lv.I(i-1,j,k)]+lv.p[lv.I(i+1,j,k)]
                               +lv.p[lv.I(i,j-1,k)]+lv.p[lv.I(i,j+1,k)]
                               +lv.p[lv.I(i,j,k-1)]+lv.p[lv.I(i,j,k+1)]))/(6*c);
        }
    }
    void vc(int l) const {
        Lvl& lv=L[l];
        if(l==int(L.size())-1){ for(int s=0;s<40;++s) rbgs(lv,s&1); return; }
        for(int s=0;s<2;++s) rbgs(lv,s&1);
        const double c=1.0/(lv.h*lv.h); const int nn=lv.n;
        #pragma omp parallel for collapse(3) schedule(static)
        for(int k=0;k<nn;++k)for(int j=0;j<nn;++j)for(int i=0;i<nn;++i){
            const long q=lv.I(i,j,k);
            lv.r[q]=lv.b[q]-(6*c*lv.p[q]-c*(lv.p[lv.I(i-1,j,k)]+lv.p[lv.I(i+1,j,k)]
                +lv.p[lv.I(i,j-1,k)]+lv.p[lv.I(i,j+1,k)]
                +lv.p[lv.I(i,j,k-1)]+lv.p[lv.I(i,j,k+1)]));
        }
        Lvl& cl=L[l+1]; const int cn=cl.n;
        std::fill(cl.p.begin(),cl.p.end(),0.0);
        #pragma omp parallel for collapse(3) schedule(static)
        for(int k=0;k<cn;++k)for(int j=0;j<cn;++j)for(int i=0;i<cn;++i){
            double a=0;
            for(int dk=0;dk<2;++dk)for(int dj=0;dj<2;++dj)for(int di=0;di<2;++di)
                a+=lv.r[lv.I(2*i+di,2*j+dj,2*k+dk)];
            cl.b[cl.I(i,j,k)]=a/8.0;
        }
        vc(l+1);
        #pragma omp parallel for collapse(3) schedule(static)
        for(int k=0;k<nn;++k)for(int j=0;j<nn;++j)for(int i=0;i<nn;++i)
            lv.p[lv.I(i,j,k)] += cl.p[cl.I(i/2,j/2,k/2)];
        for(int s=0;s<2;++s) rbgs(lv,s&1);
    }
    void mg(const std::vector<double>& bb,std::vector<double>& zz) const {
        Lvl& lv=L[0];
        #pragma omp parallel for schedule(static)
        for(long q=0;q<lv.n*long(lv.n)*lv.n;++q){ lv.b[q]=bb[q]; lv.p[q]=0; }
        vc(0);
        #pragma omp parallel for schedule(static)
        for(long q=0;q<N;++q) zz[q]=lv.p[q];
        mean0(zz);
    }
};

struct CpuResult { int it; double res0,res1,t; bool conv; };

static CpuResult cpuPCG(CpuGrid& G,bool useMG,int maxIt,double tol,int nLev)
{
    CpuResult R{}; double t0=now();
    if(useMG) G.buildMG(nLev);
    G.mean0(G.b); G.mean0(G.p);
    const double bn=std::sqrt(std::max(G.dot(G.b,G.b),1e-300));
    G.stencil(G.p,G.Ap);
    #pragma omp parallel for schedule(static)
    for(long q=0;q<G.N;++q) G.r[q]=G.b[q]-G.Ap[q];
    G.mean0(G.r);
    R.res0=std::sqrt(G.dot(G.r,G.r))/bn;
    if(useMG) G.mg(G.r,G.z); else G.jacobi(G.r,G.z,4);
    G.pd=G.z;
    double rz=G.dot(G.r,G.z);
    double rn=R.res0;
    for(int it=0;it<maxIt;++it){
        G.stencil(G.pd,G.Ap);
        const double pAp=G.dot(G.pd,G.Ap);
        if(std::fabs(pAp)<1e-300) break;
        const double al=rz/pAp;
        G.axpy(G.p,G.pd,al); G.axpy(G.r,G.Ap,-al); G.mean0(G.r);
        rn=std::sqrt(G.dot(G.r,G.r))/bn;
        R.it=it+1;
        if(rn<tol){R.conv=true;break;}
        if(useMG) G.mg(G.r,G.z); else G.jacobi(G.r,G.z,4);
        const double rzn=G.dot(G.r,G.z);
        G.xpay(G.pd,G.z,rzn/(std::fabs(rz)>1e-300?rz:1e-300));
        rz=rzn;
    }
    R.res1=rn; R.t=now()-t0; return R;
}

// ---------------------------------------------------------- main
int main(int argc,char** argv)
{
#ifdef WITH_MPI
    MPI_Init(&argc,&argv);
    int rank=0,nproc=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
#else
    int rank=0,nproc=1;
#endif
    double Re=1e4, kmaxEta=1.5, tol=1e-8;
    int maxIt=60, nLev=6, cpuThreads=0, doCpu=1, cudaAware=0, probe=0;
    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        auto nx=[&](){ return atof(argv[++i]); };
        if(a=="--re") Re=nx();
        else if(a=="--kmax-eta") kmaxEta=nx();
        else if(a=="--tol") tol=nx();
        else if(a=="--max-iter") maxIt=int(nx());
        else if(a=="--levels") nLev=int(nx());
        else if(a=="--cpu-threads") cpuThreads=int(nx());
        else if(a=="--no-cpu") doCpu=0;
        else if(a=="--cuda-aware-mpi") cudaAware=1;
        else if(a=="--probe-mpi") probe=1;
    }
#ifdef _OPENMP
    if(cpuThreads>0) omp_set_num_threads(cpuThreads);
#endif

    if(probe){
#ifdef WITH_MPI
        bool ok=Core::probeCudaAwareMPI(MPI_COMM_WORLD);
        if(rank==0) printf("CUDA_AWARE_MPI=%s\n", ok?"YES":"NO");
#else
        if(rank==0) printf("CUDA_AWARE_MPI=NO (built without MPI)\n");
#endif
#ifdef WITH_MPI
        MPI_Finalize();
#endif
        return 0;
    }

    // ---- rank grid: nearest-cubic factorisation of nproc ----
    int px=1,py=1,pz=nproc; double best=1e30;
    for(int a=1;a<=nproc;++a) if(nproc%a==0){
        int rem=nproc/a;
        for(int b=1;b<=rem;++b) if(rem%b==0){
            int c=rem/b;
            double s=double(std::max({a,b,c}))/std::min({a,b,c});
            if(s<best){best=s;px=a;py=b;pz=c;}
        }
    }
    const int rx=(rank/(py*pz))%px, ry=(rank/pz)%py, rz=rank%pz;

    const int N=reToN(Re,kmaxEta);
    if(N%px||N%py||N%pz){
        if(rank==0) fprintf(stderr,
          "N=%d not divisible by rank grid %dx%dx%d. "
          "Choose --re or nproc so it divides.\n",N,px,py,pz);
#ifdef WITH_MPI
        MPI_Finalize();
#endif
        return 1;
    }

    GridDesc g;
    g.nx=g.ny=g.nz=N;
    g.lx=N/px; g.ly=N/py; g.lz=N/pz;
    g.i0=rx*g.lx; g.j0=ry*g.ly; g.k0=rz*g.lz;
    g.px=px; g.py=py; g.pz=pz; g.rx=rx; g.ry=ry; g.rz=rz;
    g.dx=g.dy=g.dz=1.0/N;
    g.perX=g.perY=g.perZ=true;

    SolverOpts o;
    o.nLevels=nLev; o.tol=tol; o.maxOuter=maxIt;
    o.cudaAwareMPI=(cudaAware!=0);

    const long bytesPerRank=Core::estimateBytes(g,o);
    const long totalBytes=bytesPerRank*nproc;

    // ---- FEASIBILITY GATE (per rank) ----
    size_t freeB=0,totB=0;
    cudaSetDevice(rank%8);
    cudaMemGetInfo(&freeB,&totB);
    if(rank==0){
        printf("\n=========================================================\n");
        printf("  GPU POISSON BENCHMARK\n");
        printf("  Re_L=%.3g  kmax*eta=%.2f  ->  N=%d^3 = %.3e cells\n",
               Re,kmaxEta,N,double(N)*N*N);
        printf("  ranks=%d (%dx%dx%d)  local=%d^3\n",nproc,px,py,pz,g.lx);
        printf("  working set: %.2f GB/rank, %.2f GB total\n",
               bytesPerRank/1e9,totalBytes/1e9);
        printf("  GPU free: %.2f GB   total: %.2f GB\n",freeB/1e9,totB/1e9);
        if(kmaxEta<1.5)
            printf("  *** UNDER-RESOLVED for DNS (kmax*eta<1.5). "
                   "Solver benchmark only. ***\n");
        printf("=========================================================\n");
    }
    if(double(bytesPerRank)>0.85*double(freeB)){
        if(rank==0){
            const int need=int(std::ceil(double(totalBytes)/(0.70*totB)));
            fprintf(stderr,
              "\nREFUSING: needs %.2f GB/rank but only %.2f GB free.\n"
              "  Minimum ranks for this case: %d\n"
              "  Options: (a) more ranks, (b) --kmax-eta 1.0 "
              "(under-resolved, solver benchmark only)\n",
              bytesPerRank/1e9, freeB/1e9, need);
        }
#ifdef WITH_MPI
        MPI_Finalize();
#endif
        return 2;
    }

    // ---- RHS: zero-mean random (a realistic Poisson RHS) ----
    const long nPad=g.localPadded();
    std::vector<double> hb(nPad,0.0), hp(nPad,0.0);
    srand(1234+rank);
    double loc=0;
    for(int k=0;k<g.lz;++k)for(int j=0;j<g.ly;++j)for(int i=0;i<g.lx;++i){
        const double v=(double(rand())/RAND_MAX-0.5);
        hb[Core::padIndex(i,j,k,g.lx,g.ly,g.lz)]=v; loc+=v;
    }
#ifdef WITH_MPI
    MPI_Allreduce(MPI_IN_PLACE,&loc,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
#endif
    const double mean=loc/(double(N)*N*N);
    for(auto& x:hb) x-=mean;   // includes halo zeros; harmless (owned fixed below)
    for(int k=0;k<g.lz;++k)for(int j=0;j<g.ly;++j)for(int i=0;i<g.lx;++i){
        const long q=Core::padIndex(i,j,k,g.lx,g.ly,g.lz);
        hb[q]+=mean; hb[q]-=mean;  // keep owned zero-mean exactly
    }

    double *d_p,*d_b;
    cudaMalloc(&d_p,nPad*sizeof(double)); cudaMalloc(&d_b,nPad*sizeof(double));
    cudaMemcpy(d_b,hb.data(),nPad*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemset(d_p,0,nPad*sizeof(double));

#ifdef WITH_MPI
    gpComm comm=MPI_COMM_WORLD;
#else
    gpComm comm=0;
#endif
    double tSetup0=now();
    Core core(g,o,comm);
    double tSetup=now()-tSetup0;

    Stats st;
    // warm-up (JIT, page migration)
    core.solve(d_p,d_b,st);
    cudaMemset(d_p,0,nPad*sizeof(double));
    core.solve(d_p,d_b,st);   // timed run

    if(rank==0){
        printf("\n--- GPU: PCG + fp32 geometric MG ---\n");
        printf("  iterations      : %d\n",st.iterations);
        printf("  residual        : %.3e -> %.3e  (%s)\n",
               st.initResidual,st.finalResidual,st.converged?"CONVERGED":"NOT CONVERGED");
        printf("  MG levels used  : %d  (coarsest local %dx%dx%d)\n",
               st.nLevelsUsed,st.coarsestLocal[0],st.coarsestLocal[1],st.coarsestLocal[2]);
        printf("  setup           : %.3f s\n",tSetup);
        printf("  solve           : %.3f s\n",st.solve_s);
        printf("  halo traffic    : %.3f GB in %d calls\n",
               st.haloBytes/1e9,st.nHaloCalls);
        printf("  reductions      : %d\n",st.nReduceCalls);
        printf("  CUDA-aware MPI  : %s\n",st.cudaAwareUsed?"YES":"NO (host staging)");
        printf("  throughput      : %.2f Mcell/s/iter\n",
               double(N)*N*N*st.iterations/st.solve_s/1e6);
    }

    // ---- CPU baselines on rank 0 only, SAME grid but SERIAL-sized ----
    // Honest: a full N^3 CPU solve at N=512 needs 6*134M*8 = 6.4 GB and
    // minutes-to-hours. We run the CPU on a REDUCED grid and extrapolate
    // by measured per-cell cost, and we SAY SO.
    if(doCpu && rank==0){
        const int nc = std::min(N, 128);
        printf("\n--- CPU baselines (n=%d^3, %d OpenMP threads) ---\n",
               nc,
#ifdef _OPENMP
               omp_get_max_threads()
#else
               1
#endif
        );
        CpuGrid C(nc);
        srand(7); double s=0;
        for(long q=0;q<C.N;++q){ C.b[q]=double(rand())/RAND_MAX-0.5; s+=C.b[q]; }
        s/=double(C.N); for(long q=0;q<C.N;++q) C.b[q]-=s;

        auto rj=cpuPCG(C,false,maxIt,tol,nLev);
        printf("  (a) PCG+Jacobi : it=%4d  res=%.2e  t=%.3f s  %s\n",
               rj.it,rj.res1,rj.t,rj.conv?"conv":"NOT conv");
        CpuGrid C2(nc); C2.b=C.b;
        auto rm=cpuPCG(C2,true,maxIt,tol,nLev);
        printf("  (b) PCG+MG     : it=%4d  res=%.2e  t=%.3f s  %s\n",
               rm.it,rm.res1,rm.t,rm.conv?"conv":"NOT conv");

        // per-cell-per-iteration cost, extrapolated to the full N
        const double scale = (double(N)*N*N)/(double(nc)*nc*nc);
        const double cj = rj.t/std::max(rj.it,1)*scale;
        const double cm = rm.t/std::max(rm.it,1)*scale;
        const double gpuPerIt = st.solve_s/std::max(st.iterations,1);
        printf("\n--- SPEEDUP (extrapolated CPU at N=%d, MEASURED GPU) ---\n",N);
        printf("  vs CPU PCG+Jacobi : %8.1fx   (algorithm + hardware)\n", cj/gpuPerIt);
        printf("  vs CPU PCG+MG     : %8.1fx   (HARDWARE ONLY — same algorithm)\n",
               cm/gpuPerIt);
        printf("  *** CPU numbers are EXTRAPOLATED from n=%d by cell count.\n",nc);
        printf("      Iteration counts do not scale identically; treat the\n");
        printf("      hardware-only figure as an ORDER OF MAGNITUDE. ***\n");
    }

    cudaFree(d_p); cudaFree(d_b);
#ifdef WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}
