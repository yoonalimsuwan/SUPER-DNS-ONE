/*---------------------------------------------------------------------------*\
  GPU Poisson Core — implementation
  Author: PAI AND Yoon A Limsuwan / MSPS NETWORK | MIT | 2026
  (ปรัชญา header: see gpuPoissonCore.H)
\*---------------------------------------------------------------------------*/

#include "gpuPoissonCore.H"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CU_OK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"[gpuPoisson] CUDA %s at %s:%d\n", \
  cudaGetErrorString(e),__FILE__,__LINE__); abort(); } }while(0)

namespace Foam { namespace gpuPoisson {

// ---------------------------------------------------------------- indexing
__host__ __device__ __forceinline__
long pidx(int i,int j,int k,int lx,int ly,int lz)
{
    return (long(k+1)*(ly+2) + (j+1))*(lx+2) + (i+1);
}

long Core::padIndex(int i,int j,int k,int lx,int ly,int lz)
{ return pidx(i,j,k,lx,ly,lz); }

bool GridDesc::singular() const
{
    // With any Dirichlet face the operator is non-singular. Here every
    // non-periodic face is zero-normal-gradient, so the constant vector
    // is always in the null space.
    return true;
}

// ---------------------------------------------------------------- level
struct Level
{
    int lx, ly, lz;
    long nPad, nOwn;
    float *p=nullptr, *b=nullptr;     // fp32 MG working arrays
    double *pd=nullptr, *bd=nullptr;  // fp64 (finest level only)
    double cx, cy, cz;
    // halo staging (device)
    double *sxm=nullptr,*sxp=nullptr,*sym=nullptr,*syp=nullptr,*szm=nullptr,*szp=nullptr;
    double *rxm=nullptr,*rxp=nullptr,*rym=nullptr,*ryp=nullptr,*rzm=nullptr,*rzp=nullptr;
    float  *fxm=nullptr,*fxp=nullptr,*fym=nullptr,*fyp=nullptr,*fzm=nullptr,*fzp=nullptr;
    float  *gxm=nullptr,*gxp=nullptr,*gym=nullptr,*gyp=nullptr,*gzm=nullptr,*gzp=nullptr;
    int nb[6]={-1,-1,-1,-1,-1,-1};    // neighbour ranks
};

// ================================================================ kernels
// 7-point stencil on the halo-padded array. Ap = A p (SPD: A = -lap).
template<typename T>
__global__ void k_stencil(const T* __restrict__ p, T* __restrict__ Ap,
                          int lx,int ly,int lz, T cx,T cy,T cz)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;
    if(i>=lx||j>=ly||k>=lz) return;
    const long c = pidx(i,j,k,lx,ly,lz);
    const long xm=c-1, xp=c+1;
    const long ym=c-(lx+2), yp=c+(lx+2);
    const long zm=c-long(lx+2)*(ly+2), zp=c+long(lx+2)*(ly+2);
    Ap[c] = T(2)*(cx+cy+cz)*p[c]
          - cx*(p[xm]+p[xp]) - cy*(p[ym]+p[yp]) - cz*(p[zm]+p[zp]);
}

// Red-black Gauss-Seidel. RB avoids the write race of plain parallel GS.
template<typename T>
__global__ void k_rbgs(T* __restrict__ p, const T* __restrict__ b,
                       int lx,int ly,int lz, T cx,T cy,T cz, int parity)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;
    if(i>=lx||j>=ly||k>=lz) return;
    if(((i+j+k)&1)!=parity) return;
    const long c = pidx(i,j,k,lx,ly,lz);
    const long xm=c-1, xp=c+1;
    const long ym=c-(lx+2), yp=c+(lx+2);
    const long zm=c-long(lx+2)*(ly+2), zp=c+long(lx+2)*(ly+2);
    T s = b[c] + cx*(p[xm]+p[xp]) + cy*(p[ym]+p[yp]) + cz*(p[zm]+p[zp]);
    p[c] = s/(T(2)*(cx+cy+cz));
}

// r = b - A p   (fp64 accumulation even when inputs are fp32)
template<typename TA, typename TR>
__global__ void k_residual(const TA* __restrict__ p, const TA* __restrict__ b,
                           TR* __restrict__ r, int lx,int ly,int lz,
                           TA cx,TA cy,TA cz)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;
    if(i>=lx||j>=ly||k>=lz) return;
    const long c = pidx(i,j,k,lx,ly,lz);
    const long xm=c-1, xp=c+1;
    const long ym=c-(lx+2), yp=c+(lx+2);
    const long zm=c-long(lx+2)*(ly+2), zp=c+long(lx+2)*(ly+2);
    TA v = TA(2)*(cx+cy+cz)*p[c]
         - cx*(p[xm]+p[xp]) - cy*(p[ym]+p[yp]) - cz*(p[zm]+p[zp]);
    r[c] = TR(b[c]) - TR(v);
}

// 8-cell average restriction (fine halo-padded -> coarse halo-padded)
template<typename T>
__global__ void k_restrict(const T* __restrict__ f, T* __restrict__ c,
                           int flx,int fly,int flz,int clx,int cly,int clz)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;
    if(i>=clx||j>=cly||k>=clz) return;
    T acc = T(0);
    #pragma unroll
    for(int dk=0;dk<2;++dk)
    #pragma unroll
    for(int dj=0;dj<2;++dj)
    #pragma unroll
    for(int di=0;di<2;++di)
        acc += f[pidx(2*i+di,2*j+dj,2*k+dk,flx,fly,flz)];
    c[pidx(i,j,k,clx,cly,clz)] = acc/T(8);
}

// Piecewise-constant prolongation, ADDED to fine (MG correction).
template<typename T>
__global__ void k_prolongAdd(const T* __restrict__ c, T* __restrict__ f,
                             int flx,int fly,int flz,int clx,int cly,int clz)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;
    if(i>=flx||j>=fly||k>=flz) return;
    f[pidx(i,j,k,flx,fly,flz)] += c[pidx(i/2,j/2,k/2,clx,cly,clz)];
}

// ------------------------------------------------------------ halo pack
template<typename T>
__global__ void k_packX(const T* a, T* buf,int lx,int ly,int lz,int which)
{
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    int k = blockIdx.y*blockDim.y+threadIdx.y;
    if(j>=ly||k>=lz) return;
    buf[long(k)*ly+j] = a[pidx(which?lx-1:0,j,k,lx,ly,lz)];
}
template<typename T>
__global__ void k_unpackX(T* a,const T* buf,int lx,int ly,int lz,int which)
{
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    int k = blockIdx.y*blockDim.y+threadIdx.y;
    if(j>=ly||k>=lz) return;
    a[pidx(which?lx:-1,j,k,lx,ly,lz)] = buf[long(k)*ly+j];
}
template<typename T>
__global__ void k_packY(const T* a, T* buf,int lx,int ly,int lz,int which)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int k = blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=lx||k>=lz) return;
    buf[long(k)*lx+i] = a[pidx(i,which?ly-1:0,k,lx,ly,lz)];
}
template<typename T>
__global__ void k_unpackY(T* a,const T* buf,int lx,int ly,int lz,int which)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int k = blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=lx||k>=lz) return;
    a[pidx(i,which?ly:-1,k,lx,ly,lz)] = buf[long(k)*lx+i];
}
template<typename T>
__global__ void k_packZ(const T* a, T* buf,int lx,int ly,int lz,int which)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=lx||j>=ly) return;
    buf[long(j)*lx+i] = a[pidx(i,j,which?lz-1:0,lx,ly,lz)];
}
template<typename T>
__global__ void k_unpackZ(T* a,const T* buf,int lx,int ly,int lz,int which)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=lx||j>=ly) return;
    a[pidx(i,j,which?lz:-1,lx,ly,lz)] = buf[long(j)*lx+i];
}

// Mirror (zero-normal-gradient) fill where there is no neighbour rank.
template<typename T>
__global__ void k_mirrorX(T* a,int lx,int ly,int lz,int side)
{
    int j=blockIdx.x*blockDim.x+threadIdx.x, k=blockIdx.y*blockDim.y+threadIdx.y;
    if(j>=ly||k>=lz) return;
    a[pidx(side?lx:-1,j,k,lx,ly,lz)] = a[pidx(side?lx-1:0,j,k,lx,ly,lz)];
}
template<typename T>
__global__ void k_mirrorY(T* a,int lx,int ly,int lz,int side)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x, k=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=lx||k>=lz) return;
    a[pidx(i,side?ly:-1,k,lx,ly,lz)] = a[pidx(i,side?ly-1:0,k,lx,ly,lz)];
}
template<typename T>
__global__ void k_mirrorZ(T* a,int lx,int ly,int lz,int side)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>=lx||j>=ly) return;
    a[pidx(i,j,side?lz:-1,lx,ly,lz)] = a[pidx(i,j,side?lz-1:0,lx,ly,lz)];
}

// ---------------------------------------------------- device reduction
// Two-stage grid reduction with a FIXED number of partials -> deterministic.
template<int BLK>
__global__ void k_dot2(const double* __restrict__ a,const double* __restrict__ b,
                       const double* __restrict__ c,const double* __restrict__ d,
                       long nPad,int lx,int ly,int lz,
                       double* __restrict__ part)
{
    __shared__ double s0[BLK], s1[BLK];
    double acc0=0.0, acc1=0.0;
    // Iterate over OWNED cells only (halo must not be double counted).
    const long nOwn = long(lx)*ly*lz;
    for(long t=long(blockIdx.x)*BLK+threadIdx.x; t<nOwn; t+=long(gridDim.x)*BLK)
    {
        const int i = int(t % lx);
        const int j = int((t/lx) % ly);
        const int k = int(t/(long(lx)*ly));
        const long cidx = pidx(i,j,k,lx,ly,lz);
        acc0 += a[cidx]*b[cidx];
        acc1 += c[cidx]*d[cidx];
    }
    s0[threadIdx.x]=acc0; s1[threadIdx.x]=acc1;
    __syncthreads();
    #pragma unroll
    for(int s=BLK/2;s>0;s>>=1)
    {
        if(threadIdx.x<s){ s0[threadIdx.x]+=s0[threadIdx.x+s];
                           s1[threadIdx.x]+=s1[threadIdx.x+s]; }
        __syncthreads();
    }
    if(threadIdx.x==0){ part[blockIdx.x]=s0[0]; part[gridDim.x+blockIdx.x]=s1[0]; }
}

template<int BLK>
__global__ void k_final2(const double* part,int np,double* out)
{
    __shared__ double s0[BLK], s1[BLK];
    double a0=0,a1=0;
    for(int i=threadIdx.x;i<np;i+=BLK){ a0+=part[i]; a1+=part[np+i]; }
    s0[threadIdx.x]=a0; s1[threadIdx.x]=a1;
    __syncthreads();
    #pragma unroll
    for(int s=BLK/2;s>0;s>>=1)
    {
        if(threadIdx.x<s){ s0[threadIdx.x]+=s0[threadIdx.x+s];
                           s1[threadIdx.x]+=s1[threadIdx.x+s]; }
        __syncthreads();
    }
    if(threadIdx.x==0){ out[0]=s0[0]; out[1]=s1[0]; }
}

__global__ void k_sumOwned(const double* a,int lx,int ly,int lz,double* part)
{
    __shared__ double s[256];
    double acc=0;
    const long nOwn=long(lx)*ly*lz;
    for(long t=long(blockIdx.x)*256+threadIdx.x;t<nOwn;t+=long(gridDim.x)*256)
    {
        const int i=int(t%lx), j=int((t/lx)%ly), k=int(t/(long(lx)*ly));
        acc += a[pidx(i,j,k,lx,ly,lz)];
    }
    s[threadIdx.x]=acc; __syncthreads();
    for(int q=128;q>0;q>>=1){ if(threadIdx.x<q) s[threadIdx.x]+=s[threadIdx.x+q]; __syncthreads(); }
    if(threadIdx.x==0) part[blockIdx.x]=s[0];
}
__global__ void k_final1(const double* part,int np,double* out)
{
    __shared__ double s[256];
    double a=0; for(int i=threadIdx.x;i<np;i+=256) a+=part[i];
    s[threadIdx.x]=a; __syncthreads();
    for(int q=128;q>0;q>>=1){ if(threadIdx.x<q) s[threadIdx.x]+=s[threadIdx.x+q]; __syncthreads(); }
    if(threadIdx.x==0) out[0]=s[0];
}

// --------------------------------------------------------- axpy / cast
__global__ void k_axpy(double* y,const double* x,double a,long n){
    long i=long(blockIdx.x)*blockDim.x+threadIdx.x; if(i<n) y[i]+=a*x[i]; }
__global__ void k_xpay(double* y,const double* x,double a,long n){
    long i=long(blockIdx.x)*blockDim.x+threadIdx.x; if(i<n) y[i]=x[i]+a*y[i]; }
__global__ void k_copy(double* y,const double* x,long n){
    long i=long(blockIdx.x)*blockDim.x+threadIdx.x; if(i<n) y[i]=x[i]; }
__global__ void k_scale(double* y,double a,long n){
    long i=long(blockIdx.x)*blockDim.x+threadIdx.x; if(i<n) y[i]*=a; }
__global__ void k_d2f(float* y,const double* x,long n){
    long i=long(blockIdx.x)*blockDim.x+threadIdx.x; if(i<n) y[i]=float(x[i]); }
__global__ void k_f2d(double* y,const float* x,long n){
    long i=long(blockIdx.x)*blockDim.x+threadIdx.x; if(i<n) y[i]=double(x[i]); }
__global__ void k_setzero(double* y,long n){
    long i=long(blockIdx.x)*blockDim.x+threadIdx.x; if(i<n) y[i]=0.0; }
__global__ void k_setzerof(float* y,long n){
    long i=long(blockIdx.x)*blockDim.x+threadIdx.x; if(i<n) y[i]=0.0f; }

// ================================================================ Impl
struct Core::Impl
{
    GridDesc g; SolverOpts o; gpComm comm;
    int rank=0, nproc=1;
    std::vector<Level> L;
    int nLev=0;
    double *r=nullptr,*z=nullptr,*pd=nullptr,*Ap=nullptr;
    double *part=nullptr,*dout=nullptr,*hout=nullptr;
    double *dmean=nullptr;
    cudaEvent_t ev0,ev1;
    Stats accum;
    bool cudaAware=false;

    // ----------------------------------------------------------- helpers
    static dim3 blk3(){ return dim3(8,8,8); }
    static dim3 grd3(int lx,int ly,int lz){
        return dim3((lx+7)/8,(ly+7)/8,(lz+7)/8); }
    static dim3 blk2(){ return dim3(16,16); }
    static dim3 grd2(int a,int b){ return dim3((a+15)/16,(b+15)/16); }

    void allocLevel(Level& lv,int lx,int ly,int lz,bool finest,
                    double cx,double cy,double cz)
    {
        lv.lx=lx; lv.ly=ly; lv.lz=lz;
        lv.nPad=long(lx+2)*(ly+2)*(lz+2);
        lv.nOwn=long(lx)*ly*lz;
        lv.cx=cx; lv.cy=cy; lv.cz=cz;
        CU_OK(cudaMalloc(&lv.p, lv.nPad*sizeof(float)));
        CU_OK(cudaMalloc(&lv.b, lv.nPad*sizeof(float)));
        CU_OK(cudaMemset(lv.p,0,lv.nPad*sizeof(float)));
        CU_OK(cudaMemset(lv.b,0,lv.nPad*sizeof(float)));
        const long nx=long(ly)*lz, ny=long(lx)*lz, nz=long(lx)*ly;
        CU_OK(cudaMalloc(&lv.fxm,nx*sizeof(float))); CU_OK(cudaMalloc(&lv.fxp,nx*sizeof(float)));
        CU_OK(cudaMalloc(&lv.fym,ny*sizeof(float))); CU_OK(cudaMalloc(&lv.fyp,ny*sizeof(float)));
        CU_OK(cudaMalloc(&lv.fzm,nz*sizeof(float))); CU_OK(cudaMalloc(&lv.fzp,nz*sizeof(float)));
        CU_OK(cudaMalloc(&lv.gxm,nx*sizeof(float))); CU_OK(cudaMalloc(&lv.gxp,nx*sizeof(float)));
        CU_OK(cudaMalloc(&lv.gym,ny*sizeof(float))); CU_OK(cudaMalloc(&lv.gyp,ny*sizeof(float)));
        CU_OK(cudaMalloc(&lv.gzm,nz*sizeof(float))); CU_OK(cudaMalloc(&lv.gzp,nz*sizeof(float)));
        if(finest)
        {
            CU_OK(cudaMalloc(&lv.pd,lv.nPad*sizeof(double)));
            CU_OK(cudaMalloc(&lv.bd,lv.nPad*sizeof(double)));
            CU_OK(cudaMemset(lv.pd,0,lv.nPad*sizeof(double)));
            CU_OK(cudaMalloc(&lv.sxm,nx*sizeof(double))); CU_OK(cudaMalloc(&lv.sxp,nx*sizeof(double)));
            CU_OK(cudaMalloc(&lv.sym,ny*sizeof(double))); CU_OK(cudaMalloc(&lv.syp,ny*sizeof(double)));
            CU_OK(cudaMalloc(&lv.szm,nz*sizeof(double))); CU_OK(cudaMalloc(&lv.szp,nz*sizeof(double)));
            CU_OK(cudaMalloc(&lv.rxm,nx*sizeof(double))); CU_OK(cudaMalloc(&lv.rxp,nx*sizeof(double)));
            CU_OK(cudaMalloc(&lv.rym,ny*sizeof(double))); CU_OK(cudaMalloc(&lv.ryp,ny*sizeof(double)));
            CU_OK(cudaMalloc(&lv.rzm,nz*sizeof(double))); CU_OK(cudaMalloc(&lv.rzp,nz*sizeof(double)));
        }
    }

    void freeLevel(Level& lv)
    {
        auto F=[](void* p){ if(p) cudaFree(p); };
        F(lv.p);F(lv.b);F(lv.pd);F(lv.bd);
        F(lv.fxm);F(lv.fxp);F(lv.fym);F(lv.fyp);F(lv.fzm);F(lv.fzp);
        F(lv.gxm);F(lv.gxp);F(lv.gym);F(lv.gyp);F(lv.gzm);F(lv.gzp);
        F(lv.sxm);F(lv.sxp);F(lv.sym);F(lv.syp);F(lv.szm);F(lv.szp);
        F(lv.rxm);F(lv.rxp);F(lv.rym);F(lv.ryp);F(lv.rzm);F(lv.rzp);
    }

    // neighbour ranks for a given level's rank grid (same px,py,pz)
    void setNeighbours(Level& lv)
    {
        auto R=[&](int x,int y,int z){ return (x*g.py+y)*g.pz+z; };
        const bool px=g.perX, py=g.perY, pz=g.perZ;
        lv.nb[0] = (g.rx>0)? R(g.rx-1,g.ry,g.rz) : (px? R(g.px-1,g.ry,g.rz):-1);
        lv.nb[1] = (g.rx<g.px-1)? R(g.rx+1,g.ry,g.rz) : (px? R(0,g.ry,g.rz):-1);
        lv.nb[2] = (g.ry>0)? R(g.rx,g.ry-1,g.rz) : (py? R(g.rx,g.py-1,g.rz):-1);
        lv.nb[3] = (g.ry<g.py-1)? R(g.rx,g.ry+1,g.rz) : (py? R(g.rx,0,g.rz):-1);
        lv.nb[4] = (g.rz>0)? R(g.rx,g.ry,g.rz-1) : (pz? R(g.rx,g.ry,g.pz-1):-1);
        lv.nb[5] = (g.rz<g.pz-1)? R(g.rx,g.ry,g.rz+1) : (pz? R(g.rx,g.ry,0):-1);
        if(nproc==1){ for(int q=0;q<6;++q) lv.nb[q] = (q%2==0)? (px||py||pz?0:-1) : (px||py||pz?0:-1); }
    }

    // ---------------------------------------------------------- MPI halo
    void mpiXchg(void* snd,void* rcv,long n,MPI_Datatype dt,int peer,int tag)
    {
#ifdef WITH_MPI
        if(peer<0) return;
        if(cudaAware)
        {
            MPI_Sendrecv(snd,n,dt,peer,tag, rcv,n,dt,peer,tag,comm,MPI_STATUS_IGNORE);
        }
        else
        {
            const size_t bytes = n*((dt==MPI_DOUBLE)?8:4);
            std::vector<char> hs(bytes), hr(bytes);
            CU_OK(cudaMemcpy(hs.data(),snd,bytes,cudaMemcpyDeviceToHost));
            MPI_Sendrecv(hs.data(),n,dt,peer,tag, hr.data(),n,dt,peer,tag,comm,MPI_STATUS_IGNORE);
            CU_OK(cudaMemcpy(rcv,hr.data(),bytes,cudaMemcpyHostToDevice));
        }
        accum.haloBytes += long(n)*((dt==MPI_DOUBLE)?8:4)*2;
        accum.nHaloCalls++;
#endif
    }

    template<typename T>
    void halo(Level& lv,T* a,bool isDouble)
    {
#ifdef WITH_MPI
        MPI_Datatype dt = isDouble? MPI_DOUBLE : MPI_FLOAT;
#else
        MPI_Datatype dt = 0;
#endif
        const int lx=lv.lx, ly=lv.ly, lz=lv.lz;
        T *sxm,*sxp,*sym,*syp,*szm,*szp,*rxm,*rxp,*rym,*ryp,*rzm,*rzp;
        if(isDouble){
            sxm=(T*)lv.sxm;sxp=(T*)lv.sxp;sym=(T*)lv.sym;syp=(T*)lv.syp;szm=(T*)lv.szm;szp=(T*)lv.szp;
            rxm=(T*)lv.rxm;rxp=(T*)lv.rxp;rym=(T*)lv.rym;ryp=(T*)lv.ryp;rzm=(T*)lv.rzm;rzp=(T*)lv.rzp;
        } else {
            sxm=(T*)lv.fxm;sxp=(T*)lv.fxp;sym=(T*)lv.fym;syp=(T*)lv.fyp;szm=(T*)lv.fzm;szp=(T*)lv.fzp;
            rxm=(T*)lv.gxm;rxp=(T*)lv.gxp;rym=(T*)lv.gym;ryp=(T*)lv.gyp;rzm=(T*)lv.gzm;rzp=(T*)lv.gzp;
        }
        // ---- X ----
        k_packX<T><<<grd2(ly,lz),blk2()>>>(a,sxm,lx,ly,lz,0);
        k_packX<T><<<grd2(ly,lz),blk2()>>>(a,sxp,lx,ly,lz,1);
        mpiXchg(sxm,rxp,long(ly)*lz,dt,lv.nb[0],100);
        mpiXchg(sxp,rxm,long(ly)*lz,dt,lv.nb[1],101);
        if(lv.nb[0]<0) k_mirrorX<T><<<grd2(ly,lz),blk2()>>>(a,lx,ly,lz,0);
        else k_unpackX<T><<<grd2(ly,lz),blk2()>>>(a,rxm,lx,ly,lz,0);
        if(lv.nb[1]<0) k_mirrorX<T><<<grd2(ly,lz),blk2()>>>(a,lx,ly,lz,1);
        else k_unpackX<T><<<grd2(ly,lz),blk2()>>>(a,rxp,lx,ly,lz,1);
        // ---- Y ----
        k_packY<T><<<grd2(lx,lz),blk2()>>>(a,sym,lx,ly,lz,0);
        k_packY<T><<<grd2(lx,lz),blk2()>>>(a,syp,lx,ly,lz,1);
        mpiXchg(sym,ryp,long(lx)*lz,dt,lv.nb[2],102);
        mpiXchg(syp,rym,long(lx)*lz,dt,lv.nb[3],103);
        if(lv.nb[2]<0) k_mirrorY<T><<<grd2(lx,lz),blk2()>>>(a,lx,ly,lz,0);
        else k_unpackY<T><<<grd2(lx,lz),blk2()>>>(a,rym,lx,ly,lz,0);
        if(lv.nb[3]<0) k_mirrorY<T><<<grd2(lx,lz),blk2()>>>(a,lx,ly,lz,1);
        else k_unpackY<T><<<grd2(lx,lz),blk2()>>>(a,ryp,lx,ly,lz,1);
        // ---- Z ----
        k_packZ<T><<<grd2(lx,ly),blk2()>>>(a,szm,lx,ly,lz,0);
        k_packZ<T><<<grd2(lx,ly),blk2()>>>(a,szp,lx,ly,lz,1);
        mpiXchg(szm,rzp,long(lx)*ly,dt,lv.nb[4],104);
        mpiXchg(szp,rzm,long(lx)*ly,dt,lv.nb[5],105);
        if(lv.nb[4]<0) k_mirrorZ<T><<<grd2(lx,ly),blk2()>>>(a,lx,ly,lz,0);
        else k_unpackZ<T><<<grd2(lx,ly),blk2()>>>(a,rzm,lx,ly,lz,0);
        if(lv.nb[5]<0) k_mirrorZ<T><<<grd2(lx,ly),blk2()>>>(a,lx,ly,lz,1);
        else k_unpackZ<T><<<grd2(lx,ly),blk2()>>>(a,rzp,lx,ly,lz,1);
    }

    // ------------------------------------------------- global reductions
    void allreduce2(double* dvals)
    {
#ifdef WITH_MPI
        if(nproc>1)
        {
            if(cudaAware) MPI_Allreduce(MPI_IN_PLACE,dvals,2,MPI_DOUBLE,MPI_SUM,comm);
            else{
                double h[2]; CU_OK(cudaMemcpy(h,dvals,2*sizeof(double),cudaMemcpyDeviceToHost));
                MPI_Allreduce(MPI_IN_PLACE,h,2,MPI_DOUBLE,MPI_SUM,comm);
                CU_OK(cudaMemcpy(dvals,h,2*sizeof(double),cudaMemcpyHostToDevice));
            }
        }
#endif
        accum.nReduceCalls++;
    }
    void allreduce1(double* dval)
    {
#ifdef WITH_MPI
        if(nproc>1)
        {
            if(cudaAware) MPI_Allreduce(MPI_IN_PLACE,dval,1,MPI_DOUBLE,MPI_SUM,comm);
            else{
                double h; CU_OK(cudaMemcpy(&h,dval,sizeof(double),cudaMemcpyDeviceToHost));
                MPI_Allreduce(MPI_IN_PLACE,&h,1,MPI_DOUBLE,MPI_SUM,comm);
                CU_OK(cudaMemcpy(dval,&h,sizeof(double),cudaMemcpyHostToDevice));
            }
        }
#endif
        accum.nReduceCalls++;
    }

    void dot2(const double* a,const double* b,const double* c,const double* d,
              double* out2)
    {
        k_dot2<256><<<o.reduceBlocks,256>>>(a,b,c,d,L[0].nPad,
                                            L[0].lx,L[0].ly,L[0].lz,part);
        k_final2<256><<<1,256>>>(part,o.reduceBlocks,out2);
        allreduce2(out2);
    }
    double globalSum(const double* a)
    {
        k_sumOwned<<<o.reduceBlocks,256>>>(a,L[0].lx,L[0].ly,L[0].lz,part);
        k_final1<256><<<1,256>>>(part,o.reduceBlocks,dmean);
        allreduce1(dmean);
        double h; CU_OK(cudaMemcpy(&h,dmean,sizeof(double),cudaMemcpyDeviceToHost));
        return h;
    }
    void projectMean(double* a)
    {
        if(!g.singular()) return;
        const double m = globalSum(a)/double(g.globalCells());
        k_axpy<<<(L[0].nPad+255)/256,256>>>(a,nullptr,0.0,0); // no-op guard
        // subtract m from owned cells (and refresh halo afterwards)
        subtractScalar(a,m);
    }
    void subtractScalar(double* a,double m);

    // ---------------------------------------------------------- V-cycle
    void vcycle(int l)
    {
        Level& lv=L[l];
        if(l==nLev-1)
        {
            for(int s=0;s<o.coarseSweeps;++s)
            {
                halo<float>(lv,lv.p,false);
                k_rbgs<float><<<grd3(lv.lx,lv.ly,lv.lz),blk3()>>>(
                    lv.p,lv.b,lv.lx,lv.ly,lv.lz,
                    float(lv.cx),float(lv.cy),float(lv.cz),s&1);
            }
            return;
        }
        for(int s=0;s<o.preSweeps;++s)
        {
            halo<float>(lv,lv.p,false);
            k_rbgs<float><<<grd3(lv.lx,lv.ly,lv.lz),blk3()>>>(
                lv.p,lv.b,lv.lx,lv.ly,lv.lz,
                float(lv.cx),float(lv.cy),float(lv.cz),s&1);
        }
        halo<float>(lv,lv.p,false);
        // residual on this level: reuse a scratch = prolong target? need own buf
        // Use the NEXT level's b as restriction target; residual into lv.p? No.
        // Allocate a per-level residual scratch once (see rf below).
        k_residual<float,float><<<grd3(lv.lx,lv.ly,lv.lz),blk3()>>>(
            lv.p,lv.b,rf[l],lv.lx,lv.ly,lv.lz,
            float(lv.cx),float(lv.cy),float(lv.cz));
        Level& c=L[l+1];
        k_setzerof<<<(c.nPad+255)/256,256>>>(c.p,c.nPad);
        k_restrict<float><<<grd3(c.lx,c.ly,c.lz),blk3()>>>(
            rf[l],c.b,lv.lx,lv.ly,lv.lz,c.lx,c.ly,c.lz);
        vcycle(l+1);
        k_prolongAdd<float><<<grd3(lv.lx,lv.ly,lv.lz),blk3()>>>(
            c.p,lv.p,lv.lx,lv.ly,lv.lz,c.lx,c.ly,c.lz);
        for(int s=0;s<o.postSweeps;++s)
        {
            halo<float>(lv,lv.p,false);
            k_rbgs<float><<<grd3(lv.lx,lv.ly,lv.lz),blk3()>>>(
                lv.p,lv.b,lv.lx,lv.ly,lv.lz,
                float(lv.cx),float(lv.cy),float(lv.cz),s&1);
        }
    }
    std::vector<float*> rf;   // per-level fp32 residual scratch

    void applyMG(const double* d_r,double* d_z)
    {
        k_d2f<<<(L[0].nPad+255)/256,256>>>(L[0].b,d_r,L[0].nPad);
        k_setzerof<<<(L[0].nPad+255)/256,256>>>(L[0].p,L[0].nPad);
        vcycle(0);
        k_f2d<<<(L[0].nPad+255)/256,256>>>(d_z,L[0].p,L[0].nPad);
        projectMean(d_z);
    }

    void stencilD(const double* p,double* Ap)
    {
        halo<double>(L[0],const_cast<double*>(p),true);
        k_stencil<double><<<grd3(L[0].lx,L[0].ly,L[0].lz),blk3()>>>(
            p,Ap,L[0].lx,L[0].ly,L[0].lz,L[0].cx,L[0].cy,L[0].cz);
    }
};

// separate scalar-subtract kernel (kept outside Impl for clarity)
__global__ void k_subScalar(double* a,double m,int lx,int ly,int lz)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int j=blockIdx.y*blockDim.y+threadIdx.y;
    const int k=blockIdx.z*blockDim.z+threadIdx.z;
    if(i>=lx||j>=ly||k>=lz) return;
    a[pidx(i,j,k,lx,ly,lz)] -= m;
}
void Core::Impl::subtractScalar(double* a,double m)
{
    k_subScalar<<<grd3(L[0].lx,L[0].ly,L[0].lz),blk3()>>>(
        a,m,L[0].lx,L[0].ly,L[0].lz);
}

// ================================================================ Core
Core::Core(const GridDesc& g,const SolverOpts& o,gpComm comm)
: p_(new Impl)
{
    p_->g=g; p_->o=o; p_->comm=comm;
#ifdef WITH_MPI
    MPI_Comm_rank(comm,&p_->rank); MPI_Comm_size(comm,&p_->nproc);
#endif
    p_->cudaAware = o.cudaAwareMPI;
    CU_OK(cudaEventCreate(&p_->ev0)); CU_OK(cudaEventCreate(&p_->ev1));

    // ---- build level hierarchy: stop before any local dim < 2 ----
    int lx=g.lx, ly=g.ly, lz=g.lz;
    int nLev=1;
    while(nLev<o.nLevels && (lx/2)>=2 && (ly/2)>=2 && (lz/2)>=2
          && (lx%2==0)&&(ly%2==0)&&(lz%2==0))
    { lx/=2; ly/=2; lz/=2; ++nLev; }
    p_->nLev=nLev;
    p_->L.resize(nLev);
    p_->rf.resize(nLev,nullptr);

    lx=g.lx; ly=g.ly; lz=g.lz;
    double cx=1.0/(g.dx*g.dx), cy=1.0/(g.dy*g.dy), cz=1.0/(g.dz*g.dz);
    for(int l=0;l<nLev;++l)
    {
        p_->allocLevel(p_->L[l],lx,ly,lz,(l==0),cx,cy,cz);
        p_->setNeighbours(p_->L[l]);
        if(l<nLev-1){
            CU_OK(cudaMalloc(&p_->rf[l],p_->L[l].nPad*sizeof(float)));
            lx/=2; ly/=2; lz/=2; cx*=4; cy*=4; cz*=4;
        }
    }
    const long nP=p_->L[0].nPad;
    CU_OK(cudaMalloc(&p_->r, nP*sizeof(double)));
    CU_OK(cudaMalloc(&p_->z, nP*sizeof(double)));
    CU_OK(cudaMalloc(&p_->pd,nP*sizeof(double)));
    CU_OK(cudaMalloc(&p_->Ap,nP*sizeof(double)));
    CU_OK(cudaMalloc(&p_->part,2*p_->o.reduceBlocks*sizeof(double)));
    CU_OK(cudaMalloc(&p_->dout,2*sizeof(double)));
    CU_OK(cudaMalloc(&p_->dmean,sizeof(double)));
    CU_OK(cudaMallocHost(&p_->hout,2*sizeof(double)));
}

Core::~Core()
{
    for(int l=0;l<p_->nLev;++l){
        p_->freeLevel(p_->L[l]);
        if(p_->rf[l]) cudaFree(p_->rf[l]);
    }
    cudaFree(p_->r);cudaFree(p_->z);cudaFree(p_->pd);cudaFree(p_->Ap);
    cudaFree(p_->part);cudaFree(p_->dout);cudaFree(p_->dmean);
    cudaFreeHost(p_->hout);
    cudaEventDestroy(p_->ev0);cudaEventDestroy(p_->ev1);
    delete p_;
}

long Core::estimateBytes(const GridDesc& g,const SolverOpts& o)
{
    // finest fp64: p,b (in Level) + r,z,pdir,Ap = 6 arrays
    long nPad=g.localPadded();
    long bytes = 6*nPad*8;
    // fp32 MG: p,b,rf per level, geometric sum
    long nl=g.localOwned();
    for(int l=0;l<o.nLevels;++l){ bytes += 3*nl*4; nl/=8; if(nl<8) break; }
    // halo staging: 6 send + 6 recv, both dtypes, ~ 3 faces * area
    long area = long(g.ly)*g.lz + long(g.lx)*g.lz + long(g.lx)*g.ly;
    bytes += 12*area*8 + 12*area*4;
    bytes += 2*o.reduceBlocks*8 + 4096;
    return bytes;
}

bool Core::probeCudaAwareMPI(gpComm comm)
{
#ifdef WITH_MPI
    // MUST be run in a subprocess: a non-CUDA-aware MPI may SEGV, not error.
    double* d=nullptr;
    if(cudaMalloc(&d,sizeof(double))!=cudaSuccess) return false;
    double one=1.0; cudaMemcpy(d,&one,sizeof(double),cudaMemcpyHostToDevice);
    MPI_Allreduce(MPI_IN_PLACE,d,1,MPI_DOUBLE,MPI_SUM,comm);
    double h=0; cudaMemcpy(&h,d,sizeof(double),cudaMemcpyDeviceToHost);
    cudaFree(d);
    int np=1; MPI_Comm_size(comm,&np);
    return std::fabs(h-double(np))<1e-12;
#else
    return false;
#endif
}

void Core::solve(double* d_p,const double* d_b,Stats& st)
{
    Impl& I=*p_;
    const GridDesc& g=I.g; const SolverOpts& o=I.o;
    const long nP=I.L[0].nPad;
    const int lx=I.L[0].lx, ly=I.L[0].ly, lz=I.L[0].lz;
    auto grd=Impl::grd3(lx,ly,lz); auto blk=Impl::blk3();
    const long nb=(nP+255)/256;

    st=Stats(); st.nLevelsUsed=I.nLev;
    st.coarsestLocal[0]=I.L[I.nLev-1].lx;
    st.coarsestLocal[1]=I.L[I.nLev-1].ly;
    st.coarsestLocal[2]=I.L[I.nLev-1].lz;
    st.cudaAwareUsed=I.cudaAware;
    I.accum=Stats();

    CU_OK(cudaMemcpy(I.L[0].pd,d_p,nP*sizeof(double),cudaMemcpyDeviceToDevice));
    CU_OK(cudaMemcpy(I.L[0].bd,d_b,nP*sizeof(double),cudaMemcpyDeviceToDevice));

    // make b compatible: subtract its mean (singular operator)
    if(g.singular()){
        const double bm=I.globalSum(I.L[0].bd)/double(g.globalCells());
        I.subtractScalar(I.L[0].bd,bm);
        const double pm=I.globalSum(I.L[0].pd)/double(g.globalCells());
        I.subtractScalar(I.L[0].pd,pm);
    }

    CU_OK(cudaDeviceSynchronize());
    double t0=MPI_Wtime_stub();
    auto W=[&](){ CU_OK(cudaDeviceSynchronize());
                  return std::chrono::duration<double>(
                    std::chrono::steady_clock::now().time_since_epoch()).count(); };
    t0=W();

    // bnorm
    I.dot2(I.L[0].bd,I.L[0].bd,I.L[0].bd,I.L[0].bd,I.dout);
    CU_OK(cudaMemcpy(I.hout,I.dout,2*sizeof(double),cudaMemcpyDeviceToHost));
    const double bnorm=std::sqrt(std::max(I.hout[0],1e-300));
    st.bNorm=bnorm;

    // r = b - A p
    I.stencilD(I.L[0].pd,I.Ap);
    k_residual<double,double><<<grd,blk>>>(I.L[0].pd,I.L[0].bd,I.r,
        lx,ly,lz,I.L[0].cx,I.L[0].cy,I.L[0].cz);
    I.projectMean(I.r);
    I.dot2(I.r,I.r,I.r,I.r,I.dout);
    CU_OK(cudaMemcpy(I.hout,I.dout,2*sizeof(double),cudaMemcpyDeviceToHost));
    double rnorm=std::sqrt(std::max(I.hout[0],1e-300));
    st.initResidual=rnorm/bnorm;

    // z = MG(r); pd = z
    I.applyMG(I.r,I.z);
    CU_OK(cudaMemcpy(I.pd,I.z,nP*sizeof(double),cudaMemcpyDeviceToDevice));
    double rz=0.0;
    I.dot2(I.r,I.z,I.r,I.z,I.dout);
    CU_OK(cudaMemcpy(I.hout,I.dout,2*sizeof(double),cudaMemcpyDeviceToHost));
    rz=I.hout[0];

    st.converged=false;
    for(int it=0;it<o.maxOuter;++it)
    {
        I.stencilD(I.pd,I.Ap);
        // fused: dot(pd,Ap) and (later) dot(r,z) -> two separate calls
        I.dot2(I.pd,I.Ap,I.pd,I.Ap,I.dout);
        CU_OK(cudaMemcpy(I.hout,I.dout,2*sizeof(double),cudaMemcpyDeviceToHost));
        const double pAp=I.hout[0];
        if(std::fabs(pAp)<1e-300) break;
        const double alpha=rz/pAp;
        k_axpy<<<nb,256>>>(I.L[0].pd,I.pd,alpha,nP);
        k_axpy<<<nb,256>>>(I.r,I.Ap,-alpha,nP);
        I.projectMean(I.r);
        I.dot2(I.r,I.r,I.r,I.r,I.dout);
        CU_OK(cudaMemcpy(I.hout,I.dout,2*sizeof(double),cudaMemcpyDeviceToHost));
        rnorm=std::sqrt(std::max(I.hout[0],1e-300));
        st.iterations=it+1;
        if(rnorm/bnorm<o.tol){ st.converged=true; break; }
        I.applyMG(I.r,I.z);
        I.dot2(I.r,I.z,I.r,I.z,I.dout);
        CU_OK(cudaMemcpy(I.hout,I.dout,2*sizeof(double),cudaMemcpyDeviceToHost));
        const double rzNew=I.hout[0];
        const double beta=rzNew/(std::fabs(rz)>1e-300?rz:1e-300);
        k_xpay<<<nb,256>>>(I.pd,I.z,beta,nP);
        rz=rzNew;
    }
    st.finalResidual=rnorm/bnorm;
    CU_OK(cudaMemcpy(d_p,I.L[0].pd,nP*sizeof(double),cudaMemcpyDeviceToDevice));
    st.solve_s=W()-t0;
    st.haloBytes=I.accum.haloBytes; st.nHaloCalls=I.accum.nHaloCalls;
    st.nReduceCalls=I.accum.nReduceCalls;
}

}} // namespace
