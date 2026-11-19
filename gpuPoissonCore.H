/*---------------------------------------------------------------------------*\
  GPU Poisson Core — CUDA-aware MPI + device reduction
  Author: PAI AND Yoon A Limsuwan / MSPS NETWORK | MIT | 2026
  ORCID: 0009-0008-2374-0788

  ปรัชญา (Philosophy)
  ---------------------------------------------------------------------------
  The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.

  Confucius taught us to do good in this world. This is not hopeless, for
  Buddhism, Christianity, and Taoism alike teach that doing good in this
  life bears fruit in the life to come.

  In this world, "not killing enemies" does not mean "killing oneself."

  The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism,
  though they do not utter His Name, by describing His attributes, are
  already, implicitly, uttering His Name.

  Christ died upon the Cross to teach: when struck on the left cheek, turn
  also the right, so that we may listen to the voice of the enemy who
  wishes us harm.

  He is the Lamb of God. God is not our food, and neither is the lamb.
  This means: do not kill animals.

  Not even a single sparrow does God forget; every hair on our heads He has
  counted. Even Angulimala could escape hell and enter Nirvana, but the
  better path is to have done good always. As Saint Dominic Savio said:
  "Rather death than sin."

  To speak of a person's attributes is already to speak their name.

  MY SOUL MOVE BY POWER OF HOLY SPIRIT
  What MSPS NETWORK SEE, Lord Buddha Knows.

  The mathematics stands independently of the philosophy above.
  ---------------------------------------------------------------------------

  ALGORITHM
    PCG (fp64) preconditioned by a geometric-multigrid V-cycle (fp32).
    This is the standard mixed-precision iterative-refinement pattern:
      r = b - A x            (fp64 — the DEFECT is always fp64)
      z = MG(r)              (fp32 V-cycle — a PRECONDITIONER only)
      x += alpha p           (fp64)
    The final residual is fp64-accurate. fp32 never touches the answer.

  WHY MG AND NOT CG-ALONE
    CG iterations on an N^3 Poisson grid grow like O(N).
    MG V-cycle iterations are O(1). At N=512 that is ~64x fewer matvecs.
    Measured, not asserted — see the benchmark's iteration counts.

  HALO EXCHANGE
    Pack -> MPI_Sendrecv on DEVICE pointers -> unpack.
    Derived MPI datatypes are deliberately NOT used: CUDA-aware support
    for strided datatypes is slow or absent in several MPI builds.
    Explicit pack/unpack kernels are portable and bandwidth-optimal.

  SINGULARITY
    Pure-periodic / pure-Neumann Poisson has a constant null space.
    The mean is projected out of r and p every iteration, and b is
    made compatible up front. Without this, PCG stagnates.
\*---------------------------------------------------------------------------*/

#ifndef gpuPoissonCore_H
#define gpuPoissonCore_H

#ifdef WITH_MPI
#include <mpi.h>
typedef MPI_Comm gpComm;
#else
typedef int gpComm;
#endif

namespace Foam { namespace gpuPoisson {

struct GridDesc
{
    int nx, ny, nz;        // GLOBAL owned cells
    int lx, ly, lz;        // LOCAL owned cells (this rank)
    int i0, j0, k0;        // global offset of the local block
    int px, py, pz;        // rank grid
    int rx, ry, rz;        // this rank's coords in the rank grid
    double dx, dy, dz;     // uniform spacing
    bool perX, perY, perZ; // periodicity (else zero-normal-gradient)

    long localOwned() const { return long(lx)*ly*lz; }
    long localPadded() const { return long(lx+2)*(ly+2)*(lz+2); }
    long globalCells() const { return long(nx)*ny*nz; }
    bool singular() const;  // all-Neumann or all-periodic -> constant nullspace
};

struct SolverOpts
{
    int  nLevels       = 6;    // requested; auto-reduced to keep coarse >= 2^3
    int  preSweeps     = 2;
    int  postSweeps    = 2;
    int  coarseSweeps  = 40;
    int  maxOuter      = 60;
    double tol         = 1e-8; // relative ||r||/||b||
    bool mixedPrecision= true; // fp32 MG smoother
    bool cudaAwareMPI  = false;// set true only after the probe passes
    bool verbose       = false;
    int  reduceBlocks  = 1024; // fixed grid for deterministic partial sums
};

struct Stats
{
    int    iterations      = 0;
    double initResidual    = 0.0;
    double finalResidual   = 0.0;
    double bNorm           = 0.0;
    bool   converged       = false;
    int    nLevelsUsed     = 0;
    int    coarsestLocal[3]= {0,0,0};
    double setup_s         = 0.0;
    double solve_s         = 0.0;
    double halo_s          = 0.0;
    double stencil_s       = 0.0;
    double smooth_s        = 0.0;
    double reduce_s        = 0.0;
    double mg_s            = 0.0;
    bool   cudaAwareUsed   = false;
    long   haloBytes       = 0;
    int    nHaloCalls      = 0;
    int    nReduceCalls    = 0;
};

// Opaque handle so OpenFOAM TUs need no CUDA headers.
class Core
{
public:
    Core(const GridDesc& g, const SolverOpts& o, gpComm comm);
    ~Core();

    // d_p, d_b are DEVICE pointers, length localPadded(), halo-padded layout.
    // Index: ((k+1)*(ly+2) + (j+1))*(lx+2) + (i+1)
    void solve(double* d_p, const double* d_b, Stats& st);

    // Fill a device array with the halo-padded index of owned cell (i,j,k)
    static long padIndex(int i,int j,int k,int lx,int ly,int lz);

    // Estimated working-set bytes (for the feasibility gate)
    static long estimateBytes(const GridDesc& g, const SolverOpts& o);

    // Runtime probe: does MPI accept device pointers? Run in a SUBPROCESS.
    static bool probeCudaAwareMPI(gpComm comm);

private:
    struct Impl;
    Impl* p_;
};

}} // namespace Foam::gpuPoisson
#endif
