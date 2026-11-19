// ===================== GpuPoissonSolver.C =====================
#include "GpuPoissonSolver.H"
#include "fvMesh.H"
#include "Time.H"
#include "Pstream.H"
#include "PstreamGlobals.H"
#include "IOdictionary.H"
#include <cuda_runtime.h>

namespace Foam
{
    defineTypeNameAndDebug(GpuPoissonSolver, 0);
    lduMatrix::solver::addsymMatrixConstructorToTable<GpuPoissonSolver>
        addGpuPoissonSolverSymConstructorToTable_;
    lduMatrix::solver::addasymMatrixConstructorToTable<GpuPoissonSolver>
        addGpuPoissonSolverAsymConstructorToTable_;
}

using namespace Foam;

GpuPoissonSolver::GpuPoissonSolver
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& ibc,
    const FieldField<Field, scalar>& iic,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& controls
)
:
    lduMatrix::solver(fieldName, matrix, ibc, iic, interfaces, controls)
{}

GpuPoissonSolver::~GpuPoissonSolver()
{
    if (d_p_) cudaFree(d_p_);
    if (d_b_) cudaFree(d_b_);
}

void GpuPoissonSolver::buildStructured() const
{
    probed_ = true;
    structured_ = false;

    const fvMesh& mesh =
        dynamic_cast<const fvMesh&>(matrix_.mesh());

    // ---- Gate 1: hex-only, 6 faces/cell, quad faces ----
    const cellList& cs = mesh.cells();
    forAll(cs, i)
        if (cs[i].size() != 6) { reason_="non-hex cell"; return; }
    const faceList& fs = mesh.faces();
    forAll(fs, i)
        if (fs[i].size() != 4) { reason_="non-quad face"; return; }
    if (mesh.foundObject<volScalarField>("cellLevel"))
        { reason_="AMR mesh"; return; }

    // ---- Gate 2: read the Cartesian rank grid from decomposeParDict ----
    IOdictionary dp
    (
        IOobject("decomposeParDict", mesh.time().system(), mesh,
                 IOobject::MUST_READ_IF_MODIFIED, IOobject::NO_WRITE)
    );
    if (dp.getOrDefault<word>("method","none") != "simple")
        { reason_="decomposeParDict method != simple"; return; }
    const vector ng = dp.subDict("coeffs").get<vector>("n");
    gd_.px = label(ng.x()); gd_.py = label(ng.y()); gd_.pz = label(ng.z());
    if (gd_.px*gd_.py*gd_.pz != Pstream::nProcs())
        { reason_="rank grid != nProcs"; return; }

    // ---- Gate 3: infer local lattice from cell centres ----
    const vectorField cc = mesh.cellCentres();
    auto uniq=[&](int ax){
        scalarField v(cc.size());
        forAll(cc,i) v[i]=cc[i][ax];
        std::sort(v.begin(),v.end());
        scalarField u; u.append(v[0]);
        for (label i=1;i<v.size();++i)
            if (mag(v[i]-u.last())>1e-9) u.append(v[i]);
        return u;
    };
    const scalarField ux=uniq(0), uy=uniq(1), uz=uniq(2);
    gd_.lx=ux.size(); gd_.ly=uy.size(); gd_.lz=uz.size();
    if (long(gd_.lx)*gd_.ly*gd_.lz != cc.size())
        { reason_="cell centres not a brick"; return; }
    gd_.dx=ux[1]-ux[0]; gd_.dy=uy[1]-uy[0]; gd_.dz=uz[1]-uz[0];
    for (label i=2;i<ux.size();++i)
        if (mag((ux[i]-ux[i-1])-gd_.dx)>1e-6*gd_.dx)
            { reason_="non-uniform x spacing"; return; }

    // global extents + offset
    gd_.nx = gd_.lx*gd_.px; gd_.ny = gd_.ly*gd_.py; gd_.nz = gd_.lz*gd_.pz;
    const scalar xm = returnReduce(ux[0], minOp<scalar>());
    const scalar ym = returnReduce(uy[0], minOp<scalar>());
    const scalar zm = returnReduce(uz[0], minOp<scalar>());
    gd_.i0 = label(round((ux[0]-xm)/gd_.dx));
    gd_.j0 = label(round((uy[0]-ym)/gd_.dy));
    gd_.k0 = label(round((uz[0]-zm)/gd_.dz));
    gd_.rx = gd_.i0/gd_.lx; gd_.ry = gd_.j0/gd_.ly; gd_.rz = gd_.k0/gd_.lz;

    // periodicity
    gd_.perX=gd_.perY=gd_.perZ=false;
    forAll(mesh.boundaryMesh(), pi)
    {
        const word t = mesh.boundaryMesh()[pi].type();
        if (t=="cyclic" || t=="cyclicAMI")
        {
            const vector n = mesh.boundaryMesh()[pi].faceAreas()[0];
            const vector u = n/mag(n);
            if (mag(mag(u.x())-1)<1e-6) gd_.perX=true;
            if (mag(mag(u.y())-1)<1e-6) gd_.perY=true;
            if (mag(mag(u.z())-1)<1e-6) gd_.perZ=true;
        }
    }

    // ---- Gate 4: verify ldu addressing == unit lattice steps ----
    lduToPad_.setSize(cc.size(), -1);
    forAll(cc, i)
    {
        const label a=label(round((cc[i].x()-ux[0])/gd_.dx));
        const label b=label(round((cc[i].y()-uy[0])/gd_.dy));
        const label c=label(round((cc[i].z()-uz[0])/gd_.dz));
        if (a<0||a>=gd_.lx||b<0||b>=gd_.ly||c<0||c>=gd_.lz)
            { reason_="centre off lattice"; return; }
        lduToPad_[i] = gpuPoisson::Core::padIndex(a,b,c,gd_.lx,gd_.ly,gd_.lz);
    }
    const labelUList& own = mesh.lduAddr().lowerAddr();
    const labelUList& nei = mesh.lduAddr().upperAddr();
    forAll(nei, f)
    {
        const label A=lduToPad_[own[f]], B=lduToPad_[nei[f]];
        const long sx=gd_.lx+2, sy=gd_.ly+2;
        const long d = std::abs(A-B);
        if (!(d==1 || d==sx || d==sx*sy))
            { reason_="ldu face not a unit lattice step"; return; }
    }

    // ---- build the core ----
    gpuPoisson::SolverOpts so;
    so.nLevels        = controlDict_.getOrDefault<int>("nLevels", 6);
    so.preSweeps      = controlDict_.getOrDefault<int>("preSweeps", 2);
    so.postSweeps     = controlDict_.getOrDefault<int>("postSweeps", 2);
    so.coarseSweeps   = controlDict_.getOrDefault<int>("coarseSweeps", 40);
    so.maxOuter       = controlDict_.getOrDefault<int>("maxOuter", 60);
    so.tol            = controlDict_.getOrDefault<scalar>("tolerance", 1e-8);
    so.cudaAwareMPI   = controlDict_.getOrDefault<bool>("cudaAwareMPI", false);
    so.verbose        = controlDict_.getOrDefault<bool>("verbose", false);

    gpComm comm =
#ifdef WITH_MPI
        MPI_Comm(PstreamGlobals::MPICommunicators_[mesh.comm()]);
#else
        0;
#endif
    core_.reset(new gpuPoisson::Core(gd_, so, comm));

    const long nPad = gd_.localPadded();
    cudaMalloc(&d_p_, nPad*sizeof(double));
    cudaMalloc(&d_b_, nPad*sizeof(double));
    cudaMemset(d_p_, 0, nPad*sizeof(double));

    structured_ = true;
    reason_ = "structured Cartesian verified";

    if (Pstream::master())
    {
        Info<< "[GpuPoisson] ENABLED  global "
            << gd_.nx<<"x"<<gd_.ny<<"x"<<gd_.nz
            << "  local "<<gd_.lx<<"x"<<gd_.ly<<"x"<<gd_.lz
            << "  ranks "<<gd_.px<<"x"<<gd_.py<<"x"<<gd_.pz
            << "  cudaAwareMPI="<<so.cudaAwareMPI << endl;
    }
}

Foam::solverPerformance GpuPoissonSolver::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    if (!probed_) buildStructured();

    solverPerformance perf(typeName, psi.name());

    if (!structured_)
    {
        // Honest fallback: report WHY, then use the requested backup solver.
        if (Pstream::master() && debug)
            Info<< "[GpuPoisson] fallback ("<<reason_<<") -> GAMG" << endl;
        dictionary fb(controlDict_);
        fb.set("solver", word("GAMG"));
        fb.set("smoother", word("symGaussSeidel"));
        fb.set("nPreSweeps", 0);
        fb.set("nPostSweeps", 2);
        fb.set("nCellsInCoarsestLevel", 200);
        auto alt = lduMatrix::solver::New
        (
            psi.name(), matrix_, interfaceBouCoeffs_,
            interfaceIntCoeffs_, interfaces_, fb
        );
        return alt->solve(psi, source, cmpt);
    }

    const long nPad = gd_.localPadded();
    std::vector<double> hp(nPad,0.0), hb(nPad,0.0);
    forAll(psi, i)
    {
        hp[lduToPad_[i]] = psi[i];
        hb[lduToPad_[i]] = source[i];
    }
    cudaMemcpy(d_p_, hp.data(), nPad*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_, hb.data(), nPad*sizeof(double), cudaMemcpyHostToDevice);

    gpuPoisson::Stats st;
    core_->solve(d_p_, d_b_, st);

    cudaMemcpy(hp.data(), d_p_, nPad*sizeof(double), cudaMemcpyDeviceToHost);
    forAll(psi, i) psi[i] = hp[lduToPad_[i]];

    perf.initialResidual() = st.initResidual;
    perf.finalResidual()   = st.finalResidual;
    perf.nIterations()     = st.iterations;
    perf.converged()       = st.converged;

    if (controlDict_.getOrDefault<bool>("verbose", false) && Pstream::master())
    {
        Info<< "[GpuPoisson] it="<<st.iterations
            <<" res "<<st.initResidual<<" -> "<<st.finalResidual
            <<" solve="<<st.solve_s<<"s"
            <<" halo="<<double(st.haloBytes)/1e9<<"GB"
            <<" calls="<<st.nHaloCalls
            <<" reduce="<<st.nReduceCalls
            <<" levels="<<st.nLevelsUsed
            <<" coarsest="<<st.coarsestLocal[0]<<"^3"<<endl;
    }
    return perf;
}
