// ===================== GpuPoissonSolver.H =====================
#ifndef GpuPoissonSolver_H
#define GpuPoissonSolver_H

#include "lduMatrix.H"
#include "gpuPoissonCore.H"
#include <memory>

namespace Foam
{

class GpuPoissonSolver
:
    public lduMatrix::solver
{
    // structured-layout cache (built once per mesh)
    mutable bool probed_ = false;
    mutable bool structured_ = false;
    mutable word reason_;
    mutable std::shared_ptr<gpuPoisson::Core> core_;
    mutable labelList lduToPad_;      // local cell -> halo-padded index
    mutable gpuPoisson::GridDesc gd_;
    double* d_p_ = nullptr;
    double* d_b_ = nullptr;

    void buildStructured() const;

public:

    TypeName("GpuPoisson");

    GpuPoissonSolver
    (
        const word& fieldName,
        const lduMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& controls
    );

    virtual ~GpuPoissonSolver();

    virtual solverPerformance solve
    (
        scalarField& psi,
        const scalarField& source,
        const direction cmpt = 0
    ) const;
};

}
#endif
