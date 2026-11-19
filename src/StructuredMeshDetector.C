/*---------------------------------------------------------------------------*\
  OpenFOAM DNS Acceleration Bridge — StructuredMeshDetector implementation
  Author: PAI AND Yoon A Limsuwan / MSPS NETWORK | MIT | 2026
  (Philosophy header: see StructuredMeshDetector.H)
\*---------------------------------------------------------------------------*/

#include "StructuredMeshDetector.H"
#include "polyBoundaryMesh.H"
#include "fvPatch.H"
#include "globalIndex.H"
#include "Pstream.H"
#include "OFstream.H"

namespace Foam
{
    defineTypeNameAndDebug(StructuredMeshDetector, 0);
}

void Foam::StructuredMeshDetector::detect()
{
    isStructured_ = false;

    // ---- Gate 1: cell count must be a plausible brick -------------------
    const label nLocal = mesh_.nCells();
    if (nLocal < 8)
    {
        reason_ = "too few local cells (< 8)";
        return;
    }

    // ---- Gate 2: all cells must be hexahedra ---------------------------
    const cellList& cells = mesh_.cells();
    const faceList& faces = mesh_.faces();
    forAll(cells, celli)
    {
        if (cells[celli].size() != 6)
        {
            reason_ = "non-hex cell found (cell " + Foam::name(celli)
                    + " has " + Foam::name(cells[celli].size()) + " faces)";
            return;
        }
    }
    forAll(faces, facei)
    {
        if (faces[facei].size() != 4)
        {
            reason_ = "non-quadrilateral face found";
            return;
        }
    }

    // ---- Gate 3: no AMR / no refinement history ------------------------
    if (mesh_.hasObject("cellLevel"))
    {
        reason_ = "mesh carries cellLevel (AMR); structured path disabled";
        return;
    }

    // ---- Gate 4: infer extents from bounding box + spacing -------------
    if (!inferExtents())
    {
        return;   // reason_ set inside
    }

    // ---- Gate 5: verify face addressing matches the stencil ------------
    if (!verifyFaceAddressing())
    {
        reason_ = "ldu addressing does not match structured neighbour pattern";
        return;
    }

    isStructured_ = true;
    reason_ = "Cartesian structured topology verified";
}


bool Foam::StructuredMeshDetector::inferExtents()
{
    const pointField& pts = mesh_.points();
    const cellList& cells = mesh_.cells();
    const faceList& faces = mesh_.faces();

    // Cell centres give the lattice. Collect unique coordinates.
    const vectorField cc(mesh_.cellCentres());

    // Build sorted unique coordinate lists per axis with tolerance
    const scalar tol = 1e-9;
    auto uniqueCoords = [&](label axis) -> scalarField
    {
        scalarField v(cc.size());
        forAll(cc, i) v[i] = cc[i][axis];
        std::sort(v.begin(), v.end());
        scalarField u;
        u.append(v[0]);
        for (label i = 1; i < v.size(); ++i)
        {
            if (mag(v[i] - u.last()) > tol) u.append(v[i]);
        }
        return u;
    };

    const scalarField ux = uniqueCoords(0);
    const scalarField uy = uniqueCoords(1);
    const scalarField uz = uniqueCoords(2);

    layout_.lx = ux.size();
    layout_.ly = uy.size();
    layout_.lz = uz.size();

    if (layout_.lx*layout_.ly*layout_.lz != cc.size())
    {
        reason_ = "cell centres do not form a complete brick "
                  "(expected " + Foam::name(layout_.lx*layout_.ly*layout_.lz)
                + ", have " + Foam::name(cc.size()) + ")";
        return false;
    }

    // Uniform spacing check per axis
    auto spacing = [&](const scalarField& u, scalar& d) -> bool
    {
        if (u.size() < 2) { d = 1.0; return true; }
        d = u[1] - u[0];
        if (d <= 0) return false;
        for (label i = 2; i < u.size(); ++i)
        {
            if (mag((u[i] - u[i-1]) - d) > 1e-6*d) return false;
        }
        return true;
    };

    if (!spacing(ux, layout_.dx) ||
        !spacing(uy, layout_.dy) ||
        !spacing(uz, layout_.dz))
    {
        reason_ = "non-uniform spacing (graded mesh needs metric terms; "
                  "structured fast path not implemented for grading)";
        return false;
    }

    // Global extents via reduce
    layout_.nx = returnReduce(layout_.lx, maxOp<label>());
    layout_.ny = returnReduce(layout_.ly, maxOp<label>());
    layout_.nz = returnReduce(layout_.lz, maxOp<label>());

    // Local offset from the minimum global coordinate
    const scalar xMinLocal = ux[0];
    const scalar yMinLocal = uy[0];
    const scalar zMinLocal = uz[0];
    const scalar xMinGlobal = returnReduce(xMinLocal, minOp<scalar>());
    const scalar yMinGlobal = returnReduce(yMinLocal, minOp<scalar>());
    const scalar zMinGlobal = returnReduce(zMinLocal, minOp<scalar>());

    layout_.i0 = label(round((xMinLocal - xMinGlobal)/layout_.dx));
    layout_.j0 = label(round((yMinLocal - yMinGlobal)/layout_.dy));
    layout_.k0 = label(round((zMinLocal - zMinGlobal)/layout_.dz));

    // Periodicity from patch types
    const polyBoundaryMesh& pbm = mesh_.boundaryMesh();
    forAll(pbm, patchi)
    {
        const word& t = pbm[patchi].type();
        if (t == "cyclic" || t == "cyclicAMI")
        {
            const vector n = pbm[patchi].faceAreas()[0]
                           / mag(pbm[patchi].faceAreas()[0]);
            if (mag(mag(n[0]) - 1.0) < 1e-6) layout_.periodicX = true;
            if (mag(mag(n[1]) - 1.0) < 1e-6) layout_.periodicY = true;
            if (mag(mag(n[2]) - 1.0) < 1e-6) layout_.periodicZ = true;
        }
    }

    layout_.nCellsLocal = cc.size();
    return true;
}


bool Foam::StructuredMeshDetector::verifyFaceAddressing() const
{
    // For a structured brick, every internal face must connect cells whose
    // (i,j,k) differ by exactly 1 in one axis and 0 in the others.
    const labelList& own = mesh_.faceOwner();
    const labelList& nei = mesh_.faceNeighbour();
    const vectorField& cc = mesh_.cellCentres();
    const Layout& L = layout_;

    auto ijk = [&](label celli, label& i, label& j, label& k)
    {
        // Recover local indices from cell centre by nearest-lattice lookup
        i = -1; j = -1; k = -1;
        // Linear scan is O(N) per call -> instead use the arithmetic inverse
        // valid because spacing is uniform and we know the local origin.
        const vector c = cc[celli];
        i = label(round((c[0] - (cc[0][0]))/L.dx));
        j = label(round((c[1] - (cc[0][1]))/L.dy));
        k = label(round((c[2] - (cc[0][2]))/L.dz));
        // NOTE: cc[0] is not guaranteed to be the (0,0,0) cell after
        // renumbering. A robust implementation builds an explicit
        // coordinate->index hash once. We do that here.
    };

    // Build explicit hash: (i,j,k) -> cellIndex, and inverse
    HashTable<label, label> fwd;   // key = (k*ly+j)*lx+i
    labelList inv(cc.size(), -1);

    // Local origin = min over all local cell centres
    vector origin = cc[0];
    forAll(cc, i)
    {
        for (label d = 0; d < 3; ++d)
            origin[d] = min(origin[d], cc[i][d]);
    }

    forAll(cc, celli)
    {
        const label i = label(round((cc[celli][0] - origin[0])/L.dx));
        const label j = label(round((cc[celli][1] - origin[1])/L.dy));
        const label k = label(round((cc[celli][2] - origin[2])/L.dz));
        if (i < 0 || i >= L.lx || j < 0 || j >= L.ly || k < 0 || k >= L.lz)
            return false;
        const label key = (k*L.ly + j)*L.lx + i;
        if (fwd.found(key)) return false;   // duplicate lattice site
        fwd.insert(key, celli);
        inv[celli] = key;
    }

    if (label(fwd.size()) != cc.size()) return false;

    // Verify every internal face is a unit lattice step
    forAll(nei, facei)
    {
        const label a = inv[own[facei]];
        const label b = inv[nei[facei]];
        if (a < 0 || b < 0) return false;

        const label ai = a % L.lx, aj = (a/L.lx) % L.ly, ak = a/(L.lx*L.ly);
        const label bi = b % L.lx, bj = (b/L.lx) % L.ly, bk = b/(L.lx*L.ly);

        const label d = mag(ai-bi) + mag(aj-bj) + mag(ak-bk);
        if (d != 1) return false;
    }

    return true;
}


void Foam::StructuredMeshDetector::report() const
{
    Info<< "[StructuredMeshDetector] " << nl;
    if (isStructured_)
    {
        Info<< "  verdict        : STRUCTURED (fast path ENABLED)" << nl
            << "  global cells   : " << layout_.nx << " x "
                                    << layout_.ny << " x "
                                    << layout_.nz << nl
            << "  local block    : " << layout_.lx << " x "
                                    << layout_.ly << " x "
                                    << layout_.lz
            << "  at (" << layout_.i0 << "," << layout_.j0
            << "," << layout_.k0 << ")" << nl
            << "  spacing        : " << layout_.dx << ", "
                                    << layout_.dy << ", "
                                    << layout_.dz << nl
            << "  periodic       : "
            << (layout_.periodicX ? "x " : "")
            << (layout_.periodicY ? "y " : "")
            << (layout_.periodicZ ? "z " : "") << nl;
    }
    else
    {
        Info<< "  verdict        : NOT STRUCTURED (fall back to GAMG)" << nl
            << "  reason         : " << reason_ << nl;
    }
    Info<< endl;
}


Foam::StructuredMeshDetector::StructuredMeshDetector(const fvMesh& mesh)
:
    mesh_(mesh),
    detected_(false),
    isStructured_(false),
    layout_(),
    reason_("not yet detected")
{
    detect();
    detected_ = true;
}
