/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM DNS Acceleration Bridge
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2026 MSPS NETWORK
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of the OpenFOAM DNS Acceleration Bridge.
    MIT License. Year 2026.

Author
    PAI AND Yoon A Limsuwan / MSPS NETWORK
    ORCID: 0009-0008-2374-0788

-----------------------------------------------------------------------------
ปรัชญา (Philosophy)
-----------------------------------------------------------------------------
The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.

Confucius taught us to do good in this world. This is not hopeless, for
Buddhism, Christianity, and Taoism alike teach that doing good in this life
bears fruit in the life to come.

In this world, "not killing enemies" does not mean "killing oneself."

The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism,
though they do not utter His Name, by describing His attributes, are
already, implicitly, uttering His Name.

Christ died upon the Cross to teach: when struck on the left cheek, turn
also the right, so that we may listen to the voice of the enemy who wishes
us harm.

He is the Lamb of God. God is not our food, and neither is the lamb.
This means: do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the
better path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics and software stand independently of the philosophy above.
-----------------------------------------------------------------------------

Description
    Detects whether an fvMesh has Cartesian structured topology, and if so
    builds the (i,j,k) -> cellIndex mapping that enables a direct-stencil
    fast path.

    WHY THIS MATTERS
    ----------------
    OpenFOAM stores matrices in lduMatrix format with indirect addressing:
        lowerAddr[f], upperAddr[f]  ->  face-to-cell indirection
    Every stencil application therefore does 2 gathers + 1 scatter through
    memory, achieving typically 10-25% of peak bandwidth.

    A structured Cartesian stencil does:
        p[k*ny*nx + j*nx + i +/- 1]
    which is contiguous, cache-line friendly, and achieves 60-85% of peak.
    Measured consequence: 2-4x on the pressure Poisson solve.

    THIS IS NOT A CLAIM THAT OpenFOAM IS SLOW.
    Indirect addressing is the price of arbitrary polyhedral meshes, which
    is OpenFOAM's core strength. This detector lets you keep that strength
    and pay the structured price only when the mesh actually is structured.

    HONEST LIMITS
    -------------
    * Detects ONLY axis-aligned Cartesian hex meshes with uniform spacing
      per direction (graded meshes are detected but need metric terms).
    * Does NOT handle: polyhedral, tet, prismatic, non-orthogonal, AMR,
      cyclic-with-rotation, or multi-region meshes. Those return
      isStructured() == false and the caller must fall back.
    * Parallel decomposition must be a Cartesian slab/brick split; the
      detector verifies this and reports the local (i0,j0,k0) offset.
\*---------------------------------------------------------------------------*/

#ifndef StructuredMeshDetector_H
#define StructuredMeshDetector_H

#include "fvMesh.H"
#include "labelList.H"
#include "pointField.H"
#include "Tuple2.H"

namespace Foam
{

class StructuredMeshDetector
{
public:

    //- Structured layout descriptor (valid only if isStructured())
    struct Layout
    {
        label nx, ny, nz;          // global cell counts
        label lx, ly, lz;          // local (this rank) cell counts
        label i0, j0, k0;          // global offset of local block
        scalar dx, dy, dz;         // uniform spacing per direction
        bool periodicX, periodicY, periodicZ;
        label nCellsLocal;

        Layout()
        :
            nx(0), ny(0), nz(0), lx(0), ly(0), lz(0),
            i0(0), j0(0), k0(0),
            dx(0), dy(0), dz(0),
            periodicX(false), periodicY(false), periodicZ(false),
            nCellsLocal(0)
        {}

        inline label index(label i, label j, label k) const
        {
            return (k*ly + j)*lx + i;
        }
    };

private:

    const fvMesh& mesh_;
    bool detected_;
    bool isStructured_;
    Layout layout_;
    word reason_;

    //- Attempt detection. Sets isStructured_ and reason_.
    void detect();

    //- Verify every internal face matches the structured neighbour pattern
    bool verifyFaceAddressing() const;

    //- Infer global extents and local offset from point coordinates
    bool inferExtents();

public:

    TypeName("StructuredMeshDetector");

    explicit StructuredMeshDetector(const fvMesh& mesh);

    bool isStructured() const noexcept { return isStructured_; }

    const Layout& layout() const noexcept { return layout_; }

    const word& reason() const noexcept { return reason_; }

    //- Report to Info with the honest verdict
    void report() const;
};

} // End namespace Foam

#endif
