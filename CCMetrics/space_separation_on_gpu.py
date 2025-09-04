import cupy as cp
import cupyx.scipy.ndimage as cnd

_CONN_MAP = {6: 1, 18: 2, 26: 3}


def compute_voronoi_regions_fast_on_gpu(
    labels, connectivity=26, sampling=None, return_numpy=False
):
    """
    Voronoi assignment to connected components (CPU, single EDT).
    labels>0 are seeds. Returns for each voxel the ID of the nearest component.
    - connectivity: 6/18/26 (3D) via cc3d
    - sampling: voxel spacing for anisotropic distances (scipy.ndimage.distance_transform_edt)
    - compact: maps component tags to 1..K (optional)
    """
    rank = _CONN_MAP.get(connectivity, 3)

    x = cp.asarray(labels)
    if (x > 0).sum() == 0:
        out = cp.zeros_like(x, dtype=cp.int32)
        return cp.asnumpy(out) if return_numpy else out

    structure = cnd.generate_binary_structure(rank=3, connectivity=rank)
    cc, num = cnd.label(x > 0, structure=structure)

    if num == 0:
        out = cp.zeros_like(x, dtype=cp.int32)
        return cp.asnumpy(out) if return_numpy else out

    edt_input = cp.ones(cc.shape, dtype=cp.uint8)
    edt_input[cc > 0] = 0

    # Indizes der nächstgelegenen Seeds (kein Distanz-Array nötig)
    indices = cnd.distance_transform_edt(
        edt_input, sampling=sampling, return_distances=False, return_indices=True
    )

    voronoi = cc[tuple(indices)]  # Komponententag am nächstgelegenen Seed
    return cp.asnumpy(voronoi) if return_numpy else voronoi
