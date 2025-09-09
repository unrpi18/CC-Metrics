import cc3d
import numpy as np
from scipy.ndimage import distance_transform_edt, generate_binary_structure
from scipy.ndimage import label as sn_label


def compute_voronoi_regions(labels):
    """
    Compute Voronoi regions for the given labels.

    Parameters:
        labels (ndarray): Input label array.

    Returns:
        ndarray: Array of Voronoi region assignments.

    """
    cc_labels = cc3d.connected_components(labels)
    current_assignment = np.zeros_like(cc_labels, dtype="int")
    current_mins = np.ones_like(cc_labels, dtype="float") * np.inf
    for idx, cc in enumerate(np.unique(cc_labels)):
        if cc == 0:
            pass
        else:
            # Compute distance transforms from current cc
            cur_dt = distance_transform_edt(np.logical_not(cc_labels == cc))
            # Update the cc_asignment and previous minimas
            msk = cur_dt < current_mins
            current_mins[msk] = cur_dt[msk]
            current_assignment[msk] = idx
    cc_asignment = current_assignment
    return cc_asignment


def compute_voronoi_regions_fast(labels, connectivity=26, sampling=None):
    """
    Voronoi assignment to connected components (CPU, single EDT) without cc3d.
    labels>0 are seeds. Returns for each voxel the ID of the nearest component.
    - connectivity: 6/18/26 (3D)
    - sampling: voxel spacing for anisotropic distances (scipy.ndimage.distance_transform_edt)
    """

    x = np.asarray(labels)
    # Map 3D connectivity to SciPy structure connectivity
    conn_rank = {6: 1, 18: 2, 26: 3}.get(connectivity, 3)
    structure = generate_binary_structure(rank=3, connectivity=conn_rank)
    cc, num = sn_label(x > 0, structure=structure)

    if num == 0:
        return np.zeros_like(x, dtype=np.int32)

    # EDT: 0 = seeds, 1 = non-seeds
    edt_input = np.ones(cc.shape, dtype=np.uint8)
    edt_input[cc > 0] = 0

    # Indices of the nearest seeds (no distance array needed)
    indices = distance_transform_edt(
        edt_input, sampling=sampling, return_distances=False, return_indices=True
    )

    voronoi = cc[tuple(indices)]  # component tag at nearest seed
    return voronoi.astype(np.int32, copy=False)
