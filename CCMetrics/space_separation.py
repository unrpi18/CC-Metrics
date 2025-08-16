import cc3d
import numpy as np
from scipy.ndimage import distance_transform_edt


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
    Voronoi assignment to connected components (CPU, single EDT).
    labels>0 are seeds. Returns for each voxel the ID of the nearest component.
    - connectivity: 6/18/26 (3D) via cc3d
    - sampling: voxel spacing for anisotropic distances (scipy.ndimage.distance_transform_edt)
    - compact: maps component tags to 1..K (optional)
    """
    x = np.asarray(labels)
    cc = cc3d.connected_components(x, connectivity=connectivity)
    if cc.max() == 0:
        return np.zeros_like(labels)

    # EDT: 0 = Seeds, 1 = Nicht-Seeds
    edt_input = np.ones(cc.shape, dtype=np.uint8)
    edt_input[cc > 0] = 0

    # Indizes der nächstgelegenen Seeds (kein Distanz-Array nötig)
    indices = distance_transform_edt(
        edt_input, sampling=sampling, return_distances=False, return_indices=True
    )

    voronoi = cc[tuple(indices)]  # Komponententag am nächstgelegenen Seed
    return voronoi
