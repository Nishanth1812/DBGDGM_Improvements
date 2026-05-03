import numpy as np
import torch
from torch_geometric.data import Data


AAL90_ADJACENCY = {
    0:  [1, 2, 7, 8],
    1:  [0, 2, 3, 8],
    2:  [0, 1, 3, 4],
    3:  [2, 4, 5],
    4:  [2, 3, 5, 6, 11],
    5:  [3, 4, 6, 12],
    6:  [4, 5, 7, 13],
    7:  [0, 6, 8, 14],
    8:  [0, 1, 7, 9, 15],
    9:  [8, 10, 15, 16],
    10: [9, 11, 16, 17],
    11: [4, 10, 17],
    12: [5, 13, 18],
    13: [6, 12, 14, 18],
    14: [7, 13, 19],
    15: [8, 9, 16, 20],
    16: [9, 10, 15, 17, 20, 21],
    17: [10, 11, 16, 21],
    18: [12, 13, 22],
    19: [14, 22, 23],
    20: [15, 16, 24],
    21: [16, 17, 24, 25],
    22: [18, 19, 26],
    23: [19, 27],
    24: [20, 21, 28],
    25: [21, 29],
    26: [22, 27, 30],
    27: [23, 26, 30, 31],
    28: [24, 29, 32],
    29: [25, 28, 33],
    30: [26, 27, 34],
    31: [27, 34, 35],
    32: [28, 33, 36],
    33: [29, 32, 37],
    34: [35, 38, 67],
    35: [31, 34, 36, 38, 68],
    36: [32, 35, 39],
    37: [33, 38, 40],
    38: [34, 37, 39, 40, 41],
    39: [36, 38, 42],
    40: [37, 38, 42, 43],
    41: [42, 44],
    42: [39, 40, 41, 43],
    43: [40, 42, 44],
    44: [41, 43, 45],
    45: [44, 46],
    46: [45, 47],
    47: [46, 48],
    48: [47, 49],
    49: [48, 50],
    50: [49, 51],
    51: [50, 52],
    52: [51, 53],
    53: [52, 54],
    54: [53, 55],
    55: [54, 56],
    56: [55],
    57: [58, 65, 66],
    58: [57, 59, 66, 67],
    59: [58, 60, 67],
    60: [59, 61, 68],
    61: [60, 62, 68, 69],
    62: [61, 63, 69, 70],
    63: [62, 64, 70],
    64: [63, 71],
    65: [57, 66, 72],
    66: [57, 58, 65, 67, 72, 73],
    67: [34, 58, 59, 66, 68, 73],
    68: [35, 60, 61, 67, 69, 73, 74],
    69: [61, 62, 68, 70, 74],
    70: [62, 63, 69, 71, 74],
    71: [64, 70, 75],
    72: [65, 66, 76],
    73: [66, 67, 68, 77],
    74: [68, 69, 70, 77],
    75: [71, 78],
    76: [72, 77, 79],
    77: [73, 74, 76, 79],
    78: [75, 80],
    79: [76, 77, 81],
    80: [78, 82],
    81: [79, 83],
    82: [80, 83],
    83: [81, 82, 84],
    84: [83, 85],
    85: [84, 86],
    86: [85, 87],
    87: [86, 88],
    88: [87, 89],
    89: [88],
}


def build_structural_graph(smri_features: np.ndarray):
    """
    Build PyG Data object for static structural brain graph.

    Args:
        smri_features: (N_roi, F) structural features

    Returns:
        PyG Data object with adjacency from AAL90 anatomical neighbourhood
    """
    n_rois = smri_features.shape[0]

    edges = []
    for roi, neighbors in AAL90_ADJACENCY.items():
        for nb in neighbors:
            if nb > roi:
                edges.append([roi, nb])

    if len(edges) > 0:
        edge_index = np.array(edges, dtype=np.int64).T  # shape (2, E)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    data = Data(
        x=torch.from_numpy(smri_features.astype(np.float32)),
        edge_index=torch.from_numpy(edge_index).long(),
    )
    return data


def smri_to_graph(smri_batch: torch.Tensor):
    """
    Process a batch of sMRI feature matrices (B ? N_roi ? F) into list of PyG Data objects.
    Returns list of length B.
    """
    batch_size = smri_batch.size(0)
    graphs = []
    for b in range(batch_size):
        feats = smri_batch[b].numpy()
        graphs.append(build_structural_graph(feats))
    return graphs