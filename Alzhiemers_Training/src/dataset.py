import logging
import pathlib
import pickle
import time
from collections import defaultdict

import cv2
import networkx as nx
import nilearn.connectome as conn
import numpy as np


def _get_oasis_sequences(data_dir="./data"):
    """
    Scans the directory for OASIS jpg slices, grouping them by sequence.
    Returns: A list of dicts {"label": int, "slices": [list of filepaths]}
    """
    data_dir = pathlib.Path(data_dir)
    class_map = {
        "Non Demented": 0,
        "Very mild Dementia": 1,
        "Mild Dementia": 2,
        "Moderate Dementia": 3
    }

    print("[dataset] Scanning data directory for OASIS sequences...", flush=True)
    sequences = defaultdict(list)
    for class_name, label in class_map.items():
        class_dir = data_dir / class_name
        files = sorted(class_dir.rglob("*.jpg"))
        print(f"[dataset]   {class_name}: {len(files)} slices found", flush=True)
        for file in files:
            # OAS1_0028_MR1_mpr-1_104.jpg
            parts = file.stem.split('_')
            # Group by subject and scan: e.g., (2, 'OAS1_0028_MR1', 'mpr-1')
            sub = "_".join(parts[:3])
            scan = parts[3]
            slice_num = int(parts[4])
            sequences[(label, sub, scan)].append((slice_num, str(file)))

    # Sort slices for each sequence and return
    out_sequences = []
    for (label, sub, scan), slices in sorted(sequences.items(), key=lambda item: item[0]):
        slices.sort(key=lambda x: x[0])  # Sort by slice index
        slice_paths = [s[1] for s in slices]
        out_sequences.append({
            "label": label,
            "subject": sub,
            "scan": scan,
            "slices": slice_paths
        })
    print(f"[dataset] Found {len(out_sequences)} total sequences across all classes.", flush=True)
    return out_sequences


def _zscore_timeseries(X, axis=-1):
    std = np.std(X, axis=axis, keepdims=True)
    std[std == 0] = 1.0  # Avoid division by zero
    X -= np.mean(X, axis=axis, keepdims=True)
    X /= std
    return X


def _graph_to_snapshot(graph):
    edges = np.asarray(list(graph.edges()), dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)

    return {
        "edge_index": edges,
        "num_nodes": graph.number_of_nodes(),
    }


def _normalize_dataset(dataset):
    normalized = []
    changed = False

    for new_subject_idx, sample in enumerate(dataset):
        subject_idx, dynamic_graph, label = sample

        snapshots = []
        for graph in dynamic_graph:
            if isinstance(graph, dict) and "edge_index" in graph and "num_nodes" in graph:
                snapshots.append(graph)
            else:
                snapshots.append(_graph_to_snapshot(graph))
                changed = True

        if subject_idx != new_subject_idx:
            changed = True

        normalized.append([new_subject_idx, snapshots, label])

    return normalized, changed


def _extract_timeseries_from_slices(slice_paths, grid_size=15, zscore=False, dtype=np.float32):
    """
    Treats the series of slices as "time" and regional patches as "nodes".
    """
    time_len = len(slice_paths)
    nodes_len = grid_size * grid_size
    X = np.zeros((nodes_len, time_len), dtype=dtype)
    
    for t, path in enumerate(slice_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Unable to read slice image: {path}")
        h, w = img.shape
        grid_h, grid_w = h // grid_size, w // grid_size
        
        node_idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                patch = img[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                X[node_idx, t] = np.mean(patch)
                node_idx += 1
                
    if zscore: X = _zscore_timeseries(X)
    return X


def _compute_dynamic_fc(X,
                        window_size=30,
                        window_stride=10,
                        measure="correlation",
                        top_percent=10,
                        max_time=490,
                        zscore=True,
                        self_loops=False,
                        as_graph=True):
    # truncate time length
    time_len = X.shape[-1]
    max_time = max_time if (max_time and max_time <= time_len) else time_len
    X = X[:, :max_time]

    if zscore: X = _zscore_timeseries(X)

    # calculate starting timepoint for each window
    sampling_points = list(range(0, max_time - window_size + 1, window_stride))
    
    # Needs at least one point
    if not sampling_points:
         sampling_points = [0]
         window_size = max_time

    # initialize functional connectivity measure (uses Ledoit-Wolf covariance estimator)
    conn_measure = conn.ConnectivityMeasure(kind=measure, standardize='zscore_sample')

    # calculate dynamic functional connectivity within each timeseries window 
    G = []
    for idx in sampling_points:
        # calculate functional connectivity matrix
        A = conn_measure.fit_transform([X[:, idx: idx + window_size].T])[0]
        # remove self-loops
        if not self_loops: np.fill_diagonal(A, 0.)
        # calculate percentile threshold value
        threshold = np.percentile(A.flatten(), 100. - top_percent)
        # threshold 
        A[A < threshold] = 0.
        # return as networkx graph
        G += [nx.from_numpy_array(A) if as_graph else A]

    return G


def load_dataset(dataset="oasis", window_size=15, window_stride=5, measure="correlation",
                 top_percent=10, grid_size=15, data_dir="../data", **kwargs):
    # load dataset if already exists
    filename = "{}_w-{}_s-{}_m-{}_p-{}_g-{}.pkl".format(dataset, window_size, window_stride, measure, top_percent, grid_size)
    _filepath = pathlib.Path(data_dir) / dataset / filename
    if _filepath.exists():
        print(f"[dataset] Found cached dataset at {_filepath}. Loading from disk...", flush=True)
        logging.info('Found an existing .pkl dataset. Loading...')
        with open(_filepath, "rb") as input_file:
            _dataset = pickle.load(input_file)
        _dataset, cache_changed = _normalize_dataset(_dataset)
        if cache_changed:
            logging.info('Normalizing cached dataset format for stable subject indexing and faster snapshot access.')
            with open(_filepath, "wb") as output_file:
                pickle.dump(_dataset, output_file)
        print(f"[dataset] Loaded {len(_dataset)} samples from cache.", flush=True)
    # create and save dataset if does not exist
    else:
        print(f"[dataset] No cached dataset found. Starting full preprocessing...", flush=True)
        logging.info('No existing .pkl dataset. Pre-processing image slices...')
        sequences = _get_oasis_sequences(data_dir=pathlib.Path(data_dir))
        logging.info(f'Found {len(sequences)} scan sequences in {data_dir}')

        if len(sequences) == 0:
            logging.warning(f'No JPG files found properly structured under classes!')
            print("[dataset] ERROR: No JPG files found! Check data directory structure.", flush=True)
            return []

        total = len(sequences)
        print(f"[dataset] Processing {total} sequences. This may take a while...", flush=True)
        _dataset = []
        skipped = 0
        t_start = time.time()

        for idx, seq in enumerate(sequences):
            t_seq = time.time()
            pct = (idx + 1) / total * 100
            bar = '#' * (idx * 20 // total) + '-' * (20 - idx * 20 // total)
            print(f"[dataset] [{bar}] {idx+1}/{total} ({pct:.1f}%) "
                  f"| {seq['subject']} {seq['scan']} (label={seq['label']})", flush=True)

            label = seq["label"]
            print(f"[dataset]   -> Extracting timeseries from {len(seq['slices'])} slices...", flush=True)
            X = _extract_timeseries_from_slices(seq["slices"], grid_size=grid_size)

            # Skip sequences that are too short for the window size completely
            if X.shape[-1] < window_size:
                logging.warning(f"Sequence {idx} too short ({X.shape[-1]} slices). Skipping.")
                print(f"[dataset]   -> SKIPPED (only {X.shape[-1]} slices, need {window_size})", flush=True)
                skipped += 1
                continue

            print(f"[dataset]   -> Computing dynamic FC (shape={X.shape})...", flush=True)
            G = _compute_dynamic_fc(X, window_size, window_stride, measure, top_percent, **kwargs)
            subject_idx = len(_dataset)
            snapshots = [_graph_to_snapshot(graph) for graph in G]
            _dataset.append([subject_idx, snapshots, label])

            elapsed = time.time() - t_seq
            total_elapsed = time.time() - t_start
            avg_per_seq = total_elapsed / (idx + 1)
            eta = avg_per_seq * (total - idx - 1)
            print(f"[dataset]   -> Done in {elapsed:.1f}s | "
                  f"total elapsed: {total_elapsed/60:.1f}m | ETA: {eta/60:.1f}m", flush=True)

        # Ensure parent directory exists before saving
        print(f"[dataset] Preprocessing complete! {len(_dataset)} samples kept, {skipped} skipped.", flush=True)
        print(f"[dataset] Saving dataset to {_filepath}...", flush=True)
        _filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(_filepath, "wb") as output_file:
            pickle.dump(_dataset, output_file)
        print(f"[dataset] Dataset saved successfully.", flush=True)

    logging.info('Loaded dataset.')
    return _dataset


def data_loader(dataset, batch_size=10):
    num_samples = len(dataset)
    sample_idx = list(range(num_samples))
    if num_samples < batch_size:
        num_batches = 1
    else:
        num_batches = num_samples // batch_size
    batch_sample_idx = np.array_split(sample_idx, num_batches)
    for batch in batch_sample_idx:
        yield list(map(lambda x: dataset[x], batch))
