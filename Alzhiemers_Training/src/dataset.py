import logging
import pathlib
import pickle
import time
from collections import defaultdict

import cv2
import networkx as nx
import nilearn.connectome as conn
import numpy as np


def _scan_oasis_subjects(data_dir="./data"):
    """
    Scans the directory for OASIS jpg slices, grouping all scans per subject.
    Returns: A list of dicts {"label": int, "subject": str, "slices": [list of filepaths]}
    """
    data_dir = pathlib.Path(data_dir)
    class_map = {
        "Non Demented": 0,
        "Very mild Dementia": 1,
        "Mild Dementia": 2,
        "Moderate Dementia": 3
    }

    print("[dataset] Scanning data directory for OASIS subjects...", flush=True)
    subject_scans = defaultdict(lambda: defaultdict(list))
    for class_name, label in class_map.items():
        class_dir = data_dir / class_name
        files = sorted(class_dir.rglob("*.jpg"))
        print(f"[dataset]   {class_name}: {len(files)} slices found", flush=True)
        for file in files:
            parts = file.stem.split('_')
            sub = "_".join(parts[:3])
            scan = parts[3]
            slice_num = int(parts[4])
            subject_scans[(label, sub)][scan].append((slice_num, str(file)))

    def _scan_sort_key(scan_name):
        prefix, _, suffix = scan_name.rpartition('-')
        if suffix.isdigit():
            return prefix, int(suffix)
        return scan_name, 0

    out_subjects = []
    for (label, sub), scans in sorted(subject_scans.items(), key=lambda item: item[0]):
        scan_entries = []
        for scan_name in sorted(scans.keys(), key=_scan_sort_key):
            slices = scans[scan_name]
            slices.sort(key=lambda x: x[0])
            scan_entries.append({
                "name": scan_name,
                "slices": [path for _, path in slices],
            })

        out_subjects.append({
            "label": label,
            "subject": sub,
            "scans": scan_entries,
        })
    print(f"[dataset] Found {len(out_subjects)} total subjects across all classes.", flush=True)
    return out_subjects


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
    filename = "{}_subjects_w-{}_s-{}_m-{}_p-{}_g-{}.pkl".format(dataset, window_size, window_stride, measure, top_percent, grid_size)
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
        sequences = _scan_oasis_subjects(data_dir=pathlib.Path(data_dir))
        logging.info(f'Found {len(sequences)} subjects in {data_dir}')

        if len(sequences) == 0:
            logging.warning(f'No JPG files found properly structured under classes!')
            print("[dataset] ERROR: No JPG files found! Check data directory structure.", flush=True)
            return []

        total = len(sequences)
        print(f"[dataset] Processing {total} subjects. This may take a while...", flush=True)
        _dataset = []
        skipped = 0
        t_start = time.time()

        for idx, seq in enumerate(sequences):
            t_seq = time.time()
            pct = (idx + 1) / total * 100
            bar = '#' * (idx * 20 // total) + '-' * (20 - idx * 20 // total)
            total_slices = sum(len(scan['slices']) for scan in seq['scans'])
            print(f"[dataset] [{bar}] {idx+1}/{total} ({pct:.1f}%) "
                f"| {seq['subject']} scans={len(seq['scans'])} slices={total_slices} (label={seq['label']})", flush=True)

            label = seq["label"]
            snapshots = []
            skipped_scans = 0

            for scan in seq['scans']:
                print(
                    f"[dataset]   -> Extracting timeseries from scan {scan['name']} "
                    f"({len(scan['slices'])} slices)...",
                    flush=True,
                )
                X = _extract_timeseries_from_slices(scan['slices'], grid_size=grid_size)

                if X.shape[-1] < window_size:
                    logging.warning(
                        f"Subject {seq['subject']} scan {scan['name']} too short "
                        f"({X.shape[-1]} slices). Skipping scan."
                    )
                    print(
                        f"[dataset]   -> SKIPPED scan {scan['name']} "
                        f"(only {X.shape[-1]} slices, need {window_size})",
                        flush=True,
                    )
                    skipped_scans += 1
                    continue

                print(f"[dataset]   -> Computing dynamic FC for scan {scan['name']} (shape={X.shape})...", flush=True)
                G = _compute_dynamic_fc(X, window_size, window_stride, measure, top_percent, **kwargs)
                snapshots.extend(_graph_to_snapshot(graph) for graph in G)

            if not snapshots:
                logging.warning(f"Subject {seq['subject']} has no usable scans after preprocessing. Skipping subject.")
                print(f"[dataset]   -> SKIPPED subject (all {len(seq['scans'])} scans were unusable)", flush=True)
                skipped += 1
                continue

            subject_idx = len(_dataset)
            _dataset.append([subject_idx, snapshots, label])

            elapsed = time.time() - t_seq
            total_elapsed = time.time() - t_start
            avg_per_seq = total_elapsed / (idx + 1)
            eta = avg_per_seq * (total - idx - 1)
            print(f"[dataset]   -> Done in {elapsed:.1f}s | "
                  f"snapshots={len(snapshots)} | skipped_scans={skipped_scans} | "
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
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    for start_idx in range(0, len(dataset), batch_size):
        yield dataset[start_idx:start_idx + batch_size]
