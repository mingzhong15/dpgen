"""SOAP-based post-selection for DP-GEN candidate refinement.

Provides:
- SOAP descriptor computation (via dscribe)
- Dimensionality reduction (PCA / UMAP)
- Farthest Point Sampling (FPS) for uniform 2D selection
"""

import glob
import os
import warnings

import numpy as np

from dpgen import dlog
from dpgen.generator.constants import model_devi_name
from dpgen.generator.helpers import make_iter_name, _get_param_alias


def _read_frame_ase(task_path, frame_index, model_devi_engine, type_map,
                    model_devi_merge_traj, trj_freq=None, task_idx=None,
                    all_sys=None):
    """Read a single trajectory frame and return as an ASE Atoms object."""
    import dpdata
    from ase import Atoms

    if model_devi_engine == "lammps":
        if model_devi_merge_traj and all_sys is not None:
            sys = all_sys[task_idx][int(int(frame_index) / trj_freq)]
        else:
            conf_name = os.path.join(task_path, "traj", f"{frame_index}.lammpstrj")
            sys = dpdata.System(conf_name, fmt="lammps/dump", type_map=type_map)
    elif model_devi_engine == "gromacs":
        conf_name = os.path.join(task_path, "traj", f"{frame_index}.gromacstrj")
        sys = dpdata.System(conf_name, fmt="gromacs/gro", type_map=type_map)
    else:
        raise NotImplementedError(
            f"SOAP post-selection not supported for engine '{model_devi_engine}'"
        )

    # Convert dpdata System → ASE Atoms (handles dpdata < 1.1 without to_ase())
    d = sys.data
    symbols = [type_map[t] for t in d['atom_types']]
    return Atoms(
        symbols=symbols,
        positions=d['coords'][0],
        cell=d['cells'][0],
        pbc=True,
    )


def _compute_soap(ase_atoms_list, species, rcut, nmax, lmax):
    """Compute per-structure SOAP descriptors via dscribe.

    Parameters
    ----------
    ase_atoms_list : list[ase.Atoms]
        Input structures.
    species : list[str]
        Element symbols, e.g. ``["H", "C"]``.
    rcut : float
        SOAP cutoff radius.
    nmax : int
        Number of radial basis functions.
    lmax : int
        Maximum angular degree.

    Returns
    -------
    np.ndarray
        Descriptor matrix of shape ``(n_structures, soap_dim)``.
    """
    from dscribe.descriptors import SOAP

    soap = SOAP(
        species=species,
        periodic=True,
        r_cut=rcut,
        n_max=nmax,
        l_max=lmax,
        average="inner",
    )
    vectors = soap.create(ase_atoms_list)
    return np.array(vectors)


def _reduce_dim(method, X, jdata):
    """Reduce descriptor dimensionality to 2D.

    Parameters
    ----------
    method : str
        ``"pca"`` or ``"umap"``.
    X : np.ndarray
        Input of shape ``(n_samples, n_features)``.
    jdata : dict
        Run parameters (may contain UMAP/PCA hyper-parameters).

    Returns
    -------
    np.ndarray
        2D coordinates, shape ``(n_samples, 2)``.
    """
    if method == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=jdata.get("model_devi_post_select_seed"))
        return pca.fit_transform(X)

    elif method == "umap":
        pca_dim = jdata.get("model_devi_post_select_pca_dim", 32)
        umap_nn = jdata.get("model_devi_post_select_umap_n_neighbors", 15)
        umap_md = jdata.get("model_devi_post_select_umap_min_dist", 0.1)
        seed = jdata.get("model_devi_post_select_seed")

        from sklearn.decomposition import PCA
        import umap
        if pca_dim and pca_dim > 0 and X.shape[1] > pca_dim:
            X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_nn,
            min_dist=umap_md,
            random_state=seed,
        )
        return reducer.fit_transform(X)

    else:
        raise ValueError(f"Unknown dimensionality-reduction method: {method}")


def _farthest_point_sampling(points, n_select, seed=None):
    """Farthest Point Sampling in 2D space.

    Parameters
    ----------
    points : np.ndarray
        Coordinates, shape ``(n_points, 2)``.
    n_select : int
        Number of points to select.
    seed : int or None
        Random seed for the initial point.

    Returns
    -------
    list[int]
        Indices of selected points.
    """
    n = len(points)
    if n_select >= n:
        return list(range(n))
    if n_select < 1:
        return []

    rng = np.random.RandomState(seed)
    selected = [rng.randint(n)]
    dists = np.full(n, np.inf)

    for _ in range(1, n_select):
        new_dists = np.linalg.norm(points - points[selected[-1]], axis=1)
        np.minimum(dists, new_dists, out=dists)
        selected.append(int(np.argmax(dists)))
    return selected


def run_post_selection_per_system(
    iter_index, jdata, modd_path, system_id,
    fp_candidate, numb_task,
    modd_system_task, all_sys, trj_freq,
):
    """Run SOAP → dim-reduction → FPS for one system's candidate frames.

    Falls back to random selection on error.

    Parameters
    ----------
    iter_index : int
    jdata : dict
    modd_path : str
    system_id : str
    fp_candidate : list[[str, int]]
        ``[task_path, frame_index]`` pairs.
    numb_task : int
        Number of candidates to select via FPS.
    modd_system_task : list[str]
    all_sys : list or None
    trj_freq : int or None

    Returns
    -------
    list[[str, int]]
        Refined candidates in FPS selection order.
    """
    if numb_task <= 0 or not fp_candidate:
        return fp_candidate[:numb_task] if numb_task > 0 else []

    # Shortcut: if n_select is close to pool size, skip expensive SOAP
    if len(fp_candidate) <= max(numb_task, 3):
        dlog.info(f"system {system_id} : too few candidates ({len(fp_candidate)}), "
                  "skipping SOAP post-selection")
        return fp_candidate[:numb_task]

    model_devi_engine = jdata.get("model_devi_engine", "lammps")
    model_devi_merge_traj = jdata.get("model_devi_merge_traj", False)
    type_map = jdata["type_map"]

    # ----- 1. Read candidate structures -----
    ase_atoms_list = []
    candidate_keys = []
    traj_info_for_vis = []  # (task_path, frame_index, x, y, selected)

    for tt, ii in fp_candidate:
        try:
            tas = os.path.basename(tt)
            tidx = modd_system_task.index(tt) if model_devi_merge_traj else None
            atoms = _read_frame_ase(
                tt, ii, model_devi_engine, type_map,
                model_devi_merge_traj, trj_freq, tidx, all_sys,
            )
            ase_atoms_list.append(atoms)
            candidate_keys.append([tt, ii])
        except Exception as e:
            dlog.warning(f"system {system_id} : failed to read frame {ii} "
                         f"from {tt} – {e}")

    if len(ase_atoms_list) < 2:
        dlog.warning(f"system {system_id} : too few valid structures "
                     f"({len(ase_atoms_list)}), falling back to random")
        rng = np.random.RandomState(
            jdata.get("model_devi_post_select_seed")
        )
        idx = list(range(len(fp_candidate)))
        rng.shuffle(idx)
        return [fp_candidate[i] for i in idx[:numb_task]]

    # ----- 2. SOAP descriptors -----
    dlog.info(f"system {system_id} : computing SOAP for {len(ase_atoms_list)} frames ...")
    try:
        species = type_map
        rcut = jdata.get("model_devi_post_select_soap_rcut", 5.0)
        nmax = jdata.get("model_devi_post_select_soap_nmax", 8)
        lmax = jdata.get("model_devi_post_select_soap_lmax", 6)
        X = _compute_soap(ase_atoms_list, species, rcut, nmax, lmax)
    except Exception as e:
        dlog.warning(f"system {system_id} : SOAP computation failed ({e}), "
                     "falling back to random")
        rng = np.random.RandomState(
            jdata.get("model_devi_post_select_seed")
        )
        idx = list(range(len(fp_candidate)))
        rng.shuffle(idx)
        return [fp_candidate[i] for i in idx[:numb_task]]

    # ----- 3. Dimensionality reduction to 2D -----
    mode = jdata.get("model_devi_post_select_mode", "umap")
    dlog.info(f"system {system_id} : reducing to 2D via {mode.upper()} ...")
    try:
        coords_2d = _reduce_dim(mode, X, jdata)
    except Exception as e:
        dlog.warning(f"system {system_id} : {mode.upper()} failed ({e}), "
                     "falling back to random")
        rng = np.random.RandomState(
            jdata.get("model_devi_post_select_seed")
        )
        idx = list(range(len(fp_candidate)))
        rng.shuffle(idx)
        return [fp_candidate[i] for i in idx[:numb_task]]

    # ----- 4. Farthest Point Sampling -----
    n_select = min(numb_task, len(candidate_keys))
    dlog.info(f"system {system_id} : FPS selecting {n_select} / {len(candidate_keys)} frames")
    selected_idx = _farthest_point_sampling(
        coords_2d, n_select,
        seed=jdata.get("model_devi_post_select_seed"),
    )

    # ----- 5. Write per-task visualization files -----
    _save_vis_data(
        candidate_keys, coords_2d, selected_idx,
        modd_path, mode, model_devi_engine,
    )

    # Return in FPS order
    return [candidate_keys[i] for i in selected_idx]


def run_post_selection_from_cache(
    iter_index, jdata, modd_path, system_id,
    fp_candidate, numb_task,
    modd_system_task,
):
    """Run UMAP/PCA + FPS using pre-computed SOAP vectors from compute nodes.

    SOAP vectors were computed on each compute node (via ``post_md_soap.py``)
    and are expected as ``soap_vectors.npy`` + ``frame_indices.npy`` per task.
    Falls back to local computation if cache files are missing.
    """
    model_devi_merge_traj = jdata.get("model_devi_merge_traj", False)

    # Try to build a SOAP cache index from pre-computed files
    cache_available = True
    x_by_task = {}
    idx_by_task = {}
    for tt in modd_system_task:
        sv_file = os.path.join(tt, "soap_vectors.npy")
        fi_file = os.path.join(tt, "frame_indices.npy")
        if os.path.exists(sv_file) and os.path.exists(fi_file):
            x_by_task[tt] = np.load(sv_file)
            idx_by_task[tt] = np.load(fi_file)
        else:
            cache_available = False
            break

    if not cache_available or not x_by_task:
        dlog.info(f"system {system_id} : SOAP cache not available, "
                  "falling back to local SOAP computation")
        return run_post_selection_per_system(
            iter_index, jdata, modd_path, system_id,
            fp_candidate, numb_task,
            modd_system_task, None, None,
        )

    # Build candidate SOAP matrix from cache
    X_list = []
    candidate_keys = []
    for tt, frame_idx in fp_candidate:
        if tt in idx_by_task:
            arr = idx_by_task[tt]
            mask = arr == frame_idx
            pos = np.where(mask)[0]
            if len(pos) > 0 and pos[0] < len(x_by_task[tt]):
                X_list.append(x_by_task[tt][pos[0]])
                candidate_keys.append([tt, frame_idx])
                continue
        # Frame not found in cache – skip
        dlog.warning(f"system {system_id} : frame {frame_idx} from {tt} "
                     "not found in SOAP cache")
        candidate_keys.append([tt, frame_idx])
        X_list.append(None)

    # Filter out None entries
    valid = [i for i, x in enumerate(X_list) if x is not None]
    if len(valid) < 2:
        dlog.warning(f"system {system_id} : too few valid SOAP vectors "
                     f"({len(valid)}), falling back to random")
        rng = np.random.RandomState(
            jdata.get("model_devi_post_select_seed")
        )
        idx = list(range(len(fp_candidate)))
        rng.shuffle(idx)
        return [fp_candidate[i] for i in idx[:numb_task]]

    X_candidates = np.stack([X_list[i] for i in valid])
    valid_keys = [candidate_keys[i] for i in valid]

    # Dimension reduction
    mode = jdata.get("model_devi_post_select_mode", "umap")
    dlog.info(f"system {system_id} : reducing {len(valid)} SOAP vectors "
              f"to 2D via {mode.upper()} (from compute-node cache) ...")
    try:
        coords_2d = _reduce_dim(mode, X_candidates, jdata)
    except Exception as e:
        dlog.warning(f"system {system_id} : {mode.upper()} failed ({e}), "
                     "falling back to random")
        rng = np.random.RandomState(
            jdata.get("model_devi_post_select_seed")
        )
        idx = list(range(len(fp_candidate)))
        rng.shuffle(idx)
        return [fp_candidate[i] for i in idx[:numb_task]]

    # FPS
    n_select = min(numb_task, len(valid_keys))
    dlog.info(f"system {system_id} : FPS selecting {n_select} / "
              f"{len(valid_keys)} frames")
    selected_idx = _farthest_point_sampling(
        coords_2d, n_select,
        seed=jdata.get("model_devi_post_select_seed"),
    )

    # Write per-task visualization
    _save_vis_data(
        valid_keys, coords_2d, selected_idx,
        modd_path, mode, "lammps",
    )

    return [valid_keys[i] for i in selected_idx]


def _save_vis_data(candidate_keys, coords_2d, selected_idx,
                   modd_path, mode, model_devi_engine):
    """Write per-task ``umap.dat`` or ``pca.dat`` for visualisation."""
    suffix = "umap.dat" if mode == "umap" else "pca.dat"
    selected_set = set(selected_idx)

    per_task = {}
    for i, (tt, ii) in enumerate(candidate_keys):
        key = tt
        per_task.setdefault(key, []).append(
            (ii, coords_2d[i, 0], coords_2d[i, 1], 1 if i in selected_set else 0)
        )

    for tt, rows in per_task.items():
        traj_dir = os.path.join(tt, "traj")
        os.makedirs(traj_dir, exist_ok=True)
        out_path = os.path.join(traj_dir, suffix)
        try:
            rows.sort(key=lambda r: r[0])
            arr = np.array(rows, dtype=object)
            np.savetxt(
                out_path, arr,
                fmt="%d  %.6f  %.6f  %d",
                header="step        x            y            selected",
                comments="# ",
            )
        except Exception as e:
            dlog.warning(f"failed to write {out_path}: {e}")


def load_candidates(modd_path):
    """Load ``candidates.json`` from the model-devi directory.

    Returns
    -------
    dict
        ``{sys_idx: [[task_path, frame_index], ...]}``
    """
    import json
    path = os.path.join(modd_path, "candidates.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
