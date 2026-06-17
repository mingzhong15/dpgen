#!/usr/bin/env python3

"""First-principles (FP) labeling step of the DP-GEN workflow (``02.fp``).

Contains make/run/post for the FP labeling sub-step, plus the model-deviation
candidate selection logic (trust-level filtering). Dispatches on ``fp_style``
(vasp, pwscf, abacus, gaussian, siesta, cp2k, pwmat, amber/diff, cpx, custom).
"""

import copy
import glob
import json
import os
import random
import shutil
import warnings
from collections import Counter
from pathlib import Path

import dpdata
import numpy as np
import scipy.constants as pc
from numpy.linalg import norm

from dpgen import dlog
from dpgen.auto_test.lib.vasp import make_kspacing_kpoints
from dpgen.dispatcher.Dispatcher import make_submission
from dpgen.generator.constants import (
    calypso_model_devi_name,
    calypso_run_opt_name,
    cvasp_file,
    fp_name,
    fp_task_fmt,
    model_devi_name,
)
from dpgen.generator.helpers import (
    _get_param_alias,
    _get_model_suffix,
    dump_to_deepmd_raw,
    make_fp_task_name,
)
from dpgen.generator.lib.abacus_scf import (
    get_abacus_input_parameters,
    get_abacus_STRU,
    make_abacus_scf_input,
    make_abacus_scf_kpt,
    make_abacus_scf_stru,
)
from dpgen.generator.lib.cp2k import (
    make_cp2k_input,
    make_cp2k_input_from_external,
    make_cp2k_xyz,
)
from dpgen.generator.lib.ele_temp import NBandsEsti
from dpgen.generator.lib.gaussian import make_gaussian_input, take_cluster
from dpgen.generator.lib.lammps import get_all_dumped_forces, get_dumped_forces
from dpgen.generator.lib.parse_calypso import (
    _parse_calypso_dis_mtx,
    _parse_calypso_input,
)
from dpgen.generator.lib.pwmat import (
    make_pwmat_input_dict,
    make_pwmat_input_user_dict,
    write_input_dict,
)
from dpgen.generator.lib.pwscf import make_pwscf_input
from dpgen.generator.lib.siesta import make_siesta_input
from dpgen.generator.lib.utils import (
    check_api_version,
    create_path,
    make_iter_name,
    symlink_user_forward_files,
)
from dpgen.generator.lib.vasp import (
    incar_upper,
    make_vasp_incar_user_dict,
    write_incar_dict,
)
from dpgen.util import expand_sys_str, set_directory


def _to_face_dist(box_):
    box = np.reshape(box_, [3, 3])
    vol = np.abs(np.linalg.det(box))
    dists = []
    for [ii, jj] in [[0, 1], [1, 2], [2, 0]]:
        vv = np.cross(box[ii], box[jj])
        dists.append(vol / np.linalg.norm(vv))
    return np.array(dists)




def check_cluster(conf_name, fp_cluster_vacuum, fmt="lammps/dump"):
    sys = dpdata.System(conf_name, fmt)
    assert sys.get_nframes() == 1
    cell = sys.data["cells"][0]
    coord = sys.data["coords"][0]
    xlim = max(coord[:, 0]) - min(coord[:, 0])
    ylim = max(coord[:, 1]) - min(coord[:, 1])
    zlim = max(coord[:, 2]) - min(coord[:, 2])
    a, b, c = map(norm, [cell[0, :], cell[1, :], cell[2, :]])
    min_vac = min([a - xlim, b - ylim, c - zlim])
    # print([a-xlim,b-ylim,c-zlim])
    # _,r3d=miniball.get_bounding_ball(coord)

    if min_vac < fp_cluster_vacuum:
        is_bad = True
    else:
        is_bad = False
    return is_bad




def check_bad_box(conf_name, criteria, fmt="lammps/dump"):
    all_c = criteria.split(";")
    sys = dpdata.System(conf_name, fmt)
    assert sys.get_nframes() == 1
    is_bad = False
    for ii in all_c:
        [key, value] = ii.split(":")
        if key == "length_ratio":
            lengths = np.linalg.norm(sys["cells"][0], axis=1)
            ratio = np.max(lengths) / np.min(lengths)
            if ratio > float(value):
                is_bad = True
        elif key == "height_ratio":
            lengths = np.linalg.norm(sys["cells"][0], axis=1)
            dists = _to_face_dist(sys["cells"][0])
            ratio = np.max(lengths) / np.min(dists)
            if ratio > float(value):
                is_bad = True
        #
        elif key == "wrap_ratio":
            ratio = [
                sys["cells"][0][1][0] / sys["cells"][0][0][0],
                sys["cells"][0][2][1] / sys["cells"][0][1][1],
                sys["cells"][0][2][0] / sys["cells"][0][0][0],
            ]
            if np.max(np.abs(ratio)) > float(value):
                is_bad = True
        elif key == "tilt_ratio":
            ratio = [
                sys["cells"][0][1][0] / sys["cells"][0][1][1],
                sys["cells"][0][2][1] / sys["cells"][0][2][2],
                sys["cells"][0][2][0] / sys["cells"][0][2][2],
            ]
            if np.max(np.abs(ratio)) > float(value):
                is_bad = True
        else:
            raise RuntimeError("unknow key", key)
    return is_bad




def _read_model_devi_file(
    task_path: str,
    model_devi_f_avg_relative: bool = False,
    model_devi_merge_traj: bool = False,
):
    model_devi_files = glob.glob(os.path.join(task_path, "model_devi*.out"))
    if len(model_devi_files) > 1:
        model_devi_files_sorted = sorted(
            model_devi_files,
            key=lambda x: int(re.search(r"model_devi(\d+)\.out", x).group(1)),
        )
        with open(model_devi_files_sorted[0]) as f:
            first_line = f.readline()
        if not (first_line.startswith("#")):
            first_line = "#"
        num_beads = len(model_devi_files_sorted)
        model_devi_contents = []
        for file in model_devi_files_sorted:
            model_devi_contents.append(np.loadtxt(file))
        assert all(
            model_devi_content.shape[0] == model_devi_contents[0].shape[0]
            for model_devi_content in model_devi_contents
        ), (
            r"Not all beads generated the same number of lines in the model_devi${ibead}.out file. Check your pimd task carefully."
        )
        last_step = model_devi_contents[0][-1, 0]
        for ibead in range(1, num_beads):
            model_devi_contents[ibead][:, 0] = model_devi_contents[ibead][
                :, 0
            ] + ibead * (last_step + 1)
        model_devi = np.concatenate(model_devi_contents, axis=0)
        num_columns = model_devi.shape[1]
        formats = ["%12d"] + ["%22.6e"] * (num_columns - 1)
        np.savetxt(
            os.path.join(task_path, "model_devi.out"),
            model_devi,
            fmt=formats,
            header=first_line.rstrip(),
            comments="",
        )

        if not model_devi_merge_traj:
            num_digits = np.ceil(np.log10(num_beads + 1)).astype(int)
            traj_files_sorted = []
            for ibead in range(num_beads):
                traj_files = glob.glob(
                    os.path.join(
                        task_path, "traj", f"*lammpstrj{ibead + 1:0{num_digits}d}"
                    )
                )
                traj_files_sorted.append(
                    sorted(
                        traj_files,
                        key=lambda x: int(
                            re.search(r"^(\d+)\.lammpstrj", os.path.basename(x)).group(
                                1
                            )
                        ),
                    )
                )
            assert all(
                len(traj_list) == len(traj_files_sorted[0])
                for traj_list in traj_files_sorted
            ), (
                "Not all beads generated the same number of frames. Check your pimd task carefully."
            )
            for ibead in range(num_beads):
                for itraj in range(len(traj_files_sorted[0])):
                    base_path, original_filename = os.path.split(
                        traj_files_sorted[ibead][itraj]
                    )
                    frame_number = int(original_filename.split(".")[0])
                    new_filename = os.path.join(
                        base_path,
                        f"{frame_number + ibead * (int(last_step) + 1):d}.lammpstrj",
                    )
                    os.rename(traj_files_sorted[ibead][itraj], new_filename)
    model_devi = np.loadtxt(os.path.join(task_path, "model_devi.out"))
    if model_devi_f_avg_relative:
        if model_devi_merge_traj is True:
            all_traj = os.path.join(task_path, "all.lammpstrj")
            all_f = get_all_dumped_forces(all_traj)
        else:
            trajs = glob.glob(os.path.join(task_path, "traj", "*.lammpstrj"))
            all_f = []
            for ii in trajs:
                all_f.append(get_dumped_forces(ii))

        all_f = np.array(all_f)
        all_f = all_f.reshape([-1, 3])
        avg_f = np.sqrt(np.average(np.sum(np.square(all_f), axis=1)))
        model_devi[:, 4:7] = model_devi[:, 4:7] / avg_f
        np.savetxt(
            os.path.join(task_path, "model_devi_avgf.out"), model_devi, fmt="%16.6e"
        )
    return model_devi




def _select_by_model_devi_standard(
    modd_system_task: list[str],
    f_trust_lo: float,
    f_trust_hi: float,
    v_trust_lo: float,
    v_trust_hi: float,
    cluster_cutoff: float,
    model_devi_engine: str,
    model_devi_skip: int = 0,
    model_devi_f_avg_relative: bool = False,
    model_devi_merge_traj: bool = False,
    detailed_report_make_fp: bool = True,
):
    if model_devi_engine == "calypso":
        iter_name = modd_system_task[0].split("/")[0]
        _work_path = os.path.join(iter_name, model_devi_name)
        # calypso_run_opt_path = os.path.join(_work_path,calypso_run_opt_name)
        calypso_run_opt_path = glob.glob(f"{_work_path}/{calypso_run_opt_name}.*")[0]
        numofspecies = _parse_calypso_input("NumberOfSpecies", calypso_run_opt_path)
        min_dis = _parse_calypso_dis_mtx(numofspecies, calypso_run_opt_path)
    fp_candidate = []
    fp_rest_accurate = []
    fp_rest_failed = []
    cc = 0
    counter = Counter()
    counter["candidate"] = 0
    counter["failed"] = 0
    counter["accurate"] = 0
    for tt in modd_system_task:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all_conf = _read_model_devi_file(
                tt, model_devi_f_avg_relative, model_devi_merge_traj
            )

            if all_conf.shape == (7,):
                all_conf = all_conf.reshape(1, all_conf.shape[0])
            elif model_devi_engine == "calypso" and all_conf.shape == (8,):
                all_conf = all_conf.reshape(1, all_conf.shape[0])
            for ii in range(all_conf.shape[0]):
                if all_conf[ii][0] < model_devi_skip:
                    continue
                cc = int(all_conf[ii][0])
                if cluster_cutoff is None:
                    if model_devi_engine == "calypso":
                        if float(all_conf[ii][-1]) <= float(min_dis):
                            if detailed_report_make_fp:
                                fp_rest_failed.append([tt, cc])
                            counter["failed"] += 1
                            continue
                    if (
                        all_conf[ii][1] < v_trust_hi and all_conf[ii][1] >= v_trust_lo
                    ) or (
                        all_conf[ii][4] < f_trust_hi and all_conf[ii][4] >= f_trust_lo
                    ):
                        fp_candidate.append([tt, cc])
                        counter["candidate"] += 1
                    elif (all_conf[ii][1] >= v_trust_hi) or (
                        all_conf[ii][4] >= f_trust_hi
                    ):
                        if detailed_report_make_fp:
                            fp_rest_failed.append([tt, cc])
                        counter["failed"] += 1
                    elif all_conf[ii][1] < v_trust_lo and all_conf[ii][4] < f_trust_lo:
                        if detailed_report_make_fp:
                            fp_rest_accurate.append([tt, cc])
                        counter["accurate"] += 1
                    else:
                        if model_devi_engine == "calypso":
                            dlog.info(
                                "ase opt traj %s frame %d with f devi %f does not belong to either accurate, candidiate and failed "  # noqa: UP031
                                % (tt, ii, all_conf[ii][4])
                            )
                        else:
                            raise RuntimeError(
                                "md traj %s frame %d with f devi %f does not belong to either accurate, candidiate and failed, it should not happen"  # noqa: UP031
                                % (tt, ii, all_conf[ii][4])
                            )
                else:
                    idx_candidate = np.where(
                        np.logical_and(
                            all_conf[ii][7:] < f_trust_hi,
                            all_conf[ii][7:] >= f_trust_lo,
                        )
                    )[0]
                    for jj in idx_candidate:
                        fp_candidate.append([tt, cc, jj])
                    counter["candidate"] += len(idx_candidate)
                    idx_rest_accurate = np.where(all_conf[ii][7:] < f_trust_lo)[0]
                    if detailed_report_make_fp:
                        for jj in idx_rest_accurate:
                            fp_rest_accurate.append([tt, cc, jj])
                    counter["accurate"] += len(idx_rest_accurate)
                    idx_rest_failed = np.where(all_conf[ii][7:] >= f_trust_hi)[0]
                    if detailed_report_make_fp:
                        for jj in idx_rest_failed:
                            fp_rest_failed.append([tt, cc, jj])
                    counter["failed"] += len(idx_rest_failed)

    return fp_rest_accurate, fp_candidate, fp_rest_failed, counter




def _select_by_model_devi_adaptive_trust_low(
    modd_system_task: list[str],
    f_trust_hi: float,
    numb_candi_f: int,
    perc_candi_f: float,
    v_trust_hi: float,
    numb_candi_v: int,
    perc_candi_v: float,
    model_devi_skip: int = 0,
    model_devi_f_avg_relative: bool = False,
    model_devi_merge_traj: bool = False,
):
    """modd_system_task    model deviation tasks belonging to one system
    f_trust_hi
    numb_candi_f        number of candidate due to the f model deviation
    perc_candi_f        percentage of candidate due to the f model deviation
    v_trust_hi
    numb_candi_v        number of candidate due to the v model deviation
    perc_candi_v        percentage of candidate due to the v model deviation
    model_devi_skip.

    Returns
    -------
    accur               the accurate set
    candi               the candidate set
    failed              the failed set
    counter             counters, number of elements in the sets
    f_trust_lo          adapted trust level of f
    v_trust_lo          adapted trust level of v
    """
    idx_v = 1
    idx_f = 4
    accur = set()
    candi = set()
    failed = []
    coll_v = []
    coll_f = []
    for tt in modd_system_task:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_devi = _read_model_devi_file(
                tt, model_devi_f_avg_relative, model_devi_merge_traj
            )
            for ii in range(model_devi.shape[0]):
                if model_devi[ii][0] < model_devi_skip:
                    continue
                cc = int(model_devi[ii][0])
                # tt: name of task folder
                # cc: time step of the frame
                md_v = model_devi[ii][idx_v]
                md_f = model_devi[ii][idx_f]
                if md_f > f_trust_hi or md_v > v_trust_hi:
                    failed.append([tt, cc])
                else:
                    coll_v.append([model_devi[ii][idx_v], tt, cc])
                    coll_f.append([model_devi[ii][idx_f], tt, cc])
                    # now accur takes all non-failed frames,
                    # will be substracted by candidate lat er
                    accur.add((tt, cc))
    # sort
    coll_v.sort()
    coll_f.sort()
    assert len(coll_v) == len(coll_f)
    # calcuate numbers
    numb_candi_v = max(numb_candi_v, int(perc_candi_v * 0.01 * len(coll_v)))
    numb_candi_f = max(numb_candi_f, int(perc_candi_f * 0.01 * len(coll_f)))
    # adjust number of candidate
    if len(coll_v) < numb_candi_v:
        numb_candi_v = len(coll_v)
    if len(coll_f) < numb_candi_f:
        numb_candi_f = len(coll_f)
    # compute trust lo
    if numb_candi_v == 0:
        v_trust_lo = v_trust_hi
    else:
        v_trust_lo = coll_v[-numb_candi_v][0]
    if numb_candi_f == 0:
        f_trust_lo = f_trust_hi
    else:
        f_trust_lo = coll_f[-numb_candi_f][0]
    # add to candidate set
    for ii in range(len(coll_v) - numb_candi_v, len(coll_v)):
        candi.add(tuple(coll_v[ii][1:]))
    for ii in range(len(coll_f) - numb_candi_f, len(coll_f)):
        candi.add(tuple(coll_f[ii][1:]))
    # accurate set is substracted by the candidate set
    accur = accur - candi
    # convert to list
    candi = [list(ii) for ii in candi]
    accur = [list(ii) for ii in accur]
    # counters
    counter = Counter()
    counter["candidate"] = len(candi)
    counter["failed"] = len(failed)
    counter["accurate"] = len(accur)

    return accur, candi, failed, counter, f_trust_lo, v_trust_lo




def _make_fp_vasp_inner(
    iter_index,
    modd_path,
    work_path,
    model_devi_skip,
    v_trust_lo,
    v_trust_hi,
    f_trust_lo,
    f_trust_hi,
    fp_task_min,
    fp_task_max,
    fp_link_files,
    type_map,
    jdata,
):
    """iter_index          int             iter index
    modd_path           string          path of model devi
    work_path           string          path of fp
    fp_task_max         int             max number of tasks
    fp_link_files       [string]        linked files for fp, POTCAR for example
    fp_params           map             parameters for fp.
    """
    # --------------------------------------------------------------------------------------------------------------------------------------
    model_devi_engine = jdata.get("model_devi_engine", "lammps")
    if model_devi_engine == "calypso":
        iter_name = work_path.split("/")[0]
        _work_path = os.path.join(iter_name, model_devi_name)
        # calypso_run_opt_path = os.path.join(_work_path,calypso_run_opt_name)
        calypso_run_opt_path = glob.glob(f"{_work_path}/{calypso_run_opt_name}.*")[0]
        numofspecies = _parse_calypso_input("NumberOfSpecies", calypso_run_opt_path)
        min_dis = _parse_calypso_dis_mtx(numofspecies, calypso_run_opt_path)

        calypso_total_fp_num = 300
        modd_path = os.path.join(modd_path, calypso_model_devi_name)
        model_devi_skip = -1
        with open(os.path.join(modd_path, "Model_Devi.out")) as summfile:
            summary = np.loadtxt(summfile)
        summaryfmax = summary[:, -4]
        dis = summary[:, -1]
        acc = np.where((summaryfmax <= f_trust_lo) & (dis > float(min_dis)))
        fail = np.where((summaryfmax > f_trust_hi) | (dis <= float(min_dis)))
        nnan = np.where(np.isnan(summaryfmax))

        acc_num = len(acc[0])
        fail_num = len(fail[0])
        nan_num = len(nnan[0])
        tot = len(summaryfmax) - nan_num
        candi_num = tot - acc_num - fail_num
        dlog.info(
            f"summary  accurate_ratio: {acc_num * 100 / tot:8.4f}%  candidata_ratio: {candi_num * 100 / tot:8.4f}%  failed_ratio: {fail_num * 100 / tot:8.4f}%  in {tot:d} structures"
        )
    # --------------------------------------------------------------------------------------------------------------------------------------

    modd_task = glob.glob(os.path.join(modd_path, "task.*"))
    modd_task.sort()
    system_index = []
    for ii in modd_task:
        system_index.append(os.path.basename(ii).split(".")[1])

    set_tmp = set(system_index)
    system_index = list(set_tmp)
    system_index.sort()

    fp_tasks = []

    charges_recorder = []  # record charges for each fp_task
    charges_map = jdata.get("sys_charges", [])

    cluster_cutoff = jdata.get("cluster_cutoff", None)
    model_devi_adapt_trust_lo = jdata.get("model_devi_adapt_trust_lo", False)
    model_devi_f_avg_relative = jdata.get("model_devi_f_avg_relative", False)
    model_devi_merge_traj = jdata.get("model_devi_merge_traj", False)
    # skip save *.out if detailed_report_make_fp is False, default is True
    detailed_report_make_fp = jdata.get("detailed_report_make_fp", True)
    # skip bad box criteria
    skip_bad_box = jdata.get("fp_skip_bad_box")
    # skip discrete structure in cluster
    fp_cluster_vacuum = jdata.get("fp_cluster_vacuum", None)

    def _trust_limitation_check(sys_idx, lim):
        if isinstance(lim, list):
            sys_lim = lim[sys_idx]
        elif isinstance(lim, dict):
            sys_lim = lim[str(sys_idx)]
        else:
            sys_lim = lim
        return sys_lim

    for ss in system_index:
        modd_system_glob = os.path.join(modd_path, "task." + ss + ".*")
        modd_system_task = glob.glob(modd_system_glob)
        modd_system_task.sort()
        if model_devi_engine in ("lammps", "gromacs", "calypso"):
            # convert global trust limitations to local ones
            f_trust_lo_sys = _trust_limitation_check(int(ss), f_trust_lo)
            f_trust_hi_sys = _trust_limitation_check(int(ss), f_trust_hi)
            v_trust_lo_sys = _trust_limitation_check(int(ss), v_trust_lo)
            v_trust_hi_sys = _trust_limitation_check(int(ss), v_trust_hi)

            # assumed e -> v
            if not model_devi_adapt_trust_lo:
                (
                    fp_rest_accurate,
                    fp_candidate,
                    fp_rest_failed,
                    counter,
                ) = _select_by_model_devi_standard(
                    modd_system_task,
                    f_trust_lo_sys,
                    f_trust_hi_sys,
                    v_trust_lo_sys,
                    v_trust_hi_sys,
                    cluster_cutoff,
                    model_devi_engine,
                    model_devi_skip,
                    model_devi_f_avg_relative=model_devi_f_avg_relative,
                    model_devi_merge_traj=model_devi_merge_traj,
                    detailed_report_make_fp=detailed_report_make_fp,
                )
            else:
                numb_candi_f = jdata.get("model_devi_numb_candi_f", 10)
                numb_candi_v = jdata.get("model_devi_numb_candi_v", 0)
                perc_candi_f = jdata.get("model_devi_perc_candi_f", 0.0)
                perc_candi_v = jdata.get("model_devi_perc_candi_v", 0.0)
                (
                    fp_rest_accurate,
                    fp_candidate,
                    fp_rest_failed,
                    counter,
                    f_trust_lo_ad,
                    v_trust_lo_ad,
                ) = _select_by_model_devi_adaptive_trust_low(
                    modd_system_task,
                    f_trust_hi_sys,
                    numb_candi_f,
                    perc_candi_f,
                    v_trust_hi_sys,
                    numb_candi_v,
                    perc_candi_v,
                    model_devi_skip=model_devi_skip,
                    model_devi_f_avg_relative=model_devi_f_avg_relative,
                    model_devi_merge_traj=model_devi_merge_traj,
                )
                dlog.info(
                    "system {:s} {:9s} : f_trust_lo {:6.3f}   v_trust_lo {:6.3f}".format(
                        ss, "adapted", f_trust_lo_ad, v_trust_lo_ad
                    )
                )
        elif model_devi_engine == "amber":
            counter = Counter()
            counter["candidate"] = 0
            counter["failed"] = 0
            counter["accurate"] = 0
            fp_rest_accurate = []
            fp_candidate = []
            fp_rest_failed = []
            for tt in modd_system_task:
                cc = 0
                with open(os.path.join(tt, "rc.mdout")) as f:
                    skip_first = False
                    first_active = True
                    for line in f:
                        if line.startswith("     ntx     =       1"):
                            skip_first = True
                        if line.startswith(
                            "Active learning frame written with max. frc. std.:"
                        ):
                            if skip_first and first_active:
                                first_active = False
                                continue
                            model_devi = (
                                float(line.split()[-2])
                                * dpdata.unit.EnergyConversion("kcal_mol", "eV").value()
                            )
                            if model_devi < f_trust_lo:
                                # accurate
                                if detailed_report_make_fp:
                                    fp_rest_accurate.append([tt, cc])
                                counter["accurate"] += 1
                            elif model_devi > f_trust_hi:
                                # failed
                                if detailed_report_make_fp:
                                    fp_rest_failed.append([tt, cc])
                                counter["failed"] += 1
                            else:
                                # candidate
                                fp_candidate.append([tt, cc])
                                counter["candidate"] += 1
                            cc += 1

        else:
            raise RuntimeError("unknown model_devi_engine", model_devi_engine)

        # print a report
        fp_sum = sum(counter.values())

        if fp_sum == 0:
            dlog.info(f"system {ss:s} has no fp task, maybe the model devi is nan %")
            continue
        for cc_key, cc_value in counter.items():
            dlog.info(
                f"system {ss:s} {cc_key:9s} : {cc_value:6d} in {fp_sum:6d} {cc_value / fp_sum * 100:6.2f} %"
            )
        random.shuffle(fp_candidate)
        if detailed_report_make_fp:
            random.shuffle(fp_rest_failed)
            random.shuffle(fp_rest_accurate)
            with open(
                os.path.join(work_path, f"candidate.shuffled.{ss}.out"), "w"
            ) as fp:
                for ii in fp_candidate:
                    fp.write(" ".join([str(nn) for nn in ii]) + "\n")
            with open(
                os.path.join(work_path, f"rest_accurate.shuffled.{ss}.out"), "w"
            ) as fp:
                for ii in fp_rest_accurate:
                    fp.write(" ".join([str(nn) for nn in ii]) + "\n")
            with open(
                os.path.join(work_path, f"rest_failed.shuffled.{ss}.out"), "w"
            ) as fp:
                for ii in fp_rest_failed:
                    fp.write(" ".join([str(nn) for nn in ii]) + "\n")

        # set number of tasks
        accurate_ratio = float(counter["accurate"]) / float(fp_sum)
        fp_accurate_threshold = jdata.get("fp_accurate_threshold", 1)
        fp_accurate_soft_threshold = jdata.get(
            "fp_accurate_soft_threshold", fp_accurate_threshold
        )
        if accurate_ratio < fp_accurate_soft_threshold:
            this_fp_task_max = fp_task_max
        elif (
            accurate_ratio >= fp_accurate_soft_threshold
            and accurate_ratio < fp_accurate_threshold
        ):
            this_fp_task_max = int(
                fp_task_max
                * (accurate_ratio - fp_accurate_threshold)
                / (fp_accurate_soft_threshold - fp_accurate_threshold)
            )
        else:
            this_fp_task_max = 0
        # ----------------------------------------------------------------------------
        if model_devi_engine == "calypso":
            calypso_intend_fp_num_temp = (
                len(fp_candidate) / candi_num
            ) * calypso_total_fp_num
            if calypso_intend_fp_num_temp < 1:
                calypso_intend_fp_num = 1
            else:
                calypso_intend_fp_num = int(calypso_intend_fp_num_temp)
        # ----------------------------------------------------------------------------
        numb_task = min(this_fp_task_max, len(fp_candidate))
        if numb_task < fp_task_min:
            numb_task = 0

        # ----------------------------------------------------------------------------
        if (model_devi_engine == "calypso" and len(jdata.get("type_map")) == 1) or (
            model_devi_engine == "calypso"
            and len(jdata.get("type_map")) > 1
            and candi_num <= calypso_total_fp_num
        ):
            numb_task = min(this_fp_task_max, len(fp_candidate))
            if numb_task < fp_task_min:
                numb_task = 0
        elif (
            model_devi_engine == "calypso"
            and len(jdata.get("type_map")) > 1
            and candi_num > calypso_total_fp_num
        ):
            numb_task = calypso_intend_fp_num
            if len(fp_candidate) < numb_task:
                numb_task = 0
        # ----------------------------------------------------------------------------
        dlog.info(
            f"system {ss:s} accurate_ratio: {accurate_ratio:8.4f}    thresholds: {fp_accurate_soft_threshold:6.4f} and {fp_accurate_threshold:6.4f}   eff. task min and max {fp_task_min:4d} {this_fp_task_max:4d}   number of fp tasks: {numb_task:6d}"
        )
        # make fp tasks

        # read all.lammpstrj, save in all_sys for each system_index
        all_sys = []
        trj_freq = None
        netcdftraj = None
        if model_devi_merge_traj:
            for ii in modd_system_task:
                all_traj = os.path.join(ii, "all.lammpstrj")
                all_sys_per_task = dpdata.System(
                    all_traj, fmt="lammps/dump", type_map=type_map
                )
                all_sys.append(all_sys_per_task)
            model_devi_jobs = jdata["model_devi_jobs"]
            cur_job = model_devi_jobs[iter_index]
            trj_freq = int(
                _get_param_alias(cur_job, ["t_freq", "trj_freq", "traj_freq"])
            )

        count_bad_box = 0
        count_bad_cluster = 0
        fp_candidate = sorted(fp_candidate[:numb_task])

        for cc in range(numb_task):
            tt = fp_candidate[cc][0]
            ii = fp_candidate[cc][1]
            ss = os.path.basename(tt).split(".")[1]
            conf_name = os.path.join(tt, "traj")
            conf_sys = None
            if model_devi_engine == "lammps":
                if model_devi_merge_traj:
                    conf_sys = all_sys[int(os.path.basename(tt).split(".")[-1])][
                        int(int(ii) / trj_freq)
                    ]
                else:
                    conf_name = os.path.join(conf_name, str(ii) + ".lammpstrj")
                ffmt = "lammps/dump"
            elif model_devi_engine == "gromacs":
                conf_name = os.path.join(conf_name, str(ii) + ".gromacstrj")
                ffmt = "lammps/dump"
            elif model_devi_engine == "amber":
                conf_name = os.path.join(tt, "rc.nc")
                rst_name = os.path.abspath(os.path.join(tt, "init.rst7"))
            elif model_devi_engine == "calypso":
                conf_name = os.path.join(conf_name, str(ii) + ".poscar")
                ffmt = "vasp/poscar"
            else:
                raise RuntimeError("unknown model_devi engine", model_devi_engine)
            conf_name = os.path.abspath(conf_name)
            if skip_bad_box is not None:
                skip = check_bad_box(conf_name, skip_bad_box, fmt=ffmt)
                if skip:
                    count_bad_box += 1
                    continue

            if fp_cluster_vacuum is not None:
                assert fp_cluster_vacuum > 0
                skip_cluster = check_cluster(conf_name, fp_cluster_vacuum)
                if skip_cluster:
                    count_bad_cluster += 1
                    continue

            if model_devi_engine != "calypso":
                # link job.json
                job_name = os.path.join(tt, "job.json")
                job_name = os.path.abspath(job_name)

            if cluster_cutoff is not None:
                # take clusters
                jj = fp_candidate[cc][2]
                poscar_name = f"{conf_name}.cluster.{jj}.POSCAR"
                new_system = take_cluster(conf_name, type_map, jj, jdata)
                new_system.to_vasp_poscar(poscar_name)
            fp_task_name = make_fp_task_name(int(ss), cc)
            fp_task_path = os.path.join(work_path, fp_task_name)
            create_path(fp_task_path)
            fp_tasks.append(fp_task_path)
            if charges_map:
                charges_recorder.append(charges_map[int(ss)])
            cwd = os.getcwd()
            os.chdir(fp_task_path)
            if cluster_cutoff is None:
                if model_devi_engine == "lammps":
                    if model_devi_merge_traj:
                        conf_sys.to("lammps/lmp", "conf.dump")
                    else:
                        os.symlink(os.path.relpath(conf_name), "conf.dump")
                    os.symlink(os.path.relpath(job_name), "job.json")
                elif model_devi_engine == "gromacs":
                    os.symlink(os.path.relpath(conf_name), "conf.dump")
                    os.symlink(os.path.relpath(job_name), "job.json")
                elif model_devi_engine == "amber":
                    # read and write with ase
                    from ase.io.netcdftrajectory import (
                        NetCDFTrajectory,
                        write_netcdftrajectory,
                    )

                    if cc > 0 and tt == fp_candidate[cc - 1][0]:
                        # same MD task, use the same file
                        pass
                    else:
                        # not the same file
                        if cc > 0:
                            # close the old file
                            netcdftraj.close()
                        netcdftraj = NetCDFTrajectory(conf_name)
                    # write nc file
                    write_netcdftrajectory("rc.nc", netcdftraj[ii])
                    if cc >= numb_task - 1:
                        netcdftraj.close()
                    # link restart since it's necessary to start Amber
                    os.symlink(os.path.relpath(rst_name), "init.rst7")
                    os.symlink(os.path.relpath(job_name), "job.json")
                elif model_devi_engine == "calypso":
                    os.symlink(os.path.relpath(conf_name), "POSCAR")
                    fjob = open("job.json", "w+")
                    fjob.write('{"model_devi_engine":"calypso"}')
                    fjob.close()
                    # os.system('touch job.json')
                else:
                    raise RuntimeError("unknown model_devi_engine", model_devi_engine)
            else:
                os.symlink(os.path.relpath(poscar_name), "POSCAR")
                np.save("atom_pref", new_system.data["atom_pref"])
            for pair in fp_link_files:
                os.symlink(pair[0], pair[1])
            os.chdir(cwd)
        if count_bad_box > 0:
            dlog.info(
                f"system {ss:s} skipped {count_bad_box:6d} confs with bad box, {numb_task - count_bad_box:6d} remains"
            )
        if count_bad_cluster > 0:
            dlog.info(
                f"system {ss:s} skipped {count_bad_cluster:6d} confs with bad cluster, {numb_task - count_bad_cluster:6d} remains"
            )
    if model_devi_engine == "calypso":
        dlog.info(
            f"summary  accurate_ratio: {acc_num * 100 / tot:8.4f}%  candidata_ratio: {candi_num * 100 / tot:8.4f}%  failed_ratio: {fail_num * 100 / tot:8.4f}%  in {tot:d} structures"
        )
    if cluster_cutoff is None:
        cwd = os.getcwd()
        for idx, task in enumerate(fp_tasks):
            os.chdir(task)
            if model_devi_engine == "lammps":
                sys = None
                if model_devi_merge_traj:
                    sys = dpdata.System(
                        "conf.dump", fmt="lammps/lmp", type_map=type_map
                    )
                else:
                    sys = dpdata.System(
                        "conf.dump", fmt="lammps/dump", type_map=type_map
                    )
                sys.to_vasp_poscar("POSCAR")
                # dump to poscar

                if charges_map:
                    warnings.warn(
                        '"sys_charges" keyword only support for gromacs engine now.'
                    )
            elif model_devi_engine == "gromacs":
                # dump_to_poscar('conf.dump', 'POSCAR', type_map, fmt = "gromacs/gro")
                if charges_map:
                    dump_to_deepmd_raw(
                        "conf.dump",
                        "deepmd.raw",
                        type_map,
                        fmt="gromacs/gro",
                        charge=charges_recorder[idx],
                    )
                else:
                    dump_to_deepmd_raw(
                        "conf.dump",
                        "deepmd.raw",
                        type_map,
                        fmt="gromacs/gro",
                        charge=None,
                    )
            elif model_devi_engine in ("amber", "calypso"):
                pass
            else:
                raise RuntimeError("unknown model_devi engine", model_devi_engine)
            os.chdir(cwd)
    return fp_tasks




def make_vasp_incar(jdata, filename):
    if "fp_incar" in jdata.keys():
        fp_incar_path = jdata["fp_incar"]
        assert os.path.exists(fp_incar_path)
        fp_incar_path = os.path.abspath(fp_incar_path)
        fr = open(fp_incar_path)
        incar = fr.read()
        fr.close()
    elif "user_fp_params" in jdata.keys():
        incar = write_incar_dict(jdata["user_fp_params"])
    else:
        incar = make_vasp_incar_user_dict(jdata["fp_params"])
    with open(filename, "w") as fp:
        fp.write(incar)
    return incar




def make_pwmat_input(jdata, filename):
    if "fp_incar" in jdata.keys():
        fp_incar_path = jdata["fp_incar"]
        assert os.path.exists(fp_incar_path)
        fp_incar_path = os.path.abspath(fp_incar_path)
        fr = open(fp_incar_path)
        input = fr.read()
        fr.close()
    elif "user_fp_params" in jdata.keys():
        fp_params = jdata["user_fp_params"]
        node1 = fp_params["node1"]
        node2 = fp_params["node2"]
        atom_config = fp_params["in.atom"]
        ecut = fp_params["ecut"]
        e_error = fp_params["e_error"]
        rho_error = fp_params["rho_error"]
        kspacing = fp_params["kspacing"]
        flag_symm = fp_params["flag_symm"]
        os.system("command -v poscar2config.x | wc -l > 1.txt")
        fc = open("1.txt")
        flag_command = fc.read()
        fc.close()
        if int(flag_command) == 1:
            os.system("poscar2config.x < POSCAR > tmp.config")
        else:
            os.system(
                "cp ../../../out_data_post_fp_pwmat/02.fp/task.000.000000/poscar2config.x ./"
            )
            os.system("./poscar2config.x < POSCAR > tmp.config")
        os.system("rm -rf tmp.config")
        input_dict = make_pwmat_input_dict(
            node1,
            node2,
            atom_config,
            ecut,
            e_error,
            rho_error,
            icmix=None,
            smearing=None,
            sigma=None,
            kspacing=kspacing,
            flag_symm=flag_symm,
        )

        input = write_input_dict(input_dict)
    else:
        input = make_pwmat_input_user_dict(jdata["fp_params"])
    if "IN.PSP" in input or "in.psp" in input:
        with open(filename, "w") as fp:
            fp.write(input)
            fp.write("job=scf\n")
            if "OUT.MLMD" in input or "out.mlmd" in input:
                return input
            else:
                fp.write("OUT.MLMD = T")
                return input
    else:
        with open(filename, "w") as fp:
            fp.write(input)
            fp.write("job=scf\n")
            fp_pp_files = jdata["fp_pp_files"]
            for idx, ii in enumerate(fp_pp_files):
                fp.write("IN.PSP%d = %s\n" % (idx + 1, ii))  # noqa: UP031
            if "OUT.MLMD" in input or "out.mlmd" in input:
                return input
            else:
                fp.write("OUT.MLMD = T")
                return input




def make_vasp_incar_ele_temp(jdata, filename, ele_temp, nbands_esti=None):
    from pymatgen.io.vasp import Incar

    with open(filename) as fp:
        incar = fp.read()
    try:
        incar = incar_upper(Incar.from_string(incar))
    except AttributeError:
        incar = incar_upper(Incar.from_str(incar))
    incar["ISMEAR"] = -1
    incar["SIGMA"] = ele_temp * pc.Boltzmann / pc.electron_volt
    incar.write_file("INCAR")
    if nbands_esti is not None:
        nbands = nbands_esti.predict(".")
        with open(filename) as fp:
            try:
                incar = Incar.from_string(fp.read())
            except AttributeError:
                incar = Incar.from_str(fp.read())
        incar["NBANDS"] = nbands
        incar.write_file("INCAR")




def make_fp_vasp_incar(iter_index, jdata, nbands_esti=None):
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return
    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)
        make_vasp_incar(jdata, "INCAR")
        if os.path.exists("job.json"):
            with open("job.json") as fp:
                job_data = json.load(fp)
            if "ele_temp" in job_data:
                make_vasp_incar_ele_temp(
                    jdata, "INCAR", job_data["ele_temp"], nbands_esti=nbands_esti
                )
        os.chdir(cwd)




def _make_fp_pwmat_input(iter_index, jdata):
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return
    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)
        make_pwmat_input(jdata, "etot.input")
        os.system("sed -i '1,2c 4 1' etot.input")
        os.chdir(cwd)




def make_fp_vasp_cp_cvasp(iter_index, jdata):
    # Move cvasp interface to jdata
    if ("cvasp" in jdata) and (jdata["cvasp"] is True):
        pass
    else:
        return
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return
    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)
        # copy cvasp.py
        shutil.copyfile(cvasp_file, "cvasp.py")
        os.chdir(cwd)




def make_fp_vasp_kp(iter_index, jdata):
    from pymatgen.io.vasp import Incar, Kpoints

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_aniso_kspacing = jdata.get("fp_aniso_kspacing")

    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return
    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)
        # get kspacing and kgamma from incar
        assert os.path.exists("INCAR")
        with open("INCAR") as fp:
            incar = fp.read()
        try:
            standard_incar = incar_upper(Incar.from_string(incar))
        except AttributeError:
            standard_incar = incar_upper(Incar.from_str(incar))
        if fp_aniso_kspacing is None:
            try:
                kspacing = standard_incar["KSPACING"]
            except KeyError:
                raise RuntimeError("KSPACING must be given in INCAR")
        else:
            kspacing = fp_aniso_kspacing
        try:
            gamma = standard_incar["KGAMMA"]
            if isinstance(gamma, bool):
                pass
            else:
                if gamma[0].upper() == "T":
                    gamma = True
                else:
                    gamma = False
        except KeyError:
            raise RuntimeError("KGAMMA must be given in INCAR")
        # check poscar
        assert os.path.exists("POSCAR")
        # make kpoints
        ret = make_kspacing_kpoints("POSCAR", kspacing, gamma)
        try:
            kp = Kpoints.from_string(ret)
        except AttributeError:
            kp = Kpoints.from_str(ret)
        kp.write_file("KPOINTS")
        os.chdir(cwd)




def _link_fp_vasp_pp(iter_index, jdata):
    fp_pp_path = jdata["fp_pp_path"]
    fp_pp_files = jdata["fp_pp_files"]
    assert os.path.exists(fp_pp_path)
    fp_pp_path = os.path.abspath(fp_pp_path)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)

    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return
    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)
        for jj in fp_pp_files:
            pp_file = os.path.join(fp_pp_path, jj)
            os.symlink(pp_file, jj)
        os.chdir(cwd)




def sys_link_fp_vasp_pp(iter_index, jdata):
    fp_pp_path = jdata["fp_pp_path"]
    fp_pp_files = jdata["fp_pp_files"]
    fp_pp_path = os.path.abspath(fp_pp_path)
    type_map = jdata["type_map"]
    assert os.path.exists(fp_pp_path)
    assert len(fp_pp_files) == len(type_map), (
        "size of fp_pp_files should be the same as the size of type_map"
    )

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)

    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    system_idx_str = [os.path.basename(ii).split(".")[1] for ii in fp_tasks]
    system_idx_str = list(set(system_idx_str))
    system_idx_str.sort()
    for ii in system_idx_str:
        potcars = []
        sys_tasks = glob.glob(os.path.join(work_path, f"task.{ii}.*"))
        assert len(sys_tasks) != 0
        sys_poscar = os.path.join(sys_tasks[0], "POSCAR")
        sys = dpdata.System(sys_poscar, fmt="vasp/poscar")
        for ele_name in sys["atom_names"]:
            ele_idx = jdata["type_map"].index(ele_name)
            potcars.append(fp_pp_files[ele_idx])
        with open(os.path.join(work_path, f"POTCAR.{ii}"), "w") as fp_pot:
            for jj in potcars:
                with open(os.path.join(fp_pp_path, jj)) as fp:
                    fp_pot.write(fp.read())
        sys_tasks = glob.glob(os.path.join(work_path, f"task.{ii}.*"))
        cwd = os.getcwd()
        for jj in sys_tasks:
            os.chdir(jj)
            os.symlink(os.path.join("..", f"POTCAR.{ii}"), "POTCAR")
            os.chdir(cwd)




def _link_fp_abacus_pporb_descript(iter_index, jdata):
    # assume pp orbital files, numerical descrptors and model for dpks are all in fp_pp_path.
    fp_pp_path = os.path.abspath(jdata["fp_pp_path"])
    type_map = jdata["type_map"]

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)

        # get value of 'deepks_model' from INPUT
        input_param = get_abacus_input_parameters("INPUT")
        fp_dpks_model = input_param.get("deepks_model", None)
        if fp_dpks_model is not None:
            model_file = os.path.join(
                fp_pp_path, os.path.split(fp_dpks_model)[1]
            )  # only the filename
            assert os.path.isfile(model_file), (
                f"Can not find the deepks model file {model_file}, which is defined in {ii}/INPUT"
            )
            os.symlink(model_file, fp_dpks_model)  # link to the model file

        # get pp, orb, descriptor filenames from STRU
        stru_param = get_abacus_STRU("STRU")
        atom_names = stru_param["atom_names"]
        pp_files_stru = stru_param.get("pp_files", None)
        orb_files_stru = stru_param.get("orb_files", None)
        descriptor_file_stru = stru_param.get("dpks_descriptor", None)

        if pp_files_stru:
            assert "fp_pp_files" in jdata, "need to define fp_pp_files in jdata"
        if orb_files_stru:
            assert "fp_orb_files" in jdata, "need to define fp_orb_files in jdata"
        if descriptor_file_stru:
            assert "fp_dpks_descriptor" in jdata, (
                "need to define fp_dpks_descriptor in jdata"
            )

        for idx, iatom in enumerate(atom_names):
            type_map_idx = type_map.index(iatom)
            if iatom not in type_map:
                raise RuntimeError(
                    f"atom name {iatom} in STRU is not defined in type_map"
                )
            if pp_files_stru:
                src_file = os.path.join(fp_pp_path, jdata["fp_pp_files"][type_map_idx])
                assert os.path.isfile(src_file), (
                    f"Can not find the pseudopotential file {src_file}"
                )
                os.symlink(src_file, pp_files_stru[idx])
            if orb_files_stru:
                src_file = os.path.join(fp_pp_path, jdata["fp_orb_files"][type_map_idx])
                assert os.path.isfile(src_file), (
                    f"Can not find the orbital file {src_file}"
                )
                os.symlink(src_file, orb_files_stru[idx])
        if descriptor_file_stru:
            src_file = os.path.join(fp_pp_path, jdata["fp_dpks_descriptor"])
            assert os.path.isfile(src_file), (
                f"Can not find the descriptor file {src_file}"
            )
            os.symlink(src_file, descriptor_file_stru)

        os.chdir(cwd)




def _make_fp_vasp_configs(iter_index: int, jdata: dict):
    """Read the model deviation from model_devi step, and then generate the candidated structures
    in 02.fp directory.

    Currently, the formats of generated structures are decided by model_devi_eigne.

    Parameters
    ----------
    iter_index : int
        The index of iteration.
    jdata : dict
        The json data.

    Returns
    -------
    int
        The number of the candidated structures.
    """
    # TODO: we need to unify different data formats
    fp_task_max = jdata["fp_task_max"]
    model_devi_skip = jdata["model_devi_skip"]
    type_map = jdata["type_map"]
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    create_path(work_path)

    modd_path = os.path.join(iter_name, model_devi_name)
    task_min = -1
    if os.path.isfile(os.path.join(modd_path, "cur_job.json")):
        cur_job = json.load(open(os.path.join(modd_path, "cur_job.json")))
        if "task_min" in cur_job:
            task_min = cur_job["task_min"]
    else:
        cur_job = {}
    # support iteration dependent trust levels
    v_trust_lo = cur_job.get(
        "model_devi_v_trust_lo", jdata.get("model_devi_v_trust_lo", 1e10)
    )
    v_trust_hi = cur_job.get(
        "model_devi_v_trust_hi", jdata.get("model_devi_v_trust_hi", 1e10)
    )
    if cur_job.get("model_devi_f_trust_lo") is not None:
        f_trust_lo = cur_job.get("model_devi_f_trust_lo")
    else:
        f_trust_lo = jdata["model_devi_f_trust_lo"]
    if cur_job.get("model_devi_f_trust_hi") is not None:
        f_trust_hi = cur_job.get("model_devi_f_trust_hi")
    else:
        f_trust_hi = jdata["model_devi_f_trust_hi"]

    # make configs
    fp_tasks = _make_fp_vasp_inner(
        iter_index,
        modd_path,
        work_path,
        model_devi_skip,
        v_trust_lo,
        v_trust_hi,
        f_trust_lo,
        f_trust_hi,
        task_min,
        fp_task_max,
        [],
        type_map,
        jdata,
    )
    return fp_tasks




def make_fp_vasp(iter_index, jdata):
    # abs path for fp_incar if it exists
    if "fp_incar" in jdata:
        jdata["fp_incar"] = os.path.abspath(jdata["fp_incar"])
    # get nbands esti if it exists
    if "fp_nbands_esti_data" in jdata:
        nbe = NBandsEsti(jdata["fp_nbands_esti_data"])
    else:
        nbe = None
    # order is critical!
    # 1, create potcar
    sys_link_fp_vasp_pp(iter_index, jdata)
    # 2, create incar
    make_fp_vasp_incar(iter_index, jdata, nbands_esti=nbe)
    # 3, create kpoints
    make_fp_vasp_kp(iter_index, jdata)
    # 4, copy cvasp
    make_fp_vasp_cp_cvasp(iter_index, jdata)




def make_fp_pwscf(iter_index, jdata):
    work_path = os.path.join(make_iter_name(iter_index), fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    # make pwscf input
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_pp_files = jdata["fp_pp_files"]
    if "user_fp_params" in jdata.keys():
        fp_params = jdata["user_fp_params"]
        user_input = True
    else:
        fp_params = jdata["fp_params"]
        user_input = False
    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)
        sys_data = dpdata.System("POSCAR").data
        sys_data["atom_masses"] = []
        pps = []
        for iii in sys_data["atom_names"]:
            sys_data["atom_masses"].append(
                jdata["mass_map"][jdata["type_map"].index(iii)]
            )
            pps.append(fp_pp_files[jdata["type_map"].index(iii)])
        ret = make_pwscf_input(sys_data, pps, fp_params, user_input=user_input)
        with open("input", "w") as fp:
            fp.write(ret)
        os.chdir(cwd)
    # link pp files
    _link_fp_vasp_pp(iter_index, jdata)




def make_fp_abacus_scf(iter_index, jdata):
    work_path = os.path.join(make_iter_name(iter_index), fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    pporb_path = "pporb"
    # make abacus/pw/scf input
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_pp_files = jdata["fp_pp_files"]
    fp_orb_files = None
    fp_dpks_descriptor = None
    # get paramters for writting INPUT file
    fp_params = {}
    if "user_fp_params" in jdata.keys():
        fp_params = jdata["user_fp_params"]
    elif "fp_incar" in jdata.keys():
        fp_input_path = jdata["fp_incar"]
        assert os.path.exists(fp_input_path)
        fp_input_path = os.path.abspath(fp_input_path)
        fp_params = get_abacus_input_parameters(fp_input_path)
    else:
        raise RuntimeError(
            "Set 'user_fp_params' or 'fp_incar' in json file to make INPUT of ABACUS"
        )
    ret_input = make_abacus_scf_input(fp_params, extra_file_path=pporb_path)

    # Get orbital and deepks setting
    if "basis_type" in fp_params:
        if fp_params["basis_type"] == "lcao":
            assert (
                "fp_orb_files" in jdata
                and isinstance(jdata["fp_orb_files"], list)
                and len(jdata["fp_orb_files"]) == len(fp_pp_files)
            )
            fp_orb_files = jdata["fp_orb_files"]
    dpks_out_labels = fp_params.get("deepks_out_labels", 0)
    dpks_scf = fp_params.get("deepks_scf", 0)
    if dpks_out_labels or dpks_scf:
        assert "fp_dpks_descriptor" in jdata and isinstance(
            jdata["fp_dpks_descriptor"], str
        )
        fp_dpks_descriptor = jdata["fp_dpks_descriptor"]

    # get paramters for writting KPT file
    if "kspacing" not in fp_params.keys():
        if "gamma_only" in fp_params.keys():
            if fp_params["gamma_only"] == 1:
                gamma_param = {"k_points": [1, 1, 1, 0, 0, 0]}
                ret_kpt = make_abacus_scf_kpt(gamma_param)
            else:
                if "k_points" in jdata.keys():
                    ret_kpt = make_abacus_scf_kpt(jdata)
                elif "fp_kpt_file" in jdata.keys():
                    fp_kpt_path = jdata["fp_kpt_file"]
                    assert os.path.exists(fp_kpt_path)
                    fp_kpt_path = os.path.abspath(fp_kpt_path)
                    fk = open(fp_kpt_path)
                    ret_kpt = fk.read()
                    fk.close()
                else:
                    raise RuntimeError("Cannot find any k-points information")
        else:
            if "k_points" in jdata.keys():
                ret_kpt = make_abacus_scf_kpt(jdata)
            elif "fp_kpt_file" in jdata.keys():
                fp_kpt_path = jdata["fp_kpt_file"]
                assert os.path.exists(fp_kpt_path)
                fp_kpt_path = os.path.abspath(fp_kpt_path)
                fk = open(fp_kpt_path)
                ret_kpt = fk.read()
                fk.close()
            else:
                gamma_param = {"k_points": [1, 1, 1, 0, 0, 0]}
                ret_kpt = make_abacus_scf_kpt(gamma_param)
                warnings.warn(
                    "Cannot find k-points information, gamma_only will be generated."
                )

    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)
        sys_data = dpdata.System("POSCAR").data
        if "mass_map" in jdata:
            sys_data["atom_masses"] = jdata["mass_map"]
        with open("INPUT", "w") as fp:
            fp.write(ret_input)
        if "kspacing" not in fp_params.keys():
            with open("KPT", "w") as fp:
                fp.write(ret_kpt)
        ret_stru = make_abacus_scf_stru(
            sys_data,
            fp_pp_files,
            fp_orb_files,
            fp_dpks_descriptor,
            type_map=jdata["type_map"],
            pporb=pporb_path,
        )
        with open("STRU", "w") as fp:
            fp.write(ret_stru)

        if not os.path.isdir(pporb_path):
            os.makedirs(pporb_path)

        os.chdir(cwd)
    # link pp and orbital files
    _link_fp_abacus_pporb_descript(iter_index, jdata)




def make_fp_siesta(iter_index, jdata):
    work_path = os.path.join(make_iter_name(iter_index), fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    # make siesta input
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_pp_files = jdata["fp_pp_files"]
    if "user_fp_params" in jdata.keys():
        fp_params = jdata["user_fp_params"]
        user_input = True
    else:
        fp_params = jdata["fp_params"]
        user_input = False
    cwd = os.getcwd()
    for ii in fp_tasks:
        os.chdir(ii)
        sys_data = dpdata.System("POSCAR").data
        ret = make_siesta_input(sys_data, fp_pp_files, fp_params)
        with open("input", "w") as fp:
            fp.write(ret)
        os.chdir(cwd)
    # link pp files
    _link_fp_vasp_pp(iter_index, jdata)




def make_fp_gaussian(iter_index, jdata):
    work_path = os.path.join(make_iter_name(iter_index), fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    # make gaussian gjf file
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    if "user_fp_params" in jdata.keys():
        fp_params = jdata["user_fp_params"]
    else:
        fp_params = jdata["fp_params"]
    cwd = os.getcwd()

    model_devi_engine = jdata.get("model_devi_engine", "lammps")
    for ii in fp_tasks:
        os.chdir(ii)
        if model_devi_engine == "lammps":
            sys_data = dpdata.System("POSCAR").data
        elif model_devi_engine == "gromacs":
            sys_data = dpdata.System("deepmd.raw", fmt="deepmd/raw").data
            if os.path.isfile("deepmd.raw/charge"):
                sys_data["charge"] = int(np.loadtxt("deepmd.raw/charge", dtype=int))
        ret = make_gaussian_input(sys_data, fp_params)
        with open("input", "w") as fp:
            fp.write(ret)
        os.chdir(cwd)




def make_fp_cp2k(iter_index, jdata):
    work_path = os.path.join(make_iter_name(iter_index), fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    # make cp2k input
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    if "user_fp_params" in jdata.keys():
        fp_params = jdata["user_fp_params"]
    # some users might use own inputs
    # specify the input path string
    elif "external_input_path" in jdata.keys():
        fp_params = None
        exinput_path = os.path.abspath(jdata["external_input_path"])
    else:
        fp_params = jdata["fp_params"]
    cwd = os.getcwd()

    # skip bad box criteria
    skip_bad_box = jdata.get("fp_skip_bad_box")
    count_bad_box = 0

    for ii in fp_tasks:
        os.chdir(ii)
        sys_data = dpdata.System("POSCAR").data
        # make input for every task
        # if fp_params exits, make keys
        if skip_bad_box is not None:
            # Check the box directly from POSCAR
            skip = check_bad_box("POSCAR", skip_bad_box, fmt="vasp/poscar")
            if skip:
                count_bad_box += 1
                os.chdir(cwd)
                shutil.rmtree(ii)  # Clean up inconsistent task directory
                continue

        if fp_params:
            cp2k_input = make_cp2k_input(sys_data, fp_params)
        else:
            # else read from user input
            cp2k_input = make_cp2k_input_from_external(sys_data, exinput_path)
        with open("input.inp", "w") as fp:
            fp.write(cp2k_input)
        # make coord.xyz used by cp2k for every task
        cp2k_coord = make_cp2k_xyz(sys_data)
        with open("coord.xyz", "w") as fp:
            fp.write(cp2k_coord)
        os.chdir(cwd)

    if count_bad_box > 0:
        dlog.info(
            f"skipped {count_bad_box:6d} confs with bad box, {len(fp_tasks) - count_bad_box:6d} remains"
        )

    # link pp files
    _link_fp_vasp_pp(iter_index, jdata)




def make_fp_pwmat(iter_index, jdata):
    # abs path for fp_incar if it exists
    if "fp_incar" in jdata:
        jdata["fp_incar"] = os.path.abspath(jdata["fp_incar"])
    # order is critical!
    # 1, link pp files
    _link_fp_vasp_pp(iter_index, jdata)
    # 2, create pwmat input
    _make_fp_pwmat_input(iter_index, jdata)




def make_fp_amber_diff(iter_index: int, jdata: dict):
    """Run amber twice to calculate high-level and low-level potential,
    and then generate difference between them.

    Besides AMBER, one needs to install `dpamber` package, which is avaiable at
    https://github.com/njzjz/dpamber

    Currently, it should be used with the AMBER model_devi driver.

    Parameters
    ----------
    iter_index : int
        iter index
    jdata : dict
        Run parameters. The following parameters are used in this method:
            mdin_prefix : str
                The path prefix to AMBER mdin files
            qm_region : list[str]
                AMBER mask of the QM region. Each mask maps to a system.
            qm_charge : list[int]
                Charge of the QM region. Each charge maps to a system.
            high_level : str
                high level method
            low_level : str
                low level method
            fp_params : dict
                This parameters includes:
                    high_level_mdin : str
                        High-level AMBER mdin file. %qm_theory%, %qm_region%,
                        and %qm_charge% will be replace.
                    low_level_mdin : str
                        Low-level AMBER mdin file. %qm_theory%, %qm_region%,
                        and %qm_charge% will be replace.
            parm7_prefix : str
                The path prefix to AMBER PARM7 files
            parm7 : list[str]
                List of paths to AMBER PARM7 files. Each file maps to a system.

    References
    ----------
    .. [1] Development of Range-Corrected Deep Learning Potentials for Fast, Accurate Quantum
       Mechanical/Molecular Mechanical Simulations of Chemical Reactions in Solution,
       Jinzhe Zeng, Timothy J. Giese, Şölen Ekesan, and Darrin M. York, Journal of Chemical
       Theory and Computation 2021 17 (11), 6993-7009
    """
    assert jdata["model_devi_engine"] == "amber"
    work_path = os.path.join(make_iter_name(iter_index), fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    # make amber input
    cwd = os.getcwd()
    # link two mdin files and param7
    os.chdir(os.path.join(fp_tasks[0], ".."))
    mdin_prefix = jdata.get("mdin_prefix", "")
    low_level_mdin = jdata["fp_params"]["low_level_mdin"]
    low_level_mdin = os.path.join(mdin_prefix, low_level_mdin)
    high_level_mdin = jdata["fp_params"]["high_level_mdin"]
    high_level_mdin = os.path.join(mdin_prefix, high_level_mdin)
    with open(low_level_mdin) as f:
        low_level_mdin_str = f.read()
    with open(high_level_mdin) as f:
        high_level_mdin_str = f.read()

    qm_region = jdata["qm_region"]
    high_level = jdata["high_level"]
    low_level = jdata["low_level"]
    qm_charge = jdata["qm_charge"]
    # qm_theory qm_region qm_charge
    for ii, _ in enumerate(qm_region):
        mdin_new_str = (
            low_level_mdin_str.replace("%qm_theory%", low_level)
            .replace("%qm_region%", qm_region[ii])
            .replace("%qm_charge%", str(qm_charge[ii]))
        )
        with open("low_level%d.mdin" % ii, "w") as f:  # noqa: UP031
            f.write(mdin_new_str)

        mdin_new_str = (
            high_level_mdin_str.replace("%qm_theory%", high_level)
            .replace("%qm_region%", qm_region[ii])
            .replace("%qm_charge%", str(qm_charge[ii]))
        )
        with open("high_level%d.mdin" % ii, "w") as f:  # noqa: UP031
            f.write(mdin_new_str)

    parm7 = jdata["parm7"]
    parm7_prefix = jdata.get("parm7_prefix", "")
    parm7 = [os.path.join(parm7_prefix, pp) for pp in parm7]
    for ii, pp in enumerate(parm7):
        os.symlink(pp, "qmmm%d.parm7" % ii)  # noqa: UP031

    rst7_prefix = jdata.get("sys_configs_prefix", "")
    for ii, ss in enumerate(jdata["sys_configs"]):
        os.symlink(os.path.join(rst7_prefix, ss[0]), "init%d.rst7" % ii)  # noqa: UP031

    with open("qm_region", "w") as f:
        f.write("\n".join(qm_region))
    os.chdir(cwd)




def make_fp_cpx(iter_index, jdata):
    """Make input file for Quantum Espresso Car-Parrinello (cp.x) run.

    Convert the POSCAR file to cp.x input file using prepared template.

    Parameters
    ----------
    iter_index : int
        iter index
    jdata : dict
        Run parameters.
    """
    work_path = os.path.join(make_iter_name(iter_index), fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_params = jdata["fp_params"]
    input_fn = fp_params["input_fn"]
    template_fn = fp_params["template_fn"]

    with open(template_fn) as tn:
        template = tn.read()

    for ii in fp_tasks:
        itemp = copy.deepcopy(template)
        with set_directory(Path(ii)):
            system = dpdata.System("POSCAR", fmt="vasp/poscar")
            # convert POSCAR to cp.in
            cell_param = ""
            for a in system["cells"][0]:
                for v in a:
                    cell_param += f"{v:.16e} "
                cell_param += "\n"
            itemp = itemp.replace("%CELL%", cell_param)

            pos = ""
            ntypes = system.get_ntypes()
            numbs = system.get_atom_numbs()
            names = system.get_atom_names()
            coords = system["coords"][0]
            for t in range(ntypes):
                ts = sum(numbs[:t])
                for a in range(numbs[t]):
                    pos += f"{names[t]:2} {coords[a + ts, 0]:20.16f} {coords[a + ts, 1]:20.16f} {coords[a + ts, 2]:20.16f}\n"
            itemp = itemp.replace("%POSITIONS%", pos)
            with open(input_fn + ".in", "w") as fp:
                fp.write(itemp)




def make_fp_custom(iter_index, jdata):
    """Make input file for customized FP style.

    Convert the POSCAR file to custom format.

    Parameters
    ----------
    iter_index : int
        iter index
    jdata : dict
        Run parameters.
    """
    work_path = os.path.join(make_iter_name(iter_index), fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_params = jdata["fp_params"]
    input_fn = fp_params["input_fn"]
    input_fmt = fp_params["input_fmt"]

    for ii in fp_tasks:
        with set_directory(Path(ii)):
            system = dpdata.System("POSCAR", fmt="vasp/poscar")
            system.to(input_fmt, input_fn)




def make_fp(iter_index, jdata, mdata):
    """Select the candidate strutures and make the input file of FP calculation.

    Parameters
    ----------
    iter_index : int
        iter index
    jdata : dict
        Run parameters.
    mdata : dict
        Machine parameters.
    """
    fp_tasks = _make_fp_vasp_configs(iter_index, jdata)
    if len(fp_tasks) == 0:
        return
    make_fp_calculation(iter_index, jdata, mdata)




def make_fp_calculation(iter_index, jdata, mdata):
    """Make the input file of FP calculation.

    Parameters
    ----------
    iter_index : int
        iter index
    jdata : dict
        Run parameters.
    mdata : dict
        Machine parameters.
    """
    fp_style = jdata["fp_style"]
    if fp_style == "vasp":
        make_fp_vasp(iter_index, jdata)
    elif fp_style == "pwscf":
        make_fp_pwscf(iter_index, jdata)
    elif fp_style == "abacus":
        make_fp_abacus_scf(iter_index, jdata)
    elif fp_style == "siesta":
        make_fp_siesta(iter_index, jdata)
    elif fp_style == "gaussian":
        make_fp_gaussian(iter_index, jdata)
    elif fp_style == "cp2k":
        make_fp_cp2k(iter_index, jdata)
    elif fp_style == "pwmat":
        make_fp_pwmat(iter_index, jdata)
    elif fp_style == "amber/diff":
        make_fp_amber_diff(iter_index, jdata)
    elif fp_style == "cpx":
        make_fp_cpx(iter_index, jdata)
    elif fp_style == "custom":
        make_fp_custom(iter_index, jdata)
    else:
        raise RuntimeError("unsupported fp style")
    # Copy user defined forward_files
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    symlink_user_forward_files(mdata=mdata, task_type="fp", work_path=work_path)




def _vasp_check_fin(ii):
    if os.path.isfile(os.path.join(ii, "OUTCAR")):
        with open(os.path.join(ii, "OUTCAR")) as fp:
            content = fp.read()
            count = content.count("Elapse")
            if count != 1:
                return False
    else:
        return False
    return True




def _qe_check_fin(ii):
    if os.path.isfile(os.path.join(ii, "output")):
        with open(os.path.join(ii, "output")) as fp:
            content = fp.read()
            count = content.count("JOB DONE")
            if count != 1:
                return False
    else:
        return False
    return True




def _abacus_scf_check_fin(ii):
    if os.path.isfile(os.path.join(ii, "OUT.ABACUS/running_scf.log")):
        with open(os.path.join(ii, "OUT.ABACUS/running_scf.log")) as fp:
            content = fp.read()
            count = content.count("!FINAL_ETOT_IS")
            if count != 1:
                return False
    else:
        return False
    return True




def _siesta_check_fin(ii):
    if os.path.isfile(os.path.join(ii, "output")):
        with open(os.path.join(ii, "output")) as fp:
            content = fp.read()
            count = content.count("End of run")
            if count != 1:
                return False
    else:
        return False
    return True




def _gaussian_check_fin(ii):
    if os.path.isfile(os.path.join(ii, "output")):
        with open(os.path.join(ii, "output")) as fp:
            content = fp.read()
            count = content.count("termination")
            if count == 0:
                return False
    else:
        return False
    return True




def _cp2k_check_fin(ii):
    if os.path.isfile(os.path.join(ii, "output")):
        with open(os.path.join(ii, "output")) as fp:
            content = fp.read()
            count = content.count("SCF run converged")
            if count == 0:
                return False
    else:
        return False
    return True




def _pwmat_check_fin(ii):
    if os.path.isfile(os.path.join(ii, "REPORT")):
        with open(os.path.join(ii, "REPORT")) as fp:
            content = fp.read()
            count = content.count("time")
            if count != 1:
                return False
    else:
        return False
    return True




def run_fp_inner(
    iter_index,
    jdata,
    mdata,
    forward_files,
    backward_files,
    check_fin,
    log_file="fp.log",
    forward_common_files=[],
):
    fp_command = mdata["fp_command"]
    fp_group_size = mdata["fp_group_size"]
    fp_resources = mdata["fp_resources"]
    mark_failure = fp_resources.get("mark_failure", False)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)

    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    fp_style = jdata["fp_style"]
    if fp_style == "amber/diff":
        # firstly get sys_idx
        fp_command = (
            (
                "TASK=$(basename $(pwd)) && "
                "SYS1=${TASK:5:3} && "
                "SYS=$((10#$SYS1)) && "
                'QM_REGION=$(awk "NR==$SYS+1" ../qm_region) &&'
            )
            + fp_command
            + (
                " -O -p ../qmmm$SYS.parm7 -c ../init$SYS.rst7 -i ../low_level$SYS.mdin -o low_level.mdout -r low_level.rst7 "
                "-x low_level.nc -y rc.nc -frc low_level.mdfrc -inf low_level.mdinfo && "
            )
            + fp_command
            + (
                " -O -p ../qmmm$SYS.parm7 -c ../init$SYS.rst7 -i ../high_level$SYS.mdin -o high_level.mdout -r high_level.rst7 "
                "-x high_level.nc -y rc.nc -frc high_level.mdfrc -inf high_level.mdinfo && "
            )
            + (
                'dpamber corr --cutoff {:f} --parm7_file ../qmmm$SYS.parm7 --nc rc.nc --hl high_level --ll low_level --qm_region "$QM_REGION"'
            ).format(jdata["cutoff"])
        )

    fp_run_tasks = fp_tasks
    # for ii in fp_tasks :
    #     if not check_fin(ii) :
    #         fp_run_tasks.append(ii)
    run_tasks = [os.path.basename(ii) for ii in fp_run_tasks]

    user_forward_files = mdata.get("fp" + "_user_forward_files", [])
    forward_files += [os.path.basename(file) for file in user_forward_files]
    backward_files += mdata.get("fp" + "_user_backward_files", [])

    ### Submit jobs
    check_api_version(mdata)

    submission = make_submission(
        mdata["fp_machine"],
        mdata["fp_resources"],
        commands=[fp_command],
        work_path=work_path,
        run_tasks=run_tasks,
        group_size=fp_group_size,
        forward_common_files=forward_common_files,
        forward_files=forward_files,
        backward_files=backward_files,
        outlog=log_file,
        errlog=log_file,
    )
    submission.run_submission()




def run_fp(iter_index, jdata, mdata):
    fp_style = jdata["fp_style"]
    fp_pp_files = jdata.get("fp_pp_files", [])

    if fp_style == "vasp":
        forward_files = ["POSCAR", "INCAR", "POTCAR", "KPOINTS"]
        backward_files = ["fp.log", "OUTCAR", "vasprun.xml"]
        # Move cvasp interface to jdata
        if ("cvasp" in jdata) and (jdata["cvasp"] is True):
            mdata["fp_resources"]["cvasp"] = True
        if ("cvasp" in mdata["fp_resources"]) and (
            mdata["fp_resources"]["cvasp"] is True
        ):
            dlog.info("cvasp is on !")
            forward_files.append("cvasp.py")
            forward_common_files = []
        else:
            forward_common_files = []
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _vasp_check_fin,
            forward_common_files=forward_common_files,
        )
    elif fp_style == "pwscf":
        forward_files = ["input"] + fp_pp_files
        backward_files = ["output"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _qe_check_fin,
            log_file="output",
        )
    elif fp_style == "abacus":
        fp_params = {}
        if "user_fp_params" in jdata.keys():
            fp_params = jdata["user_fp_params"]
        elif "fp_incar" in jdata.keys():
            fp_input_path = jdata["fp_incar"]
            assert os.path.exists(fp_input_path)
            fp_input_path = os.path.abspath(fp_input_path)
            fp_params = get_abacus_input_parameters(fp_input_path)
        forward_files = ["INPUT", "STRU", "pporb"]
        if "kspacing" not in fp_params.keys():
            forward_files.append("KPT")
        backward_files = ["output", "OUT.ABACUS"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _abacus_scf_check_fin,
            log_file="output",
        )
    elif fp_style == "siesta":
        forward_files = ["input"] + fp_pp_files
        backward_files = ["output"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _siesta_check_fin,
            log_file="output",
        )
    elif fp_style == "gaussian":
        forward_files = ["input"]
        backward_files = ["output"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _gaussian_check_fin,
            log_file="output",
        )
    elif fp_style == "cp2k":
        forward_files = ["input.inp", "coord.xyz"]
        backward_files = ["output"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _cp2k_check_fin,
            log_file="output",
        )
    elif fp_style == "pwmat":
        forward_files = ["atom.config", "etot.input"] + fp_pp_files
        backward_files = ["REPORT", "OUT.MLMD", "output"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _pwmat_check_fin,
            log_file="output",
        )
    elif fp_style == "amber/diff":
        forward_files = ["rc.nc"]
        backward_files = [
            "low_level.mdfrc",
            "low_level.mdout",
            "high_level.mdfrc",
            "high_level.mdout",
            "output",
            "dataset",
        ]
        forward_common_files = [
            "low_level*.mdin",
            "high_level*.mdin",
            "qmmm*.parm7",
            "qm_region",
            "init*.rst7",
        ]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            None,
            log_file="output",
            forward_common_files=forward_common_files,
        )
    elif fp_style == "cpx":
        extensions = [".cel", ".evp", ".for", ".pos", ".str"]
        input_fn = jdata["fp_params"]["input_fn"]
        forward_files = [input_fn + ".in"]
        backward_files = [input_fn + ext for ext in extensions] + ["output"]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            _qe_check_fin,
            log_file="output",
        )
    elif fp_style == "custom":
        fp_params = jdata["fp_params"]
        forward_files = [fp_params["input_fn"]]
        backward_files = [fp_params["output_fn"]]
        run_fp_inner(
            iter_index,
            jdata,
            mdata,
            forward_files,
            backward_files,
            None,
            log_file="output",
        )
    else:
        raise RuntimeError("unsupported fp style")




def post_fp_check_fail(iter_index, jdata, rfailed=None):
    ratio_failed = rfailed if rfailed else jdata.get("ratio_failed", 0.05)
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return
    ntask = len(fp_tasks)
    nfail = 0

    # check fail according to the number of collected data
    sys_data = glob.glob(os.path.join(work_path, "data.*"))
    sys_data.sort()
    nframe = 0
    for ii in sys_data:
        sys_paths = expand_sys_str(ii)
        for single_sys in sys_paths:
            sys = dpdata.LabeledSystem(os.path.join(single_sys), fmt="deepmd/npy")
            nframe += len(sys)
    nfail = ntask - nframe

    rfail = float(nfail) / float(ntask)
    dlog.info("failed tasks: %6d in %6d  %6.2f %% " % (nfail, ntask, rfail * 100.0))  # noqa: UP031
    if rfail > ratio_failed:
        raise RuntimeError("find too many unsuccessfully terminated jobs")




def post_fp_vasp(iter_index, jdata, rfailed=None):
    ratio_failed = rfailed if rfailed else jdata.get("ratio_failed", 0.05)
    model_devi_engine = jdata.get("model_devi_engine", "lammps")
    if model_devi_engine != "calypso":
        model_devi_jobs = jdata["model_devi_jobs"]
        assert iter_index < len(model_devi_jobs)
    use_ele_temp = jdata.get("use_ele_temp", 0)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    system_index = []
    for ii in fp_tasks:
        system_index.append(os.path.basename(ii).split(".")[1])
    system_index.sort()
    set_tmp = set(system_index)
    system_index = list(set_tmp)
    system_index.sort()

    cwd = os.getcwd()

    tcount = 0
    icount = 0
    for ss in system_index:
        sys_outcars = glob.glob(os.path.join(work_path, f"task.{ss}.*/OUTCAR"))
        sys_outcars.sort()
        tcount += len(sys_outcars)
        all_sys = None
        all_te = []
        for oo in sys_outcars:
            try:
                _sys = dpdata.LabeledSystem(oo, type_map=jdata["type_map"])
            except Exception:
                dlog.info("Try to parse from vasprun.xml")
                try:
                    _sys = dpdata.LabeledSystem(
                        oo.replace("OUTCAR", "vasprun.xml"), type_map=jdata["type_map"]
                    )
                except Exception:
                    _sys = dpdata.LabeledSystem()
                    dlog.info("Failed fp path: {}".format(oo.replace("OUTCAR", "")))
            if len(_sys) == 1:
                # save ele_temp, if any
                if os.path.exists(oo.replace("OUTCAR", "job.json")):
                    with open(oo.replace("OUTCAR", "job.json")) as fp:
                        job_data = json.load(fp)
                    if "ele_temp" in job_data:
                        assert use_ele_temp
                        ele_temp = job_data["ele_temp"]
                        all_te.append(ele_temp)
                        if use_ele_temp == 0:
                            raise RuntimeError(
                                "should not get ele temp at setting: use_ele_temp == 0"
                            )
                        elif use_ele_temp == 1:
                            _sys.data["fparam"] = np.array(ele_temp).reshape(1, 1)
                        elif use_ele_temp == 2:
                            tile_te = np.tile(ele_temp, [_sys.get_natoms()])
                            _sys.data["aparam"] = tile_te.reshape(
                                1, _sys.get_natoms(), 1
                            )
                        else:
                            raise RuntimeError(
                                "invalid setting of use_ele_temp " + str(use_ele_temp)
                            )
                        # check if ele_temp shape is correct
                        _sys.check_data()
                if all_sys is None:
                    all_sys = dpdata.MultiSystems(_sys, type_map=jdata["type_map"])
                else:
                    all_sys.append(_sys)
            elif len(_sys) >= 2:
                raise RuntimeError("The vasp parameter NSW should be set as 1")
            else:
                icount += 1
        all_te = np.array(all_te)
        if all_sys is not None:
            sys_data_path = os.path.join(work_path, f"data.{ss}")
            all_sys.to_deepmd_raw(sys_data_path)
            all_sys.to_deepmd_npy(sys_data_path, set_size=len(sys_outcars))

    if tcount == 0:
        rfail = 0.0
        dlog.info("failed frame: %6d in %6d " % (icount, tcount))  # noqa: UP031
    else:
        rfail = float(icount) / float(tcount)
        dlog.info(
            "failed frame: %6d in %6d  %6.2f %% " % (icount, tcount, rfail * 100.0)  # noqa: UP031
        )

    if rfail > ratio_failed:
        raise RuntimeError(
            "find too many unsuccessfully terminated jobs. Too many FP tasks are not converged. Please check your input parameters (e.g. INCAR) or configuration (e.g. POSCAR) in directories 'iter.*.*/02.fp/task.*.*/.'"
        )




def post_fp_pwscf(iter_index, jdata):
    model_devi_jobs = jdata["model_devi_jobs"]
    assert iter_index < len(model_devi_jobs)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    system_index = []
    for ii in fp_tasks:
        system_index.append(os.path.basename(ii).split(".")[1])
    system_index.sort()
    set_tmp = set(system_index)
    system_index = list(set_tmp)
    system_index.sort()

    cwd = os.getcwd()
    for ss in system_index:
        sys_output = glob.glob(os.path.join(work_path, f"task.{ss}.*/output"))
        sys_input = glob.glob(os.path.join(work_path, f"task.{ss}.*/input"))
        sys_output.sort()
        sys_input.sort()

        flag = True
        for ii, oo in zip(sys_input, sys_output):
            if flag:
                _sys = dpdata.LabeledSystem(
                    oo, fmt="qe/pw/scf", type_map=jdata["type_map"]
                )
                if len(_sys) > 0:
                    all_sys = _sys
                    flag = False
                else:
                    pass
            else:
                _sys = dpdata.LabeledSystem(
                    oo, fmt="qe/pw/scf", type_map=jdata["type_map"]
                )
                if len(_sys) > 0:
                    all_sys.append(_sys)

        sys_data_path = os.path.join(work_path, f"data.{ss}")
        all_sys.to_deepmd_raw(sys_data_path)
        all_sys.to_deepmd_npy(sys_data_path, set_size=len(sys_output))




def post_fp_abacus_scf(iter_index, jdata):
    model_devi_jobs = jdata["model_devi_jobs"]
    assert iter_index < len(model_devi_jobs)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    system_index = []
    for ii in fp_tasks:
        system_index.append(os.path.basename(ii).split(".")[1])
    system_index.sort()
    set_tmp = set(system_index)
    system_index = list(set_tmp)
    system_index.sort()

    cwd = os.getcwd()
    for ss in system_index:
        sys_output = glob.glob(os.path.join(work_path, f"task.{ss}.*"))
        sys_input = glob.glob(os.path.join(work_path, f"task.{ss}.*/INPUT"))
        sys_output.sort()
        sys_input.sort()

        all_sys = None
        for ii, oo in zip(sys_input, sys_output):
            _sys = dpdata.LabeledSystem(oo, fmt="abacus/scf")
            if len(_sys) > 0:
                _sys.data["atom_types"] = np.asarray(_sys.data["atom_types"], dtype=int)
                _sys.apply_type_map(jdata["type_map"])
            if len(_sys) > 0:
                if all_sys is None:
                    all_sys = _sys
                else:
                    all_sys.append(_sys)

        if all_sys is not None:
            sys_data_path = os.path.join(work_path, f"data.{ss}")
            all_sys.to_deepmd_raw(sys_data_path)
            all_sys.to_deepmd_npy(sys_data_path, set_size=len(sys_output))




def post_fp_siesta(iter_index, jdata):
    model_devi_jobs = jdata["model_devi_jobs"]
    assert iter_index < len(model_devi_jobs)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    system_index = []
    for ii in fp_tasks:
        system_index.append(os.path.basename(ii).split(".")[1])
    system_index.sort()
    set_tmp = set(system_index)
    system_index = list(set_tmp)
    system_index.sort()

    cwd = os.getcwd()
    for ss in system_index:
        sys_output = glob.glob(os.path.join(work_path, f"task.{ss}.*/output"))
        sys_input = glob.glob(os.path.join(work_path, f"task.{ss}.*/input"))
        sys_output.sort()
        sys_input.sort()
        for idx, oo in enumerate(sys_output):
            _sys = dpdata.LabeledSystem(oo, fmt="siesta/output")
            if idx == 0:
                all_sys = _sys
            else:
                all_sys.append(_sys)

        sys_data_path = os.path.join(work_path, f"data.{ss}")
        all_sys.to_deepmd_raw(sys_data_path)
        all_sys.to_deepmd_npy(sys_data_path, set_size=len(sys_output))




def post_fp_gaussian(iter_index, jdata):
    model_devi_jobs = jdata["model_devi_jobs"]
    assert iter_index < len(model_devi_jobs)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    system_index = []
    for ii in fp_tasks:
        system_index.append(os.path.basename(ii).split(".")[1])
    system_index.sort()
    set_tmp = set(system_index)
    system_index = list(set_tmp)
    system_index.sort()

    cwd = os.getcwd()
    for ss in system_index:
        sys_output = glob.glob(os.path.join(work_path, f"task.{ss}.*/output"))
        sys_output.sort()
        for idx, oo in enumerate(sys_output):
            sys = dpdata.LabeledSystem(oo, fmt="gaussian/log")
            if len(sys) > 0:
                sys.check_type_map(type_map=jdata["type_map"])
            if jdata.get("use_atom_pref", False):
                sys.data["atom_pref"] = np.load(
                    os.path.join(os.path.dirname(oo), "atom_pref.npy")
                )
            if idx == 0:
                if jdata.get("use_clusters", False):
                    all_sys = dpdata.MultiSystems(sys, type_map=jdata["type_map"])
                else:
                    all_sys = sys
            else:
                all_sys.append(sys)
        sys_data_path = os.path.join(work_path, f"data.{ss}")
        all_sys.to_deepmd_raw(sys_data_path)
        all_sys.to_deepmd_npy(sys_data_path, set_size=len(sys_output))




def post_fp_cp2k(iter_index, jdata, rfailed=None):
    ratio_failed = rfailed if rfailed else jdata.get("ratio_failed", 0.10)
    model_devi_jobs = jdata["model_devi_jobs"]
    assert iter_index < len(model_devi_jobs)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    system_index = []
    for ii in fp_tasks:
        system_index.append(os.path.basename(ii).split(".")[1])
    system_index.sort()
    set_tmp = set(system_index)
    system_index = list(set_tmp)
    system_index.sort()

    cwd = os.getcwd()
    # tcount: num of all fp tasks
    tcount = 0
    # icount: num of converged fp tasks
    icount = 0
    for ss in system_index:
        sys_output = glob.glob(os.path.join(work_path, f"task.{ss}.*/output"))
        sys_output.sort()
        tcount += len(sys_output)
        all_sys = dpdata.MultiSystems(type_map=jdata["type_map"])
        for oo in sys_output:
            _sys = dpdata.LabeledSystem(
                oo, fmt="cp2kdata/e_f", type_map=jdata["type_map"]
            )
            if len(_sys) > 0:
                all_sys.append(_sys)
                icount += 1

        if (all_sys is not None) and (len(all_sys) > 0):
            sys_data_path = os.path.join(work_path, f"data.{ss}")
            all_sys.to_deepmd_raw(sys_data_path)
            all_sys.to_deepmd_npy(sys_data_path)

    if tcount == 0:
        rfail = 0.0
        dlog.info("failed frame: %6d in %6d " % (tcount - icount, tcount))  # noqa: UP031
    else:
        rfail = float(tcount - icount) / float(tcount)
        dlog.info(
            "failed frame: %6d in %6d  %6.2f %% "  # noqa: UP031
            % (tcount - icount, tcount, rfail * 100.0)
        )

    if rfail > ratio_failed:
        raise RuntimeError(
            "find too many unsuccessfully terminated jobs. Too many FP tasks are not converged. Please check your files in directories 'iter.*.*/02.fp/task.*.*/.'"
        )




def post_fp_pwmat(iter_index, jdata, rfailed=None):
    ratio_failed = rfailed if rfailed else jdata.get("ratio_failed", 0.05)
    model_devi_jobs = jdata["model_devi_jobs"]
    assert iter_index < len(model_devi_jobs)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    system_index = []
    for ii in fp_tasks:
        system_index.append(os.path.basename(ii).split(".")[1])
    system_index.sort()
    set_tmp = set(system_index)
    system_index = list(set_tmp)
    system_index.sort()

    cwd = os.getcwd()

    tcount = 0
    icount = 0
    for ss in system_index:
        sys_output = glob.glob(os.path.join(work_path, f"task.{ss}.*/OUT.MLMD"))
        sys_output.sort()
        tcount += len(sys_output)
        all_sys = None
        for oo in sys_output:
            _sys = dpdata.LabeledSystem(oo, type_map=jdata["type_map"])
            if len(_sys) == 1:
                if all_sys is None:
                    all_sys = _sys
                else:
                    all_sys.append(_sys)
            else:
                icount += 1
        if all_sys is not None:
            sys_data_path = os.path.join(work_path, f"data.{ss}")
            all_sys.to_deepmd_raw(sys_data_path)
            all_sys.to_deepmd_npy(sys_data_path, set_size=len(sys_output))
    dlog.info(f"failed frame number: {icount} ")
    dlog.info(f"total frame number: {tcount} ")
    reff = icount / tcount
    dlog.info(f"ratio of failed frame:  {reff:.2%}")

    if reff > ratio_failed:
        raise RuntimeError("find too many unsuccessfully terminated jobs")




def post_fp_amber_diff(iter_index, jdata):
    model_devi_jobs = jdata["model_devi_jobs"]
    assert iter_index < len(model_devi_jobs)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    system_index = []
    for ii in fp_tasks:
        system_index.append(os.path.basename(ii).split(".")[1])
    system_index.sort()
    set_tmp = set(system_index)
    system_index = list(set_tmp)
    system_index.sort()

    for ss in system_index:
        sys_output = glob.glob(os.path.join(work_path, f"task.{ss}.*"))
        sys_output.sort()
        all_sys = dpdata.MultiSystems(type_map=jdata["type_map"])
        for oo in sys_output:
            sys = dpdata.MultiSystems(type_map=jdata["type_map"]).from_deepmd_npy(
                os.path.join(oo, "dataset")
            )
            all_sys.append(sys)
        sys_data_path = os.path.join(work_path, f"data.{ss}")
        all_sys.to_deepmd_raw(sys_data_path)
        all_sys.to_deepmd_npy(sys_data_path, set_size=len(sys_output), prec=np.float64)




def post_fp_cpx(iter_index, jdata):
    """Post fp for cp.x. Collect data from qe/cp/traj labeled system.

    Parameters
    ----------
    iter_index : int
        The index of the current iteration.
    jdata : dict
        The parameter data.
    """
    model_devi_jobs = jdata["model_devi_jobs"]
    assert iter_index < len(model_devi_jobs)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    system_index = []
    for ii in fp_tasks:
        system_index.append(os.path.basename(ii).split(".")[1])
    system_index.sort()
    set_tmp = set(system_index)
    system_index = list(set_tmp)
    system_index.sort()

    fp_params = jdata["fp_params"]
    input_fn = fp_params["input_fn"]

    for ss in system_index:
        sys_output = glob.glob(os.path.join(work_path, f"task.{ss}.*"))
        sys_output.sort()
        all_sys = dpdata.MultiSystems(type_map=jdata["type_map"])
        for oo in sys_output:
            if os.path.exists(os.path.join(oo, "output")):
                sys = dpdata.LabeledSystem(os.path.join(oo, input_fn), fmt="qe/cp/traj")
                all_sys.append(sys)
        sys_data_path = os.path.join(work_path, f"data.{ss}")
        all_sys.to_deepmd_raw(sys_data_path)
        all_sys.to_deepmd_npy(sys_data_path, set_size=len(sys_output), prec=np.float64)




def post_fp_custom(iter_index, jdata):
    """Post fp for custom fp. Collect data from user-defined `output_fn`.

    Parameters
    ----------
    iter_index : int
        The index of the current iteration.
    jdata : dict
        The parameter data.
    """
    model_devi_jobs = jdata["model_devi_jobs"]
    assert iter_index < len(model_devi_jobs)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    fp_tasks = glob.glob(os.path.join(work_path, "task.*"))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return

    system_index = []
    for ii in fp_tasks:
        system_index.append(os.path.basename(ii).split(".")[1])
    system_index.sort()
    set_tmp = set(system_index)
    system_index = list(set_tmp)
    system_index.sort()

    fp_params = jdata["fp_params"]
    output_fn = fp_params["output_fn"]
    output_fmt = fp_params["output_fmt"]

    for ss in system_index:
        sys_output = glob.glob(os.path.join(work_path, f"task.{ss}.*"))
        sys_output.sort()
        all_sys = dpdata.MultiSystems(type_map=jdata["type_map"])
        for oo in sys_output:
            if os.path.exists(os.path.join(oo, output_fn)):
                sys = dpdata.LabeledSystem(os.path.join(oo, output_fn), fmt=output_fmt)
                all_sys.append(sys)
        sys_data_path = os.path.join(work_path, f"data.{ss}")
        all_sys.to_deepmd_raw(sys_data_path)
        all_sys.to_deepmd_npy(sys_data_path, set_size=len(sys_output), prec=np.float64)




def post_fp(iter_index, jdata):
    fp_style = jdata["fp_style"]
    if fp_style == "vasp":
        post_fp_vasp(iter_index, jdata)
    elif fp_style == "pwscf":
        post_fp_pwscf(iter_index, jdata)
    elif fp_style == "abacus":
        post_fp_abacus_scf(iter_index, jdata)
    elif fp_style == "siesta":
        post_fp_siesta(iter_index, jdata)
    elif fp_style == "gaussian":
        post_fp_gaussian(iter_index, jdata)
    elif fp_style == "cp2k":
        post_fp_cp2k(iter_index, jdata)
    elif fp_style == "pwmat":
        post_fp_pwmat(iter_index, jdata)
    elif fp_style == "amber/diff":
        post_fp_amber_diff(iter_index, jdata)
    elif fp_style == "cpx":
        post_fp_cpx(iter_index, jdata)
    elif fp_style == "custom":
        post_fp_custom(iter_index, jdata)
    else:
        raise RuntimeError("unsupported fp style")
    post_fp_check_fail(iter_index, jdata)
    # clean traj
    clean_traj = True
    if "model_devi_clean_traj" in jdata:
        clean_traj = jdata["model_devi_clean_traj"]
    modd_path = None
    if isinstance(clean_traj, bool):
        iter_name = make_iter_name(iter_index)
        if clean_traj:
            modd_path = os.path.join(iter_name, model_devi_name)
    elif isinstance(clean_traj, int):
        clean_index = iter_index - clean_traj
        if clean_index >= 0:
            modd_path = os.path.join(make_iter_name(clean_index), model_devi_name)
    if modd_path is not None:
        md_trajs = glob.glob(os.path.join(modd_path, "task*/traj"))
        for ii in md_trajs:
            shutil.rmtree(ii)



