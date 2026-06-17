#!/usr/bin/env python3

"""Model-deviation step of the DP-GEN workflow (``01.model_devi``).

Contains make/run/post for the model-deviation (MD exploration) sub-step.
Supports multiple MD engines (lammps, gromacs, amber, calypso) and two input
modes (native, revise_template).
"""

import copy
import glob
import json
import os
import shlex
import warnings
from collections.abc import Iterable
from pathlib import Path

import dpdata
import numpy as np
from packaging.version import Version

from dpgen import dlog
from dpgen.dispatcher.Dispatcher import make_submission
from dpgen.generator.constants import (
    calypso_model_devi_name,
    calypso_run_model_devi_file,
    calypso_run_opt_name,
    check_outcar_file,
    model_devi_conf_fmt,
    model_devi_name,
    model_devi_task_fmt,
    run_opt_file,
    train_name,
)
from dpgen.generator.helpers import (
    _get_model_suffix,
    _get_param_alias,
    expand_idx,
    make_model_devi_conf_name,
    make_model_devi_task_name,
    set_version,
)
from dpgen.generator.lib.lammps import make_lammps_input
from dpgen.generator.lib.make_calypso import (
    _make_model_devi_buffet,
    _make_model_devi_native_calypso,
)
from dpgen.generator.lib.run_calypso import run_calypso_model_devi
from dpgen.generator.lib.utils import (
    check_api_version,
    create_path,
    make_iter_name,
    symlink_user_forward_files,
)


def parse_cur_job(cur_job):
    ensemble = _get_param_alias(cur_job, ["ens", "ensemble"])
    temps = [-1]
    press = [-1]
    if "npt" in ensemble:
        temps = _get_param_alias(cur_job, ["Ts", "temps"])
        press = _get_param_alias(cur_job, ["Ps", "press"])
    elif "nvt" == ensemble or "nve" == ensemble:
        temps = _get_param_alias(cur_job, ["Ts", "temps"])
    nsteps = _get_param_alias(cur_job, ["nsteps"])
    trj_freq = _get_param_alias(cur_job, ["t_freq", "trj_freq", "traj_freq"])
    if "pka_e" in cur_job:
        pka_e = _get_param_alias(cur_job, ["pka_e"])
    else:
        pka_e = None
    if "dt" in cur_job:
        dt = _get_param_alias(cur_job, ["dt"])
    else:
        dt = None
    if "nbeads" in cur_job:
        nbeads = _get_param_alias(cur_job, ["nbeads"])
    else:
        nbeads = None
    return ensemble, nsteps, trj_freq, temps, press, pka_e, dt, nbeads




def expand_matrix_values(target_list, cur_idx=0):
    nvar = len(target_list)
    if cur_idx == nvar:
        return [[]]
    else:
        res = []
        prev = expand_matrix_values(target_list, cur_idx + 1)
        for ii in target_list[cur_idx]:
            tmp = copy.deepcopy(prev)
            for jj in tmp:
                jj.insert(0, ii)
                res.append(jj)
        return res




def parse_cur_job_revmat(cur_job, use_plm=False):
    templates = [cur_job["template"]["lmp"]]
    if use_plm:
        templates.append(cur_job["template"]["plm"])
    revise_keys = []
    revise_values = []
    if "rev_mat" not in cur_job.keys():
        cur_job["rev_mat"] = {}
    if "lmp" not in cur_job["rev_mat"].keys():
        cur_job["rev_mat"]["lmp"] = {}
    for ii in cur_job["rev_mat"]["lmp"].keys():
        revise_keys.append(ii)
        revise_values.append(cur_job["rev_mat"]["lmp"][ii])
    n_lmp_keys = len(revise_keys)
    if use_plm:
        if "plm" not in cur_job["rev_mat"].keys():
            cur_job["rev_mat"]["plm"] = {}
        for ii in cur_job["rev_mat"]["plm"].keys():
            revise_keys.append(ii)
            revise_values.append(cur_job["rev_mat"]["plm"][ii])
    revise_matrix = expand_matrix_values(revise_values)
    return revise_keys, revise_matrix, n_lmp_keys




def parse_cur_job_sys_revmat(cur_job, sys_idx, use_plm=False):
    templates = [cur_job["template"]["lmp"]]
    if use_plm:
        templates.append(cur_job["template"]["plm"])
    sys_revise_keys = []
    sys_revise_values = []
    if "sys_rev_mat" not in cur_job.keys():
        cur_job["sys_rev_mat"] = {}
    local_rev = cur_job["sys_rev_mat"].get(str(sys_idx), {})
    if "lmp" not in local_rev.keys():
        local_rev["lmp"] = {}
    for ii in local_rev["lmp"].keys():
        sys_revise_keys.append(ii)
        sys_revise_values.append(local_rev["lmp"][ii])
    n_sys_lmp_keys = len(sys_revise_keys)
    if use_plm:
        if "plm" not in local_rev.keys():
            local_rev["plm"] = {}
        for ii in local_rev["plm"].keys():
            sys_revise_keys.append(ii)
            sys_revise_values.append(local_rev["plm"][ii])
    sys_revise_matrix = expand_matrix_values(sys_revise_values)
    return sys_revise_keys, sys_revise_matrix, n_sys_lmp_keys




def find_only_one_key(lmp_lines, key):
    found = []
    for idx in range(len(lmp_lines)):
        words = lmp_lines[idx].split()
        nkey = len(key)
        if len(words) >= nkey and words[:nkey] == key:
            found.append(idx)
    if len(found) > 1:
        raise RuntimeError("found %d keywords %s" % (len(found), key))  # noqa: UP031
    if len(found) == 0:
        raise RuntimeError(f"failed to find keyword {key}")
    return found[0]




def revise_lmp_input_model(
    lmp_lines, task_model_list, trj_freq, deepmd_version="1", use_ele_temp=0, jdata=None
):
    idx = find_only_one_key(lmp_lines, ["pair_style", "deepmd"])
    graph_list = " ".join(task_model_list)

    # Check if D3 dispersion is configured
    lmp_d3 = jdata.get("lmp_d3", {}) if jdata else {}
    d3_enabled = lmp_d3.get("enable", False) if lmp_d3 else False

    if Version(deepmd_version) < Version("1"):
        if d3_enabled:
            d3_params = f"{lmp_d3['damping_function']} {lmp_d3['functional']} {lmp_d3['cutoff']} {lmp_d3['cn_cutoff']}"
            lmp_lines[idx] = (
                f"pair_style      hybrid/overlay deepmd {graph_list} {trj_freq} model_devi.out dispersion/d3 {d3_params}\n"
            )
        else:
            lmp_lines[idx] = "pair_style      deepmd %s %d model_devi.out\n" % (  # noqa: UP031
                graph_list,
                trj_freq,
            )
    else:
        # Build keywords string like in make_lammps_input
        keywords = ""
        if jdata is not None:
            if jdata.get("use_clusters", False):
                keywords += "atomic "
            if jdata.get("use_relative", False):
                keywords += "relative {} ".format(jdata["epsilon"])
            if jdata.get("use_relative_v", False):
                keywords += "relative_v {} ".format(jdata["epsilon_v"])

        if use_ele_temp == 1:
            keywords += "fparam ${ELE_TEMP}"

        if d3_enabled:
            d3_params = f"{lmp_d3['damping_function']} {lmp_d3['functional']} {lmp_d3['cutoff']} {lmp_d3['cn_cutoff']}"
            lmp_lines[idx] = (
                "pair_style      hybrid/overlay deepmd %s out_freq %d out_file model_devi.out %s dispersion/d3 %s\n"  # noqa: UP031
                % (
                    graph_list,
                    trj_freq,
                    keywords.rstrip(),
                    d3_params,
                )
            )
        else:
            lmp_lines[idx] = (
                "pair_style      deepmd %s out_freq %d out_file model_devi.out %s\n"  # noqa: UP031
                % (
                    graph_list,
                    trj_freq,
                    keywords.rstrip(),
                )
            )
    return lmp_lines




def revise_lmp_input_pair_coeff(lmp_lines, jdata=None):
    """Update pair_coeff lines for D3 support."""
    if jdata is None:
        return lmp_lines

    lmp_d3 = jdata.get("lmp_d3", {})
    d3_enabled = lmp_d3.get("enable", False) if lmp_d3 else False

    if not d3_enabled:
        return lmp_lines

    # D3 requires type maps (element symbols)
    type_map = jdata.get("type_map", [])
    type_map_str = " ".join(type_map)

    # Find pair_coeff line
    pair_coeff_idx = None
    for idx, line in enumerate(lmp_lines):
        if line.strip().startswith("pair_coeff") and "* *" in line:
            pair_coeff_idx = idx
            break

    if pair_coeff_idx is None:
        # If no pair_coeff found, add them after pair_style
        pair_style_idx = find_only_one_key(lmp_lines, ["pair_style"])
        lmp_lines.insert(pair_style_idx + 1, "pair_coeff      * * deepmd\n")
        lmp_lines.insert(
            pair_style_idx + 2, f"pair_coeff      * * dispersion/d3 {type_map_str}\n"
        )
    else:
        # Replace existing pair_coeff with D3 version
        lmp_lines[pair_coeff_idx] = "pair_coeff      * * deepmd\n"
        lmp_lines.insert(
            pair_coeff_idx + 1, f"pair_coeff      * * dispersion/d3 {type_map_str}\n"
        )

    return lmp_lines




def revise_lmp_input_neigh_modify(lmp_lines, jdata=None):
    """Add neigh_modify one N if requested."""
    if jdata is None:
        return lmp_lines

    neigh_modify_one = jdata.get("lmp_neigh_modify_one")
    if neigh_modify_one is None:
        return lmp_lines

    # Find where to insert neigh_modify one N
    # Look for existing neigh_modify lines or insert after neighbor command
    neigh_modify_found = False
    neighbor_idx = None

    for idx, line in enumerate(lmp_lines):
        if line.strip().startswith("neigh_modify") and " one " in line:
            neigh_modify_found = True
            break
        elif line.strip().startswith("neighbor"):
            neighbor_idx = idx

    if not neigh_modify_found:
        if neighbor_idx is not None:
            lmp_lines.insert(
                neighbor_idx + 1, f"neigh_modify    one {neigh_modify_one}\n"
            )
        else:
            # Insert after units command if neighbor not found
            units_idx = None
            for idx, line in enumerate(lmp_lines):
                if line.strip().startswith("units"):
                    units_idx = idx
                    break
            if units_idx is not None:
                lmp_lines.insert(
                    units_idx + 1, f"neigh_modify    one {neigh_modify_one}\n"
                )

    return lmp_lines




def revise_lmp_input_dump(lmp_lines, trj_freq, model_devi_merge_traj=False):
    idx = find_only_one_key(lmp_lines, ["dump", "dpgen_dump"])
    if model_devi_merge_traj:
        lmp_lines[idx] = (
            "dump            dpgen_dump all custom %d    all.lammpstrj id type x y z\n"  # noqa: UP031
            % trj_freq
        )
    else:
        lmp_lines[idx] = (
            "dump            dpgen_dump all custom %d traj/*.lammpstrj id type x y z\n"  # noqa: UP031
            % trj_freq
        )

    return lmp_lines




def revise_lmp_input_plm(lmp_lines, in_plm, out_plm="output.plumed"):
    idx = find_only_one_key(lmp_lines, ["fix", "dpgen_plm"])
    lmp_lines[idx] = (
        f"fix            dpgen_plm all plumed plumedfile {in_plm} outfile {out_plm}\n"
    )
    return lmp_lines




def revise_by_keys(lmp_lines, keys, values):
    for kk, vv in zip(keys, values):
        for ii in range(len(lmp_lines)):
            lmp_lines[ii] = lmp_lines[ii].replace(kk, str(vv))
    return lmp_lines




def make_model_devi(iter_index, jdata, mdata):
    # The MD engine to perform model deviation
    # Default is lammps
    model_devi_engine = jdata.get("model_devi_engine", "lammps")

    model_devi_jobs = jdata["model_devi_jobs"]
    if model_devi_engine != "calypso":
        if iter_index >= len(model_devi_jobs):
            return False
    else:
        # mode 1: generate structures according to the user-provided input.dat file, so calypso_input_path and model_devi_max_iter are needed
        run_mode = 1
        if "calypso_input_path" in jdata:
            try:
                maxiter = jdata.get("model_devi_max_iter")
            except KeyError:
                raise KeyError(
                    "calypso_input_path key exists so you should provide model_devi_max_iter key to control the max iter number"
                )
        # mode 2: control each iteration to generate structures in specific way by providing model_devi_jobs key
        else:
            try:
                maxiter = max(model_devi_jobs[-1].get("times"))
                run_mode = 2
            except KeyError:
                raise KeyError('did not find model_devi_jobs["times"] key')
        if iter_index > maxiter:
            dlog.info(f"iter_index is {iter_index} and maxiter is {maxiter}")
            return False

    if "sys_configs_prefix" in jdata:
        sys_configs = []
        for sys_list in jdata["sys_configs"]:
            # assert (isinstance(sys_list, list) ), "Currently only support type list for sys in 'sys_conifgs' "
            temp_sys_list = [
                os.path.join(jdata["sys_configs_prefix"], sys) for sys in sys_list
            ]
            sys_configs.append(temp_sys_list)
    else:
        sys_configs = jdata["sys_configs"]
    shuffle_poscar = jdata.get("shuffle_poscar", False)

    if model_devi_engine != "calypso":
        cur_job = model_devi_jobs[iter_index]
        sys_idx = expand_idx(cur_job["sys_idx"])
    else:
        cur_job = {"model_devi_engine": "calypso", "input.dat": "user_provided"}
        sys_idx = []

    if len(sys_idx) != len(list(set(sys_idx))):
        raise RuntimeError("system index should be uniq")
    conf_systems = []
    for idx in sys_idx:
        cur_systems = []
        ss = sys_configs[idx]
        for ii in ss:
            ii_systems = sorted(glob.glob(ii))
            if ii_systems == []:
                warnings.warn(
                    f"There is no system in the path {ii}. Please check if the path is correct."
                )
            cur_systems += ii_systems
        # cur_systems should not be sorted, as we may add specific constrict to the similutions
        # cur_systems.sort()
        cur_systems = [os.path.abspath(ii) for ii in cur_systems]
        conf_systems.append(cur_systems)

    iter_name = make_iter_name(iter_index)
    train_path = os.path.join(iter_name, train_name)
    train_path = os.path.abspath(train_path)
    suffix = _get_model_suffix(jdata)
    models = sorted(glob.glob(os.path.join(train_path, f"graph*{suffix}")))
    work_path = os.path.join(iter_name, model_devi_name)
    create_path(work_path)
    if model_devi_engine == "calypso":
        _calypso_run_opt_path = os.path.join(work_path, calypso_run_opt_name)
        calypso_model_devi_path = os.path.join(work_path, calypso_model_devi_name)
        create_path(calypso_model_devi_path)
        # run model devi script
        calypso_run_model_devi_script = os.path.join(
            calypso_model_devi_path, "calypso_run_model_devi.py"
        )
        shutil.copyfile(calypso_run_model_devi_file, calypso_run_model_devi_script)
        # Create work path list
        calypso_run_opt_path = []

        # mode 1: generate structures according to the user-provided input.dat file,
        # so calypso_input_path and model_devi_max_iter are needed
        if run_mode == 1:
            if jdata.get("vsc", False) and len(jdata.get("type_map")) > 1:
                # [input.dat.Li.250, input.dat.Li.300]
                one_ele_inputdat_list = glob.glob(
                    f"{jdata.get('calypso_input_path')}/input.dat.{jdata.get('type_map')[0]}.*"
                )
                if len(one_ele_inputdat_list) == 0:
                    number_of_pressure = 1
                else:
                    number_of_pressure = len(list(set(one_ele_inputdat_list)))

                # calypso_run_opt_path = ['gen_struc_analy.000','gen_struc_analy.001']
                for temp_idx in range(number_of_pressure):
                    calypso_run_opt_path.append(
                        "%s.%03d" % (_calypso_run_opt_path, temp_idx)  # noqa: UP031
                    )
            elif not jdata.get("vsc", False):
                calypso_run_opt_path.append("%s.%03d" % (_calypso_run_opt_path, 0))  # noqa: UP031

        # mode 2: control each iteration to generate structures in specific way
        # by providing model_devi_jobs key
        elif run_mode == 2:
            for iiidx, jobbs in enumerate(model_devi_jobs):
                if iter_index in jobbs.get("times"):
                    cur_job = model_devi_jobs[iiidx]

            pressures_list = cur_job.get("PSTRESS", [0.0001])
            for temp_idx in range(len(pressures_list)):
                calypso_run_opt_path.append(
                    "%s.%03d" % (_calypso_run_opt_path, temp_idx)  # noqa: UP031
                )
        # to different directory
        # calypso_run_opt_path = ['gen_struc_analy.000','gen_struc_analy.001','gen_struc_analy.002',]
        for temp_calypso_run_opt_path in calypso_run_opt_path:
            create_path(temp_calypso_run_opt_path)
            # run confs opt script
            run_opt_script = os.path.join(
                temp_calypso_run_opt_path, "calypso_run_opt.py"
            )
            shutil.copyfile(run_opt_file, run_opt_script)
            # check outcar script
            check_outcar_script = os.path.join(
                temp_calypso_run_opt_path, "check_outcar.py"
            )
            shutil.copyfile(check_outcar_file, check_outcar_script)

    for mm in models:
        model_name = os.path.basename(mm)
        if model_devi_engine != "calypso":
            os.symlink(mm, os.path.join(work_path, model_name))
        else:
            for temp_calypso_run_opt_path in calypso_run_opt_path:
                models_path = os.path.join(temp_calypso_run_opt_path, model_name)
                if not os.path.exists(models_path):
                    os.symlink(mm, models_path)

    with open(os.path.join(work_path, "cur_job.json"), "w") as outfile:
        json.dump(cur_job, outfile, indent=4)

    conf_path = os.path.join(work_path, "confs")
    create_path(conf_path)
    sys_counter = 0
    rng = np.random.default_rng()
    for ss in conf_systems:
        conf_counter = 0
        for cc in ss:
            if model_devi_engine == "lammps":
                conf_name = make_model_devi_conf_name(
                    sys_idx[sys_counter], conf_counter
                )
                poscar_name = conf_name + ".poscar"
                lmp_name = conf_name + ".lmp"
                os.symlink(cc, os.path.join(conf_path, poscar_name))
                if "sys_format" in jdata:
                    fmt = jdata["sys_format"]
                else:
                    fmt = "vasp/poscar"
                system = dpdata.System(
                    os.path.join(conf_path, poscar_name),
                    fmt=fmt,
                    type_map=jdata["type_map"],
                )
                if shuffle_poscar:
                    rng.shuffle(system.data["coords"], axis=1)
                if jdata.get("model_devi_nopbc", False):
                    system.remove_pbc()
                system.to_lammps_lmp(os.path.join(conf_path, lmp_name))
            elif model_devi_engine == "gromacs":
                pass
            elif model_devi_engine == "amber":
                # Jinzhe's specific Amber version
                conf_name = make_model_devi_conf_name(
                    sys_idx[sys_counter], conf_counter
                )
                rst7_name = conf_name + ".rst7"
                # link restart file
                os.symlink(cc, os.path.join(conf_path, rst7_name))
            conf_counter += 1
        sys_counter += 1

    input_mode = "native"
    if "calypso_input_path" in jdata:
        input_mode = "buffet"
    if "template" in cur_job:
        input_mode = "revise_template"
    use_plm = jdata.get("model_devi_plumed", False)
    use_plm_path = jdata.get("model_devi_plumed_path", False)
    if input_mode == "native":
        if model_devi_engine == "lammps":
            _make_model_devi_native(iter_index, jdata, mdata, conf_systems)
        elif model_devi_engine == "gromacs":
            _make_model_devi_native_gromacs(iter_index, jdata, mdata, conf_systems)
        elif model_devi_engine == "amber":
            _make_model_devi_amber(iter_index, jdata, mdata, conf_systems)
        elif model_devi_engine == "calypso":
            _make_model_devi_native_calypso(
                iter_index, model_devi_jobs, calypso_run_opt_path
            )  # generate input.dat automatic in each iter
        else:
            raise RuntimeError("unknown model_devi engine", model_devi_engine)
    elif input_mode == "revise_template":
        _make_model_devi_revmat(iter_index, jdata, mdata, conf_systems)
    elif input_mode == "buffet":
        _make_model_devi_buffet(
            jdata, calypso_run_opt_path
        )  # generate confs according to the input.dat provided
    else:
        raise RuntimeError("unknown model_devi input mode", input_mode)
    # Copy user defined forward_files
    symlink_user_forward_files(mdata=mdata, task_type="model_devi", work_path=work_path)
    return True




def _make_model_devi_revmat(iter_index, jdata, mdata, conf_systems):
    model_devi_jobs = jdata["model_devi_jobs"]
    if iter_index >= len(model_devi_jobs):
        return False
    cur_job = model_devi_jobs[iter_index]
    sys_idx = expand_idx(cur_job["sys_idx"])
    if len(sys_idx) != len(list(set(sys_idx))):
        raise RuntimeError("system index should be uniq")

    use_ele_temp = jdata.get("use_ele_temp", 0)
    mass_map = jdata["mass_map"]
    use_plm = jdata.get("model_devi_plumed", False)
    use_plm_path = jdata.get("model_devi_plumed_path", False)
    trj_freq = _get_param_alias(cur_job, ["t_freq", "trj_freq", "traj_freq"])

    rev_keys, rev_mat, num_lmp = parse_cur_job_revmat(cur_job, use_plm=use_plm)
    lmp_templ = cur_job["template"]["lmp"]
    lmp_templ = os.path.abspath(lmp_templ)
    if use_plm:
        plm_templ = cur_job["template"]["plm"]
        plm_templ = os.path.abspath(plm_templ)
        if use_plm_path:
            plm_path_templ = cur_job["template"]["plm_path"]
            plm_path_templ = os.path.abspath(plm_path_templ)

    iter_name = make_iter_name(iter_index)
    train_path = os.path.join(iter_name, train_name)
    train_path = os.path.abspath(train_path)
    suffix = _get_model_suffix(jdata)
    models = sorted(glob.glob(os.path.join(train_path, f"graph*{suffix}")))
    task_model_list = []
    for ii in models:
        task_model_list.append(os.path.join("..", os.path.basename(ii)))
    work_path = os.path.join(iter_name, model_devi_name)
    try:
        mdata["deepmd_version"]
    except KeyError:
        mdata = set_version(mdata)
    deepmd_version = mdata["deepmd_version"]

    sys_counter = 0
    for ss in conf_systems:
        conf_counter = 0
        task_counter = 0
        for cc in ss:
            sys_rev = cur_job.get("sys_rev_mat", None)
            total_rev_keys = rev_keys
            total_rev_mat = rev_mat
            total_num_lmp = num_lmp
            if sys_rev is not None:
                total_rev_mat = []
                sys_rev_keys, sys_rev_mat, sys_num_lmp = parse_cur_job_sys_revmat(
                    cur_job, sys_idx=sys_idx[sys_counter], use_plm=use_plm
                )
                _lmp_keys = rev_keys[:num_lmp] + sys_rev_keys[:sys_num_lmp]
                if use_plm:
                    _plm_keys = rev_keys[num_lmp:] + sys_rev_keys[sys_num_lmp:]
                    _lmp_keys += _plm_keys
                total_rev_keys = _lmp_keys
                total_num_lmp = num_lmp + sys_num_lmp
                for pub in rev_mat:
                    for pri in sys_rev_mat:
                        _lmp_mat = pub[:num_lmp] + pri[:sys_num_lmp]
                        if use_plm:
                            _plm_mat = pub[num_lmp:] + pri[sys_num_lmp:]
                            _lmp_mat += _plm_mat
                        total_rev_mat.append(_lmp_mat)
            for ii in range(len(total_rev_mat)):
                total_rev_item = total_rev_mat[ii]
                task_name = make_model_devi_task_name(
                    sys_idx[sys_counter], task_counter
                )
                conf_name = (
                    make_model_devi_conf_name(sys_idx[sys_counter], conf_counter)
                    + ".lmp"
                )
                task_path = os.path.join(work_path, task_name)
                # create task path
                create_path(task_path)
                model_devi_merge_traj = jdata.get("model_devi_merge_traj", False)
                if not model_devi_merge_traj:
                    create_path(os.path.join(task_path, "traj"))
                # link conf
                loc_conf_name = "conf.lmp"
                os.symlink(
                    os.path.join(os.path.join("..", "confs"), conf_name),
                    os.path.join(task_path, loc_conf_name),
                )
                cwd_ = os.getcwd()
                # chdir to task path
                os.chdir(task_path)
                shutil.copyfile(lmp_templ, "input.lammps")
                # revise input of lammps
                with open("input.lammps") as fp:
                    lmp_lines = fp.readlines()
                # only revise the line "pair_style deepmd" if the user has not written the full line (checked by then length of the line)
                template_has_pair_deepmd = 1
                for line_idx, line_context in enumerate(lmp_lines):
                    if (
                        (line_context[0] != "#")
                        and ("pair_style" in line_context)
                        and ("deepmd" in line_context)
                    ):
                        template_has_pair_deepmd = 0
                        template_pair_deepmd_idx = line_idx
                if template_has_pair_deepmd == 0:
                    if Version(deepmd_version) < Version("1"):
                        if len(lmp_lines[template_pair_deepmd_idx].split()) != (
                            len(models)
                            + len(["pair_style", "deepmd", "10", "model_devi.out"])
                        ):
                            lmp_lines = revise_lmp_input_model(
                                lmp_lines,
                                task_model_list,
                                trj_freq,
                                deepmd_version=deepmd_version,
                                use_ele_temp=use_ele_temp,
                                jdata=jdata,
                            )
                            # Add D3 pair_coeff and neigh_modify support for templates
                            lmp_lines = revise_lmp_input_pair_coeff(lmp_lines, jdata)
                            lmp_lines = revise_lmp_input_neigh_modify(lmp_lines, jdata)
                    else:
                        if len(lmp_lines[template_pair_deepmd_idx].split()) != (
                            len(models)
                            + len(
                                [
                                    "pair_style",
                                    "deepmd",
                                    "out_freq",
                                    "10",
                                    "out_file",
                                    "model_devi.out",
                                ]
                            )
                        ):
                            lmp_lines = revise_lmp_input_model(
                                lmp_lines,
                                task_model_list,
                                trj_freq,
                                deepmd_version=deepmd_version,
                                use_ele_temp=use_ele_temp,
                                jdata=jdata,
                            )
                            # Add D3 pair_coeff and neigh_modify support for templates
                            lmp_lines = revise_lmp_input_pair_coeff(lmp_lines, jdata)
                            lmp_lines = revise_lmp_input_neigh_modify(lmp_lines, jdata)
                # use revise_lmp_input_model to raise error message if "part_style" or "deepmd" not found
                else:
                    lmp_lines = revise_lmp_input_model(
                        lmp_lines,
                        task_model_list,
                        trj_freq,
                        deepmd_version=deepmd_version,
                        use_ele_temp=use_ele_temp,
                        jdata=jdata,
                    )

                # Add D3 pair_coeff and neigh_modify support for templates
                lmp_lines = revise_lmp_input_pair_coeff(lmp_lines, jdata)
                lmp_lines = revise_lmp_input_neigh_modify(lmp_lines, jdata)

                lmp_lines = revise_lmp_input_dump(
                    lmp_lines, trj_freq, model_devi_merge_traj
                )
                lmp_lines = revise_by_keys(
                    lmp_lines,
                    total_rev_keys[:total_num_lmp],
                    total_rev_item[:total_num_lmp],
                )
                # revise input of plumed
                if use_plm:
                    lmp_lines = revise_lmp_input_plm(lmp_lines, "input.plumed")
                    shutil.copyfile(plm_templ, "input.plumed")
                    with open("input.plumed") as fp:
                        plm_lines = fp.readlines()
                    # allow using the same list as lmp
                    # user should not use the same key name for plm
                    plm_lines = revise_by_keys(
                        plm_lines, total_rev_keys, total_rev_item
                    )
                    with open("input.plumed", "w") as fp:
                        fp.write("".join(plm_lines))
                    if use_plm_path:
                        shutil.copyfile(plm_path_templ, "plmpath.pdb")
                # dump input of lammps
                with open("input.lammps", "w") as fp:
                    fp.write("".join(lmp_lines))
                with open("job.json", "w") as fp:
                    job = {}
                    for ii, jj in zip(total_rev_keys, total_rev_item):
                        job[ii] = jj
                    json.dump(job, fp, indent=4)
                os.chdir(cwd_)
                task_counter += 1
            conf_counter += 1
        sys_counter += 1




def _make_model_devi_native(iter_index, jdata, mdata, conf_systems):
    model_devi_jobs = jdata["model_devi_jobs"]
    if iter_index >= len(model_devi_jobs):
        return False
    cur_job = model_devi_jobs[iter_index]
    ensemble, nsteps, trj_freq, temps, press, pka_e, dt, nbeads = parse_cur_job(cur_job)
    model_devi_f_avg_relative = jdata.get("model_devi_f_avg_relative", False)
    model_devi_merge_traj = jdata.get("model_devi_merge_traj", False)
    if (nbeads is not None) and model_devi_f_avg_relative:
        raise RuntimeError(
            "model_devi_f_avg_relative has not been supported for pimd. Set model_devi_f_avg_relative to False."
        )
    if (nbeads is not None) and (model_devi_merge_traj):
        raise RuntimeError(
            "model_devi_merge_traj has not been supported for pimd. Set model_devi_merge_traj to False."
        )
    if (nbeads is not None) and (not (nsteps % trj_freq == 0)):
        raise RuntimeError(
            "trj_freq should be a factor of nsteps for pimd. Please check your input."
        )
    if dt is not None:
        model_devi_dt = dt
    sys_idx = expand_idx(cur_job["sys_idx"])
    if len(sys_idx) != len(list(set(sys_idx))):
        raise RuntimeError("system index should be uniq")

    use_ele_temp = jdata.get("use_ele_temp", 0)
    model_devi_dt = jdata["model_devi_dt"]
    model_devi_neidelay = None
    if "model_devi_neidelay" in jdata:
        model_devi_neidelay = jdata["model_devi_neidelay"]
    model_devi_taut = 0.1
    if "model_devi_taut" in jdata:
        model_devi_taut = jdata["model_devi_taut"]
    model_devi_taup = 0.5
    if "model_devi_taup" in jdata:
        model_devi_taup = jdata["model_devi_taup"]
    mass_map = jdata["mass_map"]
    nopbc = jdata.get("model_devi_nopbc", False)

    iter_name = make_iter_name(iter_index)
    train_path = os.path.join(iter_name, train_name)
    train_path = os.path.abspath(train_path)
    suffix = _get_model_suffix(jdata)
    models = sorted(glob.glob(os.path.join(train_path, f"graph*{suffix}")))
    task_model_list = []
    for ii in models:
        task_model_list.append(Path(f"../{Path(ii).name}").as_posix())
    work_path = os.path.join(iter_name, model_devi_name)

    sys_counter = 0
    for ss in conf_systems:
        conf_counter = 0
        task_counter = 0
        for cc in ss:
            for tt_ in temps:
                if use_ele_temp:
                    if isinstance(tt_, list):
                        tt = tt_[0]
                        if use_ele_temp == 1:
                            te_f = tt_[1]
                            te_a = None
                        else:
                            te_f = None
                            te_a = tt_[1]
                    else:
                        assert isinstance(tt_, (float, int))
                        tt = float(tt_)
                        if use_ele_temp == 1:
                            te_f = tt
                            te_a = None
                        else:
                            te_f = None
                            te_a = tt
                else:
                    tt = tt_
                    te_f = None
                    te_a = None
                for pp in press:
                    task_name = make_model_devi_task_name(
                        sys_idx[sys_counter], task_counter
                    )
                    conf_name = (
                        make_model_devi_conf_name(sys_idx[sys_counter], conf_counter)
                        + ".lmp"
                    )
                    task_path = os.path.join(work_path, task_name)
                    # dlog.info(task_path)
                    create_path(task_path)
                    model_devi_merge_traj = jdata.get("model_devi_merge_traj", False)
                    if not model_devi_merge_traj:
                        create_path(os.path.join(task_path, "traj"))
                    loc_conf_name = "conf.lmp"
                    os.symlink(
                        os.path.join(os.path.join("..", "confs"), conf_name),
                        os.path.join(task_path, loc_conf_name),
                    )
                    cwd_ = os.getcwd()
                    os.chdir(task_path)
                    try:
                        mdata["deepmd_version"]
                    except KeyError:
                        mdata = set_version(mdata)
                    deepmd_version = mdata["deepmd_version"]
                    file_c = make_lammps_input(
                        ensemble,
                        loc_conf_name,
                        task_model_list,
                        nsteps,
                        model_devi_dt,
                        model_devi_neidelay,
                        trj_freq,
                        mass_map,
                        tt,
                        jdata=jdata,
                        tau_t=model_devi_taut,
                        pres=pp,
                        tau_p=model_devi_taup,
                        pka_e=pka_e,
                        ele_temp_f=te_f,
                        ele_temp_a=te_a,
                        nopbc=nopbc,
                        deepmd_version=deepmd_version,
                        nbeads=nbeads,
                    )
                    job = {}
                    job["ensemble"] = ensemble
                    job["press"] = pp
                    job["temps"] = tt
                    if te_f is not None:
                        job["ele_temp"] = te_f
                    if te_a is not None:
                        job["ele_temp"] = te_a
                    job["model_devi_dt"] = model_devi_dt
                    with open("job.json", "w") as _outfile:
                        json.dump(job, _outfile, indent=4)
                    os.chdir(cwd_)
                    with open(os.path.join(task_path, "input.lammps"), "w") as fp:
                        fp.write(file_c)
                    task_counter += 1
            conf_counter += 1
        sys_counter += 1




def _make_model_devi_native_gromacs(iter_index, jdata, mdata, conf_systems):
    try:
        from gromacs.fileformats.mdp import MDP
    except ImportError as e:
        raise RuntimeError(
            "GromacsWrapper>=0.8.0 is needed for DP-GEN + Gromacs."
        ) from e
    # only support for deepmd v2.0
    if Version(mdata["deepmd_version"]) < Version("2.0"):
        raise RuntimeError(
            "Only support deepmd-kit v2 or above for model_devi_engine='gromacs'"
        )
    model_devi_jobs = jdata["model_devi_jobs"]
    if iter_index >= len(model_devi_jobs):
        return False
    cur_job = model_devi_jobs[iter_index]
    dt = cur_job.get("dt", None)
    if dt is not None:
        model_devi_dt = dt
    else:
        model_devi_dt = jdata["model_devi_dt"]
    nsteps = cur_job.get("nsteps", None)
    lambdas = cur_job.get("lambdas", [1.0])
    temps = cur_job.get("temps", [298.0])

    for ll in lambdas:
        assert ll >= 0.0 and ll <= 1.0, "Lambda should be in [0,1]"

    if nsteps is None:
        raise RuntimeError("nsteps is None, you should set nsteps in model_devi_jobs!")
    # Currently Gromacs engine is not supported for different temperatures!
    # If you want to change temperatures, you should change it in mdp files.

    sys_idx = expand_idx(cur_job["sys_idx"])
    if len(sys_idx) != len(list(set(sys_idx))):
        raise RuntimeError("system index should be uniq")

    mass_map = jdata["mass_map"]

    iter_name = make_iter_name(iter_index)
    train_path = os.path.join(iter_name, train_name)
    train_path = os.path.abspath(train_path)
    suffix = _get_model_suffix(jdata)
    models = sorted(glob.glob(os.path.join(train_path, f"graph*{suffix}")))
    task_model_list = []
    for ii in models:
        task_model_list.append(os.path.join("..", os.path.basename(ii)))
    work_path = os.path.join(iter_name, model_devi_name)

    sys_counter = 0
    for ss in conf_systems:
        conf_counter = 0
        task_counter = 0
        for cc in ss:
            for ll in lambdas:
                for tt in temps:
                    task_name = make_model_devi_task_name(
                        sys_idx[sys_counter], task_counter
                    )
                    task_path = os.path.join(work_path, task_name)
                    create_path(task_path)
                    gromacs_settings = jdata.get("gromacs_settings", "")
                    for key, file in gromacs_settings.items():
                        if (
                            key != "traj_filename"
                            and key != "mdp_filename"
                            and key != "group_name"
                            and key != "maxwarn"
                        ):
                            os.symlink(
                                os.path.join(cc, file), os.path.join(task_path, file)
                            )
                    # input.json for DP-Gromacs
                    with open(os.path.join(cc, "input.json")) as f:
                        input_json = json.load(f)
                    input_json["graph_file"] = models[0]
                    input_json["lambda"] = ll
                    with open(os.path.join(task_path, "input.json"), "w") as _outfile:
                        json.dump(input_json, _outfile, indent=4)

                    # trj_freq
                    trj_freq = cur_job.get("trj_freq", 10)
                    mdp = MDP()
                    mdp.read(os.path.join(cc, gromacs_settings["mdp_filename"]))
                    mdp["nstcomm"] = trj_freq
                    mdp["nstxout"] = trj_freq
                    mdp["nstlog"] = trj_freq
                    mdp["nstenergy"] = trj_freq
                    # dt
                    mdp["dt"] = model_devi_dt
                    # nsteps
                    mdp["nsteps"] = nsteps
                    # temps
                    if "ref_t" in list(mdp.keys()):
                        mdp["ref_t"] = tt
                    else:
                        mdp["ref-t"] = tt
                    mdp.write(os.path.join(task_path, gromacs_settings["mdp_filename"]))

                    cwd_ = os.getcwd()
                    os.chdir(task_path)
                    job = {}
                    job["trj_freq"] = cur_job["trj_freq"]
                    job["model_devi_dt"] = model_devi_dt
                    job["nsteps"] = nsteps
                    with open("job.json", "w") as _outfile:
                        json.dump(job, _outfile, indent=4)
                    os.chdir(cwd_)
                    task_counter += 1
            conf_counter += 1
        sys_counter += 1




def _make_model_devi_amber(
    iter_index: int, jdata: dict, mdata: dict, conf_systems: list
):
    """Make amber's MD inputs.

    Parameters
    ----------
    iter_index : int
        iter index
    jdata : dict
        run parameters. The following parameters will be used in this method:
            model_devi_jobs : list[dict]
                The list including the dict for information of each cycle:
                    sys_idx : list[int]
                        list of systems to run
                    trj_freq : int
                        freq to dump trajectory
            low_level : str
                low level method
            cutoff : float
                cutoff radius of the DPRc model
            parm7_prefix : str
                The path prefix to AMBER PARM7 files
            parm7 : list[str]
                List of paths to AMBER PARM7 files. Each file maps to a system.
            mdin_prefix : str
                The path prefix to AMBER mdin files
            mdin : list[str]
                List of paths to AMBER mdin files. Each files maps to a system.
                The following keywords will be replaced by the actual value:
                    @freq@ : freq to dump trajectory
                    @nstlim@ : total time step to run
                    @qm_region@ : AMBER mask of the QM region
                    @qm_theory@ : The QM theory, such as DFTB2
                    @qm_charge@ : The total charge of the QM theory, such as -2
                    @rcut@ : cutoff radius of the DPRc model
                    @GRAPH_FILE0@, @GRAPH_FILE1@, ... : graph files
            qm_region : list[str]
                AMBER mask of the QM region. Each mask maps to a system.
            qm_charge : list[int]
                Charge of the QM region. Each charge maps to a system.
            nsteps : list[int]
                The number of steps to run. Each number maps to a system.
            r : list[list[float]] or list[list[list[float]]]
                Constrict values for the enhanced sampling. The first dimension maps to systems.
                The second dimension maps to confs in each system. The third dimension is the
                constrict value. It can be a single float for 1D or list of floats for nD.
            disang_prefix : str
                The path prefix to disang prefix.
            disang : list[str]
                List of paths to AMBER disang files. Each file maps to a sytem.
                The keyword RVAL will be replaced by the constrict values, or RVAL1, RVAL2, ...
                for an nD system.
    mdata : dict
        machine parameters. Nothing will be used in this method.
    conf_systems : list
        conf systems

    References
    ----------
    .. [1] Development of Range-Corrected Deep Learning Potentials for Fast, Accurate Quantum
       Mechanical/Molecular Mechanical Simulations of Chemical Reactions in Solution,
       Jinzhe Zeng, Timothy J. Giese, Şölen Ekesan, and Darrin M. York, Journal of Chemical
       Theory and Computation 2021 17 (11), 6993-7009

    inputs: restart (coords), param, mdin, graph, disang (optional)

    """
    model_devi_jobs = jdata["model_devi_jobs"]
    if iter_index >= len(model_devi_jobs):
        return False
    cur_job = model_devi_jobs[iter_index]
    sys_idx = expand_idx(cur_job["sys_idx"])
    if len(sys_idx) != len(list(set(sys_idx))):
        raise RuntimeError("system index should be uniq")

    iter_name = make_iter_name(iter_index)
    train_path = os.path.join(iter_name, train_name)
    train_path = os.path.abspath(train_path)
    work_path = os.path.join(iter_name, model_devi_name)
    # parm7 - list
    parm7 = jdata["parm7"]
    parm7_prefix = jdata.get("parm7_prefix", "")
    parm7 = [os.path.join(parm7_prefix, pp) for pp in parm7]

    # link parm file
    for ii, pp in enumerate(parm7):
        os.symlink(pp, os.path.join(work_path, "qmmm%d.parm7" % ii))  # noqa: UP031
    # TODO: consider writing input in json instead of a given file
    # mdin
    mdin = jdata["mdin"]
    mdin_prefix = jdata.get("mdin_prefix", "")
    mdin = [os.path.join(mdin_prefix, pp) for pp in mdin]

    qm_region = jdata["qm_region"]
    qm_charge = jdata["qm_charge"]
    nsteps = jdata["nsteps"]

    for ii, pp in enumerate(mdin):
        with (
            open(pp) as f,
            open(os.path.join(work_path, "init%d.mdin" % ii), "w") as fw,  # noqa: UP031
        ):
            mdin_str = f.read()
            # freq, nstlim, qm_region, qm_theory, qm_charge, rcut, graph
            mdin_str = (
                mdin_str.replace("@freq@", str(cur_job.get("trj_freq", 50)))
                .replace("@nstlim@", str(nsteps[ii]))
                .replace("@qm_region@", qm_region[ii])
                .replace("@qm_charge@", str(qm_charge[ii]))
                .replace("@qm_theory@", jdata["low_level"])
                .replace("@rcut@", str(jdata["cutoff"]))
            )
            suffix = _get_model_suffix(jdata)
            models = sorted(glob.glob(os.path.join(train_path, f"graph.*{suffix}")))
            task_model_list = []
            for ii in models:
                task_model_list.append(os.path.join("..", os.path.basename(ii)))
            # graph
            for jj, mm in enumerate(task_model_list):
                # replace graph
                mdin_str = mdin_str.replace("@GRAPH_FILE%d@" % jj, mm)  # noqa: UP031
            fw.write(mdin_str)
    # disang - list
    disang = jdata["disang"]
    disang_prefix = jdata.get("disang_prefix", "")
    disang = [os.path.join(disang_prefix, pp) for pp in disang]

    for sys_counter, ss in enumerate(conf_systems):
        for idx_cc, cc in enumerate(ss):
            task_counter = idx_cc
            conf_counter = idx_cc

            task_name = make_model_devi_task_name(sys_idx[sys_counter], task_counter)
            conf_name = make_model_devi_conf_name(sys_idx[sys_counter], conf_counter)
            task_path = os.path.join(work_path, task_name)
            # create task path
            create_path(task_path)
            # link restart file
            loc_conf_name = "init.rst7"
            if cur_job.get("restart_from_iter") is None:
                os.symlink(
                    os.path.join(os.path.join("..", "confs"), conf_name + ".rst7"),
                    os.path.join(task_path, loc_conf_name),
                )
            else:
                restart_from_iter = cur_job["restart_from_iter"]
                restart_iter_name = make_iter_name(restart_from_iter)
                os.symlink(
                    os.path.relpath(
                        os.path.join(
                            restart_iter_name, model_devi_name, task_name, "rc.rst7"
                        ),
                        task_path,
                    ),
                    os.path.join(task_path, loc_conf_name),
                )
            cwd_ = os.getcwd()
            # chdir to task path
            os.chdir(task_path)

            # reaction coordinates of umbrella sampling
            # TODO: maybe consider a better name instead of `r`?
            if "r" in jdata:
                r = jdata["r"][sys_idx[sys_counter]][conf_counter]
                # r can either be a float or a list of float (for 2D coordinates)
                if not isinstance(r, Iterable) or isinstance(r, str):
                    r = [r]
                # disang file should include RVAL, RVAL2, ...
                with (
                    open(disang[sys_idx[sys_counter]]) as f,
                    open("TEMPLATE.disang", "w") as fw,
                ):
                    tl = f.read()
                    for ii, rr in enumerate(r):
                        if isinstance(rr, Iterable) and not isinstance(rr, str):
                            raise RuntimeError(
                                "rr should not be iterable! sys: %d rr: %s r: %s"  # noqa: UP031
                                % (sys_idx[sys_counter], str(rr), str(r))
                            )
                        tl = tl.replace("RVAL" + str(ii + 1), str(rr))
                    if len(r) == 1:
                        tl = tl.replace("RVAL", str(r[0]))
                    fw.write(tl)

            with open("job.json", "w") as fp:
                json.dump(cur_job, fp, indent=4)
            os.chdir(cwd_)




def run_md_model_devi(iter_index, jdata, mdata):
    # rmdlog.info("This module has been run !")
    model_devi_exec = mdata["model_devi_command"]

    model_devi_group_size = mdata["model_devi_group_size"]
    model_devi_resources = mdata["model_devi_resources"]
    use_plm = jdata.get("model_devi_plumed", False)
    use_plm_path = jdata.get("model_devi_plumed_path", False)
    model_devi_merge_traj = jdata.get("model_devi_merge_traj", False)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, model_devi_name)
    assert os.path.isdir(work_path)

    all_task = glob.glob(os.path.join(work_path, "task.*"))
    all_task.sort()
    fp = open(os.path.join(work_path, "cur_job.json"))
    cur_job = json.load(fp)

    run_tasks_ = all_task
    # for ii in all_task:
    #     fres = os.path.join(ii, 'model_devi.out')
    #     if os.path.isfile(fres) :
    #         nlines = np.loadtxt(fres).shape[0]
    #         if nframes != nlines :
    #             run_tasks_.append(ii)
    #     else :
    #         run_tasks_.append(ii)

    run_tasks = [os.path.basename(ii) for ii in run_tasks_]
    # dlog.info("all_task is ", all_task)
    # dlog.info("run_tasks in run_model_deviation",run_tasks_)

    suffix = _get_model_suffix(jdata)
    all_models = glob.glob(os.path.join(work_path, f"graph*{suffix}"))
    model_names = [os.path.basename(ii) for ii in all_models]

    model_devi_engine = jdata.get("model_devi_engine", "lammps")
    if model_devi_engine == "lammps":
        nbeads = jdata["model_devi_jobs"][iter_index].get("nbeads")
        if nbeads is None:
            command = f"{{ if [ ! -f dpgen.restart.10000 ]; then {model_devi_exec} -i input.lammps -v restart 0; else {model_devi_exec} -i input.lammps -v restart 1; fi }}"
        else:
            command = f"{{ all_exist=true; for i in $(seq -w 1 {nbeads}); do [[ ! -f dpgen.restart${{i}}.10000 ]] && {{ all_exist=false; break; }}; done; $all_exist && {{ {model_devi_exec} -p {nbeads}x1 -i input.lammps -v restart 1; }} || {{ {model_devi_exec} -p {nbeads}x1 -i input.lammps -v restart 0; }} }}"
        command = f"/bin/bash -c {shlex.quote(command)}"
        commands = [command]

        forward_files = ["conf.lmp", "input.lammps"]
        backward_files = ["model_devi.log"]
        if nbeads is None:
            backward_files += ["model_devi.out"]
        else:
            num_digits = np.ceil(np.log10(nbeads + 1)).astype(int)
            backward_files += [
                f"model_devi{i + 1:0{num_digits}d}.out" for i in range(nbeads)
            ]
            backward_files += [f"log.lammps.{i:d}" for i in range(nbeads)]
        if model_devi_merge_traj:
            backward_files += ["all.lammpstrj"]
        else:
            forward_files += ["traj"]
            backward_files += ["traj"]

        if use_plm:
            forward_files += ["input.plumed"]
            # backward_files += ['output.plumed']
            backward_files += ["output.plumed", "COLVAR"]
            if use_plm_path:
                forward_files += ["plmpath.pdb"]
    elif model_devi_engine == "gromacs":
        gromacs_settings = jdata.get("gromacs_settings", {})
        mdp_filename = gromacs_settings.get("mdp_filename", "md.mdp")
        topol_filename = gromacs_settings.get("topol_filename", "processed.top")
        conf_filename = gromacs_settings.get("conf_filename", "conf.gro")
        index_filename = gromacs_settings.get("index_filename", "index.raw")
        type_filename = gromacs_settings.get("type_filename", "type.raw")
        ndx_filename = gromacs_settings.get("ndx_filename", "")
        # Initial reference to process pbc condition.
        # Default is em.tpr
        ref_filename = gromacs_settings.get("ref_filename", "em.tpr")
        deffnm = gromacs_settings.get("deffnm", "deepmd")
        maxwarn = gromacs_settings.get("maxwarn", 1)
        traj_filename = gromacs_settings.get("traj_filename", "deepmd_traj.gro")
        grp_name = gromacs_settings.get("group_name", "Other")
        trj_freq = cur_job.get("trj_freq", 10)

        command = "%s grompp -f %s -p %s -c %s -o %s -maxwarn %d" % (  # noqa: UP031
            model_devi_exec,
            mdp_filename,
            topol_filename,
            conf_filename,
            deffnm,
            maxwarn,
        )
        command += f"&& {model_devi_exec} mdrun -deffnm {deffnm} -cpi"
        if ndx_filename:
            command += f'&& echo -e "{grp_name}\\n{grp_name}\\n" | {model_devi_exec} trjconv -s {ref_filename} -f {deffnm}.trr -n {ndx_filename} -o {traj_filename} -pbc mol -ur compact -center'
        else:
            command += f'&& echo -e "{grp_name}\\n{grp_name}\\n" | {model_devi_exec} trjconv -s {ref_filename} -f {deffnm}.trr -o {traj_filename} -pbc mol -ur compact -center'
        command += "&& if [ ! -d traj ]; then \n mkdir traj; fi\n"
        command += f"python -c \"import dpdata;system = dpdata.System('{traj_filename}', fmt='gromacs/gro'); [system.to_gromacs_gro('traj/%d.gromacstrj' % (i * {trj_freq}), frame_idx=i) for i in range(system.get_nframes())]; system.to_deepmd_npy('traj_deepmd')\""
        _rel_model_names = " ".join([str(os.path.join("..", ii)) for ii in model_names])
        command += f"&& dp model-devi -m {_rel_model_names} -s traj_deepmd -o model_devi.out -f {trj_freq}"
        del _rel_model_names
        commands = [command]

        forward_files = [
            mdp_filename,
            topol_filename,
            conf_filename,
            index_filename,
            ref_filename,
            type_filename,
            "input.json",
            "job.json",
        ]
        if ndx_filename:
            forward_files.append(ndx_filename)
        backward_files = [
            f"{deffnm}.tpr",
            f"{deffnm}.log",
            traj_filename,
            "model_devi.out",
            "traj",
            "traj_deepmd",
        ]
    elif model_devi_engine == "amber":
        commands = [
            ("TASK=$(basename $(pwd)) && SYS1=${TASK:5:3} && SYS=$((10#$SYS1)) && ")
            + model_devi_exec
            + (
                " -O -p ../qmmm$SYS.parm7 -c init.rst7 -i ../init$SYS.mdin -o rc.mdout -r rc.rst7 -x rc.nc -inf rc.mdinfo -ref init.rst7"
            )
        ]
        forward_files = ["init.rst7", "TEMPLATE.disang"]
        backward_files = ["rc.mdout", "rc.nc", "rc.rst7", "TEMPLATE.dumpave"]
        model_names.extend(["qmmm*.parm7", "init*.mdin"])

    cwd = os.getcwd()

    user_forward_files = mdata.get("model_devi" + "_user_forward_files", [])
    forward_files += [os.path.basename(file) for file in user_forward_files]
    backward_files += mdata.get("model_devi" + "_user_backward_files", [])
    if len(run_tasks) == 0:
        raise RuntimeError(
            "run_tasks for model_devi should not be empty! Please check your files."
        )

    ### Submit jobs
    check_api_version(mdata)

    submission = make_submission(
        mdata["model_devi_machine"],
        mdata["model_devi_resources"],
        commands=commands,
        work_path=work_path,
        run_tasks=run_tasks,
        group_size=model_devi_group_size,
        forward_common_files=model_names,
        forward_files=forward_files,
        backward_files=backward_files,
        outlog="model_devi.log",
        errlog="model_devi.log",
    )
    submission.run_submission()




def run_model_devi(iter_index, jdata, mdata):
    model_devi_engine = jdata.get("model_devi_engine", "lammps")
    if model_devi_engine != "calypso":
        run_md_model_devi(iter_index, jdata, mdata)
    else:
        run_calypso_model_devi(iter_index, jdata, mdata)




def post_model_devi(iter_index, jdata, mdata):
    pass



