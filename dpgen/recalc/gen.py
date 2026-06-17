"""Recalculate FP energies for existing structures with new DFT settings.

Reads existing DeepMD npy/raw data (type.raw, coord.npy, box.npy, fparam.npy),
generates VASP task directories, runs calculations, and produces new npy data.

Usage: dpgen recalc PARAM MACHINE
"""

import glob
import json
import os
import shutil

import dpdata
import numpy as np

from dpgen import dlog
from dpgen.generator.constants import fp_name
from dpgen.generator.fp import _vasp_check_fin, make_fp_vasp, post_fp, run_fp
from dpgen.generator.lib.utils import create_path, make_iter_name
from dpgen.remote.decide_machine import convert_mdata
from dpgen.util import load_file, setup_ele_temp


def gen_recalc(args):
    jdata = load_file(args.PARAM)
    mdata = load_file(args.MACHINE)

    use_ele_temp = jdata.get("use_ele_temp", 0)
    if use_ele_temp == 1:
        setup_ele_temp(False)
    elif use_ele_temp == 2:
        setup_ele_temp(True)

    mdata = convert_mdata(mdata)

    # post_fp asserts iter_index < len(model_devi_jobs);
    # for recalc we don't use it, so supply a dummy entry.
    if not jdata.get("model_devi_jobs"):
        jdata["model_devi_jobs"] = [{}]

    iter_index = 0
    make_fp_recalc(iter_index, jdata)
    _hide_completed_tasks(iter_index)
    run_fp(iter_index, jdata, mdata)
    _restore_completed_tasks(iter_index)
    post_fp(iter_index, jdata)
    _write_record(iter_index)


def make_fp_recalc(iter_index, jdata):
    data_path = jdata["data_path"]
    systems = jdata["systems"]
    type_map = jdata["type_map"]
    use_ele_temp = jdata.get("use_ele_temp", 0)

    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    create_path(work_path)

    cwd = os.getcwd()

    for sys_idx, sys_name in enumerate(systems):
        sys_path = os.path.join(data_path, sys_name)
        if not os.path.isdir(sys_path):
            raise FileNotFoundError(
                f"System directory not found: {sys_path}"
            )

        dlog.info("Reading system %s from %s", sys_name, sys_path)
        ds = dpdata.LabeledSystem(sys_path, fmt="deepmd/npy", type_map=type_map)

        # dpdata doesn't read fparam.npy / aparam.npy; load them manually.
        fparam = None
        aparam = None
        if use_ele_temp > 0:
            npy_path = os.path.join(sys_path, "set.000", "fparam.npy")
            raw_path = os.path.join(sys_path, "fparam.raw")
            if os.path.isfile(npy_path):
                fparam = np.load(npy_path).reshape(len(ds))
                dlog.info("Loaded fparam.npy for %s", sys_name)
            elif os.path.isfile(raw_path):
                fparam = np.loadtxt(raw_path).reshape(len(ds))
                dlog.info("Loaded fparam.raw for %s", sys_name)
            else:
                dlog.warning(
                    "use_ele_temp=%d but no fparam file found for %s",
                    use_ele_temp,
                    sys_name,
                )

            if use_ele_temp == 2:
                npy_path = os.path.join(sys_path, "set.000", "aparam.npy")
                raw_path = os.path.join(sys_path, "aparam.raw")
                if os.path.isfile(npy_path):
                    aparam = np.load(npy_path)
                elif os.path.isfile(raw_path):
                    aparam = np.loadtxt(raw_path)

        for counter in range(len(ds)):
            task_path = os.path.join(
                work_path,
                f"task.{sys_idx:03d}.{counter:06d}",
            )
            create_path(task_path)
            os.chdir(task_path)

            frame = ds[counter]
            frame.to_vasp_poscar("POSCAR")

            if use_ele_temp > 0 and fparam is not None:
                ele_temp = float(fparam[counter])
                with open("job.json", "w") as fp:
                    json.dump({"ele_temp": ele_temp}, fp)

            os.chdir(cwd)

    make_fp_vasp(iter_index, jdata)


def _write_record(iter_index):
    with open("record.dpgen", "w") as fp:
        fp.write("0 6\n0 7\n0 8\n")
    dlog.info("Written record.dpgen (iter %d)", iter_index)


def _hide_completed_tasks(iter_index):
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    done = incomplete = 0
    for td in sorted(glob.glob(os.path.join(work_path, "task.*"))):
        if _vasp_check_fin(td):
            os.rename(td, td + ".done")
            done += 1
        else:
            incomplete += 1
    dlog.info(
        "Tasks: %d completed (hidden), %d incomplete (to submit)",
        done,
        incomplete,
    )


def _restore_completed_tasks(iter_index):
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, fp_name)
    restored = 0
    for td in sorted(glob.glob(os.path.join(work_path, "task.*.done"))):
        os.rename(td, td.removesuffix(".done"))
        restored += 1
    if restored:
        dlog.info("Restored %d completed tasks", restored)
