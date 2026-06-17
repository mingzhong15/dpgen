#!/usr/bin/env python3

"""DP-GEN main workflow orchestrator.

This module is the thin orchestration layer for ``dpgen run``. The heavy
per-step logic lives in dedicated sub-modules:

- :mod:`dpgen.generator.train`      -- ``00.train``      (make/run/post)
- :mod:`dpgen.generator.model_devi` -- ``01.model_devi`` (make/run/post)
- :mod:`dpgen.generator.fp`         -- ``02.fp``         (make/run/post)
- :mod:`dpgen.generator.helpers`    -- shared pure helpers
- :mod:`dpgen.generator.constants`  -- directory/format constants
- :mod:`dpgen.generator.checkpoint` -- ``record.dpgen`` resumption

For backward compatibility every public name from the sub-modules is
re-exported here, so ``from dpgen.generator.run import *`` and all historic
explicit imports (``gen_run``, ``make_train``, ``fp_name``,
``_read_model_devi_file``, ``parse_cur_job_sys_revmat``, ``update_mass_map``,
``data_system_fmt``, ``create_path``, ``make_fp_task_name``, ...) keep working
unchanged.

The default workflow is the canonical 9-step active-learning loop::

    iter.NNNNNN/
      00.train/      jj=0,1,2  make_train / run_train / post_train
      01.model_devi/ jj=3,4,5  make_model_devi / run_model_devi / post_model_devi
      02.fp/         jj=6,7,8  make_fp / run_fp / post_fp
"""

import argparse
import logging
import logging.handlers
import queue

from dpgen import SHORT_CMD, dlog
from dpgen.generator.arginfo import run_jdata_arginfo
from dpgen.generator.checkpoint import Checkpoint
from dpgen.generator.lib.utils import (
    create_path,  # noqa: F401  (re-exported for compat)
    log_iter,
    log_task,  # noqa: F401  (re-exported for compat)
    make_iter_name,
    record_iter,  # noqa: F401  (re-exported for compat)
)
from dpgen.remote.decide_machine import convert_mdata
from dpgen.util import (
    load_file,
    normalize,
    sepline,
    setup_ele_temp,
)

# Re-export the canonical 9-step functions from their new homes.
from dpgen.generator.train import (  # noqa: F401
    detect_batch_size,
    get_nframes,
    make_train,
    make_train_dp,
    post_train,
    post_train_dp,
    run_train,
    run_train_dp,
)
from dpgen.generator.model_devi import (  # noqa: F401
    _make_model_devi_amber,
    _make_model_devi_native,
    _make_model_devi_native_gromacs,
    _make_model_devi_revmat,
    expand_matrix_values,
    find_only_one_key,
    make_model_devi,
    parse_cur_job,
    parse_cur_job_revmat,
    parse_cur_job_sys_revmat,
    post_model_devi,
    revise_by_keys,
    revise_lmp_input_dump,
    revise_lmp_input_model,
    revise_lmp_input_neigh_modify,
    revise_lmp_input_pair_coeff,
    revise_lmp_input_plm,
    run_md_model_devi,
    run_model_devi,
)
from dpgen.generator.fp import (  # noqa: F401
    _abacus_scf_check_fin,
    _cp2k_check_fin,
    _gaussian_check_fin,
    _link_fp_abacus_pporb_descript,
    _link_fp_vasp_pp,
    _make_fp_pwmat_input,
    _make_fp_vasp_configs,
    _make_fp_vasp_inner,
    _pwmat_check_fin,
    _qe_check_fin,
    _read_model_devi_file,
    _select_by_model_devi_adaptive_trust_low,
    _select_by_model_devi_standard,
    _siesta_check_fin,
    _to_face_dist,
    _vasp_check_fin,
    check_bad_box,
    check_cluster,
    make_fp,
    make_fp_abacus_scf,
    make_fp_amber_diff,
    make_fp_calculation,
    make_fp_cp2k,
    make_fp_cpx,
    make_fp_custom,
    make_kspacing_kpoints,
    make_fp_gaussian,
    make_fp_pwmat,
    make_fp_pwscf,
    make_fp_siesta,
    make_fp_vasp,
    make_fp_vasp_cp_cvasp,
    make_fp_vasp_incar,
    make_fp_vasp_kp,
    make_pwmat_input,
    make_vasp_incar,
    make_vasp_incar_ele_temp,
    post_fp,
    post_fp_abacus_scf,
    post_fp_amber_diff,
    post_fp_check_fail,
    post_fp_cp2k,
    post_fp_cpx,
    post_fp_custom,
    post_fp_gaussian,
    post_fp_pwmat,
    post_fp_pwscf,
    post_fp_siesta,
    post_fp_vasp,
    run_fp,
    run_fp_inner,
    sys_link_fp_vasp_pp,
    take_cluster,
)
from dpgen.generator.helpers import (  # noqa: F401
    _check_empty_iter,
    _check_skip_train,
    _get_model_suffix,
    _get_param_alias,
    copy_model,
    dump_to_deepmd_raw,
    expand_idx,
    get_atomic_masses,
    get_job_names,
    get_sys_index,
    make_fp_task_name,
    make_model_devi_conf_name,
    make_model_devi_task_name,
    poscar_natoms,
    poscar_to_conf,
    set_version,
    update_mass_map,
)
from dpgen.generator.constants import (  # noqa: F401
    calypso_model_devi_name,
    calypso_run_model_devi_file,
    calypso_run_opt_name,
    check_outcar_file,
    cvasp_file,
    data_system_fmt,
    default_train_input_file,
    fp_name,
    fp_task_fmt,
    model_devi_conf_fmt,
    model_devi_name,
    model_devi_task_fmt,
    run_opt_file,
    template_name,
    train_name,
    train_task_fmt,
    train_tmpl_path,
)

# Default workflow: the canonical 9-step active-learning loop.
# Each entry is (name, callable). ``make_model_devi`` returns a continuation
# flag that can stop the loop (e.g. when iterations are exhausted). ``post_fp``
# has a different signature (no mdata) and is dispatched specially.
DEFAULT_WORKFLOW = [
    ("make_train", make_train),
    ("run_train", run_train),
    ("post_train", post_train),
    ("make_model_devi", make_model_devi),
    ("run_model_devi", run_model_devi),
    ("post_model_devi", post_model_devi),
    ("make_fp", make_fp),
    ("run_fp", run_fp),
    ("post_fp", post_fp),
]


def run_iter(param_file, machine_file):
    """Drive the default 9-step workflow with checkpoint-based resumption.

    Reads ``param.json`` / ``machine.json``, normalizes parameters, and loops
    over iterations. Each iteration runs the 9 tasks in order; completed tasks
    are recorded in ``record.dpgen`` so the run can resume after interruption.

    Parameters
    ----------
    param_file : str
        Path to the parameter file (json/yaml).
    machine_file : str
        Path to the machine file (json/yaml).
    """
    jdata = load_file(param_file)
    mdata = load_file(machine_file)

    jdata_arginfo = run_jdata_arginfo()
    jdata = normalize(jdata_arginfo, jdata, strict_check=False)

    update_mass_map(jdata)

    # set up electron temperature
    use_ele_temp = jdata.get("use_ele_temp", 0)
    if use_ele_temp == 1:
        setup_ele_temp(False)
    elif use_ele_temp == 2:
        setup_ele_temp(True)

    if jdata.get("pretty_print", False):
        from monty.serialization import dumpfn

        fparam = (
            SHORT_CMD
            + "_"
            + param_file.split(".")[0]
            + "."
            + jdata.get("pretty_format", "json")
        )
        dumpfn(jdata, fparam, indent=4)
        fmachine = (
            SHORT_CMD
            + "_"
            + machine_file.split(".")[0]
            + "."
            + jdata.get("pretty_format", "json")
        )
        dumpfn(mdata, fmachine, indent=4)

    if mdata.get("handlers", None):
        if mdata["handlers"].get("smtp", None):
            que = queue.Queue(-1)
            queue_handler = logging.handlers.QueueHandler(que)
            smtp_handler = logging.handlers.SMTPHandler(**mdata["handlers"]["smtp"])
            listener = logging.handlers.QueueListener(que, smtp_handler)
            dlog.addHandler(queue_handler)
            listener.start()
    # Convert mdata
    mdata = convert_mdata(mdata)

    # Run the default workflow with checkpoint resumption.
    _run_workflow(DEFAULT_WORKFLOW, jdata, mdata)


def _run_workflow(workflow, jdata, mdata):
    """Execute a workflow (list of (name, callable)) with checkpoint support.

    The behavior is preserved byte-for-byte from the original ``run_iter``:
    the loop uses the legacy ``ii * max_tasks + jj`` comparison against the
    last recorded ``(iter, task)`` to decide which steps to skip on resume.

    Parameters
    ----------
    workflow : list[tuple[str, callable]]
        Ordered workflow steps. ``make_model_devi`` returns a continuation
        flag; ``post_fp`` takes no ``mdata`` argument and is dispatched
        specially.
    jdata : dict
        Normalized run parameters.
    mdata : dict
        Converted machine parameters.
    """
    numb_task = len(workflow)
    checkpoint = Checkpoint()
    resume_point = checkpoint.get_resume_point()

    cont = True
    ii = -1
    while cont:
        ii += 1
        if ii < resume_point[0]:
            continue
        iter_name = make_iter_name(ii)
        sepline(iter_name, "=")
        for jj in range(numb_task):
            if checkpoint.is_done(ii, jj, resume_point):
                continue
            step_name = workflow[jj][0]
            step_fn = workflow[jj][1]
            task_name = "task %02d" % jj  # noqa: UP031
            sepline(f"{iter_name} {task_name}", "-")
            log_iter(step_name, ii, jj)
            if step_name == "post_fp":
                # post_fp does not take mdata.
                step_fn(ii, jdata)
            elif step_name == "make_model_devi":
                # make_model_devi returns a continuation flag.
                cont = step_fn(ii, jdata, mdata)
                if not cont:
                    break
            else:
                step_fn(ii, jdata, mdata)
            checkpoint.mark(ii, jj)


def gen_run(args):
    """Entry point for the ``dpgen run`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI args with ``PARAM``, ``MACHINE`` and optional ``debug``.
    """
    if args.PARAM and args.MACHINE:
        if args.debug:
            dlog.setLevel(logging.DEBUG)
        dlog.info("start running")
        run_iter(args.PARAM, args.MACHINE)
        dlog.info("finished")


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("PARAM", type=str, help="The parameters of the generator")
    parser.add_argument(
        "MACHINE", type=str, help="The settings of the machine running the generator"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("paramiko").setLevel(logging.WARNING)

    logging.info("start running")
    run_iter(args.PARAM, args.MACHINE)
    logging.info("finished!")


if __name__ == "__main__":
    _main()
