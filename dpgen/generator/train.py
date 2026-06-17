#!/usr/bin/env python3

"""Training step of the DP-GEN workflow (``00.train``).

Contains make/run/post for the DeepMD-kit training sub-step. Dispatches on
``mlp_engine`` (currently only ``dp`` is supported).
"""

import glob
import itertools
import json
import os
import random
import shlex
import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional

import dpdata
import numpy as np
from packaging.version import Version

from dpgen import dlog
from dpgen.dispatcher.Dispatcher import make_submission
from dpgen.generator.constants import (
    calypso_model_devi_name,
    default_train_input_file,
    fp_name,
    model_devi_name,
    train_name,
    train_task_fmt,
)
from dpgen.generator.lib.utils import (
    check_api_version,
    create_path,
    log_task,
    make_iter_name,
    symlink_user_forward_files,
)
from dpgen.generator.helpers import (
    _check_empty_iter,
    _check_skip_train,
    _get_model_suffix,
    copy_model,
    set_version,
)
from dpgen.util import (
    convert_training_data_to_hdf5,
    expand_sys_str,
)


def make_train(iter_index, jdata, mdata):
    mlp_engine = jdata.get("mlp_engine", "dp")
    if mlp_engine == "dp":
        return make_train_dp(iter_index, jdata, mdata)
    else:
        raise ValueError(f"Unsupported engine: {mlp_engine}")




def make_train_dp(iter_index, jdata, mdata):
    # load json param
    # train_param = jdata['train_param']
    train_input_file = default_train_input_file
    numb_models = jdata["numb_models"]
    init_data_prefix = os.path.abspath(jdata["init_data_prefix"])
    init_data_sys_ = jdata["init_data_sys"]
    fp_task_min = jdata["fp_task_min"]
    model_devi_jobs = jdata["model_devi_jobs"]
    use_ele_temp = jdata.get("use_ele_temp", 0)
    training_iter0_model = jdata.get("training_iter0_model_path", [])
    training_init_model = jdata.get("training_init_model", False)
    training_reuse_iter = jdata.get("training_reuse_iter")
    training_reuse_old_ratio = jdata.get("training_reuse_old_ratio", "auto")

    # if you want to use DP-ZBL potential , you have to give the path of your energy potential file
    if "srtab_file_path" in jdata.keys():
        srtab_file_path = os.path.abspath(jdata.get("srtab_file_path", None))

    if "training_reuse_stop_batch" in jdata.keys():
        training_reuse_stop_batch = jdata["training_reuse_stop_batch"]
    elif "training_reuse_numb_steps" in jdata.keys():
        training_reuse_stop_batch = jdata["training_reuse_numb_steps"]
    else:
        training_reuse_stop_batch = None

    training_reuse_start_lr = jdata.get("training_reuse_start_lr")
    training_reuse_start_pref_e = jdata.get("training_reuse_start_pref_e")
    training_reuse_start_pref_f = jdata.get("training_reuse_start_pref_f")
    model_devi_activation_func = jdata.get("model_devi_activation_func", None)
    training_init_frozen_model = (
        jdata.get("training_init_frozen_model") if iter_index == 0 else None
    )
    training_finetune_model = (
        jdata.get("training_finetune_model") if iter_index == 0 else None
    )

    auto_ratio = False
    if (
        training_reuse_iter is not None
        and isinstance(training_reuse_old_ratio, str)
        and training_reuse_old_ratio.startswith("auto")
    ):
        s = training_reuse_old_ratio.split(":")
        if len(s) == 1:
            new_to_old_ratio = 10.0
        elif len(s) == 2:
            new_to_old_ratio = float(s[1])
        else:
            raise ValueError(
                f"training_reuse_old_ratio is not correct, got {training_reuse_old_ratio}"
            )
        dlog.info(
            "Use automatic training_reuse_old_ratio to make new-to-old ratio close to %d times of the default value.",
            new_to_old_ratio,
        )
        auto_ratio = True
        number_old_frames = 0
        number_new_frames = 0

    suffix = _get_model_suffix(jdata)
    model_devi_engine = jdata.get("model_devi_engine", "lammps")
    if iter_index > 0 and _check_empty_iter(iter_index - 1, fp_task_min):
        log_task("prev data is empty, copy prev model")
        copy_model(numb_models, iter_index - 1, iter_index, suffix)
        return
    elif (
        model_devi_engine != "calypso"
        and iter_index > 0
        and _check_skip_train(model_devi_jobs[iter_index - 1])
    ):
        log_task("skip training at step %d " % (iter_index - 1))  # noqa: UP031
        copy_model(numb_models, iter_index - 1, iter_index, suffix)
        return
    else:
        iter_name = make_iter_name(iter_index)
        work_path = os.path.join(iter_name, train_name)
        copy_flag = os.path.join(work_path, "copied")
        if os.path.isfile(copy_flag):
            os.remove(copy_flag)

    # establish work path
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, train_name)
    create_path(work_path)
    # link init data
    cwd = os.getcwd()
    os.chdir(work_path)
    os.symlink(os.path.abspath(init_data_prefix), "data.init")
    # link iter data
    os.mkdir("data.iters")
    os.chdir("data.iters")
    for ii in range(iter_index):
        os.symlink(
            os.path.relpath(os.path.join(cwd, make_iter_name(ii))), make_iter_name(ii)
        )
    os.chdir(cwd)

    init_data_sys = []
    init_batch_size = []
    if "init_batch_size" in jdata:
        init_batch_size_ = list(jdata["init_batch_size"])
        if len(init_data_sys_) > len(init_batch_size_):
            warnings.warn(
                "The batch sizes are not enough. Assume auto for those not spefified."
            )
            init_batch_size.extend(
                ["auto" for aa in range(len(init_data_sys_) - len(init_batch_size))]
            )
    else:
        init_batch_size_ = ["auto" for aa in range(len(jdata["init_data_sys"]))]
    if "sys_batch_size" in jdata:
        sys_batch_size = jdata["sys_batch_size"]
    else:
        sys_batch_size = ["auto" for aa in range(len(jdata["sys_configs"]))]

    # make sure all init_data_sys has the batch size -- for the following `zip`
    assert len(init_data_sys_) <= len(init_batch_size_)
    for ii, ss in zip(init_data_sys_, init_batch_size_):
        sys_paths = expand_sys_str(os.path.join(init_data_prefix, ii))
        for single_sys in sys_paths:
            init_data_sys.append(
                Path(
                    os.path.normpath(
                        os.path.join(
                            "../data.init",
                            ii,
                            os.path.relpath(
                                single_sys, os.path.join(init_data_prefix, ii)
                            ),
                        )
                    )
                ).as_posix()
            )
            init_batch_size.append(detect_batch_size(ss, single_sys))
            if auto_ratio:
                number_old_frames += get_nframes(single_sys)
    old_range = None
    if iter_index > 0:
        for ii in range(iter_index):
            if ii == iter_index - 1:
                old_range = len(init_data_sys)
            fp_path = os.path.join(make_iter_name(ii), fp_name)
            fp_data_sys = glob.glob(os.path.join(fp_path, "data.*"))
            if model_devi_engine == "calypso":
                _modd_path = os.path.join(
                    make_iter_name(ii), model_devi_name, calypso_model_devi_name
                )
                sys_list = glob.glob(os.path.join(_modd_path, "*.structures"))
                sys_batch_size = ["auto" for aa in range(len(sys_list))]
            for jj in fp_data_sys:
                sys_idx = int(jj.split(".")[-1])
                sys_paths = expand_sys_str(jj)
                nframes = 0
                for sys_single in sys_paths:
                    nframes += dpdata.LabeledSystem(
                        sys_single, fmt="deepmd/npy"
                    ).get_nframes()
                if auto_ratio:
                    if ii == iter_index - 1:
                        number_new_frames += nframes
                    else:
                        number_old_frames += nframes
                if nframes < fp_task_min:
                    log_task(
                        "nframes (%d) in data sys %s is too small, skip" % (nframes, jj)  # noqa: UP031
                    )
                    continue
                for sys_single in sys_paths:
                    init_data_sys.append(
                        Path(
                            os.path.normpath(os.path.join("../data.iters", sys_single))
                        ).as_posix()
                    )
                    batch_size = (
                        sys_batch_size[sys_idx]
                        if sys_idx < len(sys_batch_size)
                        else "auto"
                    )
                    init_batch_size.append(detect_batch_size(batch_size, sys_single))
    # establish tasks
    jinput = jdata["default_training_param"]
    try:
        mdata["deepmd_version"]
    except KeyError:
        mdata = set_version(mdata)
    # setup data systems
    if Version(mdata["deepmd_version"]) >= Version("1") and Version(
        mdata["deepmd_version"]
    ) < Version("2"):
        # 1.x
        jinput["training"]["systems"] = init_data_sys
        jinput["training"]["batch_size"] = init_batch_size
        jinput["model"]["type_map"] = jdata["type_map"]
        # electron temperature
        if use_ele_temp == 0:
            pass
        elif use_ele_temp == 1:
            jinput["model"]["fitting_net"]["numb_fparam"] = 1
            jinput["model"]["fitting_net"].pop("numb_aparam", None)
        elif use_ele_temp == 2:
            jinput["model"]["fitting_net"]["numb_aparam"] = 1
            jinput["model"]["fitting_net"].pop("numb_fparam", None)
        else:
            raise RuntimeError("invalid setting for use_ele_temp " + str(use_ele_temp))
    elif Version(mdata["deepmd_version"]) >= Version("2") and Version(
        mdata["deepmd_version"]
    ) < Version("4"):
        # 2.x
        jinput["training"].setdefault("training_data", {})
        jinput["training"]["training_data"]["systems"] = init_data_sys
        old_batch_size = jinput["training"]["training_data"].get("batch_size", "")
        if not (
            isinstance(old_batch_size, str) and old_batch_size.startswith("mixed:")
        ):
            jinput["training"]["training_data"]["batch_size"] = init_batch_size
        jinput["model"]["type_map"] = jdata["type_map"]
        # electron temperature
        if use_ele_temp == 0:
            pass
        elif use_ele_temp == 1:
            jinput["model"]["fitting_net"]["numb_fparam"] = 1
            jinput["model"]["fitting_net"].pop("numb_aparam", None)
        elif use_ele_temp == 2:
            jinput["model"]["fitting_net"]["numb_aparam"] = 1
            jinput["model"]["fitting_net"].pop("numb_fparam", None)
        else:
            raise RuntimeError("invalid setting for use_ele_temp " + str(use_ele_temp))
    else:
        raise RuntimeError(
            "DP-GEN currently only supports for DeePMD-kit 1.x to 3.x version!"
        )
    # set training reuse model
    if auto_ratio:
        training_reuse_old_ratio = number_old_frames / (
            number_old_frames + number_new_frames * new_to_old_ratio
        )
    if training_reuse_iter is not None and iter_index >= training_reuse_iter:
        if "numb_steps" in jinput["training"] and training_reuse_stop_batch is not None:
            jinput["training"]["numb_steps"] = training_reuse_stop_batch
        elif (
            "stop_batch" in jinput["training"] and training_reuse_stop_batch is not None
        ):
            jinput["training"]["stop_batch"] = training_reuse_stop_batch
        if Version("1") <= Version(mdata["deepmd_version"]) < Version("2"):
            jinput["training"]["auto_prob_style"] = (
                "prob_sys_size; 0:%d:%f; %d:%d:%f"  # noqa: UP031
                % (
                    old_range,
                    training_reuse_old_ratio,
                    old_range,
                    len(init_data_sys),
                    1.0 - training_reuse_old_ratio,
                )
            )
        elif Version("2") <= Version(mdata["deepmd_version"]) < Version("4"):
            jinput["training"]["training_data"]["auto_prob"] = (
                "prob_sys_size; 0:%d:%f; %d:%d:%f"  # noqa: UP031
                % (
                    old_range,
                    training_reuse_old_ratio,
                    old_range,
                    len(init_data_sys),
                    1.0 - training_reuse_old_ratio,
                )
            )
        else:
            raise RuntimeError(
                "Unsupported DeePMD-kit version: {}".format(mdata["deepmd_version"])
            )
        if (
            jinput["loss"].get("start_pref_e") is not None
            and training_reuse_start_pref_e is not None
        ):
            jinput["loss"]["start_pref_e"] = training_reuse_start_pref_e
        if (
            jinput["loss"].get("start_pref_f") is not None
            and training_reuse_start_pref_f is not None
        ):
            jinput["loss"]["start_pref_f"] = training_reuse_start_pref_f
        if training_reuse_start_lr is not None:
            jinput["learning_rate"]["start_lr"] = training_reuse_start_lr

    input_files = []
    for ii in range(numb_models):
        task_path = os.path.join(work_path, train_task_fmt % ii)
        create_path(task_path)
        os.chdir(task_path)

        if "srtab_file_path" in jdata.keys():
            shutil.copyfile(srtab_file_path, os.path.basename(srtab_file_path))

        for jj in init_data_sys:
            # HDF5 path contains #
            if not (
                os.path.isdir(jj) if "#" not in jj else os.path.isfile(jj.split("#")[0])
            ):
                raise RuntimeError(
                    f"data sys {jj} does not exists, cwd is {os.getcwd()}"
                )
        os.chdir(cwd)
        # set random seed for each model
        if Version(mdata["deepmd_version"]) >= Version("1") and Version(
            mdata["deepmd_version"]
        ) < Version("4"):
            # 1.x
            if "descriptor" not in jinput["model"]:
                pass
            elif jinput["model"]["descriptor"]["type"] == "hybrid":
                for desc in jinput["model"]["descriptor"]["list"]:
                    desc["seed"] = random.randrange(sys.maxsize) % (2**32)
            elif jinput["model"]["descriptor"]["type"] == "loc_frame":
                pass
            else:
                jinput["model"]["descriptor"]["seed"] = random.randrange(
                    sys.maxsize
                ) % (2**32)
            if "fitting_net" in jinput["model"]:
                jinput["model"]["fitting_net"]["seed"] = random.randrange(
                    sys.maxsize
                ) % (2**32)
            if "type_embedding" in jinput["model"]:
                jinput["model"]["type_embedding"]["seed"] = random.randrange(
                    sys.maxsize
                ) % (2**32)
            jinput["training"]["seed"] = random.randrange(sys.maxsize) % (2**32)
        else:
            raise RuntimeError(
                "DP-GEN currently only supports for DeePMD-kit 1.x to 3.x version!"
            )
        # set model activation function
        if model_devi_activation_func is not None:
            if Version(mdata["deepmd_version"]) < Version("1"):
                raise RuntimeError(
                    "model_devi_activation_func does not suppport deepmd version",
                    mdata["deepmd_version"],
                )
            assert (
                isinstance(model_devi_activation_func, list)
                and len(model_devi_activation_func) == numb_models
            )
            if (
                len(np.array(model_devi_activation_func).shape) == 2
            ):  # 2-dim list for emd/fitting net-resolved assignment of actF
                jinput["model"]["descriptor"]["activation_function"] = (
                    model_devi_activation_func[ii][0]
                )
                jinput["model"]["fitting_net"]["activation_function"] = (
                    model_devi_activation_func[ii][1]
                )
            if (
                len(np.array(model_devi_activation_func).shape) == 1
            ):  # for backward compatibility, 1-dim list, not net-resolved
                jinput["model"]["descriptor"]["activation_function"] = (
                    model_devi_activation_func[ii]
                )
                jinput["model"]["fitting_net"]["activation_function"] = (
                    model_devi_activation_func[ii]
                )
        # dump the input.json
        with open(os.path.join(task_path, train_input_file), "w") as outfile:
            json.dump(jinput, outfile, indent=4)
        input_files.append(os.path.join(task_path, train_input_file))

    # link old models
    if iter_index > 0:
        prev_iter_name = make_iter_name(iter_index - 1)
        prev_work_path = os.path.join(prev_iter_name, train_name)
        for ii in range(numb_models):
            prev_task_path = os.path.join(prev_work_path, train_task_fmt % ii)
            old_model_files = glob.glob(os.path.join(prev_task_path, "model.ckpt*"))
            _link_old_models(work_path, old_model_files, ii)
    else:
        if isinstance(training_iter0_model, str):
            training_iter0_model = [training_iter0_model]
        iter0_models = []
        for ii in training_iter0_model:
            model_is = glob.glob(ii)
            model_is.sort()
            iter0_models += [os.path.abspath(ii) for ii in model_is]
        if training_init_model:
            assert numb_models == len(iter0_models), (
                "training_iter0_model should be provided, and the number of models should be equal to %d"  # noqa: UP031
                % numb_models
            )
        for ii in range(len(iter0_models)):
            old_model_path = os.path.join(iter0_models[ii], "model.ckpt*")
            old_model_files = glob.glob(old_model_path)
            if not len(old_model_files):
                raise FileNotFoundError(f"{old_model_path} not found!")
            _link_old_models(work_path, old_model_files, ii)
    copied_models = next(
        (
            item
            for item in (training_init_frozen_model, training_finetune_model)
            if item is not None
        ),
        None,
    )
    if copied_models is not None:
        for ii in range(len(copied_models)):
            _link_old_models(
                work_path, [copied_models[ii]], ii, basename=f"init{suffix}"
            )
    # Copy user defined forward files
    symlink_user_forward_files(mdata=mdata, task_type="train", work_path=work_path)
    # HDF5 format for training data
    if jdata.get("one_h5", False):
        convert_training_data_to_hdf5(input_files, os.path.join(work_path, "data.hdf5"))




def _link_old_models(work_path, old_model_files, ii, basename: Optional[str] = None):
    """Link the `ii`th old model given by `old_model_files` to
    the `ii`th training task in `work_path`.
    """
    task_path = os.path.join(work_path, train_task_fmt % ii)
    task_old_path = os.path.join(task_path, "old")
    create_path(task_old_path)
    cwd = os.getcwd()
    for jj in old_model_files:
        absjj = os.path.abspath(jj)
        if basename is None:
            basejj = os.path.basename(jj)
        else:
            basejj = basename
        os.chdir(task_old_path)
        os.symlink(os.path.relpath(absjj), basejj)
        os.chdir(cwd)




def detect_batch_size(batch_size, system=None):
    if isinstance(batch_size, int):
        return batch_size
    elif batch_size == "auto":
        # automaticcaly set batch size, batch_size = 32 // atom_numb (>=1, <=fram_numb)
        # check if h5 file
        format = "deepmd/npy" if "#" not in system else "deepmd/hdf5"
        s = dpdata.LabeledSystem(system, fmt=format)
        return int(
            min(np.ceil(32.0 / float(s["coords"].shape[1])), s["coords"].shape[0])
        )
    else:
        raise RuntimeError("Unsupported batch size")




def get_nframes(system):
    format = "deepmd/npy" if "#" not in system else "deepmd/hdf5"
    s = dpdata.LabeledSystem(system, fmt=format)
    return s.get_nframes()




def run_train(iter_index, jdata, mdata):
    mlp_engine = jdata.get("mlp_engine", "dp")
    if mlp_engine == "dp":
        return run_train_dp(iter_index, jdata, mdata)
    else:
        raise ValueError(f"Unsupported engine: {mlp_engine}")




def run_train_dp(iter_index, jdata, mdata):
    # print("debug:run_train:mdata", mdata)
    # load json param
    numb_models = jdata["numb_models"]
    suffix = _get_model_suffix(jdata)
    # train_param = jdata['train_param']
    train_input_file = default_train_input_file
    training_reuse_iter = jdata.get("training_reuse_iter")
    training_init_model = jdata.get("training_init_model", False)
    training_init_frozen_model = (
        jdata.get("training_init_frozen_model") if iter_index == 0 else None
    )
    training_finetune_model = (
        jdata.get("training_finetune_model") if iter_index == 0 else None
    )

    if "srtab_file_path" in jdata.keys():
        zbl_file = os.path.basename(jdata.get("srtab_file_path", None))

    if training_reuse_iter is not None and iter_index >= training_reuse_iter:
        training_init_model = True
    try:
        mdata["deepmd_version"]
    except KeyError:
        mdata = set_version(mdata)

    if (
        training_init_model
        + (training_init_frozen_model is not None)
        + (training_finetune_model is not None)
        > 1
    ):
        raise RuntimeError(
            "training_init_model, training_init_frozen_model, and training_finetune_model are mutually exclusive."
        )

    train_command = mdata.get("train_command", "dp").strip()
    # assert train_command == "dp", "The 'train_command' should be 'dp'"     # the tests should be updated to run this command
    if suffix == ".pth":
        train_command += " --pt"
    elif suffix == ".savedmodel":
        train_command += " --jax"

    # paths
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, train_name)
    # check if is copied
    copy_flag = os.path.join(work_path, "copied")
    if os.path.isfile(copy_flag):
        log_task("copied model, do not train")
        return
    # make tasks
    all_task = []
    for ii in range(numb_models):
        task_path = os.path.join(work_path, train_task_fmt % ii)
        all_task.append(task_path)
    commands = []
    if Version(mdata["deepmd_version"]) >= Version("1") and Version(
        mdata["deepmd_version"]
    ) < Version("4"):
        # 1.x
        ## Commands are like `dp train` and `dp freeze`
        ## train_command should not be None
        assert train_command
        extra_flags = ""
        init_flag = ""
        if jdata.get("dp_train_skip_neighbor_stat", False):
            extra_flags += " --skip-neighbor-stat"
        if training_init_model:
            init_flag = " --init-model old/model.ckpt"
        elif training_init_frozen_model is not None:
            init_flag = f" --init-frz-model old/init{suffix}"
        elif training_finetune_model is not None:
            init_flag = f" --finetune old/init{suffix}"
        command = f"{train_command} train {train_input_file}{extra_flags}"
        if suffix == ".pb":
            ckpt_suffix = ".index"
        elif suffix == ".pth":
            ckpt_suffix = ".pt"
        elif suffix == ".savedmodel":
            ckpt_suffix = ".jax"
        else:
            raise RuntimeError(f"Unknown suffix {suffix}")
        command = f"{{ if [ ! -f model.ckpt{ckpt_suffix} ]; then {command}{init_flag}; else {command} --restart model.ckpt; fi }}"
        command = f"/bin/sh -c {shlex.quote(command)}"
        commands.append(command)
        command = f"{train_command} freeze"
        commands.append(command)
        if jdata.get("dp_compress", False):
            commands.append(f"{train_command} compress")
    else:
        raise RuntimeError(
            "DP-GEN currently only supports for DeePMD-kit 1.x to 3.x version!"
        )

    # _tasks = [os.path.basename(ii) for ii in all_task]
    # run_tasks = []
    # for ii in all_task:
    #     check_pb = os.path.join(ii, "frozen_model.pb")
    #     check_lcurve = os.path.join(ii, "lcurve.out")
    #     if os.path.isfile(check_pb) and os.path.isfile(check_lcurve):
    #         pass
    #     else:
    #         run_tasks.append(ii)
    run_tasks = [os.path.basename(ii) for ii in all_task]

    forward_files = [train_input_file]
    if "srtab_file_path" in jdata.keys():
        forward_files.append(zbl_file)
    if training_init_model:
        if suffix == ".pb":
            forward_files += [
                os.path.join("old", "model.ckpt.meta"),
                os.path.join("old", "model.ckpt.index"),
                os.path.join("old", "model.ckpt.data-00000-of-00001"),
            ]
        elif suffix == ".pth":
            forward_files += [os.path.join("old", "model.ckpt.pt")]
        elif suffix == ".savedmodel":
            forward_files += [os.path.join("old", "model.ckpt.jax")]
        else:
            raise RuntimeError(f"Unknown suffix {suffix}")
    elif training_init_frozen_model is not None or training_finetune_model is not None:
        forward_files.append(os.path.join("old", f"init{suffix}"))

    backward_files = [
        f"frozen_model{suffix}",
        "lcurve.out",
        "train.log",
        "checkpoint",
    ]
    if jdata.get("dp_compress", False):
        backward_files.append(f"frozen_model_compressed{suffix}")

    if suffix == ".pb":
        backward_files += [
            "model.ckpt.meta",
            "model.ckpt.index",
            "model.ckpt.data-00000-of-00001",
        ]
    elif suffix == ".pth":
        backward_files += ["model.ckpt.pt"]
    elif suffix == ".savedmodel":
        backward_files += ["model.ckpt.jax"]
    else:
        raise RuntimeError(f"Unknown suffix {suffix}")

    if not jdata.get("one_h5", False):
        init_data_sys_ = jdata["init_data_sys"]
        init_data_sys = []
        for ii in init_data_sys_:
            init_data_sys.append(os.path.join("data.init", ii))
        trans_comm_data = []
        cwd = os.getcwd()
        os.chdir(work_path)
        fp_data = glob.glob(os.path.join("data.iters", "iter.*", "02.fp", "data.*"))
        for ii in itertools.chain(init_data_sys, fp_data):
            sys_paths = expand_sys_str(ii)
            for single_sys in sys_paths:
                if "#" not in single_sys:
                    trans_comm_data += glob.glob(os.path.join(single_sys, "set.*"))
                    trans_comm_data += glob.glob(os.path.join(single_sys, "type*.raw"))
                    trans_comm_data += glob.glob(os.path.join(single_sys, "nopbc"))
                else:
                    # H5 file
                    trans_comm_data.append(single_sys.split("#")[0])
    else:
        cwd = os.getcwd()
        trans_comm_data = ["data.hdf5"]
    # remove duplicated files
    trans_comm_data = list(set(trans_comm_data))
    os.chdir(cwd)

    try:
        train_group_size = mdata["train_group_size"]
    except Exception:
        train_group_size = 1

    user_forward_files = mdata.get("train" + "_user_forward_files", [])
    forward_files += [os.path.basename(file) for file in user_forward_files]
    backward_files += mdata.get("train" + "_user_backward_files", [])

    ### Submit jobs
    check_api_version(mdata)

    submission = make_submission(
        mdata["train_machine"],
        mdata["train_resources"],
        commands=commands,
        work_path=work_path,
        run_tasks=run_tasks,
        group_size=train_group_size,
        forward_common_files=trans_comm_data,
        forward_files=forward_files,
        backward_files=backward_files,
        outlog="train.log",
        errlog="train.log",
    )
    submission.run_submission()




def post_train(iter_index, jdata, mdata):
    mlp_engine = jdata.get("mlp_engine", "dp")
    if mlp_engine == "dp":
        return post_train_dp(iter_index, jdata, mdata)
    else:
        raise ValueError(f"Unsupported engine: {mlp_engine}")




def post_train_dp(iter_index, jdata, mdata):
    # load json param
    numb_models = jdata["numb_models"]
    # paths
    iter_name = make_iter_name(iter_index)
    work_path = os.path.join(iter_name, train_name)
    # check if is copied
    copy_flag = os.path.join(work_path, "copied")
    if os.path.isfile(copy_flag):
        log_task("copied model, do not post train")
        return
    # symlink models
    suffix = _get_model_suffix(jdata)
    for ii in range(numb_models):
        model_name = f"frozen_model{suffix}"
        if jdata.get("dp_compress", False):
            model_name = f"frozen_model_compressed{suffix}"

        ofile = os.path.join(work_path, "graph.%03d%s" % (ii, suffix))  # noqa: UP031
        task_file = os.path.join(train_task_fmt % ii, model_name)
        if os.path.isfile(ofile):
            os.remove(ofile)
        os.symlink(task_file, ofile)



