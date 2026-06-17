#!/usr/bin/env python3

"""Module-level constants for the DP-GEN generator.

Directory names, format strings, and path constants used across
train, model_devi, and fp sub-modules.
"""

import os

from dpgen import ROOT_PATH

template_name = "template"
train_name = "00.train"
train_task_fmt = "%03d"
train_tmpl_path = os.path.join(template_name, train_name)
default_train_input_file = "input.json"
data_system_fmt = "%03d"
model_devi_name = "01.model_devi"
model_devi_task_fmt = data_system_fmt + ".%06d"
model_devi_conf_fmt = data_system_fmt + ".%04d"
fp_name = "02.fp"
fp_task_fmt = data_system_fmt + ".%06d"
cvasp_file = os.path.join(ROOT_PATH, "generator/lib/cvasp.py")
# for calypso
calypso_run_opt_name = "gen_stru_analy"
calypso_model_devi_name = "model_devi_results"
calypso_run_model_devi_file = os.path.join(
    ROOT_PATH, "generator/lib/calypso_run_model_devi.py"
)
check_outcar_file = os.path.join(ROOT_PATH, "generator/lib/calypso_check_outcar.py")
run_opt_file = os.path.join(ROOT_PATH, "generator/lib/calypso_run_opt.py")
