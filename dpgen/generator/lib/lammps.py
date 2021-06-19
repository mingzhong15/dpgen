#!/usr/bin/env python3

import random, os, sys, dpdata
import numpy as np
import subprocess as sp
import scipy.constants as pc
from distutils.version import LooseVersion

def _sample_sphere() :
    while True:
        vv = np.array([np.random.normal(), np.random.normal(), np.random.normal()])
        vn = np.linalg.norm(vv)
        if vn < 0.2 :
            continue
        return vv / vn

def make_lammps_input(ensemble, 
                      conf_file,
                      graphs,
                      nsteps,
                      dt,
                      neidelay,
                      trj_freq, 
                      mass_map,
                      temp, 
                      jdata,
                      cur_job,
                      tau_t = 0.1,
                      pres = None,
                      tau_p = 0.5,
                      pka_e = None,
                      ele_temp_f = None,
                      ele_temp_a = None,
                      max_seed = 1000000,
                      us = None,
                      nopbc = False,
                      deepmd_version = '0.1') :
    if (ele_temp_f is not None or ele_temp_a is not None) and LooseVersion(deepmd_version) < LooseVersion('1'):
        raise RuntimeError('the electron temperature is only supported by deepmd-kit >= 1.0.0, please upgrade your deepmd-kit')
    if ele_temp_f is not None and ele_temp_a is not None:
        raise RuntimeError('the frame style ele_temp and atom style ele_temp should not be set at the same time')
       
    if us is not None and ensemble == 'msst':
        q_value = 36
        tscale = 0.01
        shock_direction = 'z'
        rlx_ens = cur_job['relax-ensemble']
        if 'q_value' in cur_job:
            q_value = cur_job['q_value']
        if 'tscale' in cur_job:
            tscale = cur_job['tscale']
        if 'direction' in cur_job:
            shock_direction = cur_job['direction']
            
    if (nsteps == list) and (len(nsteps) == 2):
        ret = "variable        NSTEPS_RLX          equal %d\n" % nsteps[0]
        ret += "variable        NSTEPS          equal %d\n" % nsteps[1]
    elif nsteps == int:
        ret = "variable        NSTEPS_RLX          equal %d\n" % nsteps
        ret += "variable        NSTEPS          equal %d\n" % nsteps
    ret+= "variable        THERMO_FREQ     equal %d\n" % trj_freq
    ret+= "variable        DUMP_FREQ       equal %d\n" % trj_freq
    ret+= "variable        TEMP            equal %f\n" % temp
    if ele_temp_f is not None:
        ret+= "variable        ELE_TEMP        equal %f\n" % ele_temp_f
    if ele_temp_a is not None:
        ret+= "variable        ELE_TEMP        equal %f\n" % ele_temp_a
    ret+= "variable        PRES            equal %f\n" % pres
    ret+= "variable        TAU_T           equal %f\n" % tau_t
    ret+= "variable        TAU_P           equal %f\n\n" % tau_p
    # ===== for fix-msst ===== #
    if ensemble == 'msst':
        ret+= "variable        us              equal %f\n" % us
        ret+= "variable        q_value         equal %f\n" % q_value
        ret+= "variable        tscale         equal %f\n\n" % tscale
        ret+= "variable        myTEMP         equal temp\n"
        ret+= "variable        myPRESS        equal press\n"
        ret+= "variable        ETOTAL         equal etotal\n"
        ret+= "variable        VOL            equal vol\n"
        ret+= "variable        RHO            equal density\n"
        ret+= "variable        STEP           equal step\n"
    # ========== #
    ret+= "\n"
    ret+= "units           metal\n"
    if nopbc:
        ret+= "boundary        f f f\n"
    else:
        ret+= "boundary        p p p\n"
    ret+= "atom_style      atomic\n"
    ret+= "\n"
    ret+= "neighbor        1.0 bin\n"
    ret+= "neigh_modify    every 10 delay 0 check no page 500000 one 50000"
    if neidelay is not None :
        ret+= "neigh_modify    delay %d\n" % neidelay
    ret+= "\n"
    ret+= "box          tilt large\n"
    ret+= "if \"${restart} > 0\" then \"read_restart dpgen.restart.*\" else \"read_data %s\"\n" % conf_file
    ret+= "change_box   all triclinic\n"
    for jj in range(len(mass_map)) :
        ret+= "mass            %d %f\n" %(jj+1, mass_map[jj])
    graph_list = ""
    for ii in graphs :
        graph_list += ii + " "
    if LooseVersion(deepmd_version) < LooseVersion('1'):
        # 0.x
        ret+= "pair_style      deepmd %s ${THERMO_FREQ} model_devi.out\n" % graph_list
    else:
        # 1.x
        keywords = ""
        if jdata.get('use_clusters', False):
            keywords += "atomic "
        if jdata.get('use_relative', False):
            eps = jdata.get('eps', 0.)
            keywords += "relative %s " % jdata['epsilon']
        if ele_temp_f is not None:
            keywords += "fparam ${ELE_TEMP}"
        if ele_temp_a is not None:
            keywords += "aparam ${ELE_TEMP}"
        ret+= "pair_style      deepmd %s out_freq ${THERMO_FREQ} out_file model_devi.out %s\n" % (graph_list, keywords)
    ret+= "pair_coeff      \n"
    ret+= "\n"
    ret+= "thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz\n"
    ret+= "thermo          ${THERMO_FREQ}\n"
    ret+= "dump            1 all custom ${DUMP_FREQ} traj/*.lammpstrj id type x y z\n"
    ret+= "restart         10000 dpgen.restart\n"
    ret+= "\n"
    if pka_e is None :
        ret+= "if \"${restart} == 0\" then \"velocity        all create ${TEMP} %d\"" % (random.randrange(max_seed-1)+1)
    else :
        sys = dpdata.System(conf_file, fmt = 'lammps/lmp')
        sys_data = sys.data
        pka_mass = mass_map[sys_data['atom_types'][0] - 1]
        pka_vn = pka_e * pc.electron_volt / \
                 (0.5 * pka_mass * 1e-3 / pc.Avogadro * (pc.angstrom / pc.pico) ** 2)
        pka_vn = np.sqrt(pka_vn)
        print(pka_vn)
        pka_vec = _sample_sphere()
        pka_vec *= pka_vn
        ret+= 'group           first id 1\n'
        ret+= 'if \"${restart} == 0\" then \"velocity        first set %f %f %f\"\n' % (pka_vec[0], pka_vec[1], pka_vec[2])
        ret+= 'fix	       2 all momentum 1 linear 1 1 1\n'
    ret+= "\n"
    if ensemble.split('-')[0] == 'npt' :
        assert (pres is not None)
        if nopbc:
            raise RuntimeError('ensemble %s is conflicting with nopbc' % ensemble)
    if ensemble == "npt" or ensemble == "npt-i" or ensemble == "npt-iso" :
        ret+= "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}\n"
    elif ensemble == 'npt-a' or ensemble == 'npt-aniso' : 
        ret+= "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P}\n"
    elif ensemble == 'npt-t' or ensemble == 'npt-tri' : 
        ret+= "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} tri ${PRES} ${PRES} ${TAU_P}\n"
    elif ensemble == "nvt" :
        ret+= "fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}\n"
    elif ensemble == 'nve' :
        ret+= "fix             1 all nve\n"
    elif ensemble == 'msst' :
        if rlx_ens == 'nvt' :
            ret+= "fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}\n"
        elif rlx_ens == 'npt' :
            ret+= "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} tri ${PRES} ${PRES} ${TAU_P}\n"
            
        ret+= "run             ${NSTEPS_RLX}\n" 
        ret+= "unfix           1\n\n"

        ret+= "fix             1 all msst %s ${us} q ${q_value} tscale ${tscale}\n" % shock_direction
        ret+= "fix             PRINT all print ${DUMP_FREQ}  \"${STEP} ${myTEMP} ${myPRESS} ${VOL} ${RHO} ${ETOTAL}\" file thermo.dat title screen no\n"
        
    else :
        raise RuntimeError("unknown emsemble " + ensemble)
    if nopbc:
        ret+= "velocity        all zero linear\n"
        ret+= "fix             fm all momentum 1 linear 1 1 1\n"
    ret+= "\n"
    ret+= "timestep        %f\n" % dt
    ret+= "run             ${NSTEPS} upto\n"
    return ret
        
# ret = make_lammps_input ("npt", "al.lmp", ['graph.000.pb', 'graph.001.pb'], 20000, 20, [27], 1000, pres = 1.0)
# print (ret)
# cvt_lammps_conf('POSCAR', 'tmp.lmp')


    
 
