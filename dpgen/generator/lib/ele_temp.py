import os

import dpdata
import numpy as np
import scipy.constants as pc


def estimate_nbands(
    poscar_path,
    potcar_path,
    ele_temp_k=None,
    scale=1.0,
    nband_min=3,
    nband_min_low=1,
    t_low=300.0,
    t_high=2000.0,
):
    """启发式估算 NBANDS，基于总价电子数、原子数与电子温度。

    当 ``ele_temp_k`` 为 None 时退化为旧公式
    ``int(N_elec / 2 * scale) + nband_min * N_atom``；
    当给定温度时，per-atom 额外能带数在
    ``nband_min_low`` (T <= t_low) 与 ``nband_min`` (T >= t_high)
    之间线性插值，物理上对应于 Fermi-Dirac 展宽的额外空带数。

    Parameters
    ----------
    poscar_path : str
        path to POSCAR
    potcar_path : str
        path to POTCAR
    ele_temp_k : float or None
        电子温度（Kelvin）。None 时退化为旧公式（与温度无关）。
    scale : float
        电子数缩放因子（默认 1.0）
    nband_min : int
        T >= t_high 时每个原子的额外能带数（默认 3）
    nband_min_low : int
        T <= t_low 时每个原子的额外能带数（默认 1）
    t_low : float
        线性插值温度下界（K，默认 300）
    t_high : float
        线性插值温度上界（K，默认 2000）

    Returns
    -------
    int
        估算的 NBANDS
    """
    zvals = NBandsEsti._get_potcar_nvalence(potcar_path)
    sys = dpdata.System(poscar_path, fmt="vasp/poscar")
    atom_numbs = sys.get_atom_numbs()
    total_elec = sum(z * n for z, n in zip(zvals, atom_numbs))
    total_atoms = sum(atom_numbs)
    base = int(total_elec / 2 * scale)
    if ele_temp_k is None:
        extra_per_atom = nband_min
    elif ele_temp_k <= t_low:
        extra_per_atom = nband_min_low
    elif ele_temp_k >= t_high:
        extra_per_atom = nband_min
    else:
        frac = (ele_temp_k - t_low) / (t_high - t_low)
        extra_per_atom = nband_min_low + (nband_min - nband_min_low) * frac
    return base + int(extra_per_atom * total_atoms)


class NBandsEsti:
    def __init__(self, test_list):
        if isinstance(test_list, list):
            ele_t = []
            vol = []
            d_nbd = []
            nbd = []
            for ii in test_list:
                res = NBandsEsti._get_res(ii)
                ele_t.append(res["ele_temp"])
                vol.append(res["vol"])
                d_nbd.append(NBandsEsti._get_default_nbands(res))
                nbd.append(res["nbands"])
            ele_t = np.array(ele_t)
            vol = np.array(vol)
            d_nbd = np.array(d_nbd)
            nbd = np.array(nbd)
            alpha = (nbd - d_nbd) / vol / ele_t**1.5
            self.err = np.std(alpha)
            self.pref = np.average(alpha)
            # print(np.average(alpha), np.std(alpha), self.err/self.pref)
            # print((ele_t), vol, d_nbd, nbd, alpha)
        elif isinstance(test_list, str):
            with open(test_list) as fp:
                self.pref = float(fp.readline())
                self.err = float(fp.readline())
        else:
            raise RuntimeError("unknown input type " + type(test_list))

    def save(self, fname):
        with open(fname, "w") as fp:
            fp.write(str(self.pref) + "\n")
            fp.write(str(self.err) + "\n")

    def predict(self, target_dir, tolerance=0.5):
        res = NBandsEsti._get_res(target_dir)
        ele_t = res["ele_temp"]
        vol = res["vol"]
        d_nbd = NBandsEsti._get_default_nbands(res)
        nbd = res["nbands"]
        esti = (self.pref + tolerance * self.err) * ele_t**1.5 * vol + d_nbd
        return int(esti) + 1

    @classmethod
    def _get_res(self, res_dir):
        res = {}
        sys = dpdata.System(os.path.join(res_dir, "POSCAR"))
        res["natoms"] = sys["atom_numbs"]
        res["vol"] = np.linalg.det(sys["cells"][0])
        res["nvalence"] = self._get_potcar_nvalence(os.path.join(res_dir, "POTCAR"))
        res["ele_temp"] = (
            self._get_incar_ele_temp(os.path.join(res_dir, "INCAR"))
            * pc.electron_volt
            / pc.Boltzmann
        )
        res["nbands"] = self._get_incar_nbands(os.path.join(res_dir, "INCAR"))
        return res

    @classmethod
    def _get_default_nbands(self, res):
        ret = 0
        for ii, jj in zip(res["natoms"], res["nvalence"]):
            ret += ii * jj // 2 + ii // 2 + 2
        return ret

    @classmethod
    def _get_potcar_nvalence(self, fname):
        with open(fname) as fp:
            pot_str = fp.read().split("\n")
        head_idx = []
        for idx, ii in enumerate(pot_str):
            if ("PAW_" in ii) and ("TITEL" not in ii):
                head_idx.append(idx)
        res = []
        for ii in head_idx:
            res.append(float(pot_str[ii + 1]))
        return res

    @classmethod
    def _get_incar_ele_temp(self, fname):
        from pymatgen.io.vasp.inputs import Incar

        incar = Incar.from_file(fname)
        return incar["SIGMA"]

    @classmethod
    def _get_incar_nbands(self, fname):
        from pymatgen.io.vasp.inputs import Incar

        incar = Incar.from_file(fname)
        return incar.get("NBANDS")
