import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
__package__ = "generator"
from .context import (
    NBandsEsti,
    estimate_nbands,
    setUpModule,  # noqa: F401
)


class TestNBandsEsti(unittest.TestCase):
    def test_predict(self):
        self.nbe = NBandsEsti(
            [
                "out_data_nbands_esti/md.010000K",
                "out_data_nbands_esti/md.020000K",
                "out_data_nbands_esti/md.040000K",
                "out_data_nbands_esti/md.080000K",
                "out_data_nbands_esti/md.160000K",
            ]
        )
        self.assertEqual(self.nbe.predict("out_data_nbands_esti/md.010000K"), 72)
        self.assertEqual(self.nbe.predict("out_data_nbands_esti/md.020000K"), 83)
        self.assertEqual(self.nbe.predict("out_data_nbands_esti/md.040000K"), 112)
        self.assertEqual(self.nbe.predict("out_data_nbands_esti/md.080000K"), 195)
        self.assertEqual(self.nbe.predict("out_data_nbands_esti/md.160000K"), 429)
        self.assertEqual(self.nbe.predict("out_data_nbands_esti/md.240000K"), 732)

    def test_save_load(self):
        self.nbe2 = NBandsEsti(
            [
                "out_data_nbands_esti/md.010000K",
                "out_data_nbands_esti/md.020000K",
                "out_data_nbands_esti/md.040000K",
                "out_data_nbands_esti/md.080000K",
                "out_data_nbands_esti/md.160000K",
            ]
        )
        self.nbe2.save("tmp.log")
        self.nbe = NBandsEsti("tmp.log")
        self.assertEqual(self.nbe.predict("out_data_nbands_esti/md.010000K"), 72)
        self.assertEqual(self.nbe.predict("out_data_nbands_esti/md.020000K"), 83)
        self.assertEqual(self.nbe.predict("out_data_nbands_esti/md.040000K"), 112)
        self.assertEqual(self.nbe.predict("out_data_nbands_esti/md.080000K"), 195)
        self.assertEqual(self.nbe.predict("out_data_nbands_esti/md.160000K"), 429)
        self.assertEqual(self.nbe.predict("out_data_nbands_esti/md.240000K"), 732)
        os.remove("tmp.log")

    def test_get_default_nbands(self):
        res = NBandsEsti._get_res("out_data_nbands_esti/md.020000K/")
        nb = NBandsEsti._get_default_nbands(res)
        self.assertEqual(nb, 66)

    def test_get_default_nbands2(self):
        res = NBandsEsti._get_res("out_data_nbands_esti/mgal/")
        nb = NBandsEsti._get_default_nbands(res)
        self.assertEqual(nb, 124)

    def test_potcar_nvalence(self):
        res = NBandsEsti._get_potcar_nvalence("out_data_nbands_esti/POTCAR.dbl")
        self.assertEqual(res, [10.0, 3.0])

    def test_incar_ele_temp(self):
        res = NBandsEsti._get_incar_ele_temp("out_data_nbands_esti/md.000300K/INCAR")
        self.assertAlmostEqual(res, 0.025851991011651636)

    def test_incar_nbands(self):
        res = NBandsEsti._get_incar_nbands("out_data_nbands_esti/md.020000K/INCAR")
        self.assertEqual(res, 81)

    def test_get_res(self):
        res = NBandsEsti._get_res("out_data_nbands_esti/md.020000K/")
        ref = {
            "natoms": [32],
            "vol": 138.55418502346618,
            "nvalence": [3.0],
            "ele_temp": 20000.0,
            "nbands": 81,
        }
        self.assertEqual(res["natoms"], ref["natoms"])
        self.assertAlmostEqual(res["vol"], ref["vol"])
        self.assertAlmostEqual(res["nvalence"][0], ref["nvalence"][0])
        self.assertEqual(len(res["nvalence"]), len(ref["nvalence"]))
        self.assertAlmostEqual(res["ele_temp"], ref["ele_temp"], places=1)
        self.assertEqual(res["nbands"], ref["nbands"])


class TestEstimateNBands(unittest.TestCase):
    """Tests for the temperature-aware heuristic ``estimate_nbands``."""

    poscar = "out_data_nbands_esti/mgal/POSCAR"
    potcar = "out_data_nbands_esti/mgal/POTCAR"
    # Mg16Al16, ZVAL=[10,3] -> N_elec = 16*10 + 16*3 = 208, N_atom = 32

    def test_estimate_nbands_legacy_no_temp(self):
        # T=None with explicit legacy params degrades to the old formula
        nb = estimate_nbands(
            self.poscar,
            self.potcar,
            ele_temp_k=None,
            scale=1.2,
            nband_min=5,
        )
        self.assertEqual(nb, int(208 / 2 * 1.2) + 5 * 32)

    def test_estimate_nbands_default_low_temp(self):
        # T <= t_low (default 300) -> use nband_min_low (default 1)
        nb = estimate_nbands(self.poscar, self.potcar, ele_temp_k=200.0)
        self.assertEqual(nb, int(208 / 2 * 1.0) + 1 * 32)

    def test_estimate_nbands_default_high_temp(self):
        # T >= t_high (default 2000) -> use nband_min (default 3)
        nb = estimate_nbands(self.poscar, self.potcar, ele_temp_k=3000.0)
        self.assertEqual(nb, int(208 / 2 * 1.0) + 3 * 32)

    def test_estimate_nbands_mid_temp(self):
        # T = 1150K is the exact midpoint of (300, 2000) -> extra_per_atom = 2.0
        nb = estimate_nbands(self.poscar, self.potcar, ele_temp_k=1150.0)
        self.assertEqual(nb, int(208 / 2 * 1.0) + 2 * 32)

    def test_estimate_nbands_monotonic(self):
        # NBANDS must be non-decreasing with T across the full window
        temps = [0.0, 200.0, 300.0, 1000.0, 1150.0, 2000.0, 5000.0]
        nbs = [estimate_nbands(self.poscar, self.potcar, ele_temp_k=T) for T in temps]
        for a, b in zip(nbs, nbs[1:]):
            self.assertGreaterEqual(b, a)
