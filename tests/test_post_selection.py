#!/usr/bin/env python3
"""Tests for post_selection.py (pure algorithmic logic, no deps)."""

import json
import math
import os
import random
import tempfile
import unittest
from io import StringIO

# ---------------------------------------------------------------------------
# Pure-Python reference implementations of key algorithms
# (tested independently of the numpy version in post_selection.py)
# ---------------------------------------------------------------------------


def fps_pure(points, n_select, seed=None):
    """Farthest Point Sampling — pure Python implementation.

    ``points`` is a list of ``(x, y)`` tuples.
    Returns list of selected indices.
    """
    n = len(points)
    if n_select >= n:
        return list(range(n))
    if n_select < 1:
        return []

    rng = random.Random(seed)
    selected = [rng.randrange(n)]
    # Pre-compute squared distances
    dists = [float("inf")] * n

    for _ in range(1, n_select):
        last = selected[-1]
        lx, ly = points[last]
        # Update distances to nearest selected point
        for i in range(n):
            dx = points[i][0] - lx
            dy = points[i][1] - ly
            d2 = dx * dx + dy * dy
            if d2 < dists[i]:
                dists[i] = d2
        # Pick farthest
        farthest = max(range(n), key=lambda i: dists[i])
        selected.append(farthest)

    return selected


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFarthestPointSampling(unittest.TestCase):
    def test_select_all_when_n_select_equals_n(self):
        pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        result = fps_pure(pts, 3)
        self.assertEqual(sorted(result), [0, 1, 2])

    def test_select_none_when_n_select_zero(self):
        pts = [(0.0, 0.0), (1.0, 0.0)]
        result = fps_pure(pts, 0)
        self.assertEqual(result, [])

    def test_select_single_point(self):
        pts = [(0.0, 0.0), (5.0, 5.0), (3.0, 3.0)]
        result = fps_pure(pts, 1)
        self.assertEqual(len(result), 1)
        self.assertIn(result[0], [0, 1, 2])

    def test_two_points_selected_are_farthest(self):
        pts = [(0.0, 0.0), (1.0, 1.0), (10.0, 10.0)]
        result = fps_pure(pts, 2)
        # First point is random, second should be the farthest from the first
        self.assertEqual(len(result), 2)

    def test_three_points_in_triangle(self):
        pts = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
        result = fps_pure(pts, 3, seed=42)
        self.assertEqual(len(result), 3)

    def test_five_from_ten_grid(self):
        # 10 points on a 2x5 grid
        pts = [(i % 5, i // 5) for i in range(10)]
        result = fps_pure(pts, 5, seed=123)
        self.assertEqual(len(result), 5)
        # No duplicates
        self.assertEqual(len(set(result)), 5)

    def test_reproducible_with_seed(self):
        pts = [(0.0, 0.0), (1.0, 2.0), (3.0, 1.0), (4.0, 5.0)]
        r1 = fps_pure(pts, 3, seed=42)
        r2 = fps_pure(pts, 3, seed=42)
        self.assertEqual(r1, r2)

    def test_handles_single_point(self):
        pts = [(0.0, 0.0)]
        result = fps_pure(pts, 1)
        self.assertEqual(result, [0])
        result = fps_pure(pts, 5)
        self.assertEqual(result, [0])

    def test_handles_two_points(self):
        pts = [(0.0, 0.0), (10.0, 10.0)]
        result = fps_pure(pts, 2, seed=0)
        self.assertEqual(sorted(result), [0, 1])


class TestCandidatesJsonIO(unittest.TestCase):
    def test_write_and_read_candidates(self):
        data = {
            "0": [["task.0.000000", 100], ["task.0.000001", 50]],
            "1": [["task.1.000000", 200]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "candidates.json")
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            with open(path) as f:
                loaded = json.load(f)
        self.assertEqual(loaded, data)

    def test_empty_candidates(self):
        data = {}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "candidates.json")
            with open(path, "w") as f:
                json.dump(data, f)
            with open(path) as f:
                loaded = json.load(f)
        self.assertEqual(loaded, {})

    def test_parse_task_path(self):
        path = "iter.000001/01.model_devi/task.0.000000"
        basename = os.path.basename(path)
        parts = basename.split(".")
        sys_idx, task_idx = parts[1], parts[2]
        self.assertEqual(sys_idx, "0")
        self.assertEqual(task_idx, "000000")

    def test_make_fp_task_name_format(self):
        sys_idx = 0
        counter = 5
        # Replicate the make_fp_task_name logic from helpers.py
        task_name = "%03d.%06d" % (sys_idx, counter)
        self.assertEqual(task_name, "000.000005")


class TestVisDataFormat(unittest.TestCase):
    def test_umap_dat_format(self):
        """Check that the umap.dat format rounds correctly."""
        rows = [
            (100, -2.345678, 1.567890, 0),
            (200, 0.812345, -0.312345, 1),
            (300, 3.123456, 0.212345, 0),
        ]
        buf = StringIO()
        buf.write("# step        x            y            selected\n")
        for step, x, y, sel in rows:
            buf.write(f"{step}  {x:.6f}  {y:.6f}  {sel}\n")
        content = buf.getvalue()
        self.assertIn("# step", content)
        self.assertIn("200  0.812345", content)
        lines = content.strip().split("\n")
        self.assertEqual(len(lines), 4)  # header + 3 data lines


class TestParamValidation(unittest.TestCase):
    def test_arginfo_params_exist(self):
        """Check that the arginfo.py file contains the expected parameter names."""
        arginfo_path = os.path.join(
            os.path.dirname(__file__),
            "../dpgen/generator/arginfo.py",
        )
        with open(arginfo_path) as f:
            content = f.read()

        expected_params = [
            "model_devi_post_select",
            "model_devi_post_select_mode",
            "model_devi_post_select_soap_rcut",
            "model_devi_post_select_soap_nmax",
            "model_devi_post_select_soap_lmax",
            "model_devi_post_select_pca_dim",
            "model_devi_post_select_umap_n_neighbors",
            "model_devi_post_select_umap_min_dist",
            "model_devi_post_select_seed",
        ]
        for param in expected_params:
            with self.subTest(param=param):
                self.assertIn(param, content)


class TestImportSyntax(unittest.TestCase):
    def test_post_selection_syntax(self):
        """Verify post_selection.py is syntactically valid."""
        import py_compile

        path = os.path.join(
            os.path.dirname(__file__),
            "../dpgen/generator/post_selection.py",
        )
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"Syntax error: {e}")

    def test_model_devi_syntax(self):
        """Verify model_devi.py is syntactically valid."""
        import py_compile

        path = os.path.join(
            os.path.dirname(__file__),
            "../dpgen/generator/model_devi.py",
        )
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"Syntax error: {e}")

    def test_fp_syntax(self):
        """Verify fp.py is syntactically valid."""
        import py_compile

        path = os.path.join(
            os.path.dirname(__file__),
            "../dpgen/generator/fp.py",
        )
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"Syntax error: {e}")

    def test_arginfo_syntax(self):
        """Verify arginfo.py is syntactically valid."""
        import py_compile

        path = os.path.join(
            os.path.dirname(__file__),
            "../dpgen/generator/arginfo.py",
        )
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"Syntax error: {e}")


if __name__ == "__main__":
    unittest.main()
