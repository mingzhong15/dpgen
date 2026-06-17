#!/usr/bin/env python3

"""Checkpoint module for DP-GEN.

Encapsulates the record.dpgen persistence mechanism used for
breakpoint resumption across iterations and tasks.
"""

import logging
import os

from dpgen import dlog

# Keep backward-compatible with original lib/utils.py interface
record_file_name = "record.dpgen"
max_tasks = 10000


class Checkpoint:
    """Manages the record.dpgen file for breakpoint resumption.

    The record file stores ``(iter, task)`` pairs as two integers per line.
    On resume, the loop skips any ``(ii, jj)`` where
    ``ii * max_tasks + jj <= last_recorded_iter * max_tasks + last_recorded_task``.

    Parameters
    ----------
    record_file : str
        Path to the record file, default ``"record.dpgen"``.
    max_tasks : int
        Max tasks per iteration, used to encode ``iter/task`` as a single
        monotonic integer for comparison.
    """

    def __init__(self, record_file: str = "record.dpgen", max_tasks: int = 10000):
        self.record_file = record_file
        self.max_tasks = max_tasks

    def get_resume_point(self) -> tuple[int, int]:
        """Read the last ``(iter, task)`` pair from the record file.

        Returns
        -------
        tuple[int, int]
            ``(iter_index, task_index)`` — defaults to ``(0, -1)`` when no
            record exists yet.
        """
        iter_rec = [0, -1]
        if os.path.isfile(self.record_file):
            with open(self.record_file) as frec:
                for line in frec:
                    iter_rec = [int(x) for x in line.split()]
            if len(iter_rec) == 0:
                raise ValueError("There should not be blank lines in record.dpgen.")
            dlog.info(
                "continue from iter %03d task %02d" % (iter_rec[0], iter_rec[1])  # noqa: UP031
            )
        return tuple(iter_rec)

    def is_done(self, ii: int, jj: int, resume_point: tuple[int, int]) -> bool:
        """Check whether ``(ii, jj)`` should be skipped (already completed).

        Parameters
        ----------
        ii : int
            Iteration index.
        jj : int
            Task index within the iteration.
        resume_point : tuple[int, int]
            The ``(iter, task)`` pair read from :meth:`get_resume_point`.

        Returns
        -------
        bool
            True if this step should be skipped.
        """
        return ii * self.max_tasks + jj <= resume_point[0] * self.max_tasks + resume_point[1]

    def mark(self, ii: int, jj: int) -> None:
        """Append a completed ``(iter, task)`` pair to the record file.

        Parameters
        ----------
        ii : int
            Iteration index.
        jj : int
            Task index within the iteration.
        """
        with open(self.record_file, "a") as frec:
            frec.write("%d %d\n" % (ii, jj))  # noqa: UP031
