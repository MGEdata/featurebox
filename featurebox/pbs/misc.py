#!/bin/bash
""" Misc functions for interacting between the OS and the pbs module """
import datetime
import functools
import os
import re
import sys
from distutils.spawn import find_executable
from fnmatch import translate
from typing import Union


class PBSError(Exception):
    """ A custom error class for pbs errors """

    def __init__(self, jobid, msg):
        self.jobid = jobid
        self.msg = msg
        super(PBSError, self).__init__()

    def __str__(self):
        return self.jobid + ": " + self.msg


def getsoftware():
    """Tries to find qsub, then sbatch. Returns "torque" if qsub
    is found, else returns "slurm" if sbatch is found, else returns
    "other" if neither is found. """
    if find_executable("qsub") is not None:
        return "torque"
    elif find_executable("sbatch") is not None:
        return "slurm"
    if find_executable("jctrl") is not None:
        return "unischeduler"
    else:
        return "other"


def run_popen(cmd: Union[str, list], first=True, join=False):
    """run command."""
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(cmd)
    q = os.popen(cmd)
    res = q.readlines()
    if res is None or len(res) == 0:
        return None
    else:
        if join:
            first = False
        if first:
            return res[0]
        elif join:
            return "".join(res)
        else:
            return res


def run_system(cmd: Union[str, list]):
    """run command."""
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(cmd)
    os.system(cmd)


def getlogin():
    """Returns os.getlogin(), else os.environ["LOGNAME"], else "?" """
    try:
        return os.getlogin()
    except OSError:
        return os.environ["LOGNAME"]


def getversion(software=None):
    """Returns the software version """
    if software is None:
        software = getsoftware()
    if software == "torque":
        opt = ["qstat", "--version"]
        sout = run_popen(opt)
        return sout.split("\n")[0].lower().lstrip("version: ")
    elif software == "slurm":
        opt = ["squeue", "--version"]
        sout = run_popen(opt)
        return sout.split("\n")[0].lstrip("slurm ")

    elif software == "unischeduler":
        opt = ["jctrl", "-V"]
        sout = run_popen(opt)
        return sout.split(",")[0].lstrip("JH Unischeduler ")

    else:
        return "0"


def seconds(wall_time):
    """Convert [[[DD:]HH:]MM:]SS to hours"""
    wtime = wall_time.split(":")
    if len(wtime) == 1:
        return 0.0
    elif len(wtime) == 2:
        return float(wtime[0]) * 60.0 + float(wtime[1])
    elif len(wtime) == 3:
        return float(wtime[0]) * 3600.0 + float(wtime[1]) * 60.0 + float(wtime[2])
    elif len(wtime) == 4:
        return (float(wtime[0]) * 24.0 * 3600.0
                + float(wtime[0]) * 3600.0
                + float(wtime[1]) * 60.0
                + float(wtime[2]))
    else:
        print("Error in wall_time format:", wall_time)
        sys.exit()


def hours(wall_time):
    """Convert [[[DD:]HH:]MM:]SS to hours"""
    wtime = wall_time.split(":")
    if len(wtime) == 1:
        return float(wtime[0]) / 3600.0
    elif len(wtime) == 2:
        return float(wtime[0]) / 60.0 + float(wtime[1]) / 3600.0
    elif len(wtime) == 3:
        return float(wtime[0]) + float(wtime[1]) / 60.0 + float(wtime[2]) / 3600.0
    elif len(wtime) == 4:
        return (float(wtime[0]) * 24.0
                + float(wtime[0])
                + float(wtime[1]) / 60.0
                + float(wtime[2]) / 3600.0)
    else:
        print("Error in wall_time format:", wall_time)
        sys.exit()


def strftimedelta(seconds):  # pylint: disable=redefined-outer-name
    """Convert seconds to D+:HH:MM:SS"""
    seconds = int(seconds)

    day_in_seconds = 24.0 * 3600.0
    hour_in_seconds = 3600.0
    minute_in_seconds = 60.0

    day = int(seconds / day_in_seconds)
    seconds -= day * day_in_seconds

    hour = int(seconds / hour_in_seconds)
    seconds -= hour * hour_in_seconds

    minute = int(seconds / minute_in_seconds)
    seconds -= minute * minute_in_seconds

    return str(day) + ":" + ("%02d" % hour) + ":" + ("%02d" % minute) + ":" + ("%02d" % seconds)


def exetime(deltatime):
    """Get the exetime string for the PBS '-a'option from a [[[DD:]MM:]HH:]SS string

       exetime string format: YYYYmmddHHMM.SS
    """
    return (datetime.datetime.now()
            + datetime.timedelta(hours=hours(deltatime))).strftime("%Y%m%d%H%M.%S")


@functools.lru_cache(10)
def shell_to_re_compile_pattern_lru(pat, trans=True, single=True):
    if trans:
        res = translate(pat)
        if single:
            res = res.replace("?s:", "?m:")
            res = res.replace(r"\Z", "")
    else:
        res = pat

    return re.compile(res)


if __name__ == "__main__":
    # res21 = find_executable("qsub")
    res1 = getversion(software=None)
