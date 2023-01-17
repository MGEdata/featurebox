#!/bin/bash
""" Misc functions for interfacing between torque and the pbs module """

import datetime
import os
import re
import time
from typing import Union

from featurebox.pbs.misc import getlogin, run_popen, shell_to_re_compile_pattern_lru


def _squeue(jobid=None, username=getlogin(), full=False):  # pylint: disable=unused-argument
    """Return the stdout of squeue minus the header lines.

       By default, 'username' is set to the current user.
       'full' is the '-f' option
       'jobid' is a string or list of strings of job ids
       'version' is a software version number, used to
            determine compatible ops
       'sformat' is a squeue format string (e.g., "%A %i %j %c")

       Returns the text of squeue, minus the header lines
    """
    if isinstance(jobid, str):
        jobid = [jobid]

    # If Full is true, we need to use scontrol:
    if full is True:
        if jobid is None:
            if username is None:
                # Clearly we want ALL THE JOBS
                sopt = ["scontrol", "show", "job"]
                sout = run_popen(sopt, first=False, join=True)
            else:
                sopt = ["scontrol", "show", "job", "-u", username]
                sout = run_popen(sopt, first=False, join=True)
        else:
            opt = ["scontrol", "show", "job"]
            if isinstance(jobid, list):
                opt = opt + jobid
            sout = run_popen(opt, first=False, join=True)

    else:
        sopt = ["squeue", "-h"]  # jump first line
        if username is not None:
            sopt += ["-u", username]
        if jobid is not None:
            if isinstance(jobid, list):
                sopt += ["--job="]
                sopt += ["'" + ",".join([str(i) for i in jobid]) + "'"]

        sout = run_popen(sopt, first=False, join=True)

    return sout


def job_id(username=None):  # pylint: disable=redefined-builtin
    """If 'name' given, returns a list of all jobs with a particular name using squeue.
       Else, if all=True, returns a list of all job ids by current user.
       Else, returns this job id from environment variable PBS_JOBID (split to get just the number).

       Else, returns None

    """
    sopt = ["squeue", "-h"]  # jump first line
    if username is not None:
        sopt += ["-u", username]

    sout = run_popen(sopt, first=False, join=False)

    if sout is None or len(sout) == 0:
        return []
    else:
        jobid = []
        for line in sout:
            if username is not None:
                if line.split()[3] == username:
                    jobid.append(line.split()[0])
                else:
                    print(line.split()[3])
            else:
                jobid.append(line.split()[0])
        return jobid


def job_rundir(jobid=None):
    """Return the directory job "id" was run in using squeue.

       Returns a dict, with id as key and rundir and value.
    """
    rundir = dict()

    if jobid is None:
        res = job_status(jobid=jobid)
        return {k: v["work_dir"] for k, v in res.items()}

    elif isinstance(jobid, (list, tuple)):
        for i in jobid:
            stdout = _squeue(jobid=i, full=True)
            if stdout is not None:
                match = re.search("WorkDir=(.*)\n", stdout)
                if match is not None:
                    rundir[i] = match.group(1)
    else:
        stdout = _squeue(jobid=jobid, full=True)
        if stdout is not None:
            match = re.search("WorkDir=(.*)\n", stdout)
            if match is not None:
                rundir[jobid] = match.group(1)
    return rundir


def job_status(jobid=None):
    """Return job status using squeue

       Returns a dict of dict, with jobid as key in outer dict.
       Inner dict contains:
       "name", "nodes", "procs", "wall_time",
       "job_status": status ("Q","C","R", etc.)
       "qstatstr": result of squeue -f jobid, None if not found
       "elapsed_time": None if not started, else seconds as int
       "start_time": None if not started, else seconds since epoch as int
       "completion_time": None if not completed, else seconds since epoch as int

       *This should be edited to return job_status_dict()'s*
    """
    status = dict()

    stdout = _squeue(jobid=jobid, full=True)

    if stdout is None:
        return {}

    sout = stdout.split("\n\n")

    for line in sout:

        m = re.search(r"Job\s?Id\s?=\s?(.*)\sJ", line)
        if m:
            jobstatus = {"jobid": m.group(1), "nodes": None, "procs": None, "wall_time": None, "qstatstr": line,
                         "elapsed_time": None, "start_time": None, "completion_time": None, "job_status": None,
                         'work_dir': None, 'submit_time': None}

            m2 = re.search(r"JobName=\s?(.*)\n", line)
            if m2:
                jobstatus["jobname"] = m2.group(1)

            # Look for the Nodes/PPN Info
            m3 = re.search(r"NumNodes=\s?([0-9]*-[0-9]*)\s", line)
            if m3:
                jobstatus["nodes"] = m3.group(1)
                m4 = re.search(r"NumCPUs=\s?([0-9]*)\s", line)
                if m4:
                    jobstatus["procs"] = int(m4.group(1))

            # Grab the job start time
            m4 = re.search(r"start_time=\s?([0-9]*-[0-9]*-[0-9]*T[0-9]*:[0-9]*:[0-9]*)\s",
                           line)
            if m4:
                if m4.group(1) != "Unknown":
                    year, month, day = m4.group(1).split("T")[0].split("-")
                    hrs, mns, scs = m4.group(1).split("T")[1].split(":")
                    start_time = datetime.datetime(year=int(year), month=int(month), day=int(day), hour=int(hrs),
                                                   minute=int(mns), second=int(scs))
                    jobstatus["start_time"] = time.mktime(start_time.timetuple())

            m8 = re.search(r"WorkDir=\s?(.*)", line)
            if m8:
                jobstatus["work_dir"] = m8.group(1)

            # Look for timing info
            m9 = re.search(r"RunTime=\s?([0-9]*:[0-9]*:[0-9]*)\s", line)
            if m9:
                if m9.group(1) != "Unknown":
                    hrs, mns, scs = m9.group(1).split(":")
                    runtime = datetime.timedelta(hours=int(hrs), minutes=int(mns), seconds=int(scs))
                    jobstatus["elapsed_time"] = runtime.seconds

                    m10 = re.search(r"TimeLimit=\s?([0-9]*:[0-9]*:[0-9]*)\s", line)
                    if m10:
                        wall_time = datetime.timedelta(hours=int(hrs), minutes=int(mns), seconds=int(scs))
                        jobstatus["wall_time"] = wall_time.seconds

            # Grab the job status
            m11 = re.search(r"JobState=\s?([a-zA-Z]*)\s", line)
            if m11:
                my_status = m11.group(1)
                if my_status == "RUNNING" or my_status == "CONFIGURING":
                    jobstatus["job_status"] = "R"
                elif my_status == "BOOT_FAIL" or my_status == "FAILED" or my_status == "NODE_FAIL" \
                        or my_status == "CANCELLED" or my_status == "COMPLETED" \
                        or my_status == "PREEMPTED" or my_status == "TIMEOUT":
                    jobstatus["job_status"] = "C"
                elif my_status == "COMPLETING" or my_status == "STOPPED":
                    jobstatus["job_status"] = "E"
                elif my_status == "PENDING" or my_status == "SPECIAL_EXIT":
                    jobstatus["job_status"] = "Q"
                elif my_status == "SUSPENDED":
                    jobstatus["job_status"] = "S"
                else:
                    jobstatus["job_status"] = "?"

            if jobstatus["job_status"] == "C":
                m13 = re.search(r"EndTime=\s?([0-9]*-[0-9]*-[0-9]*T[0-9]*:[0-9]*:[0-9]*)\s",
                                line)
                if m13:
                    year, month, day = m13.group(1).split("T")[0].split("-")
                    hrs, mns, scs = m13.group(1).split("T")[1].split(":")
                    start_time = datetime.datetime(year=int(year), month=int(month), day=int(day), hour=int(hrs),
                                                   minute=int(mns), second=int(scs))
                    jobstatus["completion_time"] = time.mktime(start_time.timetuple())

            # Grab the cluster/allocating node:
            m12 = re.search(r"AllocNode:.*=\s?(.*):.*\s", line)
            if m12:
                raw_str = m12.group(1)
                m12 = re.search(r"(.*?)(?=[^a-zA-Z0-9]*login.*)\s", raw_str)
                if m12:
                    jobstatus["cluster"] = m12.group(1)
                else:
                    jobstatus["cluster"] = raw_str

            m9 = re.search(r"SubmitTime=\s?([0-9]*:[0-9]*:[0-9]*)\s", line)
            if m9:
                jobstatus["submit_time"] = int(time.mktime(datetime.datetime.strptime(
                    m9.group(1), "%a %b %d %H:%M:%S %Y").timetuple()))

            status[jobstatus["jobid"]] = jobstatus

    return status


def submit_from_path(path: str, file: str):
    """Submit a PBS job using qsub.

       substr: The submit script string
    """
    if path is None:
        return ""
    pt = os.getcwd()
    os.chdir(path)

    if any([i in file for i in ["*", "!", "?", "["]]):
        fs = os.listdir()
        fss = "\n".join(fs)

        smatch = shell_to_re_compile_pattern_lru(file, trans=True)
        res = smatch.findall(fss)
        res = [i for i in res if i in fs]
        if len(res) != 1:
            raise FileNotFoundError(f"There are 1+ file/No file {res} with patten '{file}' "
                                    f"in path '{path}'."
                                    f"make sure run file in the path, or using more strict/relax "
                                    f"condition to filter the run file.", )
        file = res[0]

    res = run_popen(["sbatch", file])

    os.chdir(pt)

    res = res.replace("Submitted batch job ", "")
    jobid = res.replace("\n", "")
    return jobid


def delete(jobid: Union[list, str]):
    """qdel a PBS job."""
    if isinstance(jobid, str):
        res = run_popen(["scancel", jobid])
    else:
        res = run_popen(f"scancel " + " ".join(jobid))
    return jobid


def clear(jobids: list = None):
    """qdel a PBS job."""
    if jobids is None:
        jobids = job_id()
    res = run_popen(f"scancel " + " ".join(jobids))
    return jobids


def hold(jobid: Union[list, str]):
    """scontrol delay a PBS job."""
    if isinstance(jobid, (list, tuple)):
        [hold(i) for i in jobid]
        return jobid
    p = run_popen(["scontrol", "update", "JobId=", jobid, "start_time=", "now+30days"])
    return jobid


def release(jobid: Union[list, str]):
    """scontrol un-delay a PBS job."""
    if isinstance(jobid, (list, tuple)):
        [release(i) for i in jobid]
        return jobid
    else:
        res = run_popen(["scontrol", "update", "JobId=", jobid, "start_time=", "now"])
    return jobid


if __name__ == "__main__":
    # res21 = find_executable("qsub")
    res1 = job_status()
    print(res1[list(res1.keys())[-1]])
