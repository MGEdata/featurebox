#!/bin/bash
""" Misc functions for interfacing between torque and the pbs module """

import datetime
import os
import re
import time

try:
    from featurebox.pbs.misc import getlogin, run_popen, shell_to_re_compile_pattern_lru, run_system
except ImportError:
    from misc import getlogin, run_popen, shell_to_re_compile_pattern_lru, run_system
tl = time.localtime()
year = tl.tm_year


def _jjobs(jobid=None, full=False, username=getlogin()):
    opt = ["jjobs"]

    if username is not None and jobid is None:
        jobid = job_id(username=getlogin())
        opt += ["-u", username]

    if full:
        opt += ["-l"]

    if jobid is not None:
        if isinstance(jobid, str):
            jobid = [jobid]
        elif isinstance(jobid, list):
            pass
        else:
            raise ValueError("jobid must be str or list of str")
        opt += jobid

    sout = run_popen(opt, first=False, join=True)

    return sout


def job_id(username=None):
    if username is not None:
        qsout = run_popen(["jjobs", "-u", username], first=False, join=False)
    else:
        qsout = run_popen("jjobs", first=False, join=False)

    if qsout is None or len(qsout) == 0:
        return []
    else:
        if len(qsout) >= 2:

            jobid = []
            for line in qsout[1:]:
                ma = line.split(" ")[0]
                if ma != "":
                    jobid.append(ma)
            return jobid
        else:
            return []


def job_rundir(jobid=None):
    """Return the directory job "id" was run in using qstat.

       Returns a dict, with id as key and rundir and value.
    """
    rundir = dict()

    if jobid is None:
        res = job_status(jobid=jobid)
        return {k: v["work_dir"] for k, v in res.items()}

    elif isinstance(jobid, (list, tuple)):
        for i in jobid:
            stdout = _jjobs(jobid=i, full=True)
            if stdout is not None:
                stdout = stdout.replace("\n                     ", "")
                match = re.search("CWD <(.*)>, Output", stdout)
                if match is not None:
                    res = match.group(1)
                    rundir[i] = res.replace("\n", "")
    else:
        stdout = _jjobs(jobid=jobid, full=True)
        if stdout is not None:
            stdout = stdout.replace("\n                     ", "")
            match = re.search("CWD <(.*)>, Output", stdout)
            if match is not None:
                res = match.group(1)
                rundir[jobid] = res.replace("\n", "")
    return rundir


def job_status(jobid=None):
    """Return job status using qstat

       Returns a dict of dict, with jobid as key in outer dict.
       Inner dict contains:
       "name", "nodes", "procs", "wall_time",
       "job_status": status ("Q","C","R", etc.)
       "qstatstr": result of qstat -f jobid, None if not found
       "elapsed_time": None if not started, else seconds as int
       "start_time": None if not started, else seconds since epoch as int
       "completion_time": None if not completed, else seconds since epoch as int

       *This should be edited to return job_status_dict()'s*
    """
    status = dict()

    stdout = _jjobs(jobid=jobid, full=True)

    if stdout is None:
        return {}

    sout = stdout.split("-------------------------------------------------------------------------------")

    for line in sout:
        line = line.replace("\n                     ", "")

        m = re.search(r"Job <([0-9]*)>", line)
        if m:
            jobstatus = {"jobid": m.group(1), "nodes": None, "procs": None, "wall_time": None, "qstatstr": line,
                         "elapsed_time": None, "start_time": None, "completion_time": None, "job_status": None,
                         'work_dir': None, 'submit_time': None}

            m2 = re.search(r"\s*Job Name <(\S*)>,", line)
            if m2:
                jobstatus["jobname"] = m2.group(1)

            m3 = re.search(r"([0-9]*) Processors Requested", line)
            if m3:
                jobstatus["nodes"] = None
                jobstatus["procs"] = m3.group(1)

            jobstatus["wall_time"] = None

            m5 = re.search(r"(.*): Started on", line)
            if m5:
                ti_str = m5.group(1)
                ti_str = ti_str + " %s" % year
                jobstatus["start_time"] = int(time.mktime(
                    datetime.datetime.strptime(ti_str, "%a %b %d %H:%M:%S %Y").timetuple()))
            else:
                jobstatus["start_time"] = None

            m6 = re.search(r"Status <(\D*\n?\D*)>, Queue", line)
            if m6:
                my_status = m6.group(1)
                my_status = my_status.replace("\n", "")
                my_status = my_status.replace(" ", "")

                if my_status == "RUN":
                    jobstatus["job_status"] = "R"
                elif my_status == "DONE":
                    jobstatus["job_status"] = "C"
                elif my_status == "EXIT":
                    jobstatus["job_status"] = "E"
                elif my_status == "PEND":
                    jobstatus["job_status"] = "Q"
                elif my_status == "PSUSP":
                    jobstatus["job_status"] = "H"
                else:
                    jobstatus["job_status"] = "?"
            else:
                jobstatus["job_status"] = None

            if jobstatus["job_status"] == "R" and jobstatus["start_time"] is not None:
                jobstatus["elapsed_time"] = int(time.time()) - jobstatus["start_time"]
            else:
                jobstatus["elapsed_time"] = None

            m7 = re.search(r"(.*): Done successfully", line)
            if m7:
                ti_str = m7.group(1)
                ti_str = ti_str + " %s" % year
                jobstatus["completion_time"] = int(time.mktime(
                    datetime.datetime.strptime(ti_str, "%a %b %d %H:%M:%S %Y").timetuple()))

            m7 = re.search(r"(.*): Exited", line)
            if m7:
                ti_str = m7.group(1)
                ti_str = ti_str + " %s" % year
                jobstatus["completion_time"] = int(time.mktime(
                    datetime.datetime.strptime(ti_str, "%a %b %d %H:%M:%S %Y").timetuple()))

            m8 = re.search("CWD <(.*)>, Out", line, )
            if m8:
                jobstatus["work_dir"] = m8.group(1)

            m9 = re.search(r"(.*): Submitted from host", line)
            if m9:
                ti_str = m9.group(1)
                ti_str = ti_str + " %s" % year
                jobstatus["submit_time"] = int(time.mktime(
                    datetime.datetime.strptime(ti_str, "%a %b %d %H:%M:%S %Y").timetuple()))

            m10 = re.search(r"ORDER: (.*)", line)
            if m10:
                jobstatus["order"] = m10.group(1)

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

    ll = run_popen(f"jsub < {file}")

    os.chdir(pt)

    if "Job <" in ll:
        return re.search(r"Job <(\d*)>", ll).group(1)
    else:
        return None


def delete(jobid):
    """qdel a PBS job."""
    if isinstance(jobid, str):
        res = run_popen(f"jctrl kill {jobid}")
    else:
        res = run_popen(f"jctrl kill " + " ".join(jobid))
    return jobid


def clear(jobids: list = None):
    """qdel a PBS job."""
    res = run_popen(f"jctrl kill " + " ".join(jobids))
    return jobids


def hold(jobid):
    """qhold a PBS job."""
    if isinstance(jobid, str):
        res = run_popen(f"jctrl stop {jobid}")
    else:
        res = run_popen(f"jctrl stop " + " ".join(jobid))
    return jobid


def release(jobid):
    """qrls a PBS job."""
    if isinstance(jobid, str):
        res = run_popen(f"jctrl resume {jobid}")
    else:
        res = run_popen(f"jctrl resume " + " ".join(jobid))
    return jobid

# if __name__ == "__main__":
#     # res21 = find_executable("qsub")
#     res = _jjobs(jobid=None)
#     print(res)
