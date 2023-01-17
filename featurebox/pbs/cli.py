# -*- coding: utf-8 -*-

# @Time  : 2022/12/7 8:48
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License
# -*- coding: utf-8 -*-

from featurebox.pbs.job_manager import JobManager

_dos_help = """
-------------------------------
Job manager command line model
-------------------------------

Example:

1. submit jobs from paths.

    jm -sub -f pbs.run -p ~/path1 ~/path2 ~/path3 

2. submit jobs from one file contain paths (could get from commond: 'findpath'). 

    jm -sub -f "*.run" -pf paths.temp
    
3. clear all jobs:

    jm -c
    
4. delete jobs by ids:

    jm -d 334.c0 335.c0
    
5. delete jobs by ids:

    jm -d 334.c0-335.c0

6. suspend/release jobs by ids:

    jm -s 334
    
    jm -r 334

"""


def run(args, parser):
    print("-----Start-----")

    jm = JobManager(manager=None)
    jm.get_job_msg()

    if args.clear:
        jm.clear()

    if args.submit:
        if args.path_file:
            jm.submit_from_paths_file(path_file=args.path_file, file=args.file)
        else:
            jm.submit_from_path(path=args.path, file=args.file)

    if args.delete != ():
        ids = args.delete[0] if len(args.delete) == 1 else args.delete

        jm.delete(ids)
    if args.suspend != ():
        ids = args.suspend[0] if len(args.suspend) == 1 else args.suspend

        jm.hold(ids)

    if args.release != ():
        ids = args.release[0] if len(args.release) == 1 else args.release

        jm.release(ids)

    import pandas as pd
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option("display.max_colwidth", 100)
    jm.dump_job_msg()
    print(jm.print_pd())

    print("------End------")


class CLICommand:
    __doc__ = _dos_help

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-sub', '--submit', help='submit job/jobs', action="store_true")
        parser.add_argument('-p', '--path', type=str, default=[".", ], nargs="?", const=[".", ],
                            help="path or paths")
        parser.add_argument('-pf', '--path_file', type=str, default=None, help="file contains paths.")
        parser.add_argument('-f', '--file', type=str, help="run file patten", default="*.run")
        parser.add_argument('-c', '--clear', help='clear jobs.', action="store_true")
        parser.add_argument('-d', '--delete', help='delete jobs by jobid/jobids.', nargs="*", type=str, default=())
        parser.add_argument('-s', '--suspend', help='suspend/hold jobs by jobid/jobids.', nargs="*", type=str,
                            default=())
        parser.add_argument('-r', '--release', help='release jobs by jobid/jobids.', nargs="*", type=str, default=())

    @staticmethod
    def parse_args(parser):
        return parser.parse_args()

    @staticmethod
    def run(args, parser):
        run(args, parser)


def main():
    """
    Example:
        $ python this.py -p /home/dir_name
        $ python this.py -f /home/dir_name/path.temp
    """

    from mgetool.cli._formatter import Formatter
    import argparse

    parser = argparse.ArgumentParser(description=_dos_help, formatter_class=Formatter)
    CLICommand.add_arguments(parser=parser)
    args = CLICommand.parse_args(parser=parser)
    CLICommand.run(args=args, parser=parser)


if __name__ == '__main__':
    main()
