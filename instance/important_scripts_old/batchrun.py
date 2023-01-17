# -*- coding: utf-8 -*-
"""以 featurebox 中版本为准，若在其它处更改，请更新 featurebox。"""
import os

from tqdm import tqdm

# lsf_vasp_240 = """
# #BSUB -q normal
# #BSUB -n 32
# #BSUB -o %J.out
# #BSUB -e %J.err
# #BSUB -R "span[ptile=24]"
# PATH=/share/app/vasp.5.4.1-2018/bin/:$PATH
# source /share/intel/intel/bin/compilervars.sh intel64
# mpirun -np 36 vasp_std > log
# """
#
# pbs_vasp_238_239 = """
# #SBATCH -N 1
# #SBATCH -n 36
# #SBATCH --ntasks-per-node=36
# #SBATCH --output=%j.out
# #SBATCH --error=%j.err
# source /data/home/qian1/intel/bin/compilervars.sh intel64
# source /data/home/qian1/intel/mkl/bin/mklvars.sh intel64
# export PATH=$PATH:/data/home/qian1/app/vasp/vasp.5.4.4/bin
# scontrol show hostname $SLURM_JOB_NODELIST > host
# mpirun -np 36 vasp_std >log
# """
#
# pbs_python_template = """
# #SBATCH -N 1
# #SBATCH -n 16
# #SBATCH --ntasks-per-node=36
# #SBATCH --output=%j.out
# #SBATCH --error=%j.err
# python test.py run > log
# """
#
# lsf_python_template = """
# #BSUB -q normal
# #BSUB -n 16
# #BSUB -o %J.out
# #BSUB -e %J.err
# #BSUB -R "span[ptile=24]"
# python test.py run > log
# """
#
# slurm_vasp = """
# #!/bin/sh
# #SBATCH --partition=normal
# #SBATCH --job-name=smh
# #SBATCH --nodes=2
# #SBATCH --ntasks=28
#
# source /data/home/suyj/intel/bin/compilervars.sh intel64
# export PATH=/data/home/suyj/app/vasp.5.4.4-2018/bin:$PATH
# #mpirun -np $SLURM_NPROCS vasp_std | tee output
# mpirun -np 56 vasp_std | tee output
# """
#
# static = """
# ALGO = Fast
# EDIFF = 1e-06
# ENCUT = 600
# IBRION = -1
# ICHARG = 1
# ISIF = 2
# ISMEAR = -5
# ISPIN = 2
# ISTART = 1
# KSPACING = 0.2
# LAECHG = True
# LCHARG = True
# LELF = True
# LORBIT = 11
# LREAL = Auto
# LWAVE = False
# NELM = 200
# PREC = Accurate
# SYMPREC = 1e-08"""
#
# temps = {"lsf_python": lsf_python_template, "pbs_python": pbs_python_template,
#          "pbs_vasp": pbs_vasp_238_239, "lsf_vasp": lsf_vasp_240}


def upload(run_tem=None, pwd=None, filter_in=None,
           filter_out=None, existed_run_tem=None, store_name="batch.sh", **kwargs):
    """
    产生批处理提交文件。
    最方便用法: 把该文件放到算例同一文件夹下，并提供 pbs or lsf 模板给 run_tem 参数.
    Args:
        store_name: str
            存储名称
        run_tem: str, file, None
            pbs,lsf 模板。可以是文件，也可以是复制过来的字符串,该文件会复制到每个子文件夹.

        existed_run_tem: str,None
            1.不需要批量复制时, 使用每个子文件下已经存在 pbs,lsf 模板 该模板的名字必须提供 （run_tem=None, existed_run ！= None）。
            2.使用批量复制时, existed_run_tem 为每个文件夹下新 pbs,lsf 模板名字 （run_tem！=None, existed_run != None）
            3.使用批量复制时, 每个文件夹下新 pbs,lsf 模板名字默认使用源文件名字, 可能会覆盖！ （run_tem!=None, existed_run == None）
            4.不使用批量复制, 也不提供 pbs,lsf 模板名字,你用个 p!（run_tem==None, existed_run == None）

        pwd: path,None
            数据路径

        filter_in: list,str,None
            文件夹过滤条件，默认全选子文件夹
            如果为字符串，数据路径的子文件夹中包含该字符串被选择
            如果为列表，数据路径的子文件夹名在列表中的被选择

        filter_out: list,str,None
            如果为字符串，数据路径的子文件夹中包含该字符串被忽略
            如果为列表，数据路径的子文件夹名在该列表中的被忽略

        **kwargs:
            INCAR:str
                若提供，在每个子文件夹下加入该 INCAR 文件

    Returns:
        ./bash.sh 文件
    """
    job_type = "pbs"

    if "INCAR" in kwargs:
        INCAR = kwargs["INCAR"]
    else:
        INCAR = None

    if INCAR is not None:
        if os.path.isfile(INCAR):
            try:
                a = open(INCAR)
                con = a.readlines()
                INCAR = "".join(con)
            except IOError:

                raise IOError("We cannot import message from {} file".format(INCAR))

    if run_tem is None:
        assert existed_run_tem is not None, \
            "Use -r, -e or both of them !!!.\n" \
            "若把模板文件复制到所有子文件夹，如：\n" \
            " -r \***\pbs.run\n" \
            "若子文件夹已经存在模板，提供使用它们的统一文件名如：\n" \
            " -e lsf.run \n"

        job_type = "pbs"
        print("Job type is un-recognition, please modify the {} file manually, "
              "such as change 'qsub' to  'bsub <' or not.".format(store_name))

    elif os.path.isfile(run_tem):
        try:
            a = open(run_tem)
            con = a.readlines()
            run_tem = "".join(con)
        except IOError:

            raise IOError("We cannot import message from {} file".format(run_tem))

        if "SBATCH" in run_tem:
            job_type = "pbs"
        else:
            job_type = "lsf"

    # elif run_tem in temps:
    #     run_tem = temps[run_tem]

    if pwd is None:
        pwd = os.getcwd()

    files = os.listdir(pwd)  # 读入文件夹
    if isinstance(filter_in, str):
        files = [i for i in files if filter_in in i]
    if isinstance(filter_out, str):
        files = [i for i in files if filter_out not in i]

    if isinstance(filter_in, (list, tuple)):
        files = filter_in
    if isinstance(filter_out, (list, tuple)):
        [files.remove(i) for i in filter_out]

    if len(files) == 0:
        raise FileNotFoundError("There is no directory left after filtering")

    old = os.getcwd()

    for filei in tqdm(files):
        if run_tem is not None:
            try:
                existed_run_tem = existed_run_tem if existed_run_tem else "run.lsfpbs"
                os.chdir(os.path.join(pwd, str(filei)))

                f = open(existed_run_tem, "w")
                f.write(run_tem)
                f.close()
                if INCAR is not None:
                    f = open("INCAR", "w")
                    f.write(INCAR)
                    f.close()
            except NotADirectoryError:
                print(filei, "is a file and be filtered")
                files.remove(filei)
        else:
            try:
                existed_run_tem = existed_run_tem if existed_run_tem else "run.lsfpbs"
                os.chdir(os.path.join(pwd, str(filei)))
                # pwdpath = os.getcwd()
                # assert os.path.isfile(existed_run_tem), "老夫一瞅，你的{}下没有你说的这个{}文件".format(os.getcwd(),existed_run_tem)
                assert os.path.isfile(os.path.join(os.getcwd(), existed_run_tem)), \
                    "There is no {} in your {}, '{}' is the real name?".format(existed_run_tem, os.getcwd(),
                                                                               existed_run_tem)
                if INCAR is not None:
                    f = open("INCAR", "w")
                    f.write(INCAR)
                    f.close()
            except NotADirectoryError:
                print(filei, "is a file and filtered")
                files.remove(filei)
        os.chdir(old)

    if job_type == "lsf":
        job_type = "bsub < {}".format(existed_run_tem)
    else:
        job_type = "qsub {}".format(existed_run_tem)

    os.chdir(pwd)

    files = ["./" + i for i in files]
    batch_str = """#!/bin/bash
echo $dirname
for i in {}

do
cd $i
{}
cd ..
done
    """.format(files, job_type, )
    batch_str = batch_str.replace("'", "")
    batch_str = batch_str.replace("[", "")
    batch_str = batch_str.replace("]", "")
    batch_str = batch_str.replace(",", "")
    bach = open(store_name, "w")
    bach.write(batch_str)
    bach.close()
    print("{} are sored in {}".format(store_name, os.path.abspath(pwd)))
    print("OK")


def run(args, parser):
    upload(run_tem=args.run_tem, pwd=args.path_name, filter_in=args.dir_include,
           filter_out=args.dir_exclude, existed_run_tem=args.existed_run_tem, INCAR=args.INCAR,
           store_name=args.store_name)


class CLICommand:
    """
    产生任务批处理文件,请保证 -r,-e 至少存在一个.
    最方便用法:
    1.把该文件放到算例文件夹们的 父文件夹下，并提供 pbs, lsf 等文件模板给 -r 参数.

    $ featurebox batchrun -p /***/data -r /***/pbs.run

    文件格式：

    ---- data/ \n
     |-- data1/ \n
     |-- data2/ \n
     |-- pbs.run \n

    2.若算例中已经存在pbs, lsf 模板，请提供模板名字（不会包含路径）给 -e 参数.

    $ featurebox batchrun -p /***/data -e pbs.run

    文件格式：

    ---- data/ \n
     |-- data1/ \n
      ||- pbs.run \n
      ||- POSCAR \n
      ||- .... \n
     |-- data2/ \n
      ||- pbs.run \n
      ||- .... \n

    Example:

        $ featurebox batchrun -p /home/dir_name -if POSCAR
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-r', dest='run_tem', default=None,
                            help='{pbs,lsf}模板文件地址')
        parser.add_argument('-e', dest='existed_run_tem', default=None,
                            help='当子文件夹已经存在{pbs,lsf}模板文件，直接使用它。提供该统一模板文件的名字')
        parser.add_argument('-incar', dest='INCAR', default=None,
                            help='该 INCAR 也批量复制。')
        parser.add_argument('-p', '--path_name', type=str, default='.', help='所有算例地址, 默认当前地址')
        parser.add_argument('-id', '--dir_include', help='include dir name.', type=str, default=None)
        parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.', type=str, default=None)
        parser.add_argument('-o', '--store_name', help='out file name.', type=str, default="batch_run.sh")

    @staticmethod
    def run(args, parser):
        # args = args.parse_args()
        run(args, parser)


################################################################################################################
if __name__ == '__main__':
    # 命令行模式
    import argparse

    parser = argparse.ArgumentParser(description='产生任务批处理文件,请保证 -r,-e 至少存在一个.\n'
                                                 '最方便用法: \n'
                                                 '1.把该文件放到和算例同一文件夹下，并提供 pbs or lsf 等模板给 -r 参数.\n'
                                                 "python batchrun.py -r /***/pbs.run\n"
                                                 "2.若算例中已经存在pbs or lsf 模板，请提供模板名字（不包含路径）给 -e 参数.\n"
                                                 "python batchrun.py -e pbs.run \n")
    parser.add_argument('-r', dest='run_tem', default=None,
                        help='{pbs,lsf}模板文件地址')
    parser.add_argument('-e', dest='existed_run_tem', default=None,
                        help='当子文件夹已经存在{pbs,lsf}模板文件，直接使用它。提供该统一模板文件的名字')
    parser.add_argument('-incar', dest='INCAR', default=None,
                        help='该 INCAR 也批量复制')
    parser.add_argument('-p', '--path_name', type=str, default='.', help='所有算例地址, 默认当前地址')
    parser.add_argument('-id', '--dir_include', help='include dir name.', type=str, default=None)
    parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.', type=str, default=None)
    parser.add_argument('-o', '--store_name', help='out file name.', type=str, default="batch_run.sh")
    args = parser.parse_args()
    run(args, parser)
##############################################################################################################
# print(upload.__doc__)
# upload(run_tem="/share/home/skk/wcx/cam3d/Instance/Instance1/others/run.lsf", pwd="/share/home/skk/wcx/test/")
