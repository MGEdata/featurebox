
import os
import numpy as np
import pandas as pd


# Due to the pymatgen is incorrect of band gap with 2 spin. we use vaspkit for extract data.


def cal(d, store=False, store_name="temp.csv", run_cmd=True, cmds=None):
    """Run linux cmd and return result, make sure the vaspkit is installed."""
    old = os.getcwd()
    os.chdir(d)
    try:
        #
        if run_cmd:
            cmd_sys(cmds=cmds)
        # >>>
        result = read(d, store=store, store_name=store_name)
        # <<<

        os.chdir(old)
        return result
    except BaseException as e:
        print(e)
        print("Error for:", d)

        os.chdir(old)
        return None


def cal_all(d, repeat=0, store=False, store_name="temp_all.csv", run_cmd=True, cmds=None):
    data_all = []
    col = None
    for di in d:
        res = cal(di, run_cmd=run_cmd, cmds=cmds)

        if isinstance(res, pd.DataFrame):
            col = res.columns
            res = res.values
            if repeat <= 1:
                data_all.append(res)
            else:
                for i in range(repeat):
                    res2 = res
                    res2[:, 0] = res2[:, 0] + f"-S{i}"
                    data_all.append(res2)
        else:
            pass
    data_all = np.concatenate(data_all, axis=0)
    result = pd.DataFrame(data_all, columns=col)

    if store:
        result.to_csv(store_name)
        print("'{}' are sored in '{}'".format(store_name, os.getcwd()))
    return result


def cmd_sys(cmds=None):
    os.system("vaspkit -task 911")


def read(d, store=False, store_name="temp.csv", file_name="BAND_GAP"):
    with open(file_name) as f:
        res = f.readlines()

    res = [i for i in res if "(eV)" in i]

    name = [i.split(":")[0].replace("  ", "") for i in res]
    name = [i[1:] if i[0] == " " else i for i in name]
    value = [float(i.split(" ")[-1].replace("\n", "")) for i in res]

    result = {"File": str(d)}
    for ni, vi in zip(name, value):
        result.update({ni: vi})

    if round(value[1], 2) <= round(value[3], 2) <= round(value[2], 2):
        result.update({'Fermi Energy (eV) Center': (value[1] + value[2]) / 2})
    else:
        result.update({'Fermi Energy (eV) Center': value[3]})

    result = {"0": result}

    result = pd.DataFrame.from_dict(result).T

    if store:
        result.to_csv(store_name)
        print("'{}' are sored in '{}'".format(store_name, os.getcwd()))

    return result


def run(args, parser):
    from mgetool.imports import batchfile
    if args.job_type in ["S", "s"]:
        res = cal(args.path_name, store=True, store_name=args.out_name)

        print(args.path_name, res)
    else:
        assert args.job_type in ["M", "m"]
        bf = BatchFile(args.path_name, suffix=args.suffix)
        bf.filter_dir_name(include=args.dir_include, exclude=args.dir_exclude, layer=args.layer)
        bf.filter_file_name(include=args.file_include, exclude=args.file_exclude)
        bf.merge()

        fdir = bf.file_dir

        fdir.sort()

        if not fdir:
            raise FileNotFoundError("There is no dir meeting the requirements after filter.")

        if not args.abspath:
            absp = os.path.abspath(args.path)
            fdir = [i.replace(absp, ".") for i in fdir]

        if "All" not in args.out_name or "all" not in args.out_name:
            name = "All_" + args.out_name
        else:
            name = args.out_name

        cal_all(fdir, repeat=args.repeat, store=True, store_name=name)


class CLICommand:
    """
    批量提取带隙,费米能级， 保存到当前工作文件夹。 查看参数帮助使用 -h。
    在 featurebox 中运行，请使用 featurebox bgefvk ...
    若复制本脚本并单运行，请使用 python bgefvk ...

    首先，请确保 运算子文件夹(sample_i_dir)包含应有 vasp 输入输出文件。
    parent_dir(为上层目录，或者上n层目录)

    EIGENVAL is need for vaspkit.

    如果在 featurebox 中运行多个案例,请指定顶层文件夹:

    Example:

        $ featurebox bgefvk -p /home/parent_dir

    如果在 featurebox 中运行单个案例，请指定运算子文件夹:

    Example:

        $ featurebox bgefvk -t s -p /home/parent_dir/***/sample_i_dir
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-p', '--path_name', type=str, default='.')
        parser.add_argument('-t', '--job_type', type=str, default='m')
        parser.add_argument('-s', '--suffix', help='suffix of file', type=str, default=None)
        parser.add_argument('-if', '--file_include', help='include file name.', type=str, default="EIGENVAL")
        parser.add_argument('-ef', '--file_exclude', help='exclude file name.', type=str, default=None)
        parser.add_argument('-id', '--dir_include', help='include dir name.', type=str, default=None)
        parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.', type=str, default=None)
        parser.add_argument('-l', '--layer', help='dir depth,default the last layer', type=int, default=-1)
        parser.add_argument('-abspath', '--abspath', help='return abspath', type=bool, default=True)
        parser.add_argument('-repeat', '--repeat', help='repeat times', type=int, default=0)
        parser.add_argument('-o', '--out_name', help='out file name.', type=str, default="bandgap_Ef.csv")

    @staticmethod
    def run(args, parser):
        run(args, parser)


if __name__ == '__main__':
    """
    Example:
        
        $ python bgefvk.py -p /home/dir_name -if EIGENVAL
    """
    import argparse
    parser = argparse.ArgumentParser(description="Get band gaps. Examples：\n"
                                                 "python bgefvk.py -p /home/dir_name -if EIGENVAL")
    parser.add_argument('-p', '--path_name', type=str, default='.')
    parser.add_argument('-t', '--job_type', type=str, default='m')
    parser.add_argument('-s', '--suffix', help='suffix of file', type=str, default=None)
    parser.add_argument('-if', '--file_include', help='include file name.', type=str, default="EIGENVAL")
    parser.add_argument('-ef', '--file_exclude', help='exclude file name.', type=str, default=None)
    parser.add_argument('-id', '--dir_include', help='include dir name.', type=str, default=None)
    parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.', type=str, default=None)
    parser.add_argument('-l', '--layer', help='dir depth,default the last layer', type=int, default=-1)
    parser.add_argument('-abspath', '--abspath', help='return abspath', type=bool, default=True)
    parser.add_argument('-repeat', '--repeat', help='repeat times', type=int, default=0)
    parser.add_argument('-o', '--out_name', help='out file name.', type=str, default="bandgap_Ef.csv")
    args = parser.parse_args()
    run(args, parser)
