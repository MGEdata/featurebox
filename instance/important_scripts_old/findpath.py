# -*- coding: utf-8 -*-

# @Time     : 2021/10/8 14:29
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import os

from mgetool.imports import batchfile


def run(args, parser):
    # args = args.parse_args()
    if not os.path.isabs(args.path):
        args.path = os.path.abspath(args.path)
    bf = BatchFile(args.path, suffix=args.suffix)
    if " " in args.layer:
        layer = str(args.layer).split(" ")
        layer = [int(i) for i in layer]
        if len(layer)==1:
            layer = layer[0]
    else:
        layer = int(args.layer)

    if args.dir_include is not None and " " in args.dir_include:
        dir_include = str(args.dir_include).split(" ")
    else:
        dir_include = args.dir_include

    if args.dir_exclude is not None and " " in args.dir_exclude:
        dir_exclude = str(args.dir_exclude).split(" ")
    else:
        dir_exclude = args.dir_exclude

    if isinstance(layer, int):
        bf.filter_dir_name(include=dir_include, exclude=dir_exclude, layer=layer)
    else:
        if dir_include is not None and len(dir_include)==len(layer):
            for di,la in zip(dir_include, layer):
                if di !="-":
                    bf.filter_dir_name(include=di, layer=la)
        else:
            bf.filter_dir_name(include=dir_include, layer=layer)

        if dir_exclude is not None and len(dir_exclude)==len(layer):
            for di,la in zip(dir_exclude, layer):
                if di !="-":
                    bf.filter_dir_name(exclude=di, layer=la)
        else:
            bf.filter_dir_name(exclude=dir_exclude, layer=layer)

    bf.filter_file_name(include=args.file_include, exclude=args.file_exclude)
    bf.merge()

    fdir = bf.file_dir
    fdir.sort()
    os.chdir(args.path)
    if not args.abspath:
        absp = os.path.abspath(args.path)
        fdir = [i.replace(absp, ".") for i in fdir]
    with open(str(args.store_name), mode="w") as f:
        fdir = "\n".join(fdir)
        f.writelines(fdir)

    print("The paths '{}' are stored in '{}'.".format(args.store_name, str(os.getcwd())))
    print("OK")


class CLICommand:
    """
    寻找符合要求所有的子路径, 查看参数帮助使用 -h。

    如果在 mgetool 中运行:

    Example:

        $ mgetool findpath -p /home/dir_name -if POSCAR

        $ mgetool findpath  -id my_dir1 -ed my_dir2

    指定每层条件，使用单引号，保证每层个数一致，使用空格分割，使用 - 代表跳过。
    如下式代表最后一层需要是 ini_opt ，最后第三层不能是 Na .

        mgetool findpath -id '- - ini_opt' -ed 'Na - -' -l '-3 -2 -1'

    如果复制该脚本到某处，仅运行单个脚本:

    Example:

        $ python findpath.py -p /home/dir_name -if POSCAR
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-p', '--path', help='source path', type=str, default=".")
        parser.add_argument('-s', '--suffix', help='suffix of file', type=str, default=None)
        parser.add_argument('-if', '--file_include', help='include file name.', type=str, default=None)
        parser.add_argument('-ef', '--file_exclude', help='exclude file name.', type=str, default=None)
        parser.add_argument('-id', '--dir_include', help='include dir name.', type=str, default=None)
        parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.', type=str, default=None)
        parser.add_argument('-l', '--layer', help='dir depth,default the last layer', type=str, default="-1")
        parser.add_argument('-abspath', '--abspath', help='return abspath', type=bool, default=False)
        parser.add_argument('-o', '--store_name', help='out file name, default paths.temp', type=str,
                            default="paths.temp")

    @staticmethod
    def run(args, parser):
        run(args, parser)


if __name__ == '__main__':
    # 命令行模式
    import argparse

    parser = argparse.ArgumentParser(description="根据路径文件，寻找所有符合要求的文件子路径："
                                                 "python findpath.py -p /home/dir_name -if POSCAR"
                                     )

    parser.add_argument('-p', '--path', help='source path', type=str, default=".")
    parser.add_argument('-s', '--suffix', help='suffix of file', type=str, default=None)
    parser.add_argument('-if', '--file_include', help='include file name.', type=str, default=None)
    parser.add_argument('-ef', '--file_exclude', help='exclude file name.', type=str, default=None)
    parser.add_argument('-id', '--dir_include', help='include dir name.', type=str, default=None)
    parser.add_argument('-ed', '--dir_exclude', help='exclude dir name.', type=str, default=None)
    parser.add_argument('-l', '--layer', help='dir depth,default the last layer', type=str, default="-1")
    parser.add_argument('-abspath', '--abspath', help='return abspath', type=bool, default=False)
    parser.add_argument('-o', '--store_name', help='out file name,default paths.temp', type=str, default="paths.temp")
    args = parser.parse_args()
    run(args, parser)
