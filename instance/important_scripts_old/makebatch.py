# -*- coding: utf-8 -*-

# @Time     : 2021/10/8 15:23
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx
import os
import pathlib


def trans_str(string):
    lines = string.split("\\n")
    print(lines)
    lines = "\n".join(lines)
    return lines


def make_batch_from_file(path_file, cmd="", out_file_name="batch.sh",enter=True):
    if not enter:
        batch_str = """#!/bin/bash

for i in $(cat {})

do
echo $i

{}

done
        """.format(path_file, trans_str(cmd))
    else:
        batch_str = """#!/bin/bash

old_path=$(cd "$(dirname "$0")"; pwd)

for i in $(cat {})

do
cd $i
echo $(cd "$(dirname "$0")"; pwd)

{}

cd $old_path
done
        """.format(path_file, trans_str(cmd))

    batch_str = batch_str.replace("'", "")
    batch_str = batch_str.replace("[", "")
    batch_str = batch_str.replace("]", "")
    batch_str = batch_str.replace(",", "")

    path = pathlib.Path(path_file).parent
    os.chdir(path)
    bach = open(out_file_name, "w")
    bach.write(batch_str)
    bach.close()
    print("The batch file '{}' is stored in '{}'".format(out_file_name, os.getcwd()))


class CLICommand:
    """
    根据路径文件，创建循环命令， 查看帮助使用 -h。
    可以配合 mgetool findpath 使用，先 findpath 查找路径，再 makebatch 创建循环脚本。
    复杂sh脚本也可以使用该脚本创建，再手动修改具体操作命令。

    如果在 mgetool 中运行:

    Example:

        $ mgetool makebatch -f paths.temp

        $ mgetool makebatch -cmd 'jsub < $(find -name *.run)'

    此处为单引号！

    如果复制该脚本到某处，仅运行单个脚本:

    Example:

        $ python makebatch.py -f paths.temp

        $ python makebatch.py  -cmd 'cd $i \necho i \ncd ..'

    cmd 命令用单引号。换行使用\n,并且其后不要留有空格。(特殊字符 $ -cmd 仅能在脚本中使用!!!)
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-f', '--path_file', help='source path file', type=str, default="paths.temp")
        parser.add_argument('-cmd', '--commands', help='commands', type=str, default="")
        parser.add_argument('-enter', '--enter', help='enter the disk', type=bool, default=True)
        parser.add_argument('-o', '--store_name', help='out file name', type=str, default="batch.sh")

    @staticmethod
    def run(args, parser):
        make_batch_from_file(args.path_file, args.commands, args.store_name, args.enter)


if __name__ == '__main__':
    # 命令行模式
    import argparse

    parser = argparse.ArgumentParser(description="根据路径文件，创建循环命令："
                                                 "python makebatch.py -cmd 'cd $i \necho i \ncd ..'  "
                                                 "cmd 命令用单引号。换行使用\n,并且其后不要留有空格。")
    parser.add_argument('-f', '--path_file', help='source path file', type=str, default="paths.temp")
    parser.add_argument('-cmd', '--commands', help='commands', type=str, default="")
    parser.add_argument('-enter', '--enter', help='enter the disk', type=bool, default=True)
    parser.add_argument('-o', '--store_name', help='out file name', type=str, default="batch.sh")
    args = parser.parse_args()

    make_batch_from_file(args.path_file, args.commands, args.store_name, args.enter)

