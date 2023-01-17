# -*- coding: utf-8 -*-

# @Time     : 2021/8/23 21:10
# @Software : PyCharm
# @License  : GNU General Public License v3.0
# @Author   : xxx

import os
from typing import Dict, Any, Tuple
from pymatgen.io.vasp.inputs import Incar


def pop_note(params: Dict[str, Any]) -> Tuple[str, Dict]:
    if params is None:
        return "", {}
    if "notes" in params:
        notes = params["notes"]
        params.pop("notes")
    else:
        notes = ""
    return notes, params


def _ISMEAR(notes: str, params: Dict = None) -> Dict:
    """"""
    _ = params
    if any([i in notes for i in
            ["半导体", "绝缘体", "分子", "原子", "molecule", "atom", "semiconductor", "nonconductor", "insulator"]]):
        _a = {"ISMEAR": 0}
    elif any([i in notes for i in ["金属", "metal"]]):
        _a = {"ISMEAR": 1}
    elif any([i in notes for i in ["精确能量", "DOS", "态密度", "能带", "bandgap", "band gap", "半导体"]]):
        _a = {"ISMEAR": -5}
    elif any([i in notes for i in ["K点少", "K点小", "少K点", "小K点"]]):
        _a = {"ISMEAR": 0}
    else:
        _a = {"ISMEAR": 0}
    return _a


def _SIGMA(notes: str, params: Dict = None) -> Dict:
    if "ISMEAR" in params and params["ISMEAR"] == -5:
        _a = {}
    if any([i in notes for i in ["分子", "原子", "molecule", "atom", ]]):
        _a = {"SIGMA": 0.01}
    elif any([i in notes for i in ["金属", "metal"]]):
        _a = {"SIGMA": 0.10}
    else:
        _a = {}
    return _a


def _ISPIN(notes: str, params: Dict = None) -> Dict:
    if any([i in notes for i in ["自旋", "spin"]]):
        _a = {"ISPIN": 1}

    else:
        _a = {}
    return _a


def _PREC(notes: str, params: Dict = None) -> Dict:
    if any([i in notes for i in ["高精确", "high"]]):
        _a = {"PREC": "medium"}
    elif any([i in notes for i in ["低精度", "low"]]):
        _a = {"PREC": "low"}
    else:
        _a = {}
    return _a


def _LDAU(notes: str, params: Dict = None) -> Dict:
    """
    LDAU=.T. 使用LDA+U功能
    LDAUL=-1/1/2/3 分别对应不加U和p、d、f轨道加U
    LDAUU、LDAUJ 分别设置U和J值"""

    if any([i in notes for i in ["加U", "+u", "+U"]]):
        _a = {"LDAU": True}
    else:
        _a = {}

    if any([i in notes for i in ["加Up", "+Up"]]):
        _a.update({"LDAUL": 1})
    elif any([i in notes for i in ["加Ud", "+Ud"]]):
        _a.update({"LDAUL": 2})
    elif any([i in notes for i in ["加Uf", "+Uf"]]):
        _a.update({"LDAUL": 3})

    return _a


def _MAGMOM(notes: str, params: Dict = None) -> Dict:
    """默认值是每个原子的磁矩均为1
    对于含有d、f轨道的原子，需根据原子核外电子排布情况设置相应的数值
    可以设置每个原子初始磁矩方向，正值为自旋向上，负值为自旋向下。"""
    return {}


def _IVDW(notes: str, params: Dict = None) -> Dict:
    """
    10：DFT-D2方法
    11：DFT-D3方法
    推荐首选更新的DFT-D3方法。
    哪些体系需要使用此参数呢？
    在计算表面吸附(物理吸附) ; 弱相互作用占体系能量比例较大的体系,如分子晶体、层状结构体系时,要使用此参数。"""

    if any([i in notes for i in ["分子晶体", "层状"]]):
        _a = {"IVDW": 11}
    return {}


######初始参数############

def _ISTART(notes: str, params: Dict = None) -> Dict:
    _ = params
    if any([i in notes for i in ["热启动", "restart", "DOS", "态密度", "能带", "bandgap", "band gap"]]):
        _a = {"ISTART": 1}
    else:
        _a = {"ISTART": 0}
    return _a


def _ICHARG(notes: str, params: Dict = None) -> Dict:
    if any([i in notes for i in ["DOS", "态密度", "能带", "bandgap", "band gap"]]):
        _a = {"ICHARG": 11}
    elif any([i in notes for i in ["热启动"]]):
        _a = {"ICHARG": 1}
    else:
        _a = {}
    return _a


######能带#######

def _LORBIT(notes: str, params: Dict = None) -> Dict:
    if any([i in notes for i in ["DOS", "态密度", "能带", "bandgap", "band gap"]]):
        _a = {"LORBIT": 10}
    else:
        _a = {}
    return _a


def _NBANDS(notes: str, params: Dict = None) -> Dict:
    if any([i in notes for i in ["DOS", "态密度", "能带", "bandgap", "band gap"]]):
        _a = {"LORBIT": 10}
    if any([i in notes for i in ["分波", "band centor"]]):
        _a = {"LORBIT": 11}
    else:
        _a = {}
    return _a


####离子弛豫#################

def _IBRION(notes: str, params: Dict = None) -> Dict:
    """
    0：分子动力学模拟
    1：准牛顿法
    2：共轭梯度法
    5：振动频率计算
    6：弹性常数计算
    Tips：推荐设置值为2，如果初始结构和最终稳定结构接近，可以设置为1。
    """
    if any([i in notes for i in ["弛豫", "CG"]]):
        _a = {"IBRION": 2}
    elif any([i in notes for i in ["静态", "固定原子", "能带"]]):
        _a = {"IBRION": -1}
    elif any([i in notes for i in ["AIMD", ]]):
        _a = {"IBRION": 0}
    else:
        _a = {}
    return _a


def _EDIFFG(notes: str, params: Dict = None) -> Dict:
    """正值为系统能量变化（单位为eV），负值为原子上残余力（单位为：eV/埃）。多数情况用力的收敛标准判断离子步弛豫是否收敛
    Tips：三维结构可以收敛到-0.01~ -0.03 eV/埃，低维体系可以收敛到- 0.03~0.05eV/埃。"""
    return {}


def _ISIF(notes: str, params: Dict = None) -> Dict:
    """
    2：为固定体积的弛豫，适用于如表面结构优化，微量掺杂体系优化
    3：全弛豫，适用于绝大多数的三维结构优化
    4：固定体积，但是形状可变的弛豫.
    Tips：复杂的结构优化过程，可以分步优化，即初始只优化离子（ISIF=2）→优化晶胞（ISIF=6）→优化离子和晶胞（ISIF=3）。"""
    if any([i in notes for i in ["静态"]]):
        _a = {"ISIF": 0}
    elif any([i in notes for i in ["只优化离子", "弛豫"]]):
        _a = {"ISIF": 2}
    elif any([i in notes for i in ["优化晶胞", ]]):
        _a = {"ISIF": 3}
    elif any([i in notes for i in ["固定体积，形状可变", ]]):
        _a = {"ISIF": 4}
    return {}


def _NSW(notes: str, params: Dict = None) -> Dict:
    """最大的离子步的数目 or 分子动力学模拟步数
    当IBRION=1和2时，NSW代表最大的离子步数目
    当IBRION=0时，NSW代表分子动力学模拟步数"""
    if any([i in notes for i in ["优化离子", "弛豫"]]):
        _a = {"NSW": 100}
    elif any([i in notes for i in ["固定体积，形状可变", ]]):
        _a = {"NSW": 100}
    else:
        _a = {"NSW": 0}
    return _a


def _BADER(notes: str, params: Dict = None) -> Dict:
    """bader """
    if any([i in notes for i in ["bader", "巴德"]]):
        _a = {"LAECHG": True,
              "LCHARG": 11,
              }
    else:
        _a = {}
    return _a


###电子优化部分##

def _ENCUT(notes: str, params: Dict = None) -> Dict:
    """截断能"""
    _a = {"ENCUT": 500}
    return _a


def _ALGO(notes: str, params: Dict = None) -> Dict:
    """设置自洽迭代过程优化电子波函数的算法
    N：DAV算法，收敛性好，但速度慢
    V：RMM算法，收敛性差，但速度快（比N快2~3倍）
    F：以上算法的结合，综合表现与V类似
    Tips：结构偏离稳定结构较多的，建议用N，接近稳定结构的，可以用V。"""
    return {}


def _NELM(notes: str, params: Dict = None) -> Dict:
    """自洽迭代循环的最大次数，通常设置为100"""
    return {}


def _EDIFF(notes: str, params: Dict = None) -> Dict:
    """自洽迭代循环收敛标准，单位为eV,自洽迭代过程紧邻两次迭代的系统能量的差与此标准相比。
    一般设置为1E-5~1E-6，意为两次能量差小于设定的收敛标准，则自洽迭代结束，判定体系收敛。"""
    return {}


def _auto_incar(params: Dict[str, Any] = None) -> Incar:
    # kk = grep 'entropy T'  OUTCAR
    # rkk = kk/n_atom<0.001ev
    """"""
    if params is None:
        params = {}
    notes, params = pop_note(params)
    params.update(_ISMEAR(notes, params))
    params.update(_SIGMA(notes, params))
    params.update(_ISTART(notes, params))
    params.update(_ICHARG(notes, params))
    params.update(_ISPIN(notes, params))
    params.update(_PREC(notes, params))
    params.update(_ENCUT(notes, params))
    params.update(_IBRION(notes, params))
    params.update(_ISIF(notes, params))
    params.update(_IVDW(notes, params))
    params.update(_NSW(notes, params))
    params.update(_LORBIT(notes, params))
    params.update(_NBANDS(notes, params))
    params.update(_BADER(notes, params))
    # if "SYSTEM" not in params:
    #     params.update({"SYSTEM": "test_write_time_{}".format(time.time())})

    incar = Incar(params)

    return incar


def auto_incar_file(notes="", params: Dict[str, Any] = None, path=None):
    """
    Get INCAR (VASP) file.

    Examples:
    >>> auto_incar_file(notes="静态计算")

    Args:
        notes: (str), description.
        params: (dict,None), Direct and explicit parameters for INCAR file.
        path: (str, Pathlike), store path, default work path.

    """
    if params is None:
        params = {}
    if notes == "":
        pass
    else:
        params.update({"notes": notes})
    incar = _auto_incar(params)
    if path is None:
        path = os.getcwd()
    path = os.path.abspath(path)
    path = os.path.join(path, 'INCAR')
    incar.write_file(path)
    print("Incar file is stored in {}.".format(path))


class CLICommand:
    """
    自动产生 INCAR 脚本（仅供参考）. (-n 内容不要留有空格).
    Example:

    $ featurebox autoincar -n 半导体静态计算 -p {"NSW":100}

    $ featurebox autoincar -n 金属弛豫计算

    $ featurebox autoincar -n 半导体静态计算
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-n', dest='notes', default="",
                            help='一段描述该体系的话。如：“半导体静态计算。金属弛豫计算。分子低精度静态计算” ')
        parser.add_argument('-p', dest='params', default={}, type=str,
                            help='明确的 Incar 参数，字典格式，如： {"NSW":100} ')
        parser.add_argument('-s', dest='site', default=None,
                            help='文件存放位置，默认当前位置')

    @staticmethod
    def run(args, parser):
        # args = args.parse_args()
        notes = args.notes
        params = args.params
        path = args.site

        params_ = {}

        if isinstance(params, str):

            if ":" in params:
                ttp = True
                params = params.replace("{", "")
                params = params.replace("}", "")
            else:
                ttp = False

            params = params.split(",")
            if ttp:

                params = [i.split(":") for i in params]
            else:
                params = [i.split("=") for i in params]

            params = [(i[0].replace(" ", ""), i[1]) for i in params]

            params_.update(params)
            assert isinstance(params_, dict)

        auto_incar_file(notes=notes, params=params_, path=path)


################################################################################################################
if __name__ == '__main__':
    # 命令行模式
    import argparse

    parser = argparse.ArgumentParser(description='自动产生 INCAR 脚本（仅供参考）。\n'
                                                 '最方便用法: \n'
                                                 'python autoincar.py -n 半导体静态计算 -p {"NSW":100} \n'
                                                 'python autoincar.py -n 金属弛豫计算 \n')
    parser.add_argument('-n', dest='notes', default="",
                        help='一段描述该体系的话。如：半导体静态计算。金属弛豫计算。分子低精度静态计算。')
    parser.add_argument('-p', dest='params', default={}, type=str,
                        help='明确的 Incar 参数，字典格式，如： {"NSW":100} ')
    parser.add_argument('-s', dest='site', default=None,
                        help='文件存放位置，默认当前位置')

    args = parser.parse_args()
    notes = args.notes
    params = args.params
    path = args.site

    params_ = {}

    if isinstance(params, str):

        if ":" in params:
            ttp = True
            params = params.replace("{", "")
            params = params.replace("}", "")
        else:
            ttp = False

        params = params.split(",")
        if ttp:

            params = [i.split(":") for i in params]
        else:
            params = [i.split("=") for i in params]

        params = [(i[0].replace(" ", ""), i[1]) for i in params]

        params_.update(params)
        assert isinstance(params_, dict)

    auto_incar_file(notes=notes, params=params_, path=path)
