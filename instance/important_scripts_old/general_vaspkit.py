import os
from typing import Callable
import numpy as np
import pandas as pd


def cal(d, read: Callable, cmds=("ls", "pwd"), store=False, store_name="temp.csv", run_cmd=True):
    """Run linux cmd and return result, make sure the vaspkit is installed."""
    old = os.getcwd()
    os.chdir(d)
    try:
        #
        if run_cmd:
            cmd_sys(cmds)
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


def cal_all(d, read: Callable, cmds=("ls", "pwd"), repeat=0, store=False, store_name="temp_all.csv", run_cmd=True):
    data_all = []
    col = None
    for di in d:
        res = cal(di, read=read, store=False, cmds=cmds, run_cmd=run_cmd)

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

    return data_all


def cmd_sys(cmds=("vaspkit -task 911",)):
    if not cmds:
        pass
    else:
        for i in cmds:
            os.system(i)
