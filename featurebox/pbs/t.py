# # -*- coding: utf-8 -*-
# from job_manager import JobManager
#
# # @Time  : 2022/12/6 16:32
# # @Author : boliqq07
# # @Software: PyCharm
# # @License: MIT License
#
# if __name__=="__main__":
#
#     jm = JobManager(manager="unischeduler")
#     path = ["/beegfs/home/wangchangxin/data/code_test/Mo2CO2_add_zero/Mo/H/S0/ini_static",
#                  "/beegfs/home/wangchangxin/data/code_test/Mo2CO2_add_zero/Ru/H/S0/ini_static",
#                  "/beegfs/home/wangchangxin/data/code_test/Mo2CO2_add_zero/Ru/H/S1/ini_static",
#                  "/beegfs/home/wangchangxin/data/code_test/Mo2CO2_add_zero/Ru/H/S2/ini_static"]
#
#
#     jobids = jm.submit_from_path(path=path, file="gjj_wang_cpu.run")
#     jm.dump_job_msg()
#
#     jm.dump_job_msg()
#     jm.job_dir()
#
#     print(jm.job_id())
#     print(jm.job_dir())
#
#     assert len(jm.job_id()) == len(jm.job_dir())
#
#     jm.get_job_msg()
#     k = [i for i in jm.msg.keys()]
#     l = len(jm.msg)
#     print("path",k[0])
#     jobids = jm.delete(k[0])
#     jjobss = jm.re_submit_from_path(old_id=k[0], file="gjj_wang_cpu.run")
#     print("r0",jobids,jjobss)
#
#     jm.get_job_msg()
#     k = [i for i in jm.msg.keys()]
#     l = len(jm.msg)
#     jobids = jm.delete(k[:2])
#     jjobss = jm.re_submit_from_path(old_id=k[:2], file="gjj_wang_cpu.run")
#
#     print("r1",jobids, jjobss)
#
#
#
#     jm.get_job_msg()
#     k = [i for i in jm.msg.keys()]
#     jobids1 = jm.hold(k[-2:])
#     jobids2 = jm.release(jobids1)
#
#     print("r2",jobids1, jobids2)
#
#
#
#     jm.get_job_msg()
#     jobids = jm.clear()
#     assert len(jobids) > 2
#
#
#
#     jm.get_job_msg()
#     # print(jm.sparse())
#
#     jm.load_job_msg()
#     # print(jm.sparse())
#
#     print(jm.job_id())
#
#
#
#     print(jm.job_dir())
