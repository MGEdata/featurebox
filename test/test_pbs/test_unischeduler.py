#!/bin/bash
import unittest

from featurebox.pbs.job_manager import JobManager


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        if not hasattr(self, "jm"):
            self.jm = JobManager(manager="unischeduler")
            self.path = ["/beegfs/home/wangchangxin/data/code_test/Mo2CO2_add_zero/Mo/H/S0/ini_static",
            "/beegfs/home/wangchangxin/data/code_test/Mo2CO2_add_zero/Ru/H/S0/ini_static",
            "/beegfs/home/wangchangxin/data/code_test/Mo2CO2_add_zero/Ru/H/S1/ini_static",
            "/beegfs/home/wangchangxin/data/code_test/Mo2CO2_add_zero/Ru/H/S2/ini_static"]

    def test_a_submit_from_path(self):
        jobids = self.jm.submit_from_path(path=self.path, file="gjj_wang_cpu.run")
        self.jm.dump_job_msg()
        self.assertGreater(len(jobids), 0)
        self.assertGreater(len(self.jm.msg), 0)

    def test_b_delete_and_re_submit(self):
        self.jm.get_job_msg()
        k = [i for i in self.jm.msg.keys()]
        l = len(self.jm.msg)
        jobids = self.jm.delete(k[0])
        jjobss = self.jm.re_submit_from_path(old_id=k[0], file="gjj_wang_cpu.run")
        self.assertEqual(len(self.jm.msg), l)
        self.assertGreater(len(self.jm.deleted_msg), 0)
        self.assertNotEqual(jobids, jjobss)

    def test_c_delete_and_re_submit2(self):
        self.jm.get_job_msg()
        k = [i for i in self.jm.msg.keys()]
        l = len(self.jm.msg)
        jobids = self.jm.delete(k[:2])
        jjobss = self.jm.re_submit_from_path(old_id=k[:2], file="gjj_wang_cpu.run")
        self.assertEqual(len(self.jm.msg), l)
        self.assertGreater(len(self.jm.deleted_msg), 0)
        self.assertNotEqual(jobids, jjobss)
        print(jobids,jjobss)

    def test_d_hold_release(self):
        self.jm.get_job_msg()
        k = [i for i in self.jm.msg.keys()]
        jobids1 = self.jm.hold(k[-2:])
        jobids2 = self.jm.release(jobids1)
        self.assertEqual(jobids1, jobids2)
        print(jobids1, jobids2)

    def test_e_clear(self):
        self.jm.get_job_msg()
        jobids = self.jm.clear()
        assert len(jobids) > 2

    def test_f_log(self):
        self.jm.get_job_msg()
        print(self.jm.sparse())

    def test_g_load(self):
        self.jm.load_job_msg()
        print(self.jm.sparse())

    def test_h_id(self):
        print(self.jm.job_id())

    def test_i_dir(self):
        print(self.jm.job_dir())

if __name__ == '__main__':
    unittest.main()
