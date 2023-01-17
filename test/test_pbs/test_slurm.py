import unittest

from featurebox.pbs.job_manager import JobManager


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        if not hasattr(self, "jm"):
            self.jm = JobManager("slurm")
            self.path = ["/data/home/qian1/wcx/MXenes/Ti2CO2/no_add/Au/Al/test_z_16/relax_o_0",
                        "/data/home/qian1/wcx/MXenes/Ti2CO2/no_add/Au/Al/test_z_16/relax_o_10"]

    def test_a_submit_from_path(self):
        jobids = self.jm.submit_from_path(path=self.path, file="*wcx.run")
        self.jm.get_job_msg()
        self.jm.dump_job_msg()
        self.jm.job_dir()
        self.assertGreater(len(jobids), 0)
        self.assertGreater(len(self.jm.msg), 0)
        assert len(self.jm.msg) == len(self.jm.job_dir())

    def test_b_delete_and_re_submit(self):
        self.jm.get_job_msg()
        k = [i for i in self.jm.msg.keys()]
        l = len(self.jm.msg)
        jobids = self.jm.delete(k[-1])
        jjobss = self.jm.re_submit_from_path(old_id=k[-1], file="*wcx.run")
        self.assertEqual(len(self.jm.msg), l)
        self.assertGreater(len(self.jm.deleted_msg), 0)
        self.assertNotEqual(jobids, jjobss)
        print(jobids, jjobss)

    def test_c_delete_and_re_submit2(self):
        self.jm.get_job_msg()
        k = [i for i in self.jm.msg.keys()]
        l = len(self.jm.msg)
        jobids = self.jm.delete(k[-2:])
        jjobss = self.jm.re_submit_from_path(old_id=k[-2:], file="*wcx.run")
        self.assertEqual(len(self.jm.msg), l)
        self.assertGreater(len(self.jm.deleted_msg), 0)
        self.assertNotEqual(jobids, jjobss)
        print(jobids,jjobss)

    def test_d_hold_release(self):
        self.jm.get_job_msg()
        k = [i for i in self.jm.msg.keys()]
        jobids1 = self.jm.hold(k[-2:])
        jobids2 = self.jm.release(jobids1)
        print(jobids1, jobids2)
        self.assertEqual(jobids1, jobids2)

    def test_h_log(self):
        self.jm.get_job_msg()
        print(self.jm.sparse())

    def test_i_load(self):
        self.jm.load_job_msg()
        print(self.jm.sparse())

    def test_j_id(self):
        print(self.jm.job_id())

    def test_k_dir(self):
        print(self.jm.job_dir())

if __name__ == '__main__':
    unittest.main()
