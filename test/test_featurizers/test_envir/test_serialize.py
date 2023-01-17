# import unittest
#
# from monty.json import MSONable
#
# from featurebox.featurizers.envir.local_env import *
# from featurebox.utils.look_json import get_marked_class
#
#
# class AD(MSONable):
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b
#
#
# class TestGraph(unittest.TestCase):
#     def test_ser(self):
#         ad = AD(a=5, b=VoronoiNN_D())
#         #
#         # ad = serialize(VoronoiNN(cutoff=5.0))
#         afd = ad.as_dict()
#         afj = ad.to_json()
#         ad2 = ad.from_dict(afd)
#
#     def test_ser2(self):
#         a = get_marked_class(VoronoiNN, NNDict)
#         print(a.__class__)
#
#
# if __name__ == '__main__':
#     unittest.main()
