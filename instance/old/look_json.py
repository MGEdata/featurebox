# """This part is used to add as_json and to_json method for Externally class.
# In default, for one case, we add by inherit monty.json.MSONable.
# but for large mount cases, it rewrite a lot of things.
# This is a dynamic approach, For a small amount of work, we don't recommend it.
#
# Examples
# ---------
# #In definition code.
#
# >>> NNDict = mark_classes([VoronoiNN,CutOffDictNN])
# >>> for i, j in NNDict.items():
# ...    locals()[i] = j
#
# # Usage code
#
# >>> vor = get_marked_class("VoronoiNN", NNDict)
#
# The new class are marked with "_D"
# """
#
# import json
# import os
# from inspect import getfullargspec
# from typing import List, Dict
#
# from monty.json import MontyDecoder, _load_redirect
#
# REDIRECT = _load_redirect(
#     os.path.join(os.path.expanduser("~"), ".monty.yaml"))
#
#
# def as_dict(self) -> dict:
#     """
#     A JSON serializable dict representation of an object.
#     """
#     d = {"@module": self.__class__.__module__, "@class": self.__class__.__name__, "@version": None}
#
#     dk = {}
#     spec = getfullargspec(self.__init__)
#     args = spec.args
#
#     def recursive_as_dict(obj):
#         if isinstance(obj, (list, tuple)):
#             return [recursive_as_dict(it) for it in obj]
#         if isinstance(obj, dict):
#             return {kk: recursive_as_dict(vv) for kk, vv in obj.items()}
#         if hasattr(obj, "as_dict"):
#             return obj.as_dict()
#         return obj
#
#     for c in args:
#         if c != "self":
#             try:
#                 a = self.__getattribute__(c)
#             except AttributeError:
#                 try:
#                     a = self.__getattribute__("_" + c)
#                 except AttributeError:
#                     raise NotImplementedError(
#                         "Unable to automatically determine as_dict "
#                         "format from class. MSONAble requires all "
#                         "args to be present as either self.argname or "
#                         "self._argname, and kwargs to be present under"
#                         "a self.kwargs variable to automatically "
#                         "determine the dict format. Alternatively, "
#                         "you can implement both as_dict and from_dict.")
#             dk[c] = recursive_as_dict(a)
#     if hasattr(self, "kwargs"):
#         # type: ignore
#         dk.update(**getattr(self, "kwargs"))  # pylint: disable=E1101
#     if hasattr(self, "_kwargs"):
#         dk.update(**getattr(self, "_kwargs"))  # pylint: disable=E1101
#
#     d["all_args_kwargs"] = dk
#     d["tp_na"] = self.__class__.__name__
#     d["tp_mo"] = self.__class__.__module__
#     return d
#
#
# @classmethod
# def _from_dict(cls, d):
#     modname = d["tp_mo"]
#     classname = d["tp_na"]
#
#     decoded = {
#         k: MontyDecoder().process_decoded(v)
#         for k, v in d.items() if not k.startswith("@")
#     }
#
#     mod = __import__(modname, globals(), locals(), [classname], 0)
#     if hasattr(mod, classname):
#         cla = getattr(mod, classname)
#
#         if decoded["all_args_kwargs"] == {}:
#             return cla()
#         else:
#             return cla(**decoded["all_args_kwargs"])
#     else:
#         try:
#             return cls()
#         except BaseException:
#             raise TypeError("Cant find {} in {}, Please import it first.".format(classname, modname),
#                             "NNDict = mark_classes([VoronoiNN,])"
#                             "for i, j in NNDict.items():\n"
#                             "    locals()[i] = j")
#
#
# def to_json(self) -> str:
#     """
#     Returns a json string representation of the MSONable object.
#     """
#     return json.dumps(self, cls=UserMontyEncoder)
#
#
# class UserMontyEncoder(json.JSONEncoder):
#     """
#     A Json Encoder which supports the MSONable API, plus adds support for
#     numpy arrays, datetime objects, bson ObjectIds (requires bson).
#
#     Usage::
#
#         # Add it as a *cls* keyword when using json.dump
#         json.dumps(object, cls=MontyEncoder)
#     """
#
#     def default(self, o) -> dict:  # pylint: disable=E0202
#         try:
#             d = o.as_dict()
#             if "@module" not in d:
#                 d["@module"] = u"{}".format(o.__class__.__module__)
#             if "@class" not in d:
#                 d["@class"] = u"{}".format(o.__class__.__name__)
#             if "@version" not in d:
#                 d["@version"] = None
#             return d
#         except AttributeError:
#             raise TypeError
#
#

#
# def mark_classes(classes: List):
#     """
#     Batch add as_json and to_json method for Externally class.
#
#     NNDict = mark_classes([VoronoiNN,CutOffDictNN])
#
#     for i, j in NNDict.items():
#         locals()[i] = j
#
#
#     Parameters
#     ----------
#     classes:List of class.
#         not object.
#
#     Returns
#     -------
#     NNdict:
#         key is the class name with append "_D", and values is the new class with `as_json` method.
#     """
#
#     NNDict = {i.__name__ + "_D": type(i.__name__ + "_D", (i,), {"as_dict": as_dict, "to_json": to_json,
#                                                                 "from_dict": _from_dict, "tp_na": True})
#               for i in classes}
#
#     return NNDict
#
#
# def get_marked_class(nn_strategy, env_dict: Dict = None, instantiation: bool = True):
#     """
#     Just call values in NNict by,consider multiple cases at the same time.
#     ["VoronoiNN",
#     VoronoiNN,
#     VoronoiNN(),
#     "VoronoiNN_D",
#     VoronoiNN_D,
#     VoronoiNN_D(),]
#     All map VoronoiNN_D() or VoronoiNN_D.
#
#     Examples
#     ---------
#     # In definition code.
#     #
#     NNDict = mark_classes([VoronoiNN,CutOffDictNN])
#
#     for i, j in NNDict.items():
#         locals()[i] = j
#
#     # Usage code
#     #
#     vor = get_marked_class("VoronoiNN", NNDict)
#
#     Parameters
#     ----------
#     nn_strategy
#         str or class in NNDict.
#     env_dict:dict
#         dict of pre-definition, {"classname_D": class}.
#     instantiation:bool
#         return class of object.
#
#     Returns
#     -------
#     obj:
#         object or class in NNDict.
#
#     """
#     try:
#         ######old type for compatibility ####
#         if nn_strategy is None:
#             return nn_strategy
#         elif isinstance(nn_strategy, str) and nn_strategy in ["find_points_in_spheres", "find_xyz_in_spheres"]:
#             return nn_strategy
#         elif isinstance(nn_strategy, (float, int)):
#             return nn_strategy
#
#         ############restore##################
#
#         elif isinstance(nn_strategy, dict):
#             return env_dict[nn_strategy["tp_na"]](**nn_strategy['all_args_kwargs'])
#
#         ############by str name##############
#         elif isinstance(nn_strategy, str):
#             if "_D" in nn_strategy:
#                 Nei = env_dict[nn_strategy]()
#             else:
#                 Nei = env_dict[nn_strategy + "_D"]()
#         else:
#             try:
#                 nn_strategy = nn_strategy()
#             except TypeError:
#                 pass
#
#             if nn_strategy.__class__.__name__ in env_dict:
#                 Nei = nn_strategy
#             else:
#                 if nn_strategy.__class__.__name__ + "_D" in env_dict:
#                     Nei = env_dict[nn_strategy.__class__.__name__ + "_D"]()
#                     Nei.__dict__.update(nn_strategy.__dict__)
#                 else:
#                     raise TypeError("only accept str or object inherit from nn_dict.values()")
#         if instantiation:
#             return Nei
#         else:
#             return Nei.__class__
#     except (KeyError, TypeError):
#         raise TypeError("only accept str or object inherit from nn_dict.values()")
#
# # if __name__ == '__main__':
# #     class AD(MSONable):
# #         def __init__(self, a, b):
# #             self.a = a
# #             self.b = b
# #
# #
# #     NNDict = mark_classes([VoronoiNN])
# #     # for i, j in NNDict.items():
# #     #     locals()[i] = j
# #     vor = get_marked_class(VoronoiNN(), NNDict)
# #
# #     # ad = AD(a=5, b=vor)
# #     ad = vor
# #
# #     afd = ad.as_dict()
# #     afj = ad.to_json()
# #     ad2 = ad.from_dict(afd)
