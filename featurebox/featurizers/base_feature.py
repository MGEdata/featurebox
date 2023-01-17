"""Base"""
import warnings
from itertools import chain
from multiprocessing import cpu_count
from typing import List, Tuple, Iterable, Any

import numpy as np
import pandas as pd
from mgetool.tool import parallelize


class BaseFeature:
    """
    **Using a BaseFeature Class**

    That means you can embed this feature directly into ``BaseFeature`` class implement.
    ::

        class MatFeature(BaseFeature):
            def convert(spath, *x):
                ...

    ``BaseFeature`` implement :class:`sklearn.base.BaseEstimator` and :class:`sklearn.base.TransformerMixin`
    that means you can use it in a scikit-learn way.
    ::

        feature = SomeFeature()
        features = feature.fit_transform(X)

    .. note::

        The ``convert`` method should be rewrite to deal with single case. And the ``transform`` and ``fit_transform``
        will be established for list of case automatically.


    **Adding references**

    ``BaseFeature`` also provide you to retrieving proper references for a feature.
    The ``__citations__`` returns a list of papers that should be cited.
    The ``__authors__`` returns a list of people who wrote the feature.
    Also can be accessed from property ``citations`` and ``citations``.

    These operations must be implemented for each new feature:

    - ``feature_labels`` - Generates a human-meaningful x_name for each of the features. Implement this as property.

    which can be set by ``set_feature_labels``

    Also suggest to implement these two properties:

    - ``citations`` - Returns a list of citations in BibTeX format.
    - ``authors`` - Returns a list of people who contributed writing a paper.

    .. note::

        None of these operations should change the state of the feature. I.e.,
        running each method twice should no produce different results, no class
        attributes should be changed, Running one operation should not affect the
        output of another.

    """

    __authors__ = ['boliqq07']
    __citations__ = ['No citations']
    _n_jobs = 1
    _feature_labels = []

    def __init__(self, n_jobs: int = 1, *, on_errors: str = 'raise', return_type: str = 'any',
                 batch_calculate: bool = False,
                 batch_size: int = 30):
        """
        Parameters
        ----------
        batch_size: int
            size of batch.
        batch_calculate :bool
            batch_calculate or not.
        n_jobs: int
            Parallel number.
        on_errors: str
            How to handle the exceptions in a feature calculations. Can be ``nan``, ``keep``, ``raise``.
            When 'nan', return a column with np.nan. The length of column corresponding to the number of feature
            labs.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``np``,``array`` and ``df``.
            'array' and 'df' force return type to np.ndarray and pd.DataFrame respectively.
            If 'any', without type conversion .
            Default is 'any'
        """
        self.return_type = return_type
        self.n_jobs = n_jobs
        self.on_errors = on_errors
        self._kwargs = {"x_labels": None, "feature_labels_mark": None}
        self.support_ = []
        self.ndim = None
        self._feature_labels = []
        self.batch_calculate = batch_calculate
        self.batch_size = batch_size
        # import inspect
        # inspect.getfullargspec(self.convert)

    @property
    def n_jobs(self):
        """
        n_jobs: int
            Parallel number.
        """
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs: int):
        """Set the number of threads for this"""
        if n_jobs > cpu_count() or n_jobs == -1:
            self._n_jobs = cpu_count()
        else:
            self._n_jobs = n_jobs

    def fit(self, *args, **kwargs):
        """fit function in :class:`BaseFeature` are weakened and just pass parameter."""
        _ = args
        if kwargs is {}:
            pass
        else:
            self._kwargs.update(kwargs)
        return self

    def fit_transform(self, X: List, y=None, **kwargs) -> Any:
        """
        If `convert` takes multiple inputs, supply inputs as a list of tuples.

        Copy from Mixin class for all transformers in scikit-learn. TransformerMixin

        Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X: list
            list of case.
        y: None
            deprecated.
        **kwargs:
            Additional fit or transform parameters.
            feature_labels_mark: str, mark for each feature_labes. for return_type =='pd'.
            x_labels: list, mark for each row. for return_type =='pd'.

        Returns
        --------
        X_new:
            result data.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **kwargs).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **kwargs).transform(X)

    def transform_with_zip(self, *args) -> Any:
        """
        Second transform, which convert Iterables to list and run transform.

        first:
        p1s,p2s -> [(p1,p2),(p1,p2),(p1,p2),...,(p1,p2),(p1,p2)]\n
        second:
        run self.transform

        Parameters
        ----------
        args:Iterable
            each of args must be Iterable.

        Returns
        ---------
        result: any
            features for each entry.

        """
        return self.transform(list(zip(*args)))

    def transform(self, entries: List) -> Any:
        """
        Transform a list of entries. Each iterable element of entries is corresponding to the parameter of ``convert``,
        If ``convert`` takes n multiple inputs, the transform inputs should be a list or tuple (size n),\n
        [(p1,p2),(p1,p2),(p1,p2),...,(p1,p2),(p1,p2)]\n
        which can be from `zip`` or used the built-in ``transform_with_zip``.

        Parameters
        ----------
        entries: list
            A list of entries to be featured.

        Returns
        ---------
        result: any
            features for each entry.
        """

        # Check inputs
        if not isinstance(entries, Iterable):
            raise TypeError('Parameter "entries" must be a iterable object')

        # Special case: Empty list
        if hasattr(entries, "__len__") and len(entries) == 0:
            return []

        # Run the actual feature
        rets = parallelize(self.n_jobs, self._wrapper, entries, tq=True)

        ret, self.support_ = zip(*rets)

        if self.return_type == 'any':
            return ret

        if self.return_type == 'array' or self.return_type == 'np':
            return np.array(ret)

        if self.return_type == 'df' or self.return_type == 'pd':
            try:
                labels_len = len(self.feature_labels)
                if labels_len > 0:
                    labels = self.feature_labels
                else:
                    labels = None
            except (NotImplementedError, TypeError):
                labels = None

            return pd.DataFrame(ret, columns=labels, index=self._kwargs["x_labels"])

    def _wrapper(self, *args, **kwargs):
        """
        An exception wrapper for convert, used in transform and
        changes the parameter passed to convert, and return the result with an bool mark.

        Notes
        -----
        The _wrapper is only called for ``transform`` for batch data, If your want implement a specific function,
        you could just use ``convert`` and loop with ``for``.


        Parameters
        ----------
        args:
            input0 data_cluster to feature (type depends on feature).

        Returns
        -------
        result: tuple
            The first is calculated result, and the second is bool to mark data availability.
        """

        try:
            try:
                con = self.convert(*args, **kwargs)
            except TypeError as e:
                print(e)
                raise TypeError("Please check the above errors")
                # raise TypeError("Please check the above errors, If there is an un-understood error, "
                #                 "please make sure the ``tuple`` type parameter would be separate automatically,"
                #                 "if you each case of data is ``tuple`` now, and want pass it to the first argument,"
                #                 "please change it to list, turn to ``_wrapper`` for more information\n"
                #                 )

            if isinstance(con, (List, Tuple)):
                if len(con) == 2 and isinstance(con[1], bool):
                    pass
                else:
                    con = (con, True)
            else:
                con = (con, True)
            return con

        except BaseException as e:
            if self.on_errors == "nan":

                warnings.warn(e.__str__())
                print("Bad conversion for:", args)
                return np.nan, False
            elif self.on_errors == "raise":
                raise e

    def convert(self, d):
        """
        Main feature function, which has to be implemented
        in any derived feature subclass.

        Notes
        -----
        It cannot be passed np.array in default unless:

        1. useful for bond_converter.
        For np.array we check the ndim and for ndim 2, or 3.
        we decide whether to pass them the data to ``_converter``
        together or separately by ``self.ndim`` attribute. Now max support 3d.
        due to for some functions, using ``ufunc`` in numpy is very efficient.

        2. keep the size of data and simple the ``_convert``.

        Parameters
        -----------
        d:
            one input data (one sample, one case),

        Returns
        -------
        new_x:
            new x.
        """
        try:
            if isinstance(d, np.ndarray):
                now_dim = d.ndim
                self_dim = now_dim if self.ndim is None else self.ndim
                if now_dim == self_dim:
                    return np.array(self._convert(d))
                elif now_dim - self_dim == 1:
                    return np.array([self._convert(i) for i in d])
                elif now_dim - self_dim == 2:
                    return np.array([[self._convert(i) for i in di] for di in d])
                elif now_dim - self_dim == 3:
                    return np.array([[[self._convert(i) for i in dii] for dii in di] for di in d])
                else:
                    warnings.warn("The attribute `ndim` is {}, but the input data shape is {}, "
                                  "which could be cause an error".format(self_dim, now_dim), UserWarning)
                    return np.array(self._convert(d))
            else:
                return np.array(self._convert(d))
        except BaseException as e:
            print(e)
            raise ValueError("Error when try to convert: \n {}".format(str(d)))

    @property
    def feature_labels(self):
        """Generate attribute names.

        Returns:
            ([str]) attribute labels.
        """
        mark = self._kwargs["feature_labels_mark"]
        if mark is None:
            return self._feature_labels
        else:
            return ["{}_{}".format(i, mark) for i in self._feature_labels]

    def set_feature_labels(self, values: List[str]):
        """Generate attribute names.

        Returns:
            ([str]) attribute labels.
        """
        self._feature_labels = values

    @property
    def citations(self):
        """Citation(s) and reference(s) for this feature.

        Returns:
            (list) each element should be a string citation,
                ideally in BibTeX format.
        """
        return '\n'.join(self.__citations__)

    @property
    def authors(self):
        """List of implementors of the feature.

        Returns:
            (list) each element should either be a string with author x_name (e.g.,
                "Anubhav Jain") or a dictionary  with required key "x_name" and other
                keys like "email" or "institution" (e.g., {"x_name": "Anubhav
                Jain", "email": "ajain@lbl.gov", "institution": "LBNL"}).
        """

        return '\n'.join(self.__authors__)

    def _convert(self, d):
        """ input"""
        return d

    def __add__(self, other):
        raise TypeError("This method has no add")


class DummyConverter(BaseFeature):
    """
    Dummy converter as a placeholder, Do nothing.
    """

    def convert(self, d) -> np.ndarray:
        """
        Dummy convert, does nothing to input.

        Args:
            d (Any): input object

        Returns: d

        """
        d = np.array(d)

        return d.reshape(-1, 1) if d.ndim == 1 else d

    def _convert(self, d: np.ndarray) -> np.ndarray:
        pass


class ConverterCat(BaseFeature):
    """Pack the converters in to one unified approach.
    The same type Converter would merge and different would order to run.
    Thus, keeping the same type is next to each other! such as A(),A(),B(),B().

    Examples
    ----------

    >>> tmps = ConverterCat(
    ...    AtomEmbeddingMap(),
    ...    AtomEmbeddingMap("ie.json")
    ...    AtomTableMap(search_tp="name"))
    >>> tmp.convert(x)
    >>> tmp.tranmform(xs)

    """

    def __init__(self, *args: BaseFeature, force_concatenate=False,
                 n_jobs: int = 1, on_errors: str = 'raise', return_type: str = 'any'):
        """

        Parameters
        ----------
        args: Converter
            List of Converter
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        return_name = set([getattr(i, "return_type") for i in args])
        assert len(return_name) == 1, "The return_type for each Converter must be same," \
                                      " but there are {}".format(list(return_name))

        if self.return_type in ["df", "pd", "np", "array"]:
            assert force_concatenate is True, "force_concatenate=True is need to transform to table data."

        self.args = self.sums(list(args))
        self._feature_labels = list(chain(*[i.feature_labels for i in self.args]))
        self.force_concatenate = force_concatenate

    @staticmethod
    def sums(args):
        """SUM"""
        i = 0
        while i < len(args) - 1:
            try:
                args[i] = args[i] + args[i + 1]
                del args[i + 1]
            except (AssertionError, TypeError, IndexError, NotImplementedError):
                i += 1
        return args

    def convert(self, d):
        """convert and concatenate."""
        data = []
        for ci in self.args:
            data.append(ci.convert(d))
        if len(self.args) == 1:
            raw_data = data[0]
        else:
            if not self.force_concatenate:
                try:
                    raw_data = np.concatenate(data, axis=-1)
                except (ValueError, IndexError):
                    raw_data = data
            else:
                try:
                    raw_data = np.concatenate(data, axis=-1)
                except (ValueError, IndexError) as e:
                    tys = [ar.__class__.__name__ for ar in self.args]
                    print("Fall to concentrate the data from {}".format(str(tys)),
                          "The data are can not transformed to return_type={} data,".format(self.return_type),
                          "please set ``force_concatenate``= False, and concatenate them in your code manually."
                          )
                    raise e
        return raw_data


class ConverterSequence(BaseFeature):
    """Pack the converters in to one sequentially executed assembly approach.

    input -> convert1 -> temp -> convert2 -> temp -> convert3 -> output

    Notes
    -----
    There is no error checking, please make sure the ``temp`` could be passed manually !!!
    There is no error checking, please make sure the ``temp`` could be passed manually !!!
    There is no error checking, please make sure the ``temp`` could be passed manually !!!

    Examples
    ----------

    >>> tmps = ConverterCat(
    ...    AtomEmbeddingMap(),
    ...    DummyConverter()
    >>> tmp.convert(x)

    """

    def __init__(self, *args: BaseFeature, n_jobs: int = 1, on_errors: str = 'raise',
                 return_type: str = 'any'):
        """

        Parameters
        ----------
        args: Converter
            List of Converter
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        self.args = list(args)

    def convert(self, d):
        """convert batched"""
        for ci in self.args:
            d = ci.convert(d)
        return d


Converter = BaseFeature  # alias
