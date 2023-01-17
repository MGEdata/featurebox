Combination
===========

Combination to composition data from element data.::

    >>> from featurebox.featurizers.atom.mapper import AtomTableMap, AtomJsonMap
    >>> data_map = AtomJsonMap(search_tp="name", n_jobs=1)
    >>> wa = WeightedAverage(data_map, n_jobs=1,return_type="df")
    >>> x3 = [{"H": 2, "Pd": 1},{"He":1,"Al":4}]
    >>> wa.fit_transform(x3) # or
    >>> wa.fit_transform(structure_list)

             0         1         2   ...        13        14        15
    0  0.422068  0.360958  0.201433  ... -0.459164 -0.064783 -0.250939
    1  0.007163 -0.471498 -0.072860  ...  0.206306 -0.041006  0.055843
    <BLANKLINE>
    [2 rows x 16 columns]


    >>> wa.set_feature_labels(["fea_{}".format(_) for _ in range(16)])
    >>> wa.fit_transform(x3)


          fea_0     fea_1     fea_2  ...    fea_13    fea_14    fea_15
    0  0.422068  0.360958  0.201433  ... -0.459164 -0.064783 -0.250939
    1  0.007163 -0.471498 -0.072860  ...  0.206306 -0.041006  0.055843
    <BLANKLINE>
    [2 rows x 16 columns]