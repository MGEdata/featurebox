Select by Corr
================


- 1. Corr Automatically

>>> from sklearn.datasets import load_boston
>>> from featurebox.selection.corr import Corr
>>> x, y = load_boston(return_X_y=True)
>>> co = Corr(threshold=0.7,multi_index=[0,8],multi_grade=2)
>>> newx = co.fit_transform(x)
>>> print(x.shape)
>>> print(newx.shape)
>>> #(506, 13)
>>> #(506, 9)


- 2. Corr Step

>>> from sklearn.datasets import load_boston
>>> from featurebox.selection.corr import Corr
>>> x, y = load_boston(return_X_y=True)
>>> co = Corr(threshold=0.7,multi_index=[0,8],multi_grade=2)

Nn range [0,8], the features are binding in to 2 sized: [[0,1],[2,3],[4,5],[6,7]]
Corresponding to the initial 13 feature.
[0,1] -> 0;
[2,3] -> 1;
[4,5]->2;
[6,7]->3;
8->4;
9->5;
10->6;
11->7;
12->8;
13->9;

>>> co.fit(x)
>>> Corr(multi_index=[0, 8], threshold=0.7)
>>> group = co.count_cof()
>>> group[1]
>>> #[[0], [1], [2], [3], [4, 5], [4, 5], [6], [7], [8]]


In this step, you could select manually, or filter automatically as following.

>>> co.remove_coef(group[1]) # Filter automatically by machine.
>>> #[0, 1, 2, 3, 4, 6, 7, 8]

where 2 is filtered, Corresponding to the initial feature 14.
[0,1] -> 0; [2,3] -> 1; [4,5]->2; [6,7]->3; 8->4; ``[9->5]``; 10->6; 11->7; 12->8; ``13->9``;