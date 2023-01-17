Use Yourself Data
=================

>>> from featurebox.featurizers.atom.mapper import AtomJsonMap
>>> tmps = AtomJsonMap(search_tp="number",embedding_dict="your.json")
>>> tmps = AtomJsonMap(search_tp="number",embedding_dict=Your_dict)

>>> tmps = AtomTableMap(search_tp="number",tablename="your.csv")
>>> tmps = AtomJsonMap(search_tp="number",tablename=Your_pd_DataFrame)

where the search_tp is "number" or "name" depend on your data, but advise use "name" for json data.