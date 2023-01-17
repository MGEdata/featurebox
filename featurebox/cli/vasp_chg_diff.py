# -*- coding: utf-8 -*-

# @Time  : 2022/7/26 14:30
# @Author : boliqq07
# @Software: PyCharm
# @License: MIT License


"""
Due to the CHGCAR file are large, We don't use python code to get chg_diff. (CHGCAR2-CHGCAR1)
Result = f2 - f1, make sure the rank of paths in two files are matching.

file1
--------
p1\n
p2\n
...\n

file2
---------
p1'\n
p2'\n
...\n

"""

r"""
Copy follow code to form one ”chg_diff.sh“ file, and 'sh chg_diff.sh':

Notes::

    #!/bin/bash
    
    old_path=$(cd "$(dirname "$0")"; pwd)
    
    paste  paths1.temp paths2.temp| while read chg1 chg2;
    
    do
    
    echo "Try to" $chg2 "-" $chg1 ">>>"
    
    n1=${chg1//\//_}
    
    n2=${chg2//\//_}
    
    ~/bin/chgdiff.pl $chg1/CHGCAR $chg2/CHGCAR
    
    mv CHGCAR_diff $n1$n2-"CHGCAR_diff"
    
    echo $n1$n2-"CHGCAR_diff" "store in" $old_path
    
    done
"""
