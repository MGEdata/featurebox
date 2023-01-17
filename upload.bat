
@echo on
set path=D:\Anaconda3;D:\Anaconda3\Library\bin;D:\Anaconda3\Scripts;D:\Anaconda3\condabin;%path%
set path=C:\Users\Administrator\anaconda3;C:\Users\Administrator\anaconda3\Library\bin;%path%
set path=C:\Users\Administrator\anaconda3\Scripts;C:\Users\Administrator\anaconda3;\condabin;%path%

path

python setup.py sdist

twine check dist/*

twine upload dist/*

rd /s /Q dist

rd /s /Q featurebox.egg-info

pause

pause

exit