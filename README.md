# LIG-Doctor
#setting up theano
0) install gcc
1) install miniconda for python 2.7 (doesn't work with python 3)
2) install theano and pygpu (conda install theano pygpu)
3) numpy downgrade to 1.12 (conda install numpy=1.12)
4) fix a theano installation bug as described in
https://stackoverflow.com/questions/53423610/how-to-update-scan-cython-code-in-theano
(here is the file: https://github.com/Theano/Theano/blob/master/theano/scan_module/c_code/scan_perform.c)
5) create text file ~/.theanorc with lines
"[global]
optimizer_excluding=scanOp_pushout_output"

#testing the installation
python
import theano #should execute smoothly


#after all set
#model training - builds model to be saved in a file called model.npz:

"python ./LIG-Doctor.py ./mimic-90-10_02_271 ./model"



#model testing:

"python ./LIG-Doctor-test.py ./mimic-90-10_02_271 ./model.npz"


These input files are outputs of script preprocess_mimiciii.py.
We provide actual input files for reproduction of our results. So, runing preprocess_mimiciii.py is, initially, optional.
DISCLAIMER: we built the data file using the mimic-III dataset; but the file is nothing but a bunch of numbers in binary format;
in order to make sense out of the file, one must download mimic-III from https://mimic.physionet.org
