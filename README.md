# LIG-Doctor

#testing the installation\
	python
	import theano #should execute smoothly


#after all set\
#model training - builds model to be saved in a file called model.npz:\
	"python ./LIG-Doctor.py ./mimic-90-10_02_271 ./model"

#model testing:\
	"python ./LIG-Doctor-test.py ./mimic-90-10_02_271 ./model.npz"

These input files are outputs of script preprocess_mimiciii.py.
We provide actual input files for reproduction of our results. So, runing preprocess_mimiciii.py is, initially, optional.
DISCLAIMER: we built the data file using the mimic-III dataset; but the file is nothing but a bunch of numbers in binary format;
in order to make sense out of the file, one must download mimic-III from https://mimic.physionet.org
