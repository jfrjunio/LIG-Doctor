Runing the scripts, after installing theano.

TRAINING
user@machine: python ./LIG-Doctor.py ./data/mimic-90-10_01_271 ./modelFile

The file name "./data/mimic-90-10_01_271" is only a radical to the actual names:
-> Script LIG-Doctor.py will look for files "./data/mimic-90-10_01_271.test", and "./data/mimic-90-10_01_271.train"
-> Script LIG-Doctor-using-time-duration.py will look for files "./data/mimic-90-10_01_271.test", "./data/mimic-90-10_01_271.train", "./data/mimic-90-10_01_271.DURATION.train", and "./data/mimic-90-10_01_271.DURATION.test".

All these input files are outputs of script preprocess_mimiciii.py.

We provide actual input files for full reproduction of our results. So, runing preprocess_mimiciii.py is, initially, optional.

TESTING
The execution line for testing is quite similar to the training line.
The training script will produce a model file with extension XX.npz., where XX is the number of the best epoch.
user@machine: python ./LIG-Doctor.py ./data/mimic-90-10_01_271 ./modelFile.XX.npz

