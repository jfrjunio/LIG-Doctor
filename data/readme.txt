In this directory, there are 4 different randomized splits (90% train + 10% testing) of Mimic-III dataset:
-mimic-90-10_01_271
-mimic-90-10_02_271
-mimic-90-10_03_271
-mimic-90-10_04_271
->Each split spams to four files: 'fileRadical'.train, 'fileRadical'.DURATION.train, 'fileRadical'.test, and 'fileRadical'.DURATION.test
->For training/validation, all the four files of a given split are necessary; for testing, only the two test files are necessary.


This data comes from files ADMISSIONS.csv and DIAGNOSES_ICD.csv distributed by Mimic-III website: https://physionet.org/physiobank/database/mimic3cdb/

Although the data actually comes from dataset Mimic-III, it is impossible to make sense of it without the original csv files. However, it is possible to run experiments.

Mimic-III is open for all researchers, but is has controlled access, so one cannot distribute it.
More instructions on how to gain access at https://physionet.org/physiobank/database/mimic3cdb/
