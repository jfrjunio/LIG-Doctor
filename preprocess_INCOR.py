#################################################################################################
# author: junio@usp.br - Jose F Rodrigues Jr
#################################################################################################

import math
import cPickle as pickle
import random
import argparse
import entropy_analysis

#\copy (select a.person_seq, b.concept_code, b.concept_name, a.days from event a, concept b where b.domain_id = 'Condition' and a.concept_seq = b.concept_seq order by a.person_seq, a.days desc) TO '/home/junio/Desktop/Ju/INCOR/INCOR-PREPROCESSED.csv' WITH (FORMAT CSV,HEADER TRUE);

global ARGS

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('incor_file', type=str, default='INCOR-PREPROCESSED.csv', help='The Incor dataset file.')
	parser.add_argument('output_prefix', type=str, default='preprocessing', help='The output file radical name.')
	parser.add_argument('--data_partition', type=str, default='[90,10]', help='Provide an array with two values that sum up 100.')
	argsTemp = parser.parse_args()
	return argsTemp

if __name__ == '__main__':
	global ARGS
	ARGS = parse_arguments()
	partitions = [int(strDim) for strDim in ARGS.data_partition[1:-1].split(',')]
	actionOrderedIndexesMAP = {}

	#one line of the admissions file contains one admission hadm_id of one subject_id at a given time admittime
	print 'Building Maps: hadm_id to admtttime; and Map: subject_id to set of all its hadm_ids'
	subjectTOhadms_Map = {}
	hadmTOadmttime_Map = {}
	hadmTOinterval_Map = {}
	hadmToactionIDCODEs_Map = {}
	agir_ADMISSIONS_csv = open(ARGS.incor_file, 'r')
	#person_seq,concept_code,concept_name,days
	agir_ADMISSIONS_csv.readline()

	initial_number_of_admissions = 0
	previous_subject = 0
	previous_admission = 0
	for line in agir_ADMISSIONS_csv:
		initial_number_of_admissions += 1
		try:
			indexOfFirstQuote = line.index('\"')
			lastComa = line.rindex(',')
			quotesToTheEnd = line[indexOfFirstQuote:lastComa]
			quotesToTheEnd = quotesToTheEnd.replace(',','')
			line = line[0:indexOfFirstQuote] + quotesToTheEnd + line[lastComa:-1]
		except:
			None
		line = line.replace('\"','')
		tokens = line.strip().split(',')
		subject_id = int(tokens[0])
		tokens[1] = tokens[1].replace('/','')
		tokens[1] = tokens[1].replace('.', '')

		hadm_id = str(tokens[0]) + '-' + str(tokens[3])

		#keep track of the admission amount of time
		#hadmTOadmttime_Map(hadm_id) -> time of admission
		hadmTOadmttime_Map[hadm_id] = int(tokens[3])
		if subject_id == previous_subject:
			if previous_subject_time != int(tokens[3]):
				# keep track of the time since the last admission in days
				temp = hadmTOadmttime_Map[previous_admission] - int(tokens[3])
				hadmTOinterval_Map[hadm_id] = temp
		else:
			hadmTOinterval_Map[hadm_id] = 0  # 1st interval since the last admission is 0

		incorCondition_code = tokens[1]
		if not hadm_id in hadmToactionIDCODEs_Map:
			hadmToactionIDCODEs_Map[hadm_id] = set()              #use set to avoid repetitions
		hadmToactionIDCODEs_Map[hadm_id].add(incorCondition_code)

		previous_admission = hadm_id
		previous_subject = subject_id
		previous_subject_time = int(tokens[3])

		#subjectTOhadms_Map(subject_id) -> set of hadms for subject_id
		if not subject_id in subjectTOhadms_Map:
			subjectTOhadms_Map[subject_id] = set()
		subjectTOhadms_Map[subject_id].add(hadm_id)

	agir_ADMISSIONS_csv.close()

	for hadm_id in hadmToactionIDCODEs_Map.keys():
		hadmToactionIDCODEs_Map[hadm_id] = list(hadmToactionIDCODEs_Map[hadm_id])   #convert to list, as the rest of the code expects

	#since the data in the database is not necessarily time-ordered
	#here we sort the admissions (hadm_id) according to the admission time (admittime)
	#after this, we have a list subjectTOorderedHADM_IDS_Map(subject_id) -> admission-time-ordered set of actionID codes
	print 'Building Map: subject_id to admission-ordered (admittime, actionIDs set) and cleaning one-admission-only patients'
	subjectTOorderedHADM_IDS_Map = {}
	subjectTOProcHADM_IDs_Map = {}
	#for each admission hadm_id of each patient subject_id
	number_of_subjects_with_only_one_visit = 0
	for subject_id, hadmList in subjectTOhadms_Map.iteritems():
		if len(hadmList) < 2:
			number_of_subjects_with_only_one_visit += 1
			continue  #discard subjects with only 2 admissions
		#sorts the hadm_ids according to date admttime
		#only for the hadm_id in the list hadmList
		#In InCor time is measured in days before the year 2020 - hence, ordering is reversed - bigger number of days, mean older records
		sortedList = sorted([(hadmTOadmttime_Map[hadm_id], hadmToactionIDCODEs_Map[hadm_id], hadm_id) for hadm_id in hadmList],reverse = True)
		# each element in subjectTOorderedHADM_IDS_Map is a key-value (subject_id, (admittime, actionID_List, hadm_id))
		subjectTOorderedHADM_IDS_Map[subject_id] = sortedList
	print '-Number of discarded subjects with only one admission: ' + str(number_of_subjects_with_only_one_visit)
	print '-Number of subjects after ordering: ' + str(len(subjectTOorderedHADM_IDS_Map))

	print 'Converting maps to lists in preparation for dump'
	all_subjectsListOfCODEsList_LIST = []

	#for each subject_id, get its key-value (subject_id, (admittime, CODESs_List))
	for subject_id, time_ordered_CODESs_List in subjectTOorderedHADM_IDS_Map.iteritems():
		subject_list_of_CODEs_List = []
		#for each admission (admittime, CODESs_List) build lists of time and CODEs list
		for admission in time_ordered_CODESs_List:   		#each element in time_ordered_CODESs_List is a tripple (admittime, actionID_List, hadm_id)
			#here, admission = [admittime, actionID_List, hadm_id)
			subject_list_of_CODEs_List.append((admission[1],admission[2]))  #build list of lists of the admissions' CODEs of the current subject_id, stores hadm_id together
		#lists of lists, one entry per subject_id
		all_subjectsListOfCODEsList_LIST.append(subject_list_of_CODEs_List)	#build list of list of lists of the admissions' actionIDs - one entry per subject_id

	CODES_distributionMAP = entropy_analysis.writeDistributions(ARGS.incor_file, hadmToactionIDCODEs_Map, subjectTOhadms_Map, all_subjectsListOfCODEsList_LIST)
	for i, key in enumerate(CODES_distributionMAP):
		actionOrderedIndexesMAP[key[0]] = i
	entropy_analysis.computeShannonEntropyDistribution(all_subjectsListOfCODEsList_LIST, CODES_distributionMAP, ARGS.incor_file)

	#Randomize the order of the patients at the first dimension
	random.shuffle(all_subjectsListOfCODEsList_LIST)

	interval_since_last_admissionListOfLists = []
	new_all_subjectsListOfCODEsList_LIST = []
	final_number_of_admissions = 0
	#Here we convert the database codes to internal sequential codes
	print 'Converting database ids to sequential integer ids'
	procCODEstoInternalID_map = {}
	for subject_list_of_CODEs_List in all_subjectsListOfCODEsList_LIST:
		new_subject_list_of_CODEs_List = []
		interval_since_last_admissionList = []
		for CODEs_List in subject_list_of_CODEs_List:
			final_number_of_admissions += 1
			new_CODEs_List = []
			hadm_id = CODEs_List[1]
			try:
				intervalTemp = hadmTOinterval_Map[hadm_id]
			except:
				print 'oi'
			#we bypass admissions with 0 or negative intervals
			if intervalTemp < 0:
				continue
			interval_since_last_admissionList.append(intervalTemp)

			for CODE in CODEs_List[0]:
				new_CODEs_List.append(actionOrderedIndexesMAP[CODE])   #newVisit is the CODEs_List, but with the new sequential ids
			new_subject_list_of_CODEs_List.append(new_CODEs_List)		#new_subject_list_of_CODEs_List is the subject_list_of_CODEs_List, but with the id given by its frequency

		#when we bypass admissions with 0 or negative intervals, we might create patients with only one admission, which we also bypass
		if len(new_subject_list_of_CODEs_List) > 1:
			interval_since_last_admissionListOfLists.append(interval_since_last_admissionList)
			new_all_subjectsListOfCODEsList_LIST.append(new_subject_list_of_CODEs_List)	#new_all_subjectsListOfCODEsList_LIST is the all_subjectsListOfCODEsList_LIST, but with the new sequential ids

	print ''
	nCodes = len(actionOrderedIndexesMAP)
	print '-Number of actually used DIAGNOSES codes: '+ str(nCodes)
	print '-Final number of subjects: ' + str(len(new_all_subjectsListOfCODEsList_LIST))
	print '-Final number of admissions: ' + str(final_number_of_admissions)
	#Partitioning data
	if (len(partitions) >= 1):
		total_patients_dumped = 0;
		print 'Writing ' + str(partitions[0]) + '% of the patients read from file ' + ARGS.incor_file
		index_of_last_patient_to_dump = int(math.ceil(len(new_all_subjectsListOfCODEsList_LIST)*int(partitions[0])/100))
		pickle.dump(new_all_subjectsListOfCODEsList_LIST[0:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.train', 'wb'), -1)
		pickle.dump(interval_since_last_admissionListOfLists[0:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.INTERVAL.train', 'wb'), -1)
		print '   Patients from 0 to ' + str(index_of_last_patient_to_dump)
		print '   Success, file: ' + ARGS.output_prefix + '_' + str(nCodes) + '.train created'
		total_patients_dumped += index_of_last_patient_to_dump

		if (len(partitions) >= 2):
			print 'Writing ' + str(partitions[1]) + '% of the patients read from file ' + ARGS.incor_file
			index_of_first_patient_to_dump = index_of_last_patient_to_dump
			index_of_last_patient_to_dump = index_of_first_patient_to_dump + int(math.ceil(len(new_all_subjectsListOfCODEsList_LIST)*int(partitions[1])/100))
			pickle.dump(new_all_subjectsListOfCODEsList_LIST[index_of_first_patient_to_dump:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.test', 'wb'), -1)
			pickle.dump(interval_since_last_admissionListOfLists[index_of_first_patient_to_dump:index_of_last_patient_to_dump], open(ARGS.output_prefix + '_' + str(nCodes) + '.INTERVAL.test', 'wb'), -1)
			print '   Patients from ' + str(index_of_first_patient_to_dump) + ' to ' + str(index_of_last_patient_to_dump)
			print '   Success, file: ' + ARGS.output_prefix + '_' + str(nCodes) + '.test created'
			total_patients_dumped += index_of_last_patient_to_dump - index_of_first_patient_to_dump

			if (len(partitions) >= 3):
				print 'Writing ' + str(partitions[2]) + '% of the patients read from file ' + ARGS.incor_file
				index_of_first_patient_to_dump = index_of_last_patient_to_dump
				pickle.dump(new_all_subjectsListOfCODEsList_LIST[index_of_first_patient_to_dump:],open(ARGS.output_prefix + '_' + str(nCodes) + '.valid', 'wb'), -1)
				pickle.dump(interval_since_last_admissionListOfLists[index_of_first_patient_to_dump:], open(ARGS.output_prefix + '_' + str(nCodes) + '.INTERVAL.valid', 'wb'), -1)
				print '   Patients from ' + str(index_of_first_patient_to_dump) + ' to the end of the file'
				print '   Success, file: ' + ARGS.output_prefix + '_' + str(nCodes) + '.valid created'
				total_patients_dumped += len(new_all_subjectsListOfCODEsList_LIST) - total_patients_dumped
				print 'Total of dumped patients: ' + str(total_patients_dumped) + ' out of ' + str(len(new_all_subjectsListOfCODEsList_LIST))
	else:
		print 'Error, please provide data partition scheme. E.g, [80,10,10], for 80\% train, 10\% test, and 10\% validation.'