#################################################################################################
# author: junio@usp.br - Jose F Rodrigues Jr
# note: in many places, the code could be shorter, but that would just make it less comprehensible
#################################################################################################
import numpy as np
import cPickle as pickle
from collections import OrderedDict
import argparse
import theano
import theano.tensor as T
from theano import config
from sklearn import metrics

global ARGS
global tPARAMS

def numpy_floatX(data):
	return np.asarray(data, dtype=config.floatX)


def prepareHotVectors(test_tensor):
	n_visits_of_each_patientList = np.array([len(seq) for seq in test_tensor]) - 1
	number_of_patients = len(test_tensor)
	max_number_of_visits = np.max(n_visits_of_each_patientList)

	x_hotvectors_tensorf = np.zeros((max_number_of_visits, number_of_patients, ARGS.numberOfInputCodes)).astype(config.floatX)
	mask = np.zeros((max_number_of_visits, number_of_patients)).astype(config.floatX)

	for idx, (train_patient_matrix) in enumerate(test_tensor):
		for i_th_visit, visit_line in enumerate(train_patient_matrix[:-1]): #ignores the last visit, which is not part of the computation
			for code in visit_line:
				x_hotvectors_tensorf[i_th_visit, idx, code] = 1
		mask[:n_visits_of_each_patientList[idx], idx] = 1.

	x_hotvectors_tensorb = x_hotvectors_tensorf[::-1,::,::]
	return x_hotvectors_tensorf, x_hotvectors_tensorb, mask, n_visits_of_each_patientList

def loadModel():
	model = np.load(ARGS.modelFile)
	tPARAMS = OrderedDict()
	for key, value in model.iteritems():
		tPARAMS[key] = theano.shared(value, name=key)
	ARGS.numberOfInputCodes = model['fWf_0'].shape[0]
	return tPARAMS


def fMinGRU_layer(inputTensor, layerIndex, hiddenDimSize, mask=None):
	maxNumberOfVisits = inputTensor.shape[0]
	batchSize = inputTensor.shape[1]

	Wf = T.dot(inputTensor,tPARAMS['fWf_' + layerIndex])
	Wh = T.dot(inputTensor,tPARAMS['fWh_' + layerIndex])

	def stepFn(stepMask, wf, wh, h_previous):
		f = T.nnet.sigmoid(wf + T.dot(h_previous,tPARAMS['fUf_' + layerIndex]) + tPARAMS['fbf_' + layerIndex])
		h_intermediate = T.tanh(wh + T.dot(f * h_previous, tPARAMS['fUh_' + layerIndex]) + tPARAMS['fbh_' + layerIndex])
		h_new = ((1. - f) * h_previous) + f * h_intermediate
		h_new = stepMask[:, None] * h_new + (1. - stepMask)[:,None] * h_previous
		return h_new # becomes h_previous in the next iteration

	results, _ = theano.scan(fn=stepFn,  # function to execute
								   sequences=[mask, Wf, Wh],  # input to stepFn
								   outputs_info=T.alloc(numpy_floatX(0.0), batchSize, hiddenDimSize), #initial h_previous
								   name='fMinGRU_layer' + layerIndex,  # labeling for debug
								   n_steps=maxNumberOfVisits)  # number of times to execute - scan is a loop

	return results

def bMinGRU_layer(inputTensor, layerIndex, hiddenDimSize, mask=None):
	maxNumberOfVisits = inputTensor.shape[0]
	batchSize = inputTensor.shape[1]

	Wf = T.dot(inputTensor,tPARAMS['bWf_' + layerIndex])
	Wh = T.dot(inputTensor,tPARAMS['bWh_' + layerIndex])
	bStepMask = mask[::-1,::]

	def stepFn(stepMask, wf, wh, h_previous):  # .* -> element-wise multiplication; * -> matrix multiplication
		f = T.nnet.sigmoid(wf + T.dot(h_previous,tPARAMS['bUf_' + layerIndex])) + tPARAMS['bbf_' + layerIndex]
		h_intermediate = T.tanh(wh + T.dot(f * h_previous, tPARAMS['bUh_' + layerIndex]) + tPARAMS['bbh_' + layerIndex])
		h_new = ((1. - f) * h_previous) + f * h_intermediate
		h_new = stepMask[:, None] * h_new + (1. - stepMask)[:,None] * h_previous
		return h_new
	results, _ = theano.scan(fn=stepFn,  # function to execute
								   sequences=[bStepMask, Wf, Wh],  # input to stepFn
								   outputs_info=T.alloc(numpy_floatX(0.0), batchSize, hiddenDimSize),
								   # just initialization
								   name='bMinGRU_layer' + layerIndex,  # just labeling for debug
								   n_steps=maxNumberOfVisits)  # number of times to execute - scan is a loop
	return results


def build_model():
	xf = T.tensor3('xf', dtype=config.floatX)
	xb = T.tensor3('xb', dtype=config.floatX)
	mask = T.matrix('mask', dtype=config.floatX)
	maxNumberOfAdmissions = xf.shape[0]

	flowing_tensorf = xf
	flowing_tensorb = xb

	for i, hiddenDimSize in enumerate(ARGS.hiddenDimSize):
		flowing_tensorf = fMinGRU_layer(flowing_tensorf, str(i), hiddenDimSize, mask=mask)
		#flowing_tensorf = flowing_tensorf*0.5 #suggested in many places to reflect dropout used in training - it just reduces performance

	for i, hiddenDimSize in enumerate(ARGS.hiddenDimSize):
		flowing_tensorb = bMinGRU_layer(flowing_tensorb, str(i), hiddenDimSize, mask=mask)
		#flowing_tensorf = flowing_tensorf * 0.5

	flowing_tensorb = flowing_tensorb[::-1, ::, ::]
	joint_flow = T.nnet.relu(T.dot(flowing_tensorf, tPARAMS['fJ']) + T.dot(flowing_tensorb, tPARAMS['bJ']) + tPARAMS['fbb'],tPARAMS['jlrelu'])

	results, _ = theano.scan(
		lambda theFlowingTensor: T.nnet.softmax(T.nnet.relu(T.dot(theFlowingTensor, tPARAMS['W_output']) + tPARAMS['b_output'], tPARAMS['olrelu'])),
		sequences=[joint_flow],
		outputs_info=None,
		name='softmax_layer',
		n_steps=maxNumberOfAdmissions)

	results = results * mask[:, :, None]

	return xf, xb, mask, results



def load_data():
	testSet_x = np.array(pickle.load(open(ARGS.inputFileRadical+'.test', 'rb')))
	testSet_y = np.array(pickle.load(open(ARGS.inputFileRadical+'.test', 'rb')))

	def len_argsort(seq):
		return sorted(range(len(seq)), key=lambda x: len(seq[x]))

	sorted_index = len_argsort(testSet_x)
	testSet_x = [testSet_x[i] for i in sorted_index]
	testSet_y = [testSet_y[i] for i in sorted_index]

	testSet = [testSet_x, testSet_y]
	return testSet


def testModel():
	print '==> model loading'
	global tPARAMS
	tPARAMS = loadModel()

	print '==> data loading'
	testSet = load_data()

	print '==> model rebuilding'
	xf, xb, mask, MODEL = build_model()
	PREDICTOR_COMPILED = theano.function(inputs=[xf, xb, mask], outputs=MODEL, name='PREDICTOR_COMPILED')

	print '==> model execution'
	nBatches = int(np.ceil(float(len(testSet[0])) / float(ARGS.batchSize)))
	predictedY_list = []
	predictedProbabilities_list = []
	actualY_list = []

	file = open(ARGS.inputFileRadical + 'AUCROC.input.txt', 'w')
	#file.write('Data prepared for input at http://www.rad.jhmi.edu/jeng/javarad/roc/helpers/formats.html' + '\n')
	#Execute once for each batch
	for batchIndex in range(nBatches):
		batchX = testSet[0][batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
		batchY = testSet[1][batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
		xf, xb, mask, nVisitsOfEachPatient_List = prepareHotVectors(batchX)
		#retrieve the maximum number of admissions considering all the patients
		maxNumberOfAdmissions = np.max(nVisitsOfEachPatient_List)
		#make prediction
		predicted_y = PREDICTOR_COMPILED(xf, xb, mask)

		#traverse the predicted results, once for each patient in the batch
		for ith_patient in range(predicted_y.shape[1]):
			predictedPatientSlice = predicted_y[:, ith_patient, :]
			#retrieve actual y from batch tensor -> actual codes, not the hotvector
			actual_y = batchY[ith_patient][1:]
			#for each admission of the ith-patient
			for ith_admission in range(nVisitsOfEachPatient_List[ith_patient]):
				#convert array of actual answers to list
				actualY_list.append(actual_y[ith_admission])
				#retrieves ith-admission of ths ith-patient
				ithPrediction = predictedPatientSlice[ith_admission]
				#since ithPrediction is a vector of probabilties with the same dimensionality of the hotvectors
				#enumerate is enough to retrieve the original codes
				enumeratedPrediction = [codeProbability_pair for codeProbability_pair in enumerate(ithPrediction)]
				#sort everything
				sortedPredictionsAll = sorted(enumeratedPrediction, key=lambda x: x[1],reverse=True)
				#creates trimmed list up to max(maxNumberOfAdmissions,30) elements
				sortedTopPredictions = sortedPredictionsAll[0:max(maxNumberOfAdmissions,30)]
				#here we simply toss off the probability and keep only the sorted codes
				sortedTopPredictions_indexes = [codeProbability_pair[0] for codeProbability_pair in sortedTopPredictions]
				#stores results in a list of lists - after processing all batches, predictedY_list stores all the prediction results
				predictedY_list.append(sortedTopPredictions_indexes)
				predictedProbabilities_list.append(sortedPredictionsAll)

	# ---------------------------------Report results using k=[10,20,30]
	print '==> computation of prediction results with constant k'
	recall_sum = [0.0, 0.0, 0.0]

	k_list = [10,20,30]
	for ith_admission in range(len(predictedY_list)):
		ithActualYSet = set(actualY_list[ith_admission])
		for ithK, k in enumerate(k_list):
			ithPredictedY = set(predictedY_list[ith_admission][:k])
			intersection_set = ithActualYSet.intersection(ithPredictedY)
			recall_sum[ithK] += len(intersection_set) / float(len(ithActualYSet)) # this is recall because the numerator is len(ithActualYSet)

	precision_sum = [0.0, 0.0, 0.0]
	k_listForPrecision = [1,2,3]
	for ith_admission in range(len(predictedY_list)):
		ithActualYSet = set(actualY_list[ith_admission])
		for ithK, k in enumerate(k_listForPrecision):
			ithPredictedY = set(predictedY_list[ith_admission][:k])
			intersection_set = ithActualYSet.intersection(ithPredictedY)
			precision_sum[ithK] += len(intersection_set) / float(k) # this is precision because the numerator is k \in [10,20,30]

	finalRecalls = []
	finalPrecisions = []
	for ithK, k in enumerate(k_list):
		finalRecalls.append(recall_sum[ithK] / float(len(predictedY_list)))
		finalPrecisions.append(precision_sum[ithK] / float(len(predictedY_list)))

	print 'Results for Recall@' + str(k_list)
	print str(finalRecalls[0])
	print str(finalRecalls[1])
	print str(finalRecalls[2])

	print 'Results for Precision@' + str(k_listForPrecision)
	print str(finalPrecisions[0])
	print str(finalPrecisions[1])
	print str(finalPrecisions[2])


	#---------------------------------Report results using k=lenght of actual answer vector
	print '==> computation of prediction results with dynamic k=lenght of actual answer vector times [1,2,3]'
	recall_sum = [0.0, 0.0, 0.0]
	precision_sum = [0.0, 0.0, 0.0]
	multiples_list = [0, 1, 2]
	for ith_admission in range(len(predictedY_list)):
		ithActualYSet = set(actualY_list[ith_admission])
#		print '--->Admission: ' + str(ith_admission)
		for m in multiples_list:
			k = len(ithActualYSet) * (m+1)
#			print 'K: ' + str(k)
			ithPredictedY = set(predictedY_list[ith_admission][:k])
#			print 'Prediction: ' + str(ithPredictedY)
#			print 'Actual: ' + str(ithActualYSet)
			intersection_set = ithActualYSet.intersection(ithPredictedY)
#			print 'Intersection: ' + str(intersection_set)
			recall_sum[m] += len(intersection_set) / float(len(ithActualYSet))
			precision_sum[m] += len(intersection_set) / float(k) # this is precision because the numerator is ithK \in [10,20,30]

	bReportDynamic_K = False
	if bReportDynamic_K:
		finalRecalls = []
		finalPrecisions = []
		for m in multiples_list:
			finalRecalls.append(recall_sum[m] / float(len(predictedY_list)))
			finalPrecisions.append(precision_sum[m] / float(len(predictedY_list)))

		print 'Results for Recall@k*1, Recall@k*2, and Recall@k*3'
		print str(finalRecalls[0])
		print str(finalRecalls[1])
		print str(finalRecalls[2])

		print 'Results for Precision@k*1, Precision@k*2, and Precision@k*3'
		print str(finalPrecisions[0])
		print str(finalPrecisions[1])
		print str(finalPrecisions[2])

	# ---------------------------------Write data for AUC-ROC computation
	bWriteDataForAUC = False
	fullListOfTrueYOutcomeForAUCROCAndPR_list = []
	fullListOfPredictedYProbsForAUCROC_list = []
	fullListOfPredictedYForPrecisionRecall_list = []
	for ith_admission in range(len(predictedY_list)):
		ithActualY = actualY_list[ith_admission]
		nActualCodes = len(ithActualY)
		ithPredictedProbabilities = predictedProbabilities_list[ith_admission]#[0:nActualCodes]
		ithPrediction = 0
		for predicted_code, predicted_prob in ithPredictedProbabilities:
			fullListOfPredictedYProbsForAUCROC_list.append(predicted_prob)
			#for precision-recall purposes, the nActual first codes correspond to what was estimated as correct answers
			if ithPrediction < nActualCodes:
				fullListOfPredictedYForPrecisionRecall_list.append(1)
			else:
				fullListOfPredictedYForPrecisionRecall_list.append(0)

			#the list fullListOfTrueYOutcomeForAUCROCAndPR_list corresponds to the true answer, either positive or negative
			#it is used for both Precision Recall and for AUCROC
			if predicted_code in ithActualY:
				fullListOfTrueYOutcomeForAUCROCAndPR_list.append(1)
				file.write("1 " + str(predicted_prob) + '\n')
			else:
				fullListOfTrueYOutcomeForAUCROCAndPR_list.append(0)
				file.write("0 " + str(predicted_prob) + '\n')
			ithPrediction += 1
	file.close()

	#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
	print "Weighted AUC-ROC score: " + str(metrics.roc_auc_score(fullListOfTrueYOutcomeForAUCROCAndPR_list,
														fullListOfPredictedYProbsForAUCROC_list,
														average = 'weighted'))
	#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
	PRResults = metrics.precision_recall_fscore_support(fullListOfTrueYOutcomeForAUCROCAndPR_list,
														fullListOfPredictedYForPrecisionRecall_list,
														average = 'binary')
	print 'Precision: ' + str(PRResults[0])
	print 'Recall: ' + str(PRResults[1])
	print 'Binary F1 Score: ' + str(PRResults[2]) #FBeta score with beta = 1.0
	print 'Support: ' + str(PRResults[3])

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('inputFileRadical', type=str, metavar='<visit_file>', help='File radical name (the software will look for .test file) with pickled data organized as patient x admission x codes.')
	parser.add_argument('modelFile', type=str, metavar='<model_file>', help='The path to the model file .npz')
	parser.add_argument('--hiddenDimSize', type=str, default='[271]', help='Number of layers and their size - for example [100,200] refers to two layers with 100 and 200 nodes.')
	parser.add_argument('--batchSize', type=int, default=100, help='Batch size.')
	ARGStemp = parser.parse_args()
	hiddenDimSize = [int(strDim) for strDim in ARGStemp.hiddenDimSize[1:-1].split(',')]
	ARGStemp.hiddenDimSize = hiddenDimSize
	return ARGStemp

if __name__ == '__main__':
	global tPARAMS
	tPARAMS = OrderedDict()
	global ARGS
	ARGS = parse_arguments()

	testModel()
