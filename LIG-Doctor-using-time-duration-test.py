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

global ARGS
global tPARAMS

def numpy_floatX(data):
	return np.asarray(data, dtype=config.floatX)


def prepareHotVectors(test_tensor, features_tensor):
	n_visits_of_each_patientList = np.array([len(seq) for seq in test_tensor]) - 1
	number_of_patients = len(test_tensor)
	max_number_of_visits = np.max(n_visits_of_each_patientList)

	x_hotvectors_tensorf = np.zeros((max_number_of_visits, number_of_patients, ARGS.numberOfInputCodes)).astype(config.floatX)
	mask = np.zeros((max_number_of_visits, number_of_patients)).astype(config.floatX)
	feats_hotvectors_tensor = np.zeros((max_number_of_visits, number_of_patients, ARGS.numberOfFeatsCodes)).astype(config.floatX)

	for idx, (train_patient_matrix,feats_patient_matrix) in enumerate(zip(test_tensor,features_tensor)):
		for i_th_visit, visit_line in enumerate(train_patient_matrix[:-1]): #ignores the last visit, which is not part of the computation
			for code in visit_line:
				x_hotvectors_tensorf[i_th_visit, idx, code] = 1
		for i_th_visit, time in enumerate(feats_patient_matrix[:-1]): #ignores the last visit, which is not part of the computation
			feats_hotvectors_tensor[i_th_visit, idx, 0] = time
		mask[:n_visits_of_each_patientList[idx], idx] = 1.

	x_hotvectors_tensorb = x_hotvectors_tensorf[::-1,::,::]
	return x_hotvectors_tensorf, x_hotvectors_tensorb, feats_hotvectors_tensor, mask, n_visits_of_each_patientList


def min_max_normalization(time_trainSet,time_testSet):
	# we collect parameters for normalization
	tMax = np.array(np.array([(np.array(time_trainSet)).max(),(np.array(time_testSet)).max()]).max()).max()
	tMin = np.array(np.array([(np.array(time_trainSet)).min(),(np.array(time_testSet)).max()]).min()).max()
	# normalization in the range [0.1-0.8]
	for i, train in enumerate(time_trainSet):
		for j, item in enumerate(train):
			time_trainSet[i][j] = ((time_trainSet[i][j] - tMin) / float(tMax - tMin)) * 0.8 + 0.1
	for i, test in enumerate(time_testSet):
		for j, item in enumerate(test):
			time_testSet[i][j] = ((time_testSet[i][j] - tMin) / float(tMax - tMin)) * 0.8 + 0.1

def loadModel():
	model = np.load(ARGS.modelFile)
	tPARAMS = OrderedDict()
	for key, value in model.iteritems():
		tPARAMS[key] = theano.shared(value, name=key)
	ARGS.numberOfFeatsCodes = 1
	ARGS.numberOfInputCodes = model['fWf_0'].shape[0] - ARGS.numberOfFeatsCodes
	return tPARAMS


def fMinGRU_layer(inputTensor, layerIndex, hiddenDimSize, mask=None):
	maxNumberOfVisits = inputTensor.shape[0]
	batchSize = inputTensor.shape[1]

	Wf = T.dot(inputTensor,tPARAMS['fWf_' + layerIndex])
	Wh = T.dot(inputTensor,tPARAMS['fWh_' + layerIndex])

	def stepFn(stepMask, wf, wh, h_previous):
		f = T.nnet.sigmoid(wf + T.dot(h_previous,tPARAMS['fUf_' + layerIndex])) + tPARAMS['fbf_' + layerIndex]
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
	featsSlice = T.tensor3('t', dtype=config.floatX)

	flowing_tensorf = T.concatenate([featsSlice, flowing_tensorf], axis=2)
	flowing_tensorb = T.concatenate([featsSlice[::-1, ::, ::], flowing_tensorb], axis=2)

	for i, hiddenDimSize in enumerate(ARGS.hiddenDimSize):
		flowing_tensorf = fMinGRU_layer(flowing_tensorf, str(i), hiddenDimSize, mask=mask)

	for i, hiddenDimSize in enumerate(ARGS.hiddenDimSize):
		flowing_tensorb = bMinGRU_layer(flowing_tensorb, str(i), hiddenDimSize, mask=mask)

	flowing_tensorb = flowing_tensorb[::-1, ::, ::]
	joint_flow = T.nnet.relu(T.dot(flowing_tensorf, tPARAMS['fJ']) + T.dot(flowing_tensorb, tPARAMS['bJ']) + tPARAMS['fbb'],tPARAMS['jlrelu'])

	results, _ = theano.scan(
		lambda theFlowingTensor: T.nnet.softmax(T.nnet.relu(T.dot(theFlowingTensor, tPARAMS['W_output']) + tPARAMS['b_output'], tPARAMS['olrelu'])),
		sequences=[joint_flow],
		outputs_info=None,
		name='softmax_layer',
		n_steps=maxNumberOfAdmissions)
	results = results * mask[:, :, None]

	return xf, xb, featsSlice, mask, results



def load_data():
	testSet_x = np.array(pickle.load(open(ARGS.inputFileRadical+'.test', 'rb')))
	testSet_y = np.array(pickle.load(open(ARGS.inputFileRadical+'.test', 'rb')))

	def len_argsort(seq):
		return sorted(range(len(seq)), key=lambda x: len(seq[x]))

	sorted_index = len_argsort(testSet_x)
	testSet_x = [testSet_x[i] for i in sorted_index]
	testSet_y = [testSet_y[i] for i in sorted_index]

	ARGS.featsFile = ARGS.inputFileRadical + '.DURATION'
	print('Using extra features (interval/duration/type) from file ' + ARGS.featsFile)
	feats_trainSet = pickle.load(open(ARGS.featsFile+'.test', 'rb'))
	feats_testSet = pickle.load(open(ARGS.featsFile+'.test', 'rb'))

	feats_trainSet = [feats_trainSet[i] for i in sorted_index]
	feats_testSet = [feats_testSet[i] for i in sorted_index]

	min_max_normalization(feats_trainSet, feats_testSet)

	testSet = [testSet_x, testSet_y, feats_testSet]
	return testSet


def testModel():
	print '==> model loading'
	global tPARAMS
	tPARAMS = loadModel()

	print '==> data loading'
	testSet = load_data()

	print '==> model rebuilding'
	xf, xb, featsSlice, mask, MODEL = build_model()
	PREDICTOR_COMPILED = theano.function(inputs=[xf, xb, featsSlice, mask], outputs=MODEL, name='PREDICTOR_COMPILED')

	print '==> model execution'
	nBatches = int(np.ceil(float(len(testSet[0])) / float(ARGS.batchSize)))
	predictedY_list = []
	actualY_list = []
	for batchIndex in range(nBatches):
		batchX = testSet[0][batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
		batchY = testSet[1][batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
		batchT = testSet[2][batchIndex * ARGS.batchSize: (batchIndex + 1) * ARGS.batchSize]
		xf, xb, featsSlice, mask, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchT)
		predicted_y = PREDICTOR_COMPILED(xf, xb, featsSlice, mask)

		for ith_patient in range(predicted_y.shape[1]):
			predictedPatientSlice = predicted_y[:, ith_patient, :]
			actual_y = batchY[ith_patient][1:]
			for ith_admission in range(nVisitsOfEachPatient_List[ith_patient]):
				actualY_list.append(actual_y[ith_admission])

				ithPrediction = predictedPatientSlice[ith_admission]
				enumeratedPrediction = [temp for temp in enumerate(ithPrediction)]
				sortedPrediction_30Top = sorted(enumeratedPrediction, key=lambda x: x[1],reverse=True)[0:30]
				sortedPrediction_30Top_indexes = [temp[0] for temp in sortedPrediction_30Top]
				predictedY_list.append(sortedPrediction_30Top_indexes)

	print '==> computation of prediction results'
	recall_sum = [0.0,0.0,0.0]
	k_list = [10,20,30]
	for ith_admission in range(len(predictedY_list)):
		ithActualY = set(actualY_list[ith_admission])
		for ithK, k in enumerate(k_list):
			ithPredictedY = set(predictedY_list[ith_admission][:k])
			intersection_set = ithActualY.intersection(ithPredictedY)
			recall_sum[ithK] += len(intersection_set) / float(len(ithActualY))
	finalRecalls = []
	for ithK, k in enumerate(k_list):
		finalRecalls.append(recall_sum[ithK] / float(len(predictedY_list)))

	print 'Results for Precision@10, Precision@20, and Precision@30'
	print str(finalRecalls[0])
	print	str(finalRecalls[1])
	print str(finalRecalls[2])


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('inputFileRadical', type=str, metavar='<visit_file>', help='File radical name (the software will look for .test file) with pickled data organized as patient x admission x codes.')
	parser.add_argument('--featsFile', type=str, default='', help='A file containing features to concatenate to the input.')
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
