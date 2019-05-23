#################################################################################################
# author: junio@usp.br - Jose F Rodrigues-Jr
# note: in many places, the code could be shorter, but that would just make it less comprehensible
#################################################################################################
import random
import math
import cPickle as pickle
import os
from collections import OrderedDict
import argparse
import theano
import theano.tensor as T
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

global ARGS
global tPARAMS

def unzip(zipped):
	new_params = OrderedDict()
	for key, value in zipped.iteritems():
		new_params[key] = value.get_value()
	return new_params

def numpy_floatX(data):
	return np.asarray(data, dtype=config.floatX)

def getNumberOfCodes(sets):
	highestCode = 0
	for set in sets:
		for pat in set:
			for adm in pat:
				for code in adm:
					if code > highestCode:
						highestCode = code
	return (highestCode + 1)

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

def prepareHotVectors(train_tensor, labels_tensor, features_tensor):
	nVisitsOfEachPatient_List = np.array([len(seq) for seq in train_tensor]) - 1
	numberOfPatients = len(train_tensor)
	maxNumberOfAdmissions = np.max(nVisitsOfEachPatient_List)

	x_hotvectors_tensorf = np.zeros((maxNumberOfAdmissions, numberOfPatients, ARGS.numberOfInputCodes)).astype(config.floatX)
	y_hotvectors_tensor = np.zeros((maxNumberOfAdmissions, numberOfPatients, ARGS.numberOfInputCodes)).astype(config.floatX)
	mask = np.zeros((maxNumberOfAdmissions, numberOfPatients)).astype(config.floatX)
	feats_hotvectors_tensor = np.zeros((maxNumberOfAdmissions, numberOfPatients, ARGS.numberOfFeatsCodes)).astype(config.floatX)

	for idx, (train_patient_matrix,label_patient_matrix,feats_patient_matrix) in enumerate(zip(train_tensor,labels_tensor,features_tensor)):
		for i_th_visit, visit_line in enumerate(train_patient_matrix[:-1]): #ignores the last admission, which is not part of the training
			for code in visit_line:
				x_hotvectors_tensorf[i_th_visit, idx, code] = 1
		for i_th_visit, visit_line in enumerate(label_patient_matrix[1:]):  #label_matrix[1:] = all but the first admission slice, not used to evaluate (this is the answer)
			for code in visit_line:
				y_hotvectors_tensor[i_th_visit, idx, code] = 1
		for i_th_visit, time in enumerate(feats_patient_matrix[:-1]): #ignores the last admission, which is not part of the training
			feats_hotvectors_tensor[i_th_visit, idx, 0] = time
		mask[:nVisitsOfEachPatient_List[idx], idx] = 1.

	nVisitsOfEachPatient_List = np.array(nVisitsOfEachPatient_List, dtype=config.floatX)
	x_hotvectors_tensorb = x_hotvectors_tensorf[::-1,::,::] #backward tensor for bi-directional processing
	return x_hotvectors_tensorf, x_hotvectors_tensorb, y_hotvectors_tensor, feats_hotvectors_tensor, mask, nVisitsOfEachPatient_List


#initialize model tPARAMS
def init_params_BiMinGRU(previousDimSize):
	for count, hiddenDimSize in enumerate(ARGS.hiddenDimSize):  #by default: 0, 200; 1, 200 according to enumerate
		#http://philipperemy.github.io/xavier-initialization/
		xavier_variance = math.sqrt(2.0/float(previousDimSize+hiddenDimSize))
		tPARAMS['fWf_'+str(count)] = theano.shared(np.random.normal(0., xavier_variance, (previousDimSize, hiddenDimSize)).astype(config.floatX), name='fWf_'+str(count))
		tPARAMS['fUf_' + str(count)] = theano.shared(np.identity(hiddenDimSize).astype(config.floatX), name='fUf_' + str(count))
		tPARAMS['fbf_'+str(count)] = theano.shared(np.zeros(hiddenDimSize).astype(config.floatX), name='fbf_'+str(count))

		tPARAMS['fWh_'+str(count)] = theano.shared(np.random.normal(0., xavier_variance, (previousDimSize, hiddenDimSize)).astype(config.floatX), name='fWh_'+str(count))
		tPARAMS['fUh_' + str(count)] = theano.shared(np.identity(hiddenDimSize).astype(config.floatX), name='fUh_' + str(count))
		tPARAMS['fbh_'+str(count)] = theano.shared(np.zeros(hiddenDimSize).astype(config.floatX), name='fbh_'+str(count))

		tPARAMS['bWf_'+str(count)] = theano.shared(np.random.normal(0., xavier_variance, (previousDimSize, hiddenDimSize)).astype(config.floatX), name='bWf_'+str(count))
		tPARAMS['bUf_' + str(count)] = theano.shared(np.identity(hiddenDimSize).astype(config.floatX), name='bUf_' + str(count))
		tPARAMS['bbf_'+str(count)] = theano.shared(np.zeros(hiddenDimSize).astype(config.floatX), name='bbf_'+str(count))

		tPARAMS['bWh_'+str(count)] = theano.shared(np.random.normal(0., xavier_variance, (previousDimSize, hiddenDimSize)).astype(config.floatX), name='bWh_'+str(count))
		tPARAMS['bUh_' + str(count)] = theano.shared(np.identity(hiddenDimSize).astype(config.floatX), name='bUh_' + str(count))
		tPARAMS['bbh_'+str(count)] = theano.shared(np.zeros(hiddenDimSize).astype(config.floatX), name='bbh_'+str(count))

		previousDimSize = hiddenDimSize
	tPARAMS['fJ'] = theano.shared(np.identity(previousDimSize).astype(config.floatX), name='fJ')
	tPARAMS['bJ'] = theano.shared(np.identity(previousDimSize).astype(config.floatX), name='bJ')
	tPARAMS['fbb'] = theano.shared(np.zeros(previousDimSize).astype(config.floatX), name='fbb')
	tPARAMS['jlrelu'] = theano.shared(0.1, name='jlrelu')

	return previousDimSize


def fMinGRU_layer(inputTensor, layerIndex, hiddenDimSize, mask=None):
	# MinGRU: https://arxiv.org/pdf/1603.09420.pdf
	# Bidirectional RNNs: https://dl.acm.org/citation.cfm?id=2205129
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

def init_params_output_layer(previousDimSize):
	xavier_variance = math.sqrt(2.0 / float(previousDimSize + ARGS.numberOfInputCodes))
	tPARAMS['W_output'] = theano.shared(np.random.normal(0., xavier_variance, (previousDimSize, ARGS.numberOfInputCodes)).astype(config.floatX), 'W_output')
	tPARAMS['b_output'] = theano.shared(np.zeros(ARGS.numberOfInputCodes).astype(config.floatX), name='b_output')
	tPARAMS['olrelu'] = theano.shared(0.1, name='olrelu')

def dropout(nDimensionalData):
	randomS = RandomStreams(13713)
	newTensor = nDimensionalData * randomS.binomial(nDimensionalData.shape, p=ARGS.dropoutRate, dtype=nDimensionalData.dtype)
	#https://www.quora.com/How-do-you-implement-a-dropout-in-deep-neural-networks
	return newTensor

def build_model():
	xf = T.tensor3('xf', dtype=config.floatX)
	xb = T.tensor3('xb', dtype=config.floatX)
	y = T.tensor3('y', dtype=config.floatX)
	mask = T.matrix('mask', dtype=config.floatX)

	nVisitsOfEachPatient_List = T.vector('nVisitsOfEachPatient_List', dtype=config.floatX)
	maxNumberOfAdmissions = xf.shape[0]

	flowing_tensorf = xf
	flowing_tensorb = xb

	featsSlice = T.tensor3('t', dtype=config.floatX)
	# -----------
	flowing_tensorf = T.concatenate([featsSlice, flowing_tensorf], axis=2)
	flowing_tensorb = T.concatenate([featsSlice[::-1, ::, ::], flowing_tensorb], axis=2)

	for i, hiddenDimSize in enumerate(ARGS.hiddenDimSize):
		flowing_tensorf = fMinGRU_layer(flowing_tensorf, str(i), hiddenDimSize, mask=mask)
		flowing_tensorf = dropout(flowing_tensorf)

	for i, hiddenDimSize in enumerate(ARGS.hiddenDimSize):
		flowing_tensorb = bMinGRU_layer(flowing_tensorb, str(i), hiddenDimSize, mask=mask)
		flowing_tensorb = dropout(flowing_tensorb)

	#undo reverse before joining flows
	flowing_tensorb = flowing_tensorb[::-1,::,::]
	joint_flow = T.nnet.relu(T.dot(flowing_tensorf,tPARAMS['fJ']) + T.dot(flowing_tensorb,tPARAMS['bJ']) + tPARAMS['fbb'], tPARAMS['jlrelu'])

	results, _ = theano.scan(
		lambda theFlowingTensor: T.nnet.softmax(T.nnet.relu(T.dot(theFlowingTensor, tPARAMS['W_output']) + tPARAMS['b_output'], tPARAMS['olrelu'])),
		sequences=[joint_flow],
		outputs_info=None,
		name='softmax_layer',
		n_steps=maxNumberOfAdmissions)

	flowing_tensor = results * mask[:, :, None]

	epislon = 1e-8
	cross_entropy = -(y * T.log(flowing_tensor + epislon) + (1. - y) * T.log(1. - flowing_tensor + epislon))
	# the complete crossentropy equation is -1/n* sum(cross_entropy); where n is the number of elements
	# see http://neuralnetworksanddeeplearning.com/chap3.html#regularization
	prediction_loss = cross_entropy.sum(axis=2).sum(axis=0) / nVisitsOfEachPatient_List

	L2_regularized_loss = T.mean(prediction_loss) + ARGS.LregularizationAlpha*(tPARAMS['W_output'] ** 2).sum()
	MODEL = L2_regularized_loss
	return xf, xb, y, featsSlice, mask, nVisitsOfEachPatient_List, MODEL


#this code comes originally from deeplearning.net/tutorial/LSTM.html
#http://ruder.io/optimizing-gradient-descent/index.html#adadelta
#https://arxiv.org/abs/1212.5701
def addAdadeltaGradientDescent(grads, xf, xb, y, mask, nVisitsOfEachPatient_List, MODEL, featsSlice=None):
	zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in tPARAMS.iteritems()]
	running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k) for k, p in tPARAMS.iteritems()]
	running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k) for k, p in tPARAMS.iteritems()]

	zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
	rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

	TRAIN_MODEL_COMPILED = theano.function([xf, xb, y, featsSlice, mask, nVisitsOfEachPatient_List], MODEL, updates=zgup + rg2up, name='adadelta_TRAIN_MODEL_COMPILED')

	updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
	ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
	param_up = [(p, p + ud) for p, ud in zip(tPARAMS.values(), updir)]

	UPDATE_WEIGHTS_COMPILED = theano.function([], [], updates=ru2up + param_up, name='adadelta_UPDATE_WEIGHTS_COMPILED')
	return TRAIN_MODEL_COMPILED, UPDATE_WEIGHTS_COMPILED


def load_data():
	main_trainSet = pickle.load(open(ARGS.inputFileRadical+'.train', 'rb'))
	print("-> " + str(len(main_trainSet)) + " patients at dimension 0 for file: "+ ARGS.inputFileRadical + ".train dimensions ")
	main_testSet = pickle.load(open(ARGS.inputFileRadical+'.test', 'rb'))
	print("-> " + str(len(main_testSet)) + " patients at dimension 0 for file: "+ ARGS.inputFileRadical + ".test dimensions ")
	print("Note: these files carry 3D tensor data; the above numbers refer to dimension 0, dimensions 1 and 2 have irregular sizes.")

	ARGS.numberOfInputCodes = getNumberOfCodes([main_trainSet,main_testSet])
	print 'Number of diagnosis input codes: ' + str(ARGS.numberOfInputCodes)

	#uses the same data for testing, but disregarding the fist admission of each patient
	labels_trainSet = pickle.load(open(ARGS.inputFileRadical+'.train', 'rb'))
	labels_testSet = pickle.load(open(ARGS.inputFileRadical+'.test', 'rb'))

	ARGS.featsFile = ARGS.inputFileRadical + '.DURATION'
	print('Using features: ' + ARGS.featsFile)
	feats_trainSet = pickle.load(open(ARGS.featsFile+'.train', 'rb'))
	feats_testSet = pickle.load(open(ARGS.featsFile+'.test', 'rb'))

	ARGS.numberOfFeatsCodes = 1  # for time, we will need one single position
	print 'Number of features codes: ' + str(ARGS.numberOfFeatsCodes)

	train_sorted_index = sorted(range(len(main_trainSet)), key=lambda x: len(main_trainSet[x]))  #lambda x: len(seq[x]) --> f(x) return len(seq[x])
	main_trainSet = [main_trainSet[i] for i in train_sorted_index]
	labels_trainSet = [labels_trainSet[i] for i in train_sorted_index]

	test_sorted_index = sorted(range(len(main_testSet)), key=lambda x: len(main_testSet[x]))
	main_testSet = [main_testSet[i] for i in test_sorted_index]
	labels_testSet = [labels_testSet[i] for i in test_sorted_index]

	feats_trainSet = [feats_trainSet[i] for i in train_sorted_index]
	feats_testSet = [feats_testSet[i] for i in test_sorted_index]

	min_max_normalization(feats_trainSet, feats_testSet)

	trainSet = [main_trainSet, labels_trainSet, feats_trainSet]
	testSet = [main_testSet, labels_testSet, feats_testSet]

	return trainSet, testSet

#the performance computation uses the test data and returns the cross entropy measure
def performEvaluation(TEST_MODEL_COMPILED, test_Set):
	batchSize = ARGS.batchSize

	n_batches = int(np.ceil(float(len(test_Set[0])) / float(batchSize))) #default batch size is 100
	crossEntropySum = 0.0
	dataCount = 0.0
	#computes de crossEntropy for all the elements in the test_Set, using the batch scheme of partitioning
	for index in xrange(n_batches):
		batchX = test_Set[0][index * batchSize:(index + 1) * batchSize]
		batchY = test_Set[1][index * batchSize:(index + 1) * batchSize]
		batchT = test_Set[2][index * batchSize:(index + 1) * batchSize]
		xf, xb, y, featsSlice, mask, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY, batchT)
		crossEntropy = TEST_MODEL_COMPILED(xf, xb, y, featsSlice, mask, nVisitsOfEachPatient_List)

		#accumulation by simple summation taking the batch size into account
		crossEntropySum += crossEntropy * len(batchX)
		dataCount += float(len(batchX))
		#At the end, it returns the mean cross entropy considering all the batches
	return n_batches, crossEntropySum / dataCount

def train_model():
	print '==> data loading'
	trainSet, testSet = load_data()
	previousDimSize = ARGS.numberOfInputCodes

	print '==> parameters initialization'
	previousDimSize += ARGS.numberOfFeatsCodes
	print('Using neuron type Bidirectional Gated Recurrent Unit')
	previousDimSize = init_params_BiMinGRU(previousDimSize)
	init_params_output_layer(previousDimSize)

	print '==> model building'
	print('Using feature (procedure/type) information from files ' + ARGS.featsFile)
	xf, xb, y, featsSlice, mask, nVisitsOfEachPatient_List, MODEL =  build_model()
	grads = T.grad(theano.gradient.grad_clip(MODEL, -0.5, 0.5), wrt=tPARAMS.values())
	TRAIN_MODEL_COMPILED, UPDATE_WEIGHTS_COMPILED = addAdadeltaGradientDescent(grads, xf, xb, y, mask, nVisitsOfEachPatient_List, MODEL, featsSlice)

	print '==> training and validation'
	batchSize = ARGS.batchSize
	n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))
	TEST_MODEL_COMPILED = theano.function(inputs=[xf, xb, y, featsSlice, mask, nVisitsOfEachPatient_List], outputs=MODEL, name='TEST_MODEL_COMPILED')

	bestValidationCrossEntropy = 1e20
	bestValidationEpoch = 0
	bestModelFileName = ''

	iImprovementEpochs = 0
	iConsecutiveNonImprovements = 0
	epoch_counter = 0
	for epoch_counter in xrange(ARGS.nEpochs):
		iteration = 0
		trainCrossEntropyVector = []
		for index in random.sample(range(n_batches), n_batches):
			batchX = trainSet[0][index*batchSize:(index + 1)*batchSize]
			batchY = trainSet[1][index*batchSize:(index + 1)*batchSize]
			batchT = trainSet[2][index*batchSize:(index + 1)*batchSize]
			xf, xb, y, featsSlice, mask, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY, batchT)
			xf += np.random.normal(0, 0.1, xf.shape)  #add gaussian noise as a means to reduce overfitting
			xb += np.random.normal(0, 0.1, xb.shape)  #add gaussian noise as a means to reduce overfitting

			trainCrossEntropy = TRAIN_MODEL_COMPILED(xf, xb, y, featsSlice, mask, nVisitsOfEachPatient_List)
			trainCrossEntropyVector.append(trainCrossEntropy)
			UPDATE_WEIGHTS_COMPILED()
			iteration += 1
		#----------test -> uses TEST_MODEL_COMPILED
		#evaluates the network with the testSet
		print('-> Epoch: %d, mean cross entropy considering %d TRAINING batches: %f' % (epoch_counter, n_batches, np.mean(trainCrossEntropyVector)))
		nValidBatches, validationCrossEntropy = performEvaluation(TEST_MODEL_COMPILED, testSet)
		print('			 mean cross entropy considering %d VALIDATION batches: %f' % (nValidBatches, validationCrossEntropy))
		if validationCrossEntropy < bestValidationCrossEntropy:
			iImprovementEpochs += 1
			iConsecutiveNonImprovements = 0
			bestValidationCrossEntropy = validationCrossEntropy
			bestValidationEpoch = epoch_counter

			tempParams = unzip(tPARAMS)
			if os.path.exists(bestModelFileName):
				os.remove(bestModelFileName)
			np.savez_compressed(ARGS.outFile + '.' + str(epoch_counter), **tempParams)
			bestModelFileName = ARGS.outFile + '.' + str(epoch_counter) + '.npz'
		else:
			print('Epoch ended without improvement.')
			iConsecutiveNonImprovements += 1
		if iConsecutiveNonImprovements > ARGS.maxConsecutiveNonImprovements: #default is 10
			break
	#Best results
	print('--------------SUMMARY--------------')
	print('The best VALIDATION cross entropy occurred at epoch %d, the value was of %f ' % (bestValidationEpoch, bestValidationCrossEntropy))
	print('Best model file: ' + bestModelFileName)
	print 'Used features file with max number of procedures code = ' + str(ARGS.numberOfFeatsCodes)
	print('Number of improvement epochs: ' + str(iImprovementEpochs) + ' out of ' + str(epoch_counter+1) + ' possible improvements.')
	print('Note: the smaller the cross entropy, the better.')
	print('-----------------------------------')

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('inputFileRadical', type=str, metavar='<visit_file>', help='File radical name (the software will look for .train and .test files) with pickled data organized as patient x admission x codes.')
	parser.add_argument('--featsFile', type=str, default='', help='A file containing features (time duration, for instance) to concatenate to the input.')
	parser.add_argument('outFile', metavar='out_file', default='model_output', help='Any file name to store the model.')
	parser.add_argument('--maxConsecutiveNonImprovements', type=int, default=10, help='Training wiil run until reaching the maximum number of epochs without improvement before stopping the training')
	parser.add_argument('--hiddenDimSize', type=str, default='[271]', help='Number of layers and their size - for example [100,200] refers to two layers with 100 and 200 nodes.')
	parser.add_argument('--batchSize', type=int, default=100, help='Batch size.')
	parser.add_argument('--nEpochs', type=int, default=1000, help='Number of training iterations.')
	parser.add_argument('--LregularizationAlpha', type=float, default=0.001, help='Alpha regularization for L2 normalization')
	parser.add_argument('--dropoutRate', type=float, default=0.45, help='Dropout probability.')

	ARGStemp = parser.parse_args()
	hiddenDimSize = [int(strDim) for strDim in ARGStemp.hiddenDimSize[1:-1].split(',')]
	ARGStemp.hiddenDimSize = hiddenDimSize
	return ARGStemp

if __name__ == '__main__':
	global tPARAMS
	tPARAMS = OrderedDict()
	global ARGS
	ARGS = parse_arguments()

	train_model()