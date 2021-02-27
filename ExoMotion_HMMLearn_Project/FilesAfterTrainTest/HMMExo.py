import numpy as np
import pickle
import os
import coloring as col
from hmmlearn.hmm import GaussianHMM
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

tmp = np.load('ExoMotions_X.npy')
X = []
wrong = 0
right = 0
lap = 0
nrOfStates = 13
nrOfFolds = 3
 
for x in tmp:
	X.append(np.array(x, 'float32'))
	#print X
with open('ExoMotions_y.pkl', 'rb') as f:
	y = np.array(pickle.load(f))
	
for train, test in StratifiedKFold(y, n_folds= nrOfFolds):
	lap = lap + 1
	
	# Split
	X_train = [X[idx] for idx in train]
	y_train = y[train]
	X_test = [X[idx] for idx in test]
	y_test = y[test]
	
	col.printout('Round {}/{} (Train {}, Test {})\n'.format(lap, nrOfFolds, len(X_train), len(X_test)), WHITE)
	
	# Scale
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler.fit(np.vstack(X_train))
	for idx, val in enumerate(X_train):
		X_train[idx] = scaler.transform(val)
	for idx, val in enumerate(X_test):
		X_test[idx] = scaler.transform(val)
	
	unique_labels = list(set(y_train))
	hmms = [GaussianHMM(n_components= nrOfStates, covariance_type="diag", n_iter=100) for _ in unique_labels]
	for label, hmm in zip(unique_labels, hmms):
		# Get training data for that label
		X_train_label = [val for idx, val in enumerate(X_train) if y_train[idx] == label]
		y_train_label = [val for idx, val in enumerate(y_train) if y_train[idx] == label]
		
		#X.reshape(1,-1)
		hmm.fit(X_train_label)
		loglikelihood = hmm.score(X_train_label[0])
	
	# Classify unkown motion
	for X_test_curr, y_test_curr in zip(X_test, y_test):
		scores = []
		for hmm in hmms:
			scores.append(hmm.score(X_test_curr))
		
		predicted_label = unique_labels[np.argmax(scores)]
		
		if (predicted_label != y_test_curr):
			wrong = wrong + 1
			col.printout('predicted {}, is {}\n'.format(predicted_label, y_test_curr), RED)
		else:
			right = right + 1
			print 'predicted {}, is {}'.format(predicted_label, y_test_curr)
			
hitRate = right / float(wrong+right) * 100
col.printout('\nWrong predictions: {0:3d}\nRight predictions: {1:3d}\nTotal hit rate ({2:2d} States): {3:.2f}%\n'.format(wrong, right, nrOfStates,hitRate), GREEN)

idLabel = 0
for hmm in hmms:
	col.printout('\nTransition Matrix - {}\n'.format(unique_labels[idLabel]), BLUE)
	idLabel = idLabel + 1
	transMatrix = hmm.transmat_
	i = 0
	while (i < nrOfStates):
		j = 0
		print '\t',
		while (j < nrOfStates):
			print '{0:0.4f}\t'.format(transMatrix[i][j]),
			j = j +1
		print ''
		i = i + 1



