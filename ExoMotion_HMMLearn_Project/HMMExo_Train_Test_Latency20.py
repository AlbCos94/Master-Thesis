# pylint: disable = missing-docstring, relative-import, logging-not-lazy

import pdb # debugging library
import pickle
import os
import datetime
import logging
import warnings
import time
import argparse
import csv
import sys
import re
from operator import itemgetter, add

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from Component import ArgParseError, Component, Field, getListIdxFromName
import Motion_Datareader_Train_Test as Datareader
import coloring as col
from TrainingStatistics import TrainingStatistics


warnings.filterwarnings("ignore", category=DeprecationWarning)

''' General documentation
HMMExo_Train_Test is used to train an HMM-based model to predict/detect motions
performed by an exoskeleton user. Data from the internal sensors of the
exo (collected into windows by Motion_Datareader_Train_Test) is used to
train the model.
Stratified k-fold cross validation is used to train and evaluate several
models. The model with the highest prediction rate is selected and saved
for later (online) usage.
HMMExo_Train_Test offers to select the testpersons and modalities for
which the model is trained.

For usage instructions refer to the README or run this script with -h.
'''

''' Parameters for HMM '''
NUM_STATES = 6
NUM_FOLDS = 5
MAX_FITTING_TRIES = 10
windowsmotion = None
window_step_size = None
window_size = None
pathCSV='9999'
''' Parameters for modular input '''
# Components of 'exo1'
ZMD1_COMP = Component('ZMD', 1, 8, [])  # name, numSensors, valuesPerSensor
IMU1_COMP = Component('IMU', 1, 3, [])
JA1_COMP = Component('JA', 1, 4, [])  # JointAngle
JT1_COMP = Component('JT', 1, 4, [])  # JointTorque

# Components of 'exo2'
# ZMD
X_FIELD = Field('x', 0)  # name, begin (inclusive start index)
Y_FIELD = Field('y', 1)
Z_FIELD = Field('z', 2)
ZMD2_COMP = Component('ZMD', 7, 3, fields=[X_FIELD, Y_FIELD, Z_FIELD])
# IMU
LIN_ACC_COMP = Component('l', 3, 3, fields=[]) # LinearAcceleration
QUATERNION_COMP = Component('q', 3, 4, fields=[]) # Quaternion
IMU2_COMP = Component('IMU', 3, 7, components=[QUATERNION_COMP, LIN_ACC_COMP])
# Euler Angles
EULER_COMP = Component('e', 3, 3, fields=[])
IMU3_COMP = Component('IMU', 3, 10, components=[QUATERNION_COMP, LIN_ACC_COMP, EULER_COMP])

# List of available exo versions
EXO1 = Component('exo1', 1, 19, components=[ZMD1_COMP, IMU1_COMP, JA1_COMP, JT1_COMP])
EXO2 = Component('exo2', 1, 42, components=[ZMD2_COMP, IMU2_COMP])
EXO2_EULER = Component('exo2euler', 1, 51, components=[ZMD2_COMP, IMU3_COMP])
VERSIONS = [EXO1, EXO2, EXO2_EULER]

''' File name parameters '''
HMM_FILE = 'HMMBestTrained.npy'
SCALER_FILE = 'Scaler.pkl'



# if file exists, it will be deleted
if (os.path.exists(pathCSV)):
    os.remove(pathCSV)

def main():
    
    outputFileParam = args.outputFileParam
    global pathCSV
    pathCSV = 'LogFilesLatency/Latencytest/' + outputFileParam + '.csv'
    
    # track the run time
    startTime = time.time()
    modelDir = os.path.join(args.basepath, 'MotionPrediction', args.test_model)
    test_only = bool(args.test_model)
    writeDir = getWriteDir(args.basepath, args.testpersons, args.modalities, test_only)

    global NUM_STATES
    NUM_STATES = args.states
    print NUM_STATES
    global window_step_size
    window_step_size = args.stepsize
    print window_step_size
    global window_size
    window_size = args.windowsize
    print window_size
    
    # windows contains the keys 'X' and 'y', where X contains the data and y the motion label of the
    # window. Read only the columns selected via command-line input.
    windows = readWindows(args.basepath, args.testpersons,
                          args.exoversion, args.modalities, args.motions)
    # scale windows
    scale(windows, writeDir, modelDir, test_only)

    if args.test_model:
        # only do testing
        statistics = testAll(windows, modelDir)
    else:
        # train models and select the best one
        hmmFile, statistics = trainAndTest(windows)
        saveBestHmm(hmmFile, statistics, writeDir)
    createLogFiles(writeDir, statistics)

    # end time
    endTime = time.time()
    #duration of program
    duration = endTime - startTime

    col.printout("Start time of program: {} sec\n".format(startTime))
    col.printout("End time of program: {} sec\n".format(endTime))
    col.printout("Duration of program: {} sec\n".format(duration))

'''
Filter the motions in X and y by removing all windows whose label is
listed in one of the following files:
BASEPATH/Invalid_Motions.csv
BASEPATH/{testperson}/Short_Recordings.csv
BASEPATH/{testperson}/EMG/Invalid_Recodings.txt
If --motions is specified, remove all motions that are not mentioned
'''
def filterMotions(X, y, testperson, basepath, prefixes):
    
    # get short/invalid motions
    falseMotions = getFalseMotions(testperson, basepath)
    # expand prefixed in selectedMotions
    # filter
    windowsmotionzip = zip(X, y)
    windowsmotionzip[:] = [(x, (label,motion)) for (x, (label, motion)) in windowsmotionzip
                  if not motion in falseMotions
                  and (not prefixes or prefixes and isSubstring(motion, prefixes))]
    X[:], y[:] = zip(*windowsmotionzip) if windowsmotionzip else ([], [])
    global windowsmotion
    windowsmotion = zip(*y)[1]
    windows = zip(X, y)
    windows[:] = [(x, label) for (x, (label, motion)) in windows
                  if not motion in falseMotions
                  and (not prefixes or prefixes and isSubstring(motion, prefixes))]
    X[:], y[:] = zip(*windows) if windows else ([], [])


def isSubstring(string, strList):
    """ Return true if any of the strings in strList is a substring of string. """
    for substring in strList:
        if substring in string:
            return True
    return False

'''
Return all invalid and short motions for the specified testperson name.
'''
def getFalseMotions(testperson, basepath):
    invMotions = readInvalidMotions(testperson, basepath)
    shortRecordings = readShortRecordings(testperson, basepath)
    invRecordings = readInvalidRecordings(testperson, basepath)
    return list(set(invMotions + shortRecordings + invRecordings))

def readInvalidMotions(testperson, basepath):
    filename = os.path.normpath(basepath + '/' + 'Invalid_Motions.csv')
    motions = []
    if os.path.exists(filename):
        with open(filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Testperson'] == testperson:
                    motions.append(row['Motion'])
    return motions

def readShortRecordings(testperson, basepath):
    filename = os.path.normpath(basepath + '/' + testperson + '/Short_Recordings.csv')
    motions = []
    if os.path.exists(filename):
        with open(filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                motions.append(row['Motion'])
    return motions

def readInvalidRecordings(testperson, basepath):
    filename = os.path.normpath(basepath + '/' + testperson + '/EMG/Invalid_Recodings.txt')
    motions = []
    if os.path.exists(filename):
        with open(filename) as f:
            for line in f:
                line = line.strip()
                # B1_mE_LX_001 -> B1_mE_L1_001, B1_mE_L2_001
                motions += [line.replace('X', '1'), line.replace('X', '2')]
    return motions

def parseArgs():
    ''' parse command-line arguments '''
    parser = argparse.ArgumentParser()
    # add relevant arguments to the parser
    parser.add_argument('-b', '--basepath',
                        help='folder where all the experiment data lies'
                        + ' The testperson folders are subfolders of BASEPATH.'
                        + ' Possible names for testperson folders are: '
                        + str([name + 'xy' for name in Datareader.TESTPERSONS])
                        + ' for new data or "SegmentedData_Sorted*" for old data.',
                        default=Datareader.BASEPATH)
    parser.add_argument('-t', '--testpersons', help='Select the testpersons '
                        + 'whose data is used to train the HMM. Use only integers.'
                        + ' If -t is not specified, all testpersons are selected.'
                        + ' Use "-t 1,3,5" to select the testpersons 1, 3 and 5. Leading zeros '
                        + 'do not need to be added to the number, "-t 1" for example will select '
                        + 'a testperson named "ID01" or "Proband 00001".',
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('-v', '--exoversion', help='Select the exo version '
                        + 'with wich the data was created. [exo1|exo2]',
                        default=Datareader.EXO_STD)
    parser.add_argument('modalities', nargs='*',
                        type=lambda s: [item for item in s.split('=')],
                        help='Select the modalities for training the HMM. '
                        + '"ZMD" selects all ZMD values, "ZMD=xz" selects '
                        + 'the x and z values of all ZMD sensors and '
                        + '"ZMD=1x,2,3y" selects the x value of the first '
                        + 'sensor, the y value of the third sensor and all '
                        + 'values of the second sensor. Usage with IMU is '
                        + 'similiar: possible values are "l" for Linear'
                        + 'Accelerations and "q" for Quaternion.')
    parser.add_argument('-m', '--motions', help='Select the motions labels for which the HMM is '
                        + 'trained. If -m is not specified, all labels are selected.'
                        + ' Use "-m B1,B2_oE" for example to select all motions who have B1 or '
                        + 'B2_oE somewhere in their name. Or use "-m mE,B3_oE_L2" to select all '
                        + 'motion with the exo (mE) as well as all motions starting with B3_oE_L2.',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('-l', '--loglevel', default='WARNING',
                        help='Logging level [debug, info, warning]')
    parser.add_argument('-s', '--states', default=14, type=int,
                        help='Select number of states')
    parser.add_argument('--stepsize', default=10, type=int,
                        help='Define window step size')
    parser.add_argument('--windowsize', default=500, type=int,
                        help='Define window size')  
    parser.add_argument('-f', '--outputFileParam', default='9999', type=str,
                        help='Define OutputFile parameter :)')                    
    parser.add_argument('--test-model', help='Test the existing model that was trained at the'
                        + 'specified datetime.', default='')
    return parser.parse_args()

def initLogging(loglevel):
    ''' Initialize logging. '''
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=numeric_level)
    return logging.getLogger(__name__)

def selectColumns(exoversion, modalities):
    """ Decide which window columns (aka which feature vector) to use

    :param exoversion:
    :param modalities:
    :return: columns

    >>> selectColumns('exo1', [])
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    >>> selectColumns('exo2', [['ZMD']])  # 7 sensors, each yielding 3 values
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    >>> selectColumns('exo2', [['ZMD', 'x']])  # x values of all 7 sensors
    [0, 3, 6, 9, 12, 15, 18]
    >>> selectColumns('exo2', [['ZMD', '1']])  # all values of first sensor
    [0, 1, 2]
    >>> selectColumns('exo2', [['ZMD', '1x,2,3y']])
    [0, 3, 4, 5, 7]
    >>> selectColumns('exo2', [['ZMD', '1x']]) # x value of first sensor
    [0]

    """
    exo = VERSIONS[getListIdxFromName(exoversion, VERSIONS)]
    columns = []
    for mod in modalities:
        try:
            comp = exo.getComponent(mod[0])
            if len(mod) == 1:
                columns += comp.getAll()
                continue

            # extract the desired sub-components, fields and sensors
            subs = mod[1].split(',')
            for sub in subs:
                match = re.match(r'[0-9]+', sub)
                if match:
                    sensor = int(match.group()) - 1
                    sub = sub[len(match.group()):]
                else:
                    sensor = None
                if not sub:
                    # get all values for this sensor
                    columns += comp.getFromSensor(sensor)
                    continue
                # the remaining entries in sub should be fields or components
                for char in sub:
                    columns += comp.getPartFromSensor(char, sensor)
        except (ArgParseError, ValueError):
            # catch ValueError in order to omit traceback that is not helpful for the user
            sys.tracebacklimit = 0
            raise
    # if no modalities where selected, use all values
    if not columns:
        columns = exo.getAll()
    # sort the columns and make them unique
    return sorted(list(set(columns)))

def readWindows(basepath, testpersons, exoversion, modalities, motions):
    '''
    Read windows in npy and pkl files for all testpersons in basepath.

    Returns a tuple of a dictionary (windows) and a scaler instance. The windows dictionary's keys
    are the testperson names and whose values are dictionaries with the keys 'X' and 'y'.
    windows[testperson]['X'] is a list of numpy arrays where each array represents one window and
    each row in an array is the sensor data at one timestamp.
    windows[testperson]['y'] is numpy array of strings where each string is the label of the
    corresponding window in 'X'.
    '''
    windows = {}
    # select the columns
    columns = selectColumns(exoversion, modalities)
    visitedTestpersons = set([])
    # read the data
    for testperson in sorted(os.listdir(basepath)):
        if (not os.path.isdir(basepath + '/' + testperson)
                or not Datareader.isTestpersonDir(testperson)):
            continue
        personNr = Datareader.getTestpersonNumber(testperson)
        if testpersons and personNr not in testpersons:
            continue
        try:
            readWindowsForTestperson(basepath, testperson, motions, windows, columns)
            visitedTestpersons.add(personNr)
        except IOError:
            continue
            
    #pdb.set_trace()
    if not windows:
        col.printout('There are no valid windows for the specified testpersons!\n', col.RED)
        sys.exit(1)

    # check whether windows for all specified testpersons were found
    if testpersons and not visitedTestpersons == set(testpersons):
        col.printout('No windows could be found for the following testperson IDs: '
                     + '{}.\n'.format(list(set(testpersons) - visitedTestpersons))
                     + 'Please move the data to an appropriate folder and run '
                     + 'Motion_Datareader_Train_Test.py on it!\n', col.RED)
        sys.exit(1)
    return windows

def readWindowsForTestperson(basepath, testperson, motions, windows, columns):
    dirpath = os.path.normpath(basepath + '/' + testperson+ '/MotionPrediction')
    # Read sensor values of all windows
    xFile = np.load(dirpath + '/ExoMotions_X_Train_Test_{}.npy'.format(testperson))
    XNew = []
    for x in xFile:
        # shape of x: (number of timestamps, number of sensor values)
        x = np.array(x, 'float32')
        # only use the selected columns (aka sensor modalities)
        assert max(columns) < x.shape[1]
        x = x[:, columns]
        XNew.append(x)
    # Read motions for all windows
    with open(dirpath + '/ExoMotions_y_Train_Test_{}.pkl'.format(testperson), 'rb') as yFile:
        yNew = pickle.load(yFile)
    # filter short/invalid motions from X and y
    filterMotions(XNew, yNew, testperson, basepath, motions)
    if yNew:
        # save windows seperately for each testperson
        windows[testperson] = {'X': XNew, 'y': np.array(yNew)}

def generateFoldIndices(numFolds, windows):
    indices = {}
    skf = StratifiedKFold(numFolds, shuffle=True)
    for testperson in windows:
        idxGenerator = skf.split(windows[testperson]['X'], windows[testperson]['y'])
        indices[testperson] = [idxTuple for idxTuple in idxGenerator]
    return indices

def splitData(train, test, windows):
    """
    Split the windows into train and test datasets.

    Return a tuple (trainWindows, testWindows). Each of the dictionaries contains entries for the
    data ('X') and the labels ('y').

    :param train: list of indexes in windows['y']/windows['X'] that belong to the trainings dataset
    :param test: list of indexes in windows['y']/windows['X'] that belong to the test dataset
    :param windows: dictionary with the entries 'X' (list of sensor data for each window) and 'y'
                    (window labels)
    """
    # Split data into training and test data for this lap
    trainWindows = {}
    # X[train] not possible because X is list of numpy arrays and not a numpy array itself
    trainWindows['X'] = [windows['X'][idx] for idx in train]
    trainWindows['y'] = windows['y'][train]
    testWindows = {}
    testWindows['X'] = [windows['X'][idx] for idx in test]
    testWindows['y'] = windows['y'][test]

    return (trainWindows, testWindows)

def createScaler(windows):
    """ Returns a scaler fitted to the windows. """
    # Create new scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # Compute the minimum and maximum to be used for later scaling
    # Min/Max is computed for every type of sensor value, eg. ZMD0, IMU1 etc.
    X = reduce(add, [subjectWindows['X'] for subjectWindows in windows.values()])
    try:
        scaler.fit(np.vstack(X))
    except ValueError:
        col.printout('Scaling of windows not possible, there are no valid windows!\n', col.RED)
        sys.exit(1)
    return scaler

def readScaler(readDir):
    """ Reads a scaler from file and returns it. """
    filename = os.path.join(readDir, SCALER_FILE)
    if not os.path.exists(filename):
        col.printout('No scaler file found!\n', col.RED)
        col.printout('Maybe the data in {} is from a test-only run.\n'.format(readDir), col.RED)
        sys.exit(1)
    with open(filename, 'rb') as scalerFile:
        scaler = pickle.load(scalerFile)
    return scaler

def scaleWindows(windows, scaler):
    """ Scale and save the X data in train- and testWindows. """
    # Scaling features of val according to feature_range.
    for testperson, oldWindows in windows.items():
        windows[testperson]['X'] = [scaler.transform(val) for val in oldWindows['X']]

def scale(windows, writeDir, readDir, test_only):
    # obtain scaler
    if test_only:
        scaler = readScaler(readDir)
        featureSize = windows.values()[0]['X'][0].shape[1]
        if scaler.scale_.size != featureSize:
            # shape of the original scaled data is different from the received data
            col.printout('Received data has a different number of features than the '
                         + 'training data!\n', col.RED)
            col.printout('Did you train with the same sensor combination you are using now?\n',
                         col.RED)
            sys.exit(1)
    else:
        scaler = createScaler(windows)
    # do the actual scaling
    scaleWindows(windows, scaler)
    # save scaler if necessary
    if not test_only:
        saveScaler(writeDir, scaler)

def trainModel(trainWindows):
    # fully connected hmms. Use one hmm for each possible motion.
    unique_labels = list(set(trainWindows['y']))
    hmms = [(label, GaussianHMM(n_components=NUM_STATES,
                                covariance_type="diag", n_iter=100))
            for label in unique_labels]

    hmms[:] = [hmm for hmm in hmms if trainHMM(hmm, trainWindows)]
    # make sure that not all HMMs were deleted
    assert hmms
    return hmms

def trainHMM((label, hmm), trainWindows):
    X_train_label = [val for idx, val in enumerate(trainWindows['X'])
                     if trainWindows['y'][idx] == label]
    # try several times to fit the model, hoping for a succesful initialization
    for tryNr in range(MAX_FITTING_TRIES):
        try:
            # fit: training HMM
            hmm.fit(X_train_label)
            logger.debug('Successfully fitted HMM to model for '
                         + 'label {} at try {}'.format(label, tryNr))
            return True
        except ValueError:
            logger.debug('Could not fit HMM to model for '
                         + 'label {} at try {}'.format(label, tryNr))
    # no HMM could be fitted to this label's data
    logger.warning('Deleting label {} from model'.format(label)
                   + ', because no HMM could be fit')
    return False

def testModel(hmms, testWindows, statistics):
    """
    Classify unkown motion aka test the model that was trained.

    For each test data pair (feature vector + motion label) and each
    HMM get the likelihood that the HMM would predict the motion label
    when given the feature vector. The predicted label is then the
    label associated with the HMM that scores the highest likelihood.
    """
    for testperson in testWindows:
        testModelForTestperson(testperson, hmms, testWindows[testperson], statistics)

def testModelForTestperson(testperson, hmms, testWindows, statistics):
    """
    Classify unknown motion aka test the model that was trained.

    For each test data pair (feature vector + motion label) and each
    HMM get the likelihood that the HMM would predict the motion label
    when given the feature vector. The predicted label is then the
    label associated with the HMM that scores the highest likelihood.
    """
    global pathCSV
    global windowsmotion
    global window_step_size
    print "window step size" + str(window_step_size)
    global window_size
    print "window size" + str(window_size)
    #print len(windowsmotion)
    #print len(testWindows['y'])
    #sys.exit()
    y_test_curr_motion_old = windowsmotion[0]
    #y_test_curr_motion = None
    step_counter = 0
    k = 0
    i = 0
    o = 0
    prediction_counter = 0
    counter_offset = 20
    for X_test_curr, y_test_curr in zip(testWindows['X'], testWindows['y']):
        y_test_curr_motion = windowsmotion[i]
        # the score method calculates the loglikelihood
        predictionstart = int(round(time.time() * 1000))
        scores = [(label, hmm.score(X_test_curr)) for label, hmm in hmms]

        # chose motion with highest score
        predicted_label = max(scores, key=itemgetter(1))[0]
        statistics.report_score(testperson, scores, predicted_label, y_test_curr)
        predictiontime = int(round(time.time() * 1000))-predictionstart
        if k == 0:
            if y_test_curr_motion_old == y_test_curr_motion:
                if predicted_label == y_test_curr:
                    step_counter += window_step_size
                    prediction_counter += 1
                    if prediction_counter > counter_offset:
                        step_counter -= counter_offset*window_step_size
                        ####-1000 Da 1s lang 0er in den Files stehen.
                        timetoprediction = (step_counter+window_size+predictiontime)-1000
                        print 'Prediction ' + str(y_test_curr_motion) + ' correct after' + str(timetoprediction)
                        #print step_counter
                        with open(pathCSV, 'a') as csvfile:
                            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                            csvwriter.writerow([y_test_curr_motion, timetoprediction, predictiontime, step_counter, predicted_label])
                        k = 1
                        prediction_counter = 0
                        o = 0
                else:
                    if o < 2:
                        step_counter += window_step_size
                        o +=1
                    else:
                        prediction_counter = 0
                        step_counter += window_step_size
                        o = 0
        if y_test_curr_motion_old != y_test_curr_motion:
            step_counter = 0
            k = 0
        # compares predicted movement with actual movement
        if logger.getEffectiveLevel() == logging.INFO:
            if predicted_label != y_test_curr:
                col.printout('predicted {}, is {}\n'.format(predicted_label, y_test_curr), col.WHITE)
            #col.printout('predicted {}, is {}\n'.format(predicted_label, y_test_curr),
            #             col.RED if predicted_label != y_test_curr else col.WHITE)
        y_test_curr_motion_old = y_test_curr_motion
        i += 1
        
def concatenateWindows(oldWindows, newWindows):
    assert set(oldWindows.keys()) == set(['X', 'y'])
    oldWindows['X'] += newWindows['X']
    if oldWindows['y'].size != 0:
        oldWindows['y'] = np.hstack((oldWindows['y'], newWindows['y']))
    else:
        oldWindows['y'] = newWindows['y']

def prepareLap(lap, foldIndices, windows):
    """
    Devide windows into train and test folds according to the index lists train and test.

    Return the train and test folds. The train fold is a list of all training windows for all
    testpersons, whereas the test fold is a directory containing the test windows for this lap
    seperated by testperson.
    """
    trainWindows = {'X': [], 'y': np.array([])}
    testWindows = {}
    for testperson in windows:
        train, test = splitData(foldIndices[testperson][lap][0],
                                foldIndices[testperson][lap][1],
                                windows[testperson])
        # concatenate the training windows
        concatenateWindows(trainWindows, train)
        # save the test windows seperately for each testperson
        testWindows[testperson] = test
        assert testWindows.keys() == foldIndices.keys()
    lenTestWindows = [len(t['X']) for t in testWindows.values()]
    numTestWindows = reduce(add, lenTestWindows)
    col.printout('Round {}/{} (Train {}, Test {} (per subject: {})\n'
                 .format(lap + 1, NUM_FOLDS, len(trainWindows['X']), numTestWindows,
                         lenTestWindows),
                 col.WHITE)

    return (trainWindows, testWindows)

def runLap((trainWindows, testWindows), statistics):
    hmms = trainModel(trainWindows)

    testModel(hmms, testWindows, statistics)

    return hmms

def getWriteDir(basepath, testpersons, modalities, test_only):
    """ Create folder to write training data to and add the specification to README.txt. """
    writeBase = os.path.normpath(basepath + '/MotionPrediction')
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    writeDir = os.path.normpath(writeBase + '/' + date) + '/'
    if not os.path.isdir(writeDir):
        os.makedirs(writeDir)

    # save parameters of this run in README file
    with open(writeBase + '/README.txt', 'a') as readme:
        mods = modalities if modalities else 'all'
        persons = testpersons if testpersons else 'all'
        readme.write(date + ' -> modalities: {}, testpersons: {}'.format(
            mods, persons))
        if test_only:
            readme.write(', test results only!')
        readme.write('\n')
    return writeDir

def trainAndTest(windows):
    ''' train  and test HMM with stratified k-fold cross-validation '''
    # List of all models that are trained. Each model consists of several HMMs.
    hmmFile = []
    # provide the statistics for the following test runs with the number of folds and the testperson
    # names
    statistics = TrainingStatistics(NUM_FOLDS, windows.keys())

    foldIndices = generateFoldIndices(NUM_FOLDS, windows)
    for lap in range(NUM_FOLDS):
        statistics.next_lap()
        folds = prepareLap(lap, foldIndices, windows)
        hmms = runLap(folds, statistics)
        hmmFile.append(hmms)

        ''' Some statistics and debug output '''
        col.printout('Accuracy per subject: {}\n'.format(statistics.testperson_accuracies(lap)),
                     col.GREEN)
        col.printout('Total accuracy in this round: {:.2f}%\n'.format(statistics.lap_accuracy(lap)),
                     col.GREEN)

    right, wrong = statistics.right_wrong_predictions()
    alignment = max(len(str(right)), len(str(wrong)))
    col.printout('\nWrong predictions: {:>{}}\n'.format(wrong, alignment), col.RED)
    col.printout('Right predictions: {:>{}}\n'.format(right, alignment), col.GREEN)
    col.printout('Total hit rate ({} states): {:.2f}%\n'.format(NUM_STATES, \
                 statistics.total_accuracy()), col.GREEN)

    return (hmmFile, statistics)

def testAll(windows, readDir):
    # read the model from file
    hmms = np.load(os.path.join(readDir, HMM_FILE))

    # initialize statistics
    statistics = TrainingStatistics(1, windows.keys())
    statistics.next_lap()

    # test the model on the received windows
    lenTestWindows = [len(t['X']) for t in windows.values()]
    numTestWindows = reduce(add, lenTestWindows)
    col.printout('Testing the model on {} windows (per subject: {}):\n'
                 .format(numTestWindows, lenTestWindows))
    testModel(hmms, windows, statistics)

    # Some statistics and debug output
    col.printout('Accuracy per subject: {}\n'.format(statistics.testperson_accuracies(0)),
                 col.GREEN)
    col.printout('Total accuracy in this round: {:.2f}%\n'.format(statistics.lap_accuracy(0)),
                 col.GREEN)

    right, wrong = statistics.right_wrong_predictions()
    alignment = max(len(str(right)), len(str(wrong)))
    col.printout('\nWrong predictions: {:>{}}\n'.format(wrong, alignment), col.RED)
    col.printout('Right predictions: {:>{}}\n'.format(right, alignment), col.GREEN)
    return statistics

def createLogFiles(writeDir, statistics):
    """ Create files that document this training run. """
    # create HMMExo_online_Statistic file
    with open(writeDir + "HMMExo_Train_Test_Statistic.txt", "w") as statistic:
        statistic.write("Format: ('Predicted', 'TrueValue'): "
                        + "number of wrong predictions for that lap and testperson \n\n")
        pairs = statistics.all_prediction_pairs()
        for (lap, testperson), values in pairs.items():
            statistic.write("Lap {}, Testperson {}:\n".format(lap, testperson))
            for stat, number in values:
                statistic.write('{}: {}x \n'.format(stat, number))

    # create Loglikelihood file
    with open(writeDir + "Loglikelihood.txt", "w") as loglikeFile:
        for log in statistics.all_logs():
            loglikeFile.write("{}\n".format(log))

def saveBestHmm(hmmFile, statistics, writeDir):
    """ Print some info on the success of the trained HMMs. """
    # find best HMM
    hitRatePerIteration = statistics.lap_accuracies()
    hmmPos = np.argmax(hitRatePerIteration)
    bestHmm = hmmFile[hmmPos]

    #if logger.getEffectiveLevel() == logging.INFO:
        # print transition matrix only if INFO level was selected
    #    for label, hmm in bestHmm:
    #        col.printout('\nTransition Matrix - {}\n'.format(label), col.BLUE)
    #        transMatrix = hmm.transmat_
            # TODO: Replace with numpy output function if result is similiar
    #        for i in range(0, NUM_STATES):
    #            print '\t',
    #            for j in range(0, NUM_STATES):
    #                print '{0:0.4f}\t'.format(transMatrix[i][j]),
    #            print ''

    ''' Save the HMM with highest accuracy to use it for online prediction. '''
    col.printout("hitrateperiter: {}\n".format(hitRatePerIteration))
    col.printout("hmmpos : {}\n".format(hmmPos))
    np.save(writeDir + HMM_FILE, bestHmm)

def saveScaler(writeDir, scaler):
    """ Save the scaler so it can be used to scale data for online prediction. """
    with open(writeDir + '/' + SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    args = parseArgs()
    print args.modalities
    logger = initLogging(args.loglevel)
    main()
