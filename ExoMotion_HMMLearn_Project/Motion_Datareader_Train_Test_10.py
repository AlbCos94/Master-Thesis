"""
Module for data reading.

Motion_Datareader_Train_Test:
    Read data, combine modalites and create windows.
BASEPATH:
    Standard path to data from the exoskeleton.
TESTPERSONS:
    List of valid beginnings of testperson names.
isTestpersonDir(testpersonDir):
    Returns True if the folder testpersonDir is a valid testperson name.
getTestpersonNumber(testperson):
    Returns the ID of a testperson which is located at the end of its name.
"""

# pylint: disable = unused-import, relative-import, superfluous-parens

from __future__ import division
import os
import pickle
import re
import pdb
import json
import csv
import argparse
import logging
import sys
from operator import xor
import transforms3d.quaternions as trafo3d
import numpy as np  # Import numpy

import derived_features as df # added

from Window2 import Window
import coloring as col
from utilOwn import insert_or_add
np.set_printoptions(threshold='nan')
''' Parameters for sensor fusion and windowing '''
# number of values per modality. Necessary for filling up the vector in
# combineModalities if one modality is missing.
# TODO: automatically infer the number when reading the data

# changed quaternion to 27
# MOD_LENGTHS -->  dictionary that shows the number of sensors of each type, that we have in total
MOD_LENGTHS = {"IMU": {None: 3,
                       "Quaternion": 27,  # 3*9=27
                       "LinearAccelarations": 9,
                       "Calculated_Euler_Angles": 9},
               "DAQ": {"KIT0A001": 12,
                       "KIT0A002": 9},
               "ZMD": {None: 8},
               "JointAngle": {None: 4},
               "JointTorque": {None: 4},
               "Moment": {"xyz":3}
               }

#MOD_ORDER --> dictionary that shows with which exoskeleton are we working with, the sensors that each one has and the data that provides
MOD_ORDER = {'exo1': [('ZMD', None),
                      ('IMU', None),
                      ('JointAngle', None),
                      ('JointTorque', None)],
             'exo2': [('DAQ', 'KIT0A001'),
                      ('DAQ', 'KIT0A002'),
                      ('IMU', 'Quaternion'),
                      ('IMU', 'LinearAccelarations')],
             'exo2euler': [('DAQ', 'KIT0A001'),
                           ('DAQ', 'KIT0A002'),
                           ('IMU', 'Quaternion'),
                           ('IMU', 'LinearAccelarations'),
                           ('IMU', 'Calculated_Euler_Angles')],
             'exo2calc_moment': [('DAQ', 'KIT0A001'), # version to read the forces to calculate moments and save them on CSV files
                           ('DAQ', 'KIT0A002')],
             'exo2moment': [('Moment','xyz')] # version to use the moments of the knee as a derived feature
             }

# Standard exo version
EXO_STD = 'exo2euler' # by default we are working with this exoskeleton

# Delay until next window is creatd. Unit ms.
#NEXT_WINDOW_STEP = 100
NEXT_WINDOW_STEP = 10 # each window is created every 10 seconds
WINDOW_SIZE = 500 # by default each window collect the data gathered during 500ms

''' Parameters necessary for readOldData aka exo1 data format ''' # old data... maybe it is nos useful anymore
# position in the splitted blank space list, where the time stamp is written
TIMESTAMP_POINT = 1
# position where the data type is written, depend on data convention
DATATYPE_POINT_NEW = 2
DATATYPE_POINT_OLD = 0
# position where the data stars, depend on data convention
VALUE_POINT_NEW = 3
VALUE_POINT_OLD = 1

# modalities used for old data format
MODALITIES = ["ZMD", "IMU", "JointAngle", "JointTorque"]

# names by which the sensor data is identified in the data files of
# the exo1 format (old format)
SENSOR_NAME = {"IMU": {"Identity": "AngleDifferenz:",
                       "New": "IMU:",
                       "Old": "AngleDifferenz:"},
               "ZMD": {"Identity": "ZMD:",
                       "New": "ZMD:",
                       "Old": "ZMD:"},
               "JointAngle": {"Identity": ":",
                              "New": "JointAng:",
                              "Old": ":"},
               "JointTorque": {"Identity": ":",
                               "New": "JointTorque:",
                               "Old": ":"}}

# dummy data has to be added to ensure consistent data format
DUMMY_DATA = {"IMU": [],
              "ZMD": [],
              "JointAngle": ['0', '0'],
              "JointTorque": ['0', '0']}

''' Parameters for file selection and statistics '''
# structure of the files necessary to sort the data properly
TESTPERSONS = ['ID', 'Proband ', 'Proband', 'SegmentedData_']

BASEPATH = '/common/homes/students/costa/Documents/DATA/RenameNeutral_moment'

# '/common/share/Vicon_Data/Vicon/EXO/PassiveExo/IROS_Aufnahmen_2018/HMMTest/ALL_Interpoliert/RenameNeutral_new/' --> another one that said Isabel

# '/common/homes/students/costaDocuments/DATA' --> Personal

# '/common/share/Vicon_Data/Vicon/EXO/PassiveExo/IROS_Aufnahmen_2018/HMMTest/ALL_Interpoliert/RenameNeutral_corrected/'

IGNORE_DIRS = ['CSV', 'Backup_Ordner', 'EMG', 'Interpoliert',
               'Session 1', 'Session1', 'MotionPrediction', 'Python', 'Matlab', 'BackUp',
               'Invalid_Motions.csv','MotionPrediction_master','MotionPrediction_WS100','MotionPrediction_WS300']

# files with less than RECORDING_LENGTH_THRESHOLD timestamps or with
# more than TIMESTAMP_DISTANCE ms average distance between timestamps are
# listed in the Short_Recordings.csv file
RECORDING_LENGTH_THRESHOLD = 40
TIMESTAMP_DISTANCE = 100

# ignore testfiles
IGNORE_FILENAMES = ['Test', 'test', 't_', 'Trial', 'neutral']


class Motion_Datareader_Train_Test():
    """
    Read data, combine modalites and create windows.

    The Motion_Datareader has three purposes:
    1.  Read raw sensor data from text files and store it in a sensible manner.
        This means sorted by testperson, motion and modality. The collected
        data can even be stored in csv files (one per person, motion and
        modality) for analysis with different tools. --> to be stored it is needed to set the command line argument '--csv_
    2.  Combine the data from different modalities for one recording. A
        recording is generally specified by testperson and modality. The
        main part is the algorithm with decides which timestamps from which
        modalities to combine. This is important because different sensors
        are sampled with different sample rates.
    3.  Create windows to use them as input for ML training. Windows are
        created per testperson and motion using a sliding-window approach.

    For usage instructions refer to the README or run this script with -h.
    """

    # pylint: disable = old-style-class, too-many-instance-attributes
    # The class style and the number of attributes all seem reasonable

    def __init__(self):

        # dict after pre-processing (combining the modalities), used for
        # windowing
        self.dictTotal = {}
        # Dict that holds all sensor values after parsing. The keys are
        # the names of the sensor modalities, the values are dicts them-
        # selves
        self.allData = {}
        # Contains the windowed data (X and y) sorted by testperson.
        # X contains the actual data for each window, while y contains tuples
        # (label, motionName) for each window
        self.windows = {}

        # the path used for reading and writing data
        self.datapath = BASEPATH

        # flag that indicates whether the new or old data formats are used
        self.flag_old = False
        # flag that indicates whether renamed or original data should be read
        self.flag_rename = False
        # string that indicates which exoversion to use
        self.exoversion = EXO_STD

        # number of opened files
        self.openedFiles = 0
        #list of all closed windows
        self.all_data = []

        # all motions where one or more modalities are missing
        self.invalidMotions = []
        # all motions with less than RECORDING_LENGTH_THRESHOLD sensor values
        self.shortRecordings = {}

        self.logger = logging.getLogger(__name__)

    def readAllData(self, testpersonDir):
        """
        Reads all data from the directory where the data for all test
        person lies. Saves the data in the following hierarchy:
        testperson, movement, sensor modality (e.g. IMU),
        valuetype (e.g. Quaternion), values as list [(timeoffset, values)].
        For new data the time is the offset from the beginning of the
        motion, for old data it is absolute time.
        If testpersonDir is a folder containing several testpersonDirs,
        each of them will be read recursively.

        Example dict:

        {'ID01':                                              # testperson
            {'B1_mE_L2_0001':                                 # movement 1
                {'IMU':                                       # sensor modality
                    {'Quaternion':                            # valuetype
                        [(0, ['0.063049', ... ]),
                        (30, ['0.063477', ... ]),
                        (59, ['0.063599', ... ]),
                        (91, ['0.063538', ... ])],
                     'LinearAccelarations':
                        [(0, ['0.000000', ... ]),
                        (30, ['-0.070000', ... ]),
                        (59, ['-0.070000', ... ]),
                        (91, ['-0.070000', ... ])]},
                'DAQ':                                        # sensor modality
                    {u'KIT0A002':                             # valuetype
                        [(0, [1.38581, ... ]),
                        (12, [1.38581, ... ]),
                        (18, [1.37957, ... ])],
                     u'KIT0A001':
                        [(4, [-0.22543, ... ]),
                        (15, [-0.22543, ... ]),
                        (24, [-0.231645, ... ])]}},
            'B1_mE_L2_0002':                                  # movement 2
                {'IMU':
                    {'Quaternion':
                        [(0, ['-0.102173', ... ]),
                        (25, ['-0.102478', ... ]),
                        (53, ['-0.102600', ... ]),
                        (86, ['-0.102844', ... ]),
                    'LinearAccelarations':
                        [(0, ['-0.070000', ... ]),
                        (25, ['-0.090000', ... ]),
                        (53, ['0.000000', ... ]),
                        (86, ['0.000000', ... ])]},
                'DAQ':
                    {u'KIT0A002':
                        [(5, [0.89271, ... ]),
                        (15, [0.89271, ... ]),
                        (27, [0.89271, ... ])],
                    u'KIT0A001':
                        [(0, [-0.77232, ... ]),
                        (9, [-0.759891, ... ]),
                        (20, [-0.766105, ... ])]}}
            ...}
        }
        # doctest ?
        >>> reader = Motion_Datareader_Train_Test()
        >>> path = 'tests/testfiles/ID01'
        >>> reader.readAllData(path)
        Finished reading all data in directory tests/testfiles/ID01/DAQ
        Finished reading all data in directory tests/testfiles/ID01/IMU
        >>> movements = reader.allData['ID01'].keys()
        >>> movements
        ['B1_mE_L2_0001', 'B1_mE_L2_0002']
        >>> modalities = reader.allData['ID01'][movements[0]].keys()
        >>> modalities
        ['IMU', 'DAQ']
        >>> reader.readAllData(path)  # doctest: +NORMALIZE_WHITESPACE
        There already exists data for testperson ID01, movement B1_mE_L2_0001 \
            and sensor modality DAQ!
        There already exists data for testperson ID01, movement B1_mE_L2_0002 \
            and sensor modality DAQ!
        There already exists data for testperson ID01, movement B1_mE_L2_0001 \
            and sensor modality IMU!
        There already exists data for testperson ID01, movement B1_mE_L2_0002 \
            and sensor modality IMU!

        :param testpersonDir:
        :return:
        """
        # identify the testperson
        testperson = getTestpersonFromPath(testpersonDir) # with the path to that person folder, we gett the id of that person
        dataRead = False

        for entry in sorted(os.listdir(testpersonDir)): # we parse each folded --> DAQ, IMU, 'MotionPrediction' of each user
            # able to identify several testpersonDirs in the given path
            # (could happen if only basepath, but not inputfolder was
            # specified!)
            path = os.path.normpath(testpersonDir + '/' + entry)

            # recursively search for data files
            if entry in IGNORE_DIRS: #' we take care that the folder we parse is not one wrong'
                continue
            elif os.path.isdir(path): # we check if that path exists
                self.readAllData(path)
            elif xor(self.flag_rename, renameInPath(path)):
                continue
            else:
                # read the sensor values
                (modality, values) = self.readSensorFile(path) # read all the values for a certain modality (IMU or DAQ)
                if values is None:
                    continue

                # identify a measurement by the testperson and the
                # performed movement
                movement = self.getMotionNameFromFilename(entry)
                #print movement
                #print modality
                #print values
                #sys.exit()

                # save the data in allData
                if testperson not in self.allData:
                    self.allData[testperson] = {movement: {modality: values}}
                elif movement not in self.allData[testperson]:
                    self.allData[testperson][movement] = {modality: values}
                elif modality not in self.allData[testperson][movement]:
                    self.allData[testperson][movement][modality] = values
                elif self.flag_old:
                    self.allData[testperson][movement][modality] += values
                else:
                    print ('There already exists data for testperson '
                           + testperson + ', movement ' + movement
                           + ' and sensor modality ' + modality + '!')
                    continue

                dataRead = True # to show that we have ended to read the data

        if dataRead:
            print('Finished reading all data in directory ' + testpersonDir)


    def readSensorFile(self, filePath, modalityName=None):
        """
        Read the data from file filePath.

        The data may be from OptoForce sensors (modalityName = 'DAQ') or from
        IMU sensors (Quaterionen + LinearAccelerations, modalityName = 'IMU').

        >>> reader = Motion_Datareader_Train_Test()
        >>> basepath = 'tests/testfiles/ID01'
        >>> daqFilename = basepath + '/DAQ/B1_mE_L2_0002_optoForce.txt'
        >>> imuFilename = basepath + '/IMU/B1_mE_L2_0001_imu.txt'
        >>> modality, values = reader.readSensorFile(daqFilename)
        >>> modality
        'DAQ'
        >>> values.keys()
        [u'KIT0A002', u'KIT0A001']
        >>> values['KIT0A001']  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [(0, [-0.77232, ...]), (9, [-0.759891, ...]), ...]
        >>> modality, values = reader.readSensorFile(imuFilename)
        >>> modality
        'IMU'
        >>> sorted(values.items())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [('LinearAccelarations',
            [(0, ['0.000000', ...]),
            (30, ['-0.070000', ...]),
            (59, ['-0.070000', ...]),
            (91, ['-0.070000', ...]),
            ...]),
        ('Quaternion',
            [(0, ['0.063049', ...]),
            (30, ['0.063477', ...]),
            (59, ['0.063599', ...]),
            (91, ['0.063538', ...]),
            ...])]

        :param filePath:
        :param modalityName:
        :return: (sensor_name ['DAQ' or 'IMU'], dict)
        """
        if self.flag_old:
            validModalities = '"IMU", "ZMD", "JointAngle", "JointTorque"'
        else:
            validModalities = '"IMU", "DAQ"'

        if self.flag_old:
            return readOldData(filePath, modalityName)
        if modalityName == 'DAQ' or 'DAQ' in filePath.split('/'):
            return 'DAQ', readDAQ(filePath)
        if modalityName == 'IMU' or 'IMU' in filePath.split('/'):
            # zu dictSensors['IMU']['B2_oE...'] hinzufuegen
            return 'IMU', readNewIMU(filePath)
        else:
            self.logger.info('modalityName has to be %s or None. \
            If None is given, modalityName has to appear in the filePath!', validModalities)
            self.logger.info('filePath: %s', filePath)
            if not self.flag_old:
                self.logger.info('Did you forget to specifiy "--old"?')
            return None, None

    def getMotionNameFromFilename(self, filename):
        """ Return the motion name that is part of filename.

        >>> reader = Motion_Datareader_Train_Test()
        >>> reader.getMotionNameFromFilename('B1-mE_L2_0001_optoForce.txt')
        'B1-mE_L2_0001'
        >>> reader.getMotionNameFromFilename('B1-mE_L2_0002_imu.txt')
        'B1-mE_L2_0002'

        """
        # TODO: could be helper function? No, because we need the self.flag_old.
        #       Maybe a refactoring is possible, so that the flag is not needed here.

        # remove the file ending and replace any other '.' by '_'
        namelist = reduce(lambda l, r: l + '_' + r,
                          filename.split('.')[:-1])
        if self.flag_old:
            # don't include the numbering after the motion name
            return namelist.split('_')[0]
        return reduce(lambda l, r: l + '_' + r, namelist.split('_')[:-1])

    def writeAllCSV(self, targetDir):
        """
        Write the data collected by readAllData() into several csv files.

        There will be one file for each combination of testperson, movement,
        and sensor type. The files are saved in subfolders 'CSV' of the
        appropriate testperson's modality directory
        (e.g. 'Proband 1/IMU/CSV/').

        If the --rename flag was set and hence the files in folder Rename
        are used, then the CSV folder will be put into e.g.
        'Proband 1/IMU/Rename/CSV/

        :param targetDir:
        :return:
        """
        for testperson in self.allData:
            for movement in self.allData[testperson]:
                for modality in self.allData[testperson][movement]:
                    if self.flag_rename:
                        writeDir = os.path.join(
                            targetDir, testperson, modality, 'Rename', 'CSV')
                    else:
                        writeDir = os.path.join(
                            targetDir, testperson, modality, 'CSV')
                    if not os.path.isdir(writeDir):
                        os.makedirs(writeDir)
                        self.logger.info('Created directory %s!', writeDir)

                    if self.flag_old:
                        writeCSV(writeDir, movement, modality, None,
                                 self.allData[testperson][movement][modality])
                    else:
                        for valuetype in self.allData[testperson][movement][modality]:
                            writeCSV(writeDir, movement, modality, valuetype,
                                     self.allData[testperson][movement][modality][valuetype])
            self.logger.info('Finished writing CSV for testperson %s', testperson)


     #########################################

    def readCSVmoment(self, filePath, ignore_names):
        """
        Read the data in the CSV_moment folders located in the DAQ folders of each person
        The data is used to create a dictionary  --> self.allData

        There will be one file for each combination of testperson, movement,
        and sensor type. The files are saved in subfolders 'CSV' of the
        appropriate testperson's modality directory
        (e.g. 'Proband 1/IMU/CSV/').

        If the --rename flag was set and hence the files in folder Rename
        are used, then the CSV folder will be put into e.g.
        'Proband 1/IMU/Rename/CSV/

        :param targetDir:
        :return:

        """
        modality= 'Moment'

        for testperson in sorted(os.listdir(filePath)):  # we check every folder that is in that path
            if testperson in ignore_names:  # for person with ID_num / only ID_num files are considered
                continue
            else:  # inside the folder of a person
                path_CSV_moment = os.path.normpath(filePath + '/' + testperson + '/' + 'DAQ' + '/' + 'CSV_moment')  # we look into the CSV_moment folder
                for file_motion in sorted(os.listdir(path_CSV_moment)):  # inside the folder of DAQ checking each motion and repetition

                    #movement = self.getMotionNameFromFilename(file_motion) # take the name of the motion from the .csv file
                    movement=file_motion[:-4]

                    values=[]

                    with open(path_CSV_moment+'/'+file_motion) as csvfile:

                        file = csv.reader(csvfile, delimiter=',', quotechar='|')
                        for row in file:
                            timestamp = int(row[0])
                            moments = [float(row[1]),float(row[2]),float(row[3])]
                            values = values + [(timestamp,moments)]

                    dict_values={'xyz':values}

                    if testperson not in self.allData:
                        self.allData[testperson] = {movement: {modality: dict_values}}
                    elif movement not in self.allData[testperson]:
                        self.allData[testperson][movement] = {modality: dict_values}
                    elif modality not in self.allData[testperson][movement]:
                        self.allData[testperson][movement][modality] = dict_values
                    else:
                        print ('There already exists data for testperson '
                               + testperson + ', movement ' + movement
                               + ' and sensor modality ' + modality + '!')
                        continue


        '''
                    # save the data in allData
        if testperson not in self.allData:
            self.allData[testperson] = {movement: {modality: values}}
        elif movement not in self.allData[testperson]:
            self.allData[testperson][movement] = {modality: values}
        elif modality not in self.allData[testperson][movement]:
            self.allData[testperson][movement][modality] = values
        elif self.flag_old:
            self.allData[testperson][movement][modality] += values
        else:
            print ('There already exists data for testperson '
                   + testperson + ', movement ' + movement
                   + ' and sensor modality ' + modality + '!')
            continue

        for testperson in self.allData:
            for movement in self.allData[testperson]:
                for modality in self.allData[testperson][movement]:
                    if self.flag_rename:
                        writeDir = os.path.join(
                            targetDir, testperson, modality, 'Rename', 'CSV')
                    else:
                        writeDir = os.path.join(
                            targetDir, testperson, modality, 'CSV')
                    if not os.path.isdir(writeDir):
                        os.makedirs(writeDir)
                        self.logger.info('Created directory %s!', writeDir)

                    if self.flag_old:
                        writeCSV(writeDir, movement, modality, None,
                                 self.allData[testperson][movement][modality])
                    else:
                        for valuetype in self.allData[testperson][movement][modality]:
                            writeCSV(writeDir, movement, modality, valuetype,
                                     self.allData[testperson][movement][modality][valuetype])
            self.logger.info('Finished writing CSV for testperson %s', testperson)
        '''

    #####################################################################

    def combineModalities(self, testperson, motion):
        """ Combine sensor modalities into feature vectors.

        Create a combined vector of (timestamp, [values]) tuples of all
        modalities and valuetypes for this testperson and motion. The order
        is (ZMD, DAQ.KITA001, DAQ.KITA002, IMU, IMU.LinearAccelerations,
        JointAngle, JointTorque) minus those modalities that are not present
        in allData.

        :param testperson:
        :param motion:
        :return:

        """
        # sort the remaining modalities and value types in the order
        # specified in the comment
        order = MOD_ORDER[self.exoversion] # acces to the kind of exoeskeleton we are dealing with


        # case exo2euler: order=
        #[('DAQ', 'KIT0A001'),
        #('DAQ', 'KIT0A002'),
        # ('IMU', 'Quaternion'),
        # ('IMU', 'LinearAccelarations'),
        # ('IMU', 'Calculated_Euler_Angles')]}

        # Get a list of the sensor values for this person and motion.
        # Each entry contains the sensor values for one modality.
        # The list entries are sorted according to order.
        sortedDirs = self.__sortModalities(order, testperson, motion)

        # combine the modalities to form feature vectors
        # reduce --> apply combineDicts to every list in sortedDirs
        combinedVector = makeTimestampsRelative(reduce(combineDicts, sortedDirs))
        #print combinedVector

        self.dictTotal[testperson].append({'motion': motion,
                                           'data': combinedVector})
        #print self.dictTotal[testperson]
        #sys.exit()                                  

        # check distances between timestamps; only for debug purposes
        self.logger.debug('Testperson %s, motion %s', testperson, motion)
        if self.logger.getEffectiveLevel() <= logging.DEBUG:
            for (mod, modData) in zip(order, sortedDirs):
                if modData:
                    self.logger.debug(
                        '{} -> avg time between timestamps: '.format(mod)
                        + '{} ms'.format(calcAvgTimestampDelta(modData))
                        + ', start: {} ms'.format(modData[0][0])
                        + ', end: {} ms'.format(modData[-1][0])
                        + ', {} timestamps'.format(len(modData)))
                else:
                    self.logger.debug('{} -> no data'.format(mod))

        # Collect a list of very short recordings for each testperson
        num = len(combinedVector)
        if num < RECORDING_LENGTH_THRESHOLD and not ignoreMotion(motion):
            insert_or_add(self.shortRecordings, testperson, [], [(motion, num)])

    def __sortModalities(self, order, testperson, motion):
        """
        Return list of sensor data sorted by modality according to order.

        Return a list of the sensor values for this person and motion.
        Each entry contains the sensor values for one modality.
        The list entries are sorted according to order.

        e.g. if order is 'exo2' the resulting sortedDirs would be a list
        of lengt 4 as specified via MOD_ORDER['exo2'] with

        sortedDirs[0]  # DAQ - KIT0A001
        sortedDirs[1]  # DAQ - KIT0A002
        sortedDirs[2]  # IMU - LinearAccelarations
        sortedDirs[3]  # DAQ - Quaternion

        sortedDirs[0] contains 400 tuples of (timestamp, [values])
        e.g. sortedDirs[0][0] would be
        [4,  [-0.22543, -2.8904, -3.39265, -1.7623 ...]]

        """
        assert self.allData[testperson][motion]
        # collect the data in the specified order and insert mock entries
        # if no matching data exists
        sortedDirs = [self.allData[testperson][motion].get(modality, {}).get(valueType, [])
                      if valueType is not None
                      else self.allData[testperson][motion].get(modality, [])
                      for modality, valueType in order]
        #print self.allData[testperson][motion][modality]
        #print self.allData[testperson][motion].get(modality, {})
        #print self.allData[testperson][motion].get(modality, {}).get(valueType, [])
        #sys.exit()

        # create mock entries where necessary
        timeSource = max(sortedDirs, key=len)
        missingModalities = []
        for idx, (modality, valueType) in enumerate(order):
            if sortedDirs[idx]:
                # real data was collected, no need for mocks
                continue
            mockEntry = [0.0 for _ in range(MOD_LENGTHS[modality][valueType])]
            sortedDirs[idx] = [(t, mockEntry) for (t, _) in timeSource]
            # more readable text for missing modalities
            missingModalities.append(modality + ':' + valueType if valueType else modality)

        # save invalid motions
        if missingModalities and not ignoreMotion(motion):
            self.invalidMotions.append(
                {'testperson': testperson, 'motion': motion,
                 'modalities': missingModalities})

        return sortedDirs

    def createWindows(self):
        """
        Convert the data in self.allData into windows.

        Windows are saved in self.allData, sorted by testperson and motion.
        Windows are created using a sliding window approach.
        They contain all feature vectors that occurred during the time span
        covered by the window.

        On the intern program flow:
        createWindows()
            calls: combineModalities(testperson, motion) for every motion
                   --> responsible for 'data' in dictTotal of testperson
                   --> 'data' contains combinedVector of (timestamp, [values])
                   tuples of all modalities and valuetypes for this testperson
                   and motion. The order is (ZMD, DAQ.KITA001, DAQ.KITA002,
                   IMU, IMU.LinearAccelerations, JointAngle, JointTorque)
                   minus those modalities that are not present in allData
            calls: createWindowsForTestperson() for every testperson

        >>> basepath = 'tests/testfiles/'
        >>> testperson = 'ID01'
        >>> reader = Motion_Datareader_Train_Test()
        >>> reader.exoversion = 'exo2'
        >>> reader.readAllData(basepath + testperson)
        Finished reading all data in directory tests/testfiles/ID01/DAQ
        Finished reading all data in directory tests/testfiles/ID01/IMU
        >>> reader.datapath = basepath  # otherwise createWindows uses BASEPATH
        >>> reader.createWindows()  # doctest: +ELLIPSIS
        Creating windows for ID01
        ...
        <BLANKLINE>
        >>> motions = ['B1_mE_L2_0001', 'B1_mE_L2_0002']
        >>> type(reader.dictTotal['ID01'])
        <type 'list'>
        >>> type(reader.dictTotal['ID01'][0])
        <type 'dict'>
        >>> reader.dictTotal['ID01'][0].keys()
        ['motion', 'data']
        >>> reader.dictTotal[testperson][0]['motion']
        'B1_mE_L2_0001'
        >>> reader.dictTotal[testperson][0]['data']  # doctest: +ELLIPSIS
        [(4, array([...], dtype=float32))]
        >>> reader.invalidMotions
        []
        >>> reader.windows[testperson].keys()
        ['y', 'X', 'numberWindows']
        >>> reader.windows[testperson]['numberWindows']
        [32, 36]

        # X is a list of numpy arrays which each contains the data
        # for one window. Their size is {number of timestamps in the
        # window} x {number of different sensorvalues}.
        # y is a list of motion names for each window
        >>> reader.windows[testperson]['y']  # doctest: +ELLIPSIS
        [('B1_mE', 'B1_mE_L2_0001'), ... ('B1_mE', 'B1_mE_L2_0002')]
        >>> len(reader.windows[testperson]['y'])
        68
        >>> len(reader.windows[testperson]['X'])
        68
        >>> reader.windows[testperson]['X'][0]  # doctest: +ELLIPSIS
        array([[ -2.25429997e-01, ...]], dtype=float32)
        >>> reader.windows[testperson]['X'][0].shape
        (20, 49)

        num timestamps x num sensorvalues

        """
        for testperson in self.allData:
            self.dictTotal[testperson] = []
            for motion in self.allData[testperson]: # exists that motion for that person
                if ignoreMotion(motion):
                    continue
                self.combineModalities(testperson, motion) #Create a combined vector of (timestamp, [values]) tuples of all
                # modalities and valuetypes for this testperson and motion.

        for testperson in self.dictTotal:
            self.windows[testperson] = {} # inicialization
            self.createWindowsForTestperson(testperson) # function that creates the windows

        self.logger.info('Finished creating windows')

    def createWindowsForTestperson(self, testperson):
        """
        Convert all data for testperson into windows.

        Windows are saved in self.allData, sorted by testperson and motion.
        Windows are created using a sliding window approach.
        They contain all feature vectors that occured during the time span
        covered by the window.
        """
        assert testperson in self.dictTotal # to check that the person is in dictTotal. If it isnt, it takes false and the program stops
        print('Creating windows for {}'.format(testperson))

        # if overview file already exists, it will be deleted
        writeDir = os.path.normpath(self.datapath + '/' + testperson + '/MotionPrediction')
        overviewPath = os.path.normpath(writeDir
                                        + "/ExoMotions_X_Train_Test_{}.txt".format(testperson))
        if not os.path.exists(writeDir):
            os.makedirs(writeDir)
        if os.path.exists(overviewPath):
            os.remove(overviewPath)

        # number of windows for each motion name
        numberWindows = []
        # X is a list of numpy arrays which each contains the data
        # for one window. Their size is {number of timestamps in the
        # window} x {number of different sensor values}.
        # y is a list of motion names for each window
        windows = {'X': [], 'y': []}

        # create windows seperately for each motion and collect them into X
        for entry in self.dictTotal[testperson]:
            numWindows = self.__createWindowsForMotion(testperson, entry, windows, overviewPath)
            numberWindows.append(numWindows)

        assert len(windows['X']) == len(windows['y']) # we have the same number of windows and labels related to these windows
        col.printout('Testperson {}:'.format(testperson), col.BLUE)
        print('''
        number of windows completed per motion: {}
        {} completed windows in total
        '''.format(numberWindows, len(windows['X'])))

        self.windows[testperson]['numberWindows'] = numberWindows
        self.windows[testperson]['X'] = windows['X']
        self.windows[testperson]['y'] = windows['y']
        #print windows
        #sys.exit()

    def __createWindowsForMotion(self, testperson, motionData, windows, overviewPath):
        # get meta data
        data = motionData['data']

        # start -- time when the latest window was started
        # last -- last timestamp that was passed into a window
        # current -- current timestamp
        time = {'start': 0, 'last': 0, 'current': 0}
        # When a new movement starts, the windows with the old movement will be deleted
        listAllWindows = []
        numWindows = {'created': 0, 'finished': 0, 'deleted': 0}

        for tupel in data:

            time['current'] = tupel[0]

            # if 100ms passed, if there was no window before or if
            # an earlier timestamp appears, a new window will be
            # created.
            if (not listAllWindows or (time['current'] >= time['start'] + NEXT_WINDOW_STEP)
                    or (time['current'] < time['last'])):
                time['start'] = time['current']
                windObj = Window(time['start'], WINDOW_SIZE, motionData['motion'])
                listAllWindows.append(windObj)
                numWindows['created'] += 1

            for win in listAllWindows: # for each window in listAllWindows
                # checks if window is still active and adds the current zmd values
                if win.isActive(time['current']):
                    win.addValues(tupel[1])

                # write closed window and create overview file
                else:
                    if closeWindow(win, time['last'], windows, overviewPath):
                        numWindows['finished'] += 1
                    else:
                        numWindows['deleted'] += 1

                    # remove inactive window
                    listAllWindows.remove(win)

            #go to the next tupel
            time['last'] = time['current']

        # use average timestamp distance as metric to discard motions
        deltaTime = calcAvgTimestampDelta(data)
        num = len(data)
        if deltaTime >= TIMESTAMP_DISTANCE:
            insert_or_add(self.shortRecordings, testperson, [], [(motionData['motion'], num)])
        # output for debugging purposes/data introspection
        self.logger.debug('{}: '.format(motionData['motion'])
                          + '{} windows created, '.format(numWindows['created'])
                          + '{} windows deleted'.format(numWindows['deleted'])
                          + ', {} windows finished, '.format(numWindows['finished'])
                          + 'length of recording: {} ms, '.format(data[-1][0] - data[0][0])
                          + 'avg time between timestamps: {} ms'.format(deltaTime))

        return numWindows['finished']

    def writeWindows(self, targetDir):
        """
        Write all windows into the targetDir and save statistic info.

        Windows are saved separately for each testperson in the folder
        targetDir/{testperson}/MotionPrediction. Generate two window files:
        ExoMotions_X_Train_Test_{testperson}.npy -- feature vectors of all windows
        ExoMotions_y_Train_Test_{testperson}.pkl -- motion labels for all windows
        """

        print('Starting to write windows')
        for (testperson, data) in self.windows.items():

            writeDir = os.path.normpath(targetDir + '/' + testperson + '/MotionPrediction')
            

            print("save npy for {}".format(testperson))
            np.save(writeDir + '/' + 'ExoMotions_X_Train_Test_{}.npy'.format(testperson), data['X'])

            pickleFile = writeDir + '/' + 'ExoMotions_y_Train_Test_{}.pkl'.format(testperson)
            with open(pickleFile, 'wb') as f:
                pickle.dump(data['y'], f)

            labels = sorted(list(set([label for label, _ in data['y']])))
            self.logger.debug('Testperson {} has motion labels: {}'.format(testperson, labels))
        print('Finished writing windows')

        self.writeStatistics()

    def writeStatistics(self):
        """
        Save motion names with invalid or short recordings.

        A short recording is any motion with less than RECORDING_LENGTH_THRESHOLD timestamps. This
        information is saved separately for every testperson. The file is saved in
        self.datapath/{testperson}/Short_Recordings.csv and contains the name and length of all
        short motions for that testperson.
        An invalid motion is any motion for whom one or more modalities are missing in the provided
        data. Those motions are saved together with the affected testperson in
        self.datapath/Invalid_Motions.csv.
        """

        print('Starting to write statistics files')

        # print list of short recordings for each testperson
        for testperson in self.allData:
            path = os.path.normpath(self.datapath + '/' + testperson + "/Short_Recordings.csv")
            if os.path.exists(path):
                os.remove(path)
            if testperson not in self.shortRecordings:
                continue
            with open(path, 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(['Motion', 'Length of Recording'])
                for (motion, num) in self.shortRecordings[testperson]:
                    writer.writerow([motion, num])
        if self.shortRecordings:
            print('Finished writing Short_Recordings.csv for each testperson')

        # Print the invalid motions into a file. All motions missing
        # one or more modalities are considered invalid.
        writepath = os.path.normpath(self.datapath + '/' + 'Invalid_Motions.csv')
        if os.path.exists(writepath):
            os.remove(writepath)
        if self.invalidMotions:
            with open(writepath, 'wb') as f:
                fieldnames = ['testperson', 'motion', 'modalities']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({'testperson': 'Testperson',
                                 'motion': 'Motion', 'modalities': 'Missing modalities'})
                writer.writerows(self.invalidMotions)
            print('Finished writing Invalid_Motions.csv' + ' at {}'.format(writepath))

        print('Finished writing statistics files where necessary')

######################## Helper functions ##############################

def closeWindow(win, lastConvertedTime, windows, overviewPath):
    """
    Close the window win at timestamp lastConvertedTime.

    If the window's size is valid, add its data to X and y and return True.
    Return false otherwise.
    """
    # close inactive windows
    if win.sizeCheck(lastConvertedTime):
        all_data = win.getValues()
        #print (all_data[0])
        #print (all_data[21])
        #print (all_data[22])
        #print (all_data[23])
        #print (all_data[24])
        #print (all_data[25])
        #print (all_data[26])
        #print (all_data[27])
        #print (all_data[28])
        #print (all_data[29])
        #print (all_data[30])
        #print len(all_data)
        #print len(all_data[0])
        #sys.exit()


        # CALCULATION OF THE DIFFERENCES BETWEEN VALUES
        #all_data = np.diff(all_data)       # removed


        #print("Diff_Data" + str(all_data))
        #print("Shape" + str(len(all_data)))
        #print(len(all_data[0]))
        winData = np.array(all_data).T
        #print winData
        #sys.exit()
        #print("LengthTransposed Data" + str(len(winData)))
        windows['X'].append(winData)
        motionName = win.getMotionName()
        windows['y'].append((labelFromMotionname(motionName), motionName))
        # create ExoMotions_X_Train_Test.txt file for debugging.
        # Parameters of the current window will be written to the file.
        with open(overviewPath, "a") as overview:
            overview.write("Motion Name: {}\n".format(win.getMotionName()))
            overview.write("Start time of window: "
                           + "{}\n".format(win.getStartTime()))
            overview.write(
                "Last converted Time: {}\n".format(lastConvertedTime))
            overview.write("Length of window: "
                           + "{}\n".format(lastConvertedTime
                                           - win.getStartTime()))
            overview.write('\n')

        return True

    return False


def labelFromMotionname(motionName):
    """

    :param motionName:
    :return:

    >>> labelFromMotionname('B5_oE_L2_004')
    'B5_oE'
    >>> labelFromMotionname('B5_oE__L1__004')
    'B5_oE'
    >>> labelFromMotionname('Squatting_0003')
    'Squatting'
    >>> labelFromMotionname('Squatting_Backwards_43_0003')
    'Squatting_Backwards_43'
    """
    patternStartB = 'B[0-9]+_(m|o)E_(_)?L[0-9]_(_)?[0-9]+'
    patternNamePlusNumber = r'\w+_[0-9]+'
    if re.match(patternStartB, motionName):
        # names from Annika's recordings with the exo
        return reduce(lambda l, r: l + '_' + r, motionName.split('_')[:2])
    elif re.match(patternNamePlusNumber, motionName):
        # any motion name that still has a number at the end
        return reduce(lambda l, r: l + '_' + r, motionName.split('_')[:-1])
    return motionName


def addValues(motionName, dictionary, time, values):
    """
    Add the tuple (time, values) to dictionary[motionName].

    >>> d = {}
    >>> addValues('Forward', d, 0, [1, 2, 3])
    >>> addValues('Backward', d, 0, [3, 2, 1])
    >>> d == {'Forward': [(0, [1, 2, 3])], 'Backward': [(0, [3, 2, 1])]}
    True
    >>> addValues('Forward', d, 5, [4, 5, 6])
    >>> d == {'Forward': [(0, [1, 2, 3]), (5, [4, 5, 6])], 'Backward': [(0, [3, 2, 1])]}
    True
    """
    insert_or_add(dictionary, motionName, [], [(time, values)])


def calcAvgTimestampDelta(data):
    """ Calculate average distance between timestamps of the tuples in data.

    :param data:
    :return:
    """
    timestamps = [t for (t, _) in data]
    deltas = [x - y for (x, y) in zip(timestamps[1:], timestamps[:-1])]
    return np.mean(deltas)


def combineDicts(left, right):
    """
    Combine the provided dicts left and right by considering their timestamps.

    It is needed to combine value list from different sensors which may have
    different timestamps.
    The returned dictionary contains lists of (timestamp, numpy vector) tuples.
    Each vector contains the values from left and right that correspond to the
    timestamp. If necessary values from several timestamps are averaged by
    arithmetic mean.
    """
    result = []

    listLeft = []
    listRight = []

    leftIdx = 0
    rightIdx = 0

    oneBigger = False
    twoBigger = False
    # indicates whether the side changed on which the bigger times are
    changed = False

    #pdb.set_trace()

    while (leftIdx < len(left)) and (rightIdx < len(right)):
        (timeOne, valueOne) = left[leftIdx]
        (timeTwo, valueTwo) = right[rightIdx]

        # Add values to the respective listXY until a common
        # timestamp is reached. Calculate the weighted mean
        # of each list and interpret this mean as the sensor
        # data for the common timestamp
        if timeOne < timeTwo:
            if oneBigger:
                changed = True
                oneBigger = False
            twoBigger = True
            listLeft += [valueOne]
            leftIdx += 1
        elif timeTwo < timeOne:
            if twoBigger:
                changed = True
                twoBigger = False
            oneBigger = True
            listRight += [valueTwo]
            rightIdx += 1
        else: # timeOne == timeTwo
            listLeft += [valueOne]
            listRight += [valueTwo]
            leftIdx += 1
            rightIdx += 1

            # Concatenate the weighted sensor values of dictOne
            # and dictTwo at this timestamp.
            # combinedVector is a numpy-float32-array.
            combinedVector = np.append(meanValue(listLeft), meanValue(listRight))
            result += [(timeOne, combinedVector)]

            listLeft = []
            listRight = []

            oneBigger = False
            twoBigger = False

        # Check if enough values were collected to calculate the
        # mean if if no common timestamp was reached
        if changed:
            combinedVector = np.append(meanValue(listLeft), meanValue(listRight))
            result += [(min(timeOne, timeTwo), combinedVector)]
            listLeft = []
            listRight = []

            oneBigger = False
            twoBigger = False
            changed = False

        # Collect any usable data left after the last common timestamp/ mean calculation
        if leftIdx == len(left) and rightIdx < len(right) and listLeft:
            (timeTwo, valueTwo) = right[rightIdx]
            combinedVector = np.append(meanValue(listLeft), meanValue([valueTwo]))
            result += [(timeTwo, combinedVector)]
        elif rightIdx == len(right) and leftIdx < len(left) and listRight:
            (timeOne, valueOne) = left[leftIdx]
            combinedVector = np.append(meanValue([valueOne]), meanValue(listRight))
            result += [(timeOne, combinedVector)]
    return result


def meanValue(vecList):
    """ Calculate the weighted mean of the vectors in vecList.

    Return the mean vector as numpy array.

    :param vecList:
    :return:
    """
    assert vecList  # asserts that vecList is not empty
    vecLength = len(vecList[0])
    mean = np.array(vecList[0], dtype=np.float32)
    
    #print "veclist"
    #print vecList
    for vec in vecList[1:]:
        #print "vec"
        #print vec
        assert len(vec) == vecLength, 'All lists in {} must have the same size!'.format(vecList)
        npVec = np.array(vec, dtype=np.float32)
        mean = (mean + npVec) / 2
        #print "mean"
        #print mean
        #sys.exit()
    
    return mean


def read_json(filename):
    """ Generator function to read a json-encoded file. """
    with open(filename, 'rU') as fileObj:
        for line in fileObj:
            yield json.loads(line.strip())


def readDAQ(filePath):
    """ Read data from an OptoForce (DAQ) file.

    Read values from a OptoForce (DAQ) file and saves them together
    with offset timestamps. Returns a dict which for each sensor name
    contains a list of (offset timestamp [ms], values) tuples.
    Returns None if file format is not correct.

    >>> imuPath = 'tests/testfiles/ID01/IMU/B1_mE_L2_0001_imu.txt'
    >>> readDAQ(imuPath) is None
    True

    >>> daqPath = 'tests/testfiles/ID01/DAQ/B1_mE_L2_0001_optoForce.txt'
    >>> dictSensor = readDAQ(daqPath)
    >>> sensors = sorted(dictSensor.keys())
    >>> sensors
    [u'KIT0A001', u'KIT0A002']
    >>> sensorData = sorted(dictSensor['KIT0A002'])
    >>> for time, val in sensorData:
    ...     print(time, val)  # doctest: +ELLIPSIS
    (0, [1.38581, 2.00506, ... -0.167172])
    (12, [1.38581, 2.00506, ... -0.167172])
    (18, [1.37957, 2.0113, ... -0.167172])
    ...
    >>> sensorValuesListFirstTimestep = sensorData[0][1]
    >>> len(sensorValuesListFirstTimestep)
    9

    :param filePath: string
    :return: None if file format is not correct, else dictSensor
    """
    logging.info('Reading DAQ file %s', filePath)

    # assert that the given file has the right format
    with open(filePath, 'rU') as filename:
        line = filename.readline().split('"')
        if 'daq' not in line:
            logging.info('%s is not a DAQ file!', filePath)
            return None

    data = read_json(filePath)
    data = list(data)

    # identify all unique sensor names, like KIT0A001 etc.
    sensorNames = list(set([line['daq'] for line in data]))
    dictSensor = {name: [] for name in sensorNames}

    # get startTime to calculate time offsets. Assumes that all
    # measurements in one file where taken on the same day
    startTime = getTime(data[0]['timestamp'].split(' ')[1])

    # collect all data in a dictionary for different further uses
    for line in data:
        time = getTime(line['timestamp'].split(' ')[1])
        values = np.array(line['data']).flatten().tolist()
        dictSensor[line['daq']] += [(time, values)]

    return dictSensor


def readNewIMU(filePath):
    """ Read data from an IMU file.

    Read values from an IMU file and saves them together
    with offset timestamps. Returns a dict which for each sensor name
    (Quaternion, LinearAccelerations) contains a list of
    (offset timestamp [ms], values) tuples.
    Returns None if file format is not correct.

    >>> daqPath = 'tests/testfiles/ID01/DAQ/B1_mE_L2_0001_optoForce.txt'
    >>> readNewIMU(daqPath) is None
    True

    >>> imuPath = 'tests/testfiles/ID01/IMU/B1_mE_L2_0001_imu.txt'
    >>> dictSensor = readNewIMU(imuPath)
    >>> sensors = sorted(dictSensor.keys())
    >>> sensors
    ['LinearAccelarations', 'Quaternion']
    >>> quaternionData = sorted(dictSensor['Quaternion'])
    >>> for time, val in quaternionData:
    ...     print(time, val)  # doctest: +ELLIPSIS
    (0, ['0.063049', '-0.738647', ... '0.341431'])
    (30, ['0.063477', '-0.738586', ... '0.341431'])
    (59, ['0.063599', '-0.738586', ... '0.341431'])
    (91, ['0.063538', '-0.738525', ... '0.341431'])
    ...
    >>> accelData = sorted(dictSensor['LinearAccelarations'])
    >>> for time, val in accelData:
    ...     print(time, val)  # doctest: +ELLIPSIS
    (0, ['0.000000', '-0.010000', ... '-0.050000', '0.070000'])
    (30, ['-0.070000', '-0.090000', ... '0.000000', '0.070000'])
    (59, ['-0.070000', '0.010000', ... '0.080000'])
    (91, ['-0.070000', '0.030000', ... '0.010000'])
    ...

    :param filePath: string
    :return: None if file format is not correct, else dictSensor
    """
    logging.info('Reading IMU file %s', filePath)

    dictSensor = {}

    with open(filePath, 'rU') as filename:
        firstLine = filename.readline().split(' ')
        if len(firstLine) < 3 or firstLine[2] != 'IMU:':
            logging.info('%s is not an IMU file!', filePath)
            return None

        # get startTime to calculate time offsets. Assumes that all
        # measurements in one file where taken on the same day
        startTime = getTime(firstLine[1])
        filename.seek(0) # first line has to be processed again later

        for line in filename:
            seperated = line.strip().split(' ')
            time = getTime(seperated[1])

            if ('Quaternion:' not in seperated
                    or 'LinearAccelarations:' not in seperated):
                logging.debug('Line %s does not contain Quaternion and LinearAccelaration values',
                              line)
                return None

            # find the positions where Quaternion and LinearAcc values begin
            quadIndex = seperated.index('Quaternion:')
            linAccIndex = seperated.index('LinearAccelarations:')
            try:
                eulerIndex = seperated.index('Calculated_Euler_Angles:')
                insert_or_add(dictSensor, 'Calculated_Euler_Angles', [], [
                    (time, seperated[eulerIndex + 1 :])])
            except ValueError:
                eulerIndex = len(seperated)
            quat1 = seperated[quadIndex + 1 : quadIndex + 5]
            quat2 = seperated[quadIndex + 5 : quadIndex + 9]
            quat3 = seperated[quadIndex + 9 : linAccIndex]
            #print quat1
            quat1_f = []
            for i in quat1:
                quat1_f.append(float(i))
            quat2_f = []
            for i in quat2:
                quat2_f.append(float(i))
            quat3_f = []
            for i in quat3:
                quat3_f.append(float(i))
                
            #print(quat1_f)
            #print(quat2_f)
            #print(quat3_f)
            mat1 = trafo3d.quat2mat(quat1_f)
            mat2 = trafo3d.quat2mat(quat2_f)
            mat3 = trafo3d.quat2mat(quat3_f)
            flatmat1 = mat1.flatten()
            flatmat2 = mat2.flatten()
            flatmat3 = mat3.flatten()
            #print(mat1)
            #print(flatmat1)
            flatmatAll = [flatmat1, flatmat2, flatmat3]
            #print flatmatAll
            flatTotal = []
            for i in flatmatAll:
                for k in i:
                    flatTotal.append(k)
                    
            #print flatTotal
            
            #insert_or_add(dictSensor, 'Quaternion', [], [
            #    (time, seperated[quadIndex + 1 : linAccIndex])])
            insert_or_add(dictSensor, 'Quaternion', [], [
                (time, flatTotal)])
            insert_or_add(dictSensor, 'LinearAccelarations', [], [
                (time, seperated[linAccIndex + 1 : eulerIndex])])

    return dictSensor


def readOldData(filePath, modalityName):
    """
    Read one file in the old data format.

    Return the modality name and a list of (timestamp, values) tuples.
    """
    # check for a correct modality name or find it if None was given
    if modalityName and modalityName not in MODALITIES:
        print ('When reading data in the old data format the '
               + 'modality name has to be in {}!').format(MODALITIES)
        return (modalityName, None)
    elif not modalityName:
        splitPath = filePath.split('/')
        for mod in MODALITIES:
            if mod in splitPath:
                modalityName = mod
                break
        # no fitting folder was found
        if not modalityName:
            logging.info('When reading data in the old data format, the '
                         + 'data files have to be in a folder whose name is in '
                         + '{}!'.format(MODALITIES))
            return (modalityName, None)

    values = []
    with open(filePath, 'r') as datei:
        logging.info("open " + modalityName + " file " + filePath.split('/')[-1])

        # parse and read the file
        convertedTime = 0
        for zeile in datei:
            liste = zeile[:-1].split(" ")

            # get time
            if (liste[0] == 'Timestamp:') or (re.search("[0-9][0-9]/", liste[0])):
                convertedTime = getTime(liste[TIMESTAMP_POINT])

            # get values
            if ((liste[0] == SENSOR_NAME[modalityName]["Identity"])
                    or (re.search("[0-9][0-9]/", liste[0]))):
                # get joint torque values new data layout
                if liste[DATATYPE_POINT_NEW] == SENSOR_NAME[modalityName]["New"]:
                    valuesToAdd = DUMMY_DATA[modalityName] + liste[VALUE_POINT_NEW:]
                    values += [(convertedTime, valuesToAdd)]

                # get joint torque values old data layout
                elif liste[DATATYPE_POINT_OLD] == SENSOR_NAME[modalityName]["Old"]:
                    valuesToAdd = liste[VALUE_POINT_OLD: -1]
                    values += [(convertedTime, valuesToAdd)]

    return (modalityName, values)

def makeTimestampsRelative(timeValueList):
    startTime = timeValueList[0][0]
    return [(time - startTime, value) for time, value in timeValueList]


def writeCSV(dirpath, movement, modality, valueType, valueList):
    """ Write the data in valueList into a csv file.

    The file's name is composed of movement, modality (e.g. IMU) and
    valueType (e.g. Quaternion) as follows:
    dirpath/movement_modality_valueType.csv

    :param dirpath: path to CSV directory where file will be written
    :param movement: name of the movement
    :param modality: name of modality [eg IMU or DAQ]
    :param valueType: type of the modalities values (eg Quaternion)
    :param valueList: the actual values
    :return: None
    """
    filepath = os.path.normpath(dirpath + '/' + movement + '_' + modality)
    if valueType is not None:
        filepath += '_' + valueType
    filepath += '.csv'

    # include the time offset in the value list for the timestamp
    outputList = [[time] + values for (time, values) in valueList]

    # write the list as comma-seperated rows in filename
    with open(filepath, 'wb') as filename:
        wr = csv.writer(filename)
        wr.writerows(outputList)


def getTime(timestamp):
    """ Return time from timestamp string in ms.

    >>> getTime("14:50:34.659")
    53434659
    >>> getTime("00:00:00.000")
    0
    >>> getTime("00:00:00.100")
    100

    :param timestamp: string of the form "xx:xx:xx.xxx" [hours:minutes:seconds]
    :return: time in ms
    """
    # split string
    timeStampSplit = timestamp.split(":")
    secSplit = timeStampSplit[2].split(".")

    # convert timestamp in ms
    currentHour = int(timeStampSplit[0]) * 3600000
    currentMin = int(timeStampSplit[1]) * 60000
    currentSec = int(secSplit[0]) * 1000
    currentMilliSec = int(secSplit[1])
    convertedTime = currentHour + currentMin + currentSec + currentMilliSec
    return convertedTime


def ignoreMotion(motionname):
    """ Return true if motion motionname should be ignored for all statistics
    files.

    Motionnames that should be ignored must start with one of the prefices
    listed in IGNORE_FILENAMES.
    currently: IGNORE_FILENAMES = ['Test', 'test', 't_', 'Trial', 'neutral']

    >>> ignoreMotion('Test12')
    True
    >>> ignoreMotion('Test')
    True
    >>> ignoreMotion('t_something')
    True
    >>> ignoreMotion('topic500') # not every motionname starting with 't' is invalid
    False
    >>> ignoreMotion('neutral')
    True
    >>> ignoreMotion('neutral_aa')
    True
    >>> ignoreMotion('Oneneutralmotion')
    False

    """
    assert IGNORE_FILENAMES
    pattern = reduce(lambda l, r: l + '|' + re.escape(r),
                     IGNORE_FILENAMES, re.escape(IGNORE_FILENAMES[0]))
    return bool(re.match(pattern, motionname))


def isTestpersonDir(testpersonDir):
    """ Return true if testpersonDir is a valid testperson name.

    Possible names for testperson folders are: 'IDxy', 'Proband xy',
    'Probandxy', 'SegmentedData_xy' for new data or "SegmentedData_Sorted*"
    for old data.

    Valid names are stored in global variable TESTPERSONS.

    >>> isTestpersonDir('ID01')
    True
    >>> isTestpersonDir('TestpersonDir')
    False

    :param testpersonDir: string
    :return: bool
    """
    assert TESTPERSONS
    if testpersonDir is None:
        return False
    pattern = reduce(lambda l, r: l + '|' + re.escape(r),
                     TESTPERSONS, TESTPERSONS[0])
    pattern = '(' + pattern + r')\w+'
    return bool(re.match(pattern, testpersonDir))


def getTestpersonFromPath(testpersonDir):
    """
    Identify the testperson name in the path testpersonDir.

    Identifies the testperson by scanning the file system path.
    Returns the name or identifier of the testperson (e.g. 'ID01') or None
    if no such identifier was found.

    >>> getTestpersonFromPath('/some/path/ID01')
    'ID01'
    >>> getTestpersonFromPath('/some/path/noTestpersonDir') is None
    True
    >>> getTestpersonFromPath('ID01')
    'ID01'

    """
    testperson = [x for x in testpersonDir.split('/') if isTestpersonDir(x)]
    return testperson[-1] if testperson else None


def getTestpersonNumber(testperson):
    """ Return the identifying number of this testperson from its name.

    Requires that the number be at the end of the testperson's name.
    If the name contains no number, 0 is returned.

    >>> getTestpersonNumber('ID01')
    1
    >>> getTestpersonNumber('noNumber')
    0
    >>> getTestpersonNumber('Proband 12')
    12

    """
    match = re.search(r'[0-9]+$', testperson)
    return int(match.group()) if match else 0


def renameInPath(path):
    """ Return true if the path contains a folder named 'Rename'.

    >>> renameInPath('/path/to/Rename')
    True
    >>> renameInPath('/path/Rename/to')
    True
    >>> renameInPath('/path/to/something/else')
    False
    >>> renameInPath('/path/rename/to')
    True

    """
    # matching not case-sensitive
    return 'rename' in path.lower().split('/')


def parseArgs(args):
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser()
    # add relevant arguments to the parser
    parser.add_argument(
        '--csv',
        action='store_true',
        help='store the data in csv files. Location is \
        BASEPATH/{testperson}/{modality}/{motion}.csv'
    )
    parser.add_argument(
        '-b',
        '--basepath',
        default=BASEPATH,
        help='folder where all the experiment data lies. The testperson \
        folders are subfolders of BASEPATH. Possible names for testperson \
        folders are: %s for new data or "SegmentedData_Sorted*" for old data. \
        The default BASEPATH is %s.'
             % (str([name + 'xy' for name in TESTPERSONS]), BASEPATH)
    )
    parser.add_argument(
        '-i',
        '--inputfolder',
        help='subfolder of BASEPATH to read data from.',
        default=''
    )
    parser.add_argument(
        '--old',
        action='store_true',
        help='input files are in the old data format (format used by Adrian).'
    )
    parser.add_argument(
        '-l',
        '--loglevel',
        default='WARNING',
        help='Logging level [debug, info, warning]'
    )
    parser.add_argument(
        '--rename',
        action='store_true',
        help='Only read data in folder "rename".'
    )
    parser.add_argument(
        '--exoversion',
        help='Select the exo version with which the data was created. [exo1|exo2|exo2euler|exo2calc_moment|exo2moment]',
        default=EXO_STD)
    parser.add_argument(
        '-w', '--windowSize',
        default=500,
        help='Select window size'
    )
    return parser.parse_args(args)


def initLogging(loglevel):
    """ Initialize logging. """
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=numeric_level)
    return logging.getLogger(__name__)


def main(args):
    """ Read data and, depending on user input, either save it or create windows. """
    # start execution
    # Instance to the class Motion_Datareader_Train_test --> 'motionDatareader'
        # read raw data from the txt files
        # combine the data for different modalities
        # create windows to be used by the train and test code
    motionDatareader = Motion_Datareader_Train_Test() #object of the class 'Motion_Datareader_train_test

    logging.info("Python script started.")

    # check the data format
    if args.old or args.exoversion == 'exo1':
        motionDatareader.flag_old = True
        args.exoversion = 'exo1'
    if args.rename:
        motionDatareader.flag_rename = True
    motionDatareader.exoversion = args.exoversion
    global WINDOW_SIZE # t
    WINDOW_SIZE = int(args.windowSize) # This variable contains the windows size in ms

    # Read all data from datapath and its subfolders
    motionDatareader.datapath = args.basepath # We set the path where we are going to read and write the data

    ''' sometimes is not needed to read all the data
    motionDatareader.readAllData(
        os.path.normpath(args.basepath + '/' + args.inputfolder)) # it is set the direction where all the data lies (with all its subjects)
    '''

    # Contains the windowed data (X and y) sorted by testperson.
    # X contains the actual data for each window, while y contains tuples
    # (label, motionName) for each window

    # check for arguments and execute functions accordingly
    if args.csv: # depending on the command line --csv is set, we only convert all the documents to .csv or we create the windows
        #read all the data
        motionDatareader.readAllData(
            os.path.normpath(args.basepath + '/' + args.inputfolder))  # it is set the direction where all the data lies (with all its subjects)
        # write the collected data into csv files.
        motionDatareader.writeAllCSV(args.basepath)
    elif args.exoversion == 'exo2calc_moment': # calculation of the moments based on the force values --> then they are stored as .csv files in each user
        # creation of a derived feature --> moment on the knee --> saved as CSV files
        df.writeMomentCSV(BASEPATH,IGNORE_DIRS) # moments on the knee are calculated and stored on CSV files life new features based on the force values
    elif args.exoversion == 'exo2moment':
        motionDatareader.readCSVmoment(args.basepath, IGNORE_DIRS)
        motionDatareader.createWindows() # create the Windows
        motionDatareader.writeWindows(args.basepath) # store the Windows to be used later by HMMExo_Train_Test
        s=1
    else:
        #read all the data
        motionDatareader.readAllData(
            os.path.normpath(args.basepath + '/' + args.inputfolder))  # it is set the direction where all the data lies (with all its subjects)
        motionDatareader.createWindows()
        motionDatareader.writeWindows(args.basepath)


######################## Program flow ##################################

if __name__ == "__main__":
    args = parseArgs(sys.argv[1:]) # to handle command line arguments // ( the sys.argv[0] argument is the python script) . from sys.argv[1] : end there are the other list of arguments
    logger = initLogging(args.loglevel)
    main(args) # execution of the main function


