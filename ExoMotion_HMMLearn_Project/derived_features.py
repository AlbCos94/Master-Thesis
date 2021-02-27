
import numpy as np
import math
import os
import Motion_Datareader_Train_Test_10 as Datareader
import csv
import shutil


def writeMomentCSV( path , ignore_names):
    '''
    :param path: directory where there are the folders of each person
    :return: creates CSV for each person and motion of the moments on the knee based on the force sensor files
    '''

    for testperson in sorted(os.listdir( path )):  # we check every folder that is in that path

        ignore_dir=['CSV','CSV_moments']

        if testperson in ignore_names: # for person with ID_num / only ID_num files are considered
            continue
        else:   #inside the folder of a person
            path_forces=os.path.normpath(path + '/' + testperson + '/' + 'DAQ') # we look directly into the person's file

            if os.path.exists(path_forces+ '/' + 'CSV_moment'):  # all the old data of moments is removed
                shutil.rmtree(path_forces+ '/' + 'CSV_moment')

            for file_motion in sorted(os.listdir(path_forces)):  #inside the folder of DAQ checking each motion and repetition

                if file_motion not in ignore_dir: #''!='CSV' and file_motion!='CSV_moment':
                    path_data=os.path.normpath(path_forces + '/' + file_motion)
                    dictForces= Datareader.readDAQ(path_data) # store the data of the .txt force file in a dictionary
                    moments= moment_calc(dictForces) # from a dictionary of forces, it returns a list of the moments based on
                    motion = getMotionNameFromFilename(file_motion) # we get the string of 'motion_num_repetition' from the file's name
                    writeCSV(path_forces,motion,moments) # the CSV files are created --> folder: 'CSV_moment'

        print ('moments of '+ testperson + ' calculated and saved')


def writeCSV(path, motion, moments):
    """ Write the data in valueList into a csv file.
    The file's name is composed of movement and repetition
    dirpath/movement_modality_num_repetition.csv

    :param path: path where are we going to write the .csv files
    :param motion: name of the motion (e.g. DropObject)
    :param moments: list of tuples: [(timeStamp1, [Mx1,My1,Mz1]) , (timeStamp2, [Mx2,My2,Mz2]), ... ]

    :return: write the CSV file
    """
    directory = os.path.normpath(path + '/' + 'CSV_moment')

    if not os.path.exists(directory): # if the folder doesn't exist , it is created
        os.mkdir(directory)

    filepath = directory + '/' + motion + '.csv'

    with open(filepath, 'a+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for element in moments:
            filewriter.writerow([element[0]] + element[1]) # writes timestamp + moment values


def moment_calc( dictSensor ):
    '''
    calculation of the moments based on force sensors values
    :param dictSensor: dictionary with force sensor values and the timestamps--> keys are u'KIT0A001 and u'KIT0A002
    :return: moments --> list of tuples: [(timeStamp1, [Mx1,My1,Mz1]) , (timeStamp2, [Mx2,My2,Mz2]), ... ]
                            moment values in X, Y and Z axis on the knee caused by the forces captured by the force sensors and the timestamp when they were captured.
    '''
    #distances exoskeleton (millimeters)
    #thigh and shank distances
    alpha=37.5* (math.pi/180.0) # angle [degrees] --> [radians]
    d1 = 30.5
    d2 = 47.5
    d3 = 120.5
    d4 = 97
    d5 = 48.5
    d6 = 32
    d7 = 41
    d8 = 29.5
    d9 = 31.5
    d10 = 29
    d11 = 32


    #transformation matrix to change the force basis --> new reference system ( y axis along the leg, perpendicular to the ground )
    #                                                           ( z axis along the foot )
    C1 = np.array([[1, 0, 0], [0, 1, 0],[0, 0, 1]])
    C2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    C3 = np.array([[-math.cos(alpha), 0, -math.sin(alpha)], [0, 1, 0], [math.sin(alpha), 0, -math.cos(alpha)]])
    C4 = np.array([[-math.cos(alpha), 0, math.sin(alpha)], [0, 1, 0], [-math.sin(alpha), 0, -math.cos(alpha)]])
    C6 = np.array([[1, 0, 0], [0, 1, 0],[0, 0, 1]])

    C5 = np.array([[-1, 0, 0], [0, 1, 0],[0, 0, -1]])
    C7 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    num_elements= len(dictSensor[u'KIT0A001'])
    moments=[]

    for i in range(num_elements):
        # timestamp when the data was captured
        ts = dictSensor[u'KIT0A001'][i][0]
        # logical order as in the .txt files
        FS1 = np.array(dictSensor[u'KIT0A001'][i][1][0:3])
        FS2 = np.array(dictSensor[u'KIT0A001'][i][1][3:6])
        FS3 = np.array(dictSensor[u'KIT0A001'][i][1][6:9])
        FS4 = np.array(dictSensor[u'KIT0A001'][i][1][9:12])
        FS5 = np.array(dictSensor[u'KIT0A002'][i][1][0:3])
        FS6 = np.array(dictSensor[u'KIT0A002'][i][1][3:6])
        FS7 = np.array(dictSensor[u'KIT0A002'][i][1][6:9])

        # transppse of the vector
        FS1 = FS1[np.newaxis, :].T
        FS2 = FS2[np.newaxis, :].T
        FS3 = FS3[np.newaxis, :].T
        FS4 = FS4[np.newaxis, :].T
        FS5 = FS5[np.newaxis, :].T
        FS6 = FS6[np.newaxis, :].T
        FS7 = FS7[np.newaxis, :].T

        # forces in the new basis
        FS1 = C1.dot(FS1)
        FS2 = C2.dot(FS2)
        FS3 = C3.dot(FS3)
        FS4 = C4.dot(FS4)
        FS5 = C5.dot(FS5)
        FS6 = C6.dot(FS6)
        FS7 = C7.dot(FS7)

        # knee moments calculation
        # upper leg
        Mx1 = FS1[1]*d2 + FS1[2]*d3 - FS2[1]*d1 + FS2[2]*d4 + FS3[2]*d5 - FS3[1]*d10*math.cos(alpha) + FS4[2]*d5 - FS4[1]*d10*math.cos(alpha) + FS6[2]*d5 + FS6[1]*d11
        My1 = -FS1[0]*d2 + FS2[0]*d1 + FS3[0]*d10*math.cos(alpha) - FS3[2]*d10*math.sin(alpha) + FS4[0]*d10*math.cos(alpha) + FS4[2]*d10*math.sin(alpha) - FS6[0]*d11
        Mz1 = -FS1[0]*d3 - FS2[0]*d4 + FS3[1]*d10*math.sin(alpha) - FS4[1]*d10*math.sin(alpha) - FS6[0]*d5 - FS3[0]*d5 -FS4[0]*d5
        #lower leg
        Mx2 = -d7*FS7[2] - d6*FS5[2] - d8*FS5[1] + d9*FS7[1]
        My2 = d8*FS5[0] - d9*FS7[0]
        Mz2 = d6*FS5[0] + d7*FS7[0]
        #total moments
        Mx=float(Mx1+Mx2)
        My=float(My1+My2)
        Mz=float(Mz1+Mz2)

        moments=moments+[(ts,[Mx,My,Mz])]

    return moments


def getMotionNameFromFilename(filename):
    # Return the motion name that is part of filename.

    # remove the file ending and replace any other '.' by '_'
    namelist = reduce(lambda l, r: l + '_' + r,
        filename.split('.')[:-1])

    return reduce(lambda l, r: l + '_' + r, namelist.split('_')[:-1])