# First installation
virtualenv .

source bin/activate

## only on lab PCs
pip install -U pip

## everyone
pip install numpy

pip install -r requirements.txt

## install armarx python bindings into your virtualenv (only needed for HMMExo_Online_Test.py)
Create a file named 'armarx.pth' in `[this_dir]/lib/python2.7/site-packages`. Put the following line in it:
`/usr/lib/python2.7/dist-packages/`

After that you can go to the point: Create the DataMatrix and Label Files

## Troubleshooting
There might be problems with installing scipy. In that case it has to be installed manually:
`pip install scipy`

If pip install does not work, see
https://stackoverflow.com/questions/22878109/error-installing-scipy-library-through-pip-on-python-3-compile-failed-with-err


# Start Program
source bin/activate

# Create the DataMatrix and Label Files
python Motion_Datareader.py
## usage
usage: Motion_Datareader_Train_Test.py [-h] [--csv] [-b BASEPATH]
                                       [-i INPUTFOLDER] [--old] [-l LOGLEVEL]

optional arguments:

- -h, --help: show this help message and exit
- --csv: store the data in csv files. Location is
         `BASEPATH/{testperson}/{modality}/{motion}.csv`
- -b BASEPATH, --basepath BASEPATH: 
    folder where all the experiment data lies. The
    testperson folders are subfolders of BASEPATH.
    Possible names for testperson folders are: ['IDxy',
    'Proband xy', 'Probandxy'] for new data or
    "SegmentedData_Sorted*" for old data. The default
    BASEPATH is `/common/share/Vicon_Data/Vicon/SFB_Exo_Vorstudie/Exo_Aufnahmen`.
- -i INPUTFOLDER, --inputfolder INPUTFOLDER:
    subfolder of BASEPATH to read data from.
- --old: input files are in the old data format (format used by Adrian).
- -l LOGLEVEL, --loglevel LOGLEVEL: Logging level [debug, info, warning]

## usage examples
__python Motion_Datareader_Train_Test.py__

Data in BASEPATH and its subfolders is read and converted to windows. The windows are saved as pkl/npy files in `BASEPATH/{testperson}/MotionPrediction`.

__python Motion_Datareader_Train_Test.py -h__

print a small help message

__python Motion_Datareader_Train_Test.py --csv__

 read all data in `/common/share/Vicon_Data/Vicon/SFB_Exo_Vorstudie/Exo_Aufnahmen/` (aka default BASEPATH) and save them as csv in `BASEPATH/{testperson}/{modality}/{motion}.csv`

__python Motion_Datareader_Train_Test.py --csv -i ID01/DAQ/__
or __python Motion_Datareader_Train_Test.py --csv --inputfolder ID07__

read all data in `/common/share/Vicon_Data/Vicon/SFB_Exo_Vorstudie/Exo_Aufnahmen/ID01/DAQ` resp. `/common/share/Vicon_Data/Vicon/SFB_Exo_Vorstudie/Exo_Aufnahmen/ID07` and save them as csv

__python Motion_Datareader_Train_Test.py --csv -i ID01/DAQ/ -b /home/folder/__
or __python Motion_Datareader_Train_Test.py --csv -b /home/folder/__

read all data in `/home/folder/ID01/DAQ/` resp. `/home/folder/` and same them as csv. Long version of -b is --basepath.

__python Motion_Datareader_Train_Test.py -i ID01__

read all data in /common/share/Vicon_Data/Vicon/SFB_Exo_Vorstudie/Exo_Aufnahmen/ID01 and convert it into windows.

When reading old data, the basepath should be `path/to/exo_motion_prediction`. Any subfolders starting with SegmentedData_Sorted
are considered testperson folders. If a special folder is wanted, for example SegmentedData_Sorted_Small (which contains old data), use:

__python Motion_Datareader_Train_Test.py --old -b path/to/exo_motion_prediction -i SegmentedData_Sorted_Small__

## Result
Files in `BASEPATH/[Testperson]/MotionPrediction`
- ExoMotions_X.npy  ->  Matrix with all Motions
- ExoMotions_y.pkl  ->  Labels from the Motions

# Train and test the HMM

python HMMExo_Train_Test.py

## Usage
HMMExo_Train_Test.py [-h] [-b BASEPATH] [-t TESTPERSONS]
                            [-v EXOVERSION] [-l LOGLEVEL]
                            [modalities [modalities ...]]

positional arguments:

- modalities: Select the modalities for training the HMM. "ZMD"
    selects all ZMD values, "ZMD=xz" selects the x and z
    values of all ZMD sensors and "ZMD=1x,2,3y" selects
    the x value of the first sensor, the y value of the
    third sensor and all values of the second sensor.
    Usage with IMU is similiar: possible values are "l"
    for LinearAccelerations and "q" for Quaternion.

optional arguments:

- -h, --help: show this help message and exit
- -b BASEPATH, --basepath BASEPATH:
                    folder where all the experiment data lies The
                    testperson folders are subfolders of BASEPATH.
                    Possible names for testperson folders are: ['IDxy',
                    'Proband xy', 'Probandxy'] for new data or
                    "SegmentedData_Sorted*" for old data.
- -t TESTPERSONS, --testpersons TESTPERSONS:
                    Select the testpersons whose data is used to train the
                    HMM. Use only integers. If -t is not specified, all
                    testpersons are selected. Use "-t 1,3,5" to select the
                    testpersons 1, 3 and 5. Leading zeros do not need to
                    be added to the number, "-t 1" for example will select
                    a testperson named "ID01" or "Proband 00001".
- -v EXOVERSION, --exoversion EXOVERSION:
                    Select the exo version with wich the data was created.
                    [exo1|exo2]
- -m MOTIONS, --motions MOTIONS:
                    Select the motions labels for which the HMM is
                    trained. If -m is not specified, all labels are
                    selected. Use "-m B1,B2_oE" for example to select all
                    motions whose names begin with B1 or B2_oE.
- -l LOGLEVEL, --loglevel LOGLEVEL:
                    Logging level [debug, info, warning]

# Usage examples
__python HMMExo_Train_Test.py -t 2,7 IMU__

Train the HMM with the data from testperson 2 and 7 and with only the IMU values

__python HMMExo_Train_Test.py IMU=1l,q ZMD__

Train the HMM with the data form all testpersons using the IMU:LinearAccelerations values from IMU sensor 1 and all ZMD values

## Notes
With 27 Trainingsmotions and 13 Testmotions per round, 7 states and a maximum number of 100 Iterations, the HMM has a hit rate of 95%.

With 56 Trainingsmotions and 28 Testmotions per round, 10 states and a maximum number of 100 Iterations, the HMM has a hit rate of 89.29% - 90.48%
