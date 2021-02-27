import numpy as np
import os.path
import csv
from operator import itemgetter



def createSoftMaxFile(writeDir, scores, probabilities, testperson, y_test_current):


    motions = list(map(itemgetter(0), scores)) # we take the labels of the motions as a list
    score_values = list(map(itemgetter(1), scores))

    file_path_csv = writeDir + "/" + "score_values.csv"  # csv file with the current score values and the soft_max values (probabilities)

    i_predicted = score_values.index(max(score_values))
    i_probability = motions.index(y_test_current)

    motion_predicted = motions[i_predicted]

    values=[]
    values.append(y_test_current) # first column contains the real motions
    values.append(motion_predicted) # second column contains the motions predicted
    values+=score_values  # second column contains the motions predicted
    values+=probabilities  # second column contains the motions predicted

    with open(file_path_csv, 'a+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        #filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(values)


    '''
    
    if motion_predicted != y_test_current:
        motions.append(y_test_current)
        motions.append("Error")
        difference = max(probabilities) - probabilities[i_probability]# difference between the wrong motion chosen vs the correct one
        probabilities.append(' ')
        probabilities.append(difference)
        #std_user.motion_errors[y_test_current] += 1
        #std_user.motion_errors[y_test_current] += 1

    else:
        motions.append(y_test_current)
        motions.append("Correct")
        probabilities.append(' ')
        probabilities.append(0)
        #std_user.motion_errors[y_test_current] += 1

    with open(file_path_csv, 'a+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(motions)
        filewriter.writerow(score_values)
        filewriter.writerow(probabilities)
    '''



def createComparisonFile(writeDir, scores, probabilities, testperson, y_test_current): # REMOVE


    motions = list(map(itemgetter(0), scores)) # we take the labels of the motions as a list
    score_values = list(map(itemgetter(1), scores))


    file_path= writeDir + "/" + "Comparison_predicted_y.txt"       # txt file
    file_path_csv = writeDir + "/" + "Comparison_predicted_y.csv"  # csv file


    i_predicted = score_values.index(max(score_values))
    i_probability = motions.index(y_test_current)

    motion_predicted = motions[i_predicted]

    if motion_predicted != y_test_current:
        motions.append(y_test_current)
        motions.append("Error")
        difference = max(probabilities[0:2]) - probabilities[i_probability]# difference between the wrong motion chosen vs the correct one
        probabilities.append(' ')
        probabilities.append(difference)
        #std_user.motion_errors[y_test_current] += 1
        #std_user.motion_errors[y_test_current] += 1

    else:
        motions.append(y_test_current)
        motions.append("Correct")
        probabilities.append(' ')
        probabilities.append(0)
        #std_user.motion_errors[y_test_current] += 1

    with open(file_path_csv, 'a+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(motions)
        filewriter.writerow(score_values)
        filewriter.writerow(probabilities)


    predictedY = open(file_path, "a+") # if the file doesn't exist, it is created
    predictedY.write('LABELS PREDICTED\t' + ' ' * 4 + 'SUBJECT: ' + testperson + ' ' * 4 + 'CURRENT_LABEL: ' + y_test_current + '\n')
    predictedY.write(str(scores) + '\n')
    predictedY.write(str(probabilities) + ('\n') * 2)

    '''
    if os.path.exists(file_path) :
        predictedY= open(file_path, "a+")
        predictedY.write('LABELS PREDICTED' + ' ' * 4 + 'SUBJECT: ' + testperson + ' ' * 4 + 'CURRENT_LABEL: ' + y_test_current + '\n')
        predictedY.write(str(scores) + '\n')
        predictedY.write(str(probabilities) + ('\n')*2)

    else :
        with open(writeDir + "/" + "comparison_predicted_y.txt", "w") as predictedY:
            predictedY.write('LABELS PREDICTED' + ' ' * 4 + 'SUBJECT: ' + testperson + ' ' * 4 + 'CURRENT_LABEL: '+ y_test_current+ '\n')
            predictedY.write(str(scores)+'\n')
            predictedY.write(str(probabilities)+('\n')*2)
    '''

def softmax(scores):
    scores_values = list(map(itemgetter(1), scores)) # we only take the values
    exps = np.exp(scores_values)
    probabilities = exps/sum(exps)


    # for ERROR

    #s=sum(probabilities)

    return probabilities.tolist() # convert to list

class CLikelihoods_analysis:

    subject = "void"
    motions = [] # list of motions we analise
    y_test_current = '' # label we want to predict 

    w_predictions = 0.0  # wrong predictions in total
    r_predictions = 0.0  # right predictions in total
    total_precision = 0.0  # total precision
    avg_error_offset_cs = 0.0  # average error offset in the current scores
    avg_error_offset_sm = 0.0  # average error offset in the sofmax scores
    motion_errors = {}  # motions error classification for each motion
    motion_tests = {}  # motions tests for each motion
    motion_precision = {}
    motion_offsets_errors_cs = {}  # using current scores
    motion_max_offset_error_cs = {}  # using current scores
    motion_offsets_errors_sm = {}  # using softmax
    motion_max_offset_error_sm = {}  # using softmax

    """
    pattern:
    motion_tests = {'WalkingForward': 0, 'WalkingBackward': 0, 'TurnLeftSmall': 0,'TurnRightSmall': 0, 'LiftObjectUP' : 0, 'DropObject': 0,
                      'SidestepsLeft': 0, 'SidestepsRight': 0, 'GoingUpstairs': 0, 'GoingDownstairs': 0, 'GoingDownstairsBackwards': 0,
              'SitDownArmsCrossed': 0, 'StandUpArmsCrossed': 0, 'stand': 0}
    motion_precision = {'WalkingForward': 0, 'WalkingBackward': 0, 'TurnLeftSmall': 0,'TurnRightSmall': 0, 'LiftObjectUP' : 0, 'DropObject' : 0,
                      'SidestepsLeft': 0, 'SidestepsRight': 0, 'GoingUpstairs': 0, 'GoingDownstairs': 0, 'GoingDownstairsBackwards' : 0,
              'SitDownArmsCrossed': 0, 'StandUpArmsCrossed': 0, 'stand': 0}
    motion_offsets_errors = {'WalkingForward' : 0, 'WalkingBackward' : 0, 'TurnLeftSmall': 0,'TurnRightSmall': 0, 'LiftObjectUP': 0, 'DropObject': 0,
                      'SidestepsLeft': 0, 'SidestepsRight': 0, 'GoingUpstairs': 0, 'GoingDownstairs': 0, 'GoingDownstairsBackwards': 0,
              'SitDownArmsCrossed': 0, 'StandUpArmsCrossed': 0, 'stand': 0}
    """

    def __init__(self, subject, motions):
        self.subject = subject  # name of the subject analysed
        self.motions = motions  # name of all considered motions

        for motion in motions:
            self.motion_errors[motion] = 0.0
            self.motion_tests[motion] = 0.0
            self.motion_precision[motion] = 0.0
            self.motion_offsets_errors_sm[motion] = 0.0
            self.motion_max_offset_error_sm[motion] = 0.0
            self.motion_offsets_errors_cs[motion] = 0.0
            self.motion_max_offset_error_cs[motion] = 0.0

    def read_csv(self, file_path_csv):

        with open(file_path_csv, 'r') as csvFile:

            reader = csv.reader(csvFile)

            for row in reader:
                real_motion = row[0]
                classified_motion = row[1]

                self.motion_tests[real_motion] += 1

                if real_motion != classified_motion: # error in the calssification
                    self.w_predictions += 1
                    self.motion_errors[real_motion] += 1
                    current_values = row[2:1+(len(row)/2)]
                    softmax_values = row[1+(len(row)/2):]

                    i_real_motion = self.motions.index(real_motion)
                    i_classified_motion = self.motions.index(classified_motion)

                    value_real_motion_current = float(current_values[i_real_motion])
                    value_real_motion_softmax = float(softmax_values[i_real_motion])

                    value_classified_motion_current = float(current_values[i_classified_motion])
                    value_classified_motion_softmax = float(softmax_values[i_classified_motion])

                    difference = value_classified_motion_softmax - value_real_motion_softmax
                    difference2 = value_classified_motion_current - value_real_motion_current

                    self.motion_offsets_errors_sm[real_motion] += difference  # difference between the wrong motion chosen vs the correct one
                    self.motion_offsets_errors_cs[real_motion] += difference2

                    if float(difference) > self.motion_max_offset_error_sm[real_motion]:
                        self.motion_max_offset_error_sm[real_motion] = difference

                    if float(difference2) > self.motion_max_offset_error_cs[real_motion]:
                        self.motion_max_offset_error_cs[real_motion] = difference2

                else:
                    self.r_predictions += 1

        for motion in self.motion_offsets_errors_sm:

            if float(self.motion_errors[motion]) > 0:
                self.motion_offsets_errors_sm[motion] = float(self.motion_offsets_errors_sm[motion]) / float(self.motion_errors[motion])  # average offset when we make an error per each motion
                self.motion_offsets_errors_cs[motion] = float(self.motion_offsets_errors_cs[motion]) / float(self.motion_errors[motion])  # average offset when we make an error per each motion

            self.motion_precision[motion] = (1 - (float(self.motion_errors[motion]) / float(self.motion_tests[motion])))

        self.total_precision = float(self.r_predictions) / (float(self.r_predictions) + float(self.w_predictions))

        if len(self.motion_offsets_errors_sm) > 0:
           self.avg_error_offset_sm = float(sum(self.motion_offsets_errors_sm.values())) / float(len(self.motion_offsets_errors_sm))
        if len(self.motion_offsets_errors_cs) > 0:
           self.avg_error_offset_cs = float(sum(self.motion_offsets_errors_cs.values())) / float(len(self.motion_offsets_errors_cs))


    def print_results(self, writeDir):
        file_path_txt = writeDir + "/" + "Results_comparison.txt"  # txt file

        results = open(file_path_txt, "a+")  # if the file doesn't exist, it is created

        results.write("USER: " + self.subject +'\n'+'\n')

        for motion in self.motion_errors:
            results.write("Motion: " + motion + '\n')
            results.write("errors " + "= " + str(self.motion_errors[motion]) + '\n')
            results.write("precision in the classification (%) " + "= " + str(self.motion_precision[motion]*100.0) + '\n')
            results.write("average offset error using current scores " + "= " + str(self.motion_offsets_errors_cs[motion]) + '\n')  # average offset in each error
            results.write("max offset error using current scores " + "= " + str(self.motion_max_offset_error_cs[motion]) + '\n')  # max offset in each error
            results.write("average offset error using softmax (%) " + "= " + str(self.motion_offsets_errors_sm[motion]*100.0) + '\n')  # average offset in each error
            results.write("max offset error using softmax (%) " + "= " + str(self.motion_max_offset_error_sm[motion]*100.0) + '\n'+'\n')  # max offset in each error




        results.write('\n')
        results.write("RESULTS" + '\n')
        results.write("total wrong predictions = " + str(self.w_predictions) + '\n')
        results.write("total right predictions = " + str(self.r_predictions) + '\n')
        results.write("total precision (%) = " + str(self.total_precision*100.0)+'\n')
        results.write("avg offset error using current scores = " + str(self.avg_error_offset_cs) + '\n')
        results.write("avg offset error using softmax (%) = " + str(self.avg_error_offset_sm*100.0) + '\n')


        ''' 
        predictedY.write(
            'LABELS PREDICTED\t' + ' ' * 4 + 'SUBJECT: ' + testperson + ' ' * 4 + 'CURRENT_LABEL: ' + y_test_current + '\n')
        predictedY.write(str(scores) + '\n')
        predictedY.write(str(probabilities) + ('\n') * 2)

        for motion in self.motion_errors:
                                                
            print( "errors in "+motion +"= " + str( self.motion_errors[motion] ) + '\n' )
            print( "average offset error in "+ motion +"= " + str( self.motion_offsets_errors_sm[motion]) + '\n') #average offset in each error
            print( "precision in classifying " + motion + "= " + str(self.motion_precision[motion]) + '\n' )
        '''
