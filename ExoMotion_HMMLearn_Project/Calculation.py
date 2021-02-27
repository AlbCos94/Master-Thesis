from __future__ import division
import sys
import os

class Calculation():
    #Input "python" in terminal is not count, counting starts from position 2
    #sys.argv[0] = python program
    #sys.argv[1] = count right predictions
    #sys.argv[2] = count wrong predictions	
    if (len(sys.argv) != 3):
        print "Please insert count of right (as first) and count of wrong (as second) predictionns behind the python file!"
        exit()
    else:
        right = sys.argv[1]
        wrong = sys.argv[2]
        hitRate = 0

    # calculate accuracy
    def calc(self):
        Calculation.hitRate = int(Calculation.right) / float(int(Calculation.wrong)+int(Calculation.right)) * 100


        print "right: %s" %Calculation.right
        print "wrong: %s" %Calculation.wrong
        print "hitrate: %s" %Calculation.hitRate
        
# program flow
calculate = Calculation()
calculate.calc()
