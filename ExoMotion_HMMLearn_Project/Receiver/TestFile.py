#import IMUReceiver

class TestFile():
    counterZMD = 0

    def checkZMD(self):
        if (len(IMUReceiver.reportedIMUValues) != 0):
            print "reported imu values: %s" %IMUReceiver.reportedIMUValues
            TestFile.counterZMD = len(IMUReceiver.reportedZMDValues)
        else:
            print "no values"

print "entered test"
testf = TestFile()
testf.checkZMD()