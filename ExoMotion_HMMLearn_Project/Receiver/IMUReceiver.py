#!/usr/bin/python

import time
import TestFile

from armarx.interface_helper import load_armarx_slice, register_object, using_topic
load_armarx_slice('MMMCapture', 'units/ArduinoIMUUnit.ice')

from armarx import ArduinoIMUUnitListener

class IMUReceiver(ArduinoIMUUnitListener):
    reportedIMUValues = []
    counterIMU = 0
    #def reportForceSensorValues(self, pos1x, pos1y, pos1z, quaternion1, pos2x, pos2y, pos2z, quaternion2, current=None):
    def reportIMUSensorValues(self, imuDiff0, imuDiff1, imuDiff2, current=None):
        cur_time = time.time()
        time_diff = cur_time - getattr(self, 'last_time', 0.0)
        self.last_time = cur_time
        IMUReceiver.reportedIMUValues = [(imuDiff0, imuDiff1, imuDiff2)]
        print 'Received measurement: %f %f %f (%.2f ms since last measurement)' % (imuDiff0, imuDiff1, imuDiff2, time_diff * 1000.0)

        if (len(IMUReceiver.reportedIMUValues) != 0):
            print "reported imu values: %s" % IMUReceiver.reportedIMUValues
            IMUReceiver.counterIMU = len(IMUReceiver.reportedIMUValues)
        else:
            print "no values"

	
receiver_instance = IMUReceiver()
proxy = register_object(receiver_instance, 'IMUReceiver')
using_topic(proxy, 'ArduinoIMUUnit')

while True:
    pass
