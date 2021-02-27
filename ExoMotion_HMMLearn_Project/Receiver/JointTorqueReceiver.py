#!/usr/bin/python

import time

from armarx.interface_helper import load_armarx_slice, register_object, using_topic
load_armarx_slice('MMMCapture', 'units/ARMAR5ActorUnit.ice')

from armarx import ARMAR5ActorUnitTorqueListener
class JointTorqueReceiver(ARMAR5ActorUnitTorqueListener):
    def reportTorqueSensorValues(self, thigh_joint, lowerleg_joint, current=None):
        cur_time = time.time()
        time_diff = cur_time - getattr(self, 'last_time', 0.0)
        self.last_time = cur_time

        print 'Received measurement: %f %f (%.2f ms since last measurement)' % (thigh_joint, lowerleg_joint, time_diff * 1000.0)

receiver_instance = JointTorqueReceiver()
proxy = register_object(receiver_instance, 'JointTorqueReceiver')
using_topic(proxy, 'ARMAR5ActorUnitJointTorque')

while True:
    pass

