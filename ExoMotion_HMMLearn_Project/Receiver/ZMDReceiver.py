#!/usr/bin/python

import time

from armarx.interface_helper import load_armarx_slice, register_object, using_topic
load_armarx_slice('MMMCapture', 'units/ZMDMeasurementUnit.ice')

from armarx import ZMDMeasurementUnitListener

class ZMDReceiver(ZMDMeasurementUnitListener):
    def reportForceSensorValues(self, currentForce1, currentForce2, currentForce3, currentForce4, currentForce5, currentForce6, currentForce7, currentForce8, current=None):
        cur_time = time.time()
        time_diff = cur_time - getattr(self, 'last_time', 0.0)
        self.last_time = cur_time

        print 'Received measurement: %f %f %f %f %f %f %f %f (%.2f ms since last measurement)' % (currentForce1, currentForce2, currentForce3, currentForce4, currentForce5, currentForce6, currentForce7, currentForce8, time_diff * 1000.0)

receiver_instance = ZMDReceiver()
proxy = register_object(receiver_instance, 'ZMDReceiver3')
using_topic(proxy, 'ZMDMeasurementUnit')

while True:
    pass

