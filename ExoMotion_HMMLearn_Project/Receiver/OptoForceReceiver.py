#!/usr/bin/python

from __future__ import print_function
import logging

import time
import Ice

from armarx.interface_helper import load_armarx_slice, register_object, using_topic
from armarx.interface_helper import ice_helper
load_armarx_slice('RobotAPI', 'units/OptoForceUnit.ice')
                                                                        
from armarx import TimestampBase  


from armarx import OptoForceUnitListener


logger = logging.getLogger(__name__)

class OptoForceReceiver(OptoForceUnitListener):
    
    
    def __init__(self):
        super(OptoForceUnitListener, self).__init__() 
        self.reportedOptoForceValues = []
        self.counterOptoForce = 0
        
    #def reportForceSensorValues(self, pos1x, pos1y, pos1z, quaternion1, pos2x, pos2y, pos2z, quaternion2, current=None):
    def reportSensorValues(self, device, name, fx, fy, fz, timestamp, current=None):
        logger.info('report sensor values')
 
        print ('Received measurement: %f %f %f (ms since last measurement)' % (fx, fy, fz))
	
    
    
class Timestamp(TimestampBase):                                            
    pass                                                                         
                                                                                   
                                                                                   
class TimestampFactory(Ice.ObjectFactory):                                    
    def create(self, t):                                                         
        return Timestamp()          
        
        
        
def main():    
    logging.basicConfig(level=logging.DEBUG)
    ice_helper.iceCommunicator.addObjectFactory(TimestampFactory(), Timestamp.ice_staticId())


    receiver_instance = OptoForceReceiver()
    proxy = register_object(receiver_instance, 'OptoForceReceiver')
    using_topic(proxy, 'OptoForceValues')

    #while True:
    while not ice_helper.iceCommunicator.isShutdown():
        pass
        
        
if __name__ == '__main__':
    main()
