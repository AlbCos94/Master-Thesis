from operator import xor

class ArgParseError(ValueError):
    """
    Raise if errors occur while parsing a component structure.

    The field 'validNames' supplies the possible correct options that may be chosen instead.
    """
    def __init__(self, message, validNames):
        super(ArgParseError, self).__init__(message)
        self.validNames = validNames

'''
A component describes a modality (e.g. IMU) or a sub-modality (e.g.
LinearAccelerations). It can either contain other components or fields
in a certain order. The order has to be the same order that was used for
the sensor fusion. It can be used to
extract certain modalities from a data window, which means that the
indices of the corresponding columns in the window matrix are returned.
'''
class Component(object):
    def __init__(self, name, numSensors, valuesPerSensor,
                 fields=None, components=None):
        # a component can either consist of fields XOR components
        assert xor(fields is None, components is None)
        # name of this component, e.g. ZMD or Quaternion
        self.name = name
        # list of all fields that can be extracted from each sensor.
        # The order of the fields is not important.
        if fields is not None:
            for field in fields:
                # assure that a field knows the correct number of values
                # that each sensor provides
                field.setValuesPerSensor(valuesPerSensor)
        self.fields = fields
        # sorted list of all sub-components this component is comprised of
        if components is not None:
            assert components
            sumSizes = 0
            for component in components:
                sumSizes += component.size()
            assert sumSizes == valuesPerSensor * numSensors
        self.components = components
        # number of sensors that measure data for this modality
        self.numSensors = numSensors
        # number of values that each sensor returns
        self.valuesPerSensor = valuesPerSensor
        # start index of this component in the higher-level component list
        self.setStartIdx(0)

    '''
    Get the number of indices belonging to this component.
    '''
    def size(self):
        return self.numSensors * self.valuesPerSensor

    '''
    Recursively sets the start indices of this component and of all
    sub-components.
    Necessary to ensure that the start index of all components and fields
    is absolute. This means that the start index represents the absolute
    position of the modality in the complete feature vector.
    '''
    def setStartIdx(self, idx):
        self.startIdx = idx
        if self.components is None:
            return
        idxSum = 0
        for comp in self.components:
            comp.setStartIdx(idxSum + self.startIdx)
            idxSum += comp.size()

    '''
    Get all the indices belonging to this component, starting with self.startIdx.
    '''
    def getAll(self):
        return [item + self.startIdx for item in range(self.size())]

    '''
    Get all value indices for the specified sensor.
    '''
    def getFromSensor(self, sensorIdx):
        if sensorIdx < 0 or sensorIdx >= self.numSensors:
            raise ValueError('Sensor indices for component {}'.format(self.name)
                             + ' have to be between [0, {})'.format(self.numSensors))
        if self.fields is not None:
            start = sensorIdx * self.valuesPerSensor + self.startIdx
            return range(start, start + self.valuesPerSensor)
        else:
            idxList = []
            for comp in self.components:
                idxList += comp.getFromSensor(sensorIdx)
            return idxList

    '''
    Get all value indices for the specified sensors.
    '''
    def getFromSensors(self, sensorIdxList):
        idxList = []
        for idx in sensorIdxList:
            idxList += self.getFromSensor(idx)
        return idxList

    '''
    The same as getComponent/getField, but decides which one to use.
    Raise ArgParseError if self has no sub-component or field with the name 'name'.
    '''
    def getPart(self, name):
        if self.fields is not None:
            return self.getField(name)
        else:
            return self.getComponent(name)

    '''
    The same as getComponentFromSensor/getFieldFromSensor, but decides
    which one to use.
    Raise ArgParseError if self has no sub-component or field with the name 'name'.
    '''
    def getPartFromSensor(self, name, sensorIdx=None):
        if sensorIdx is None:
            part = self.getPart(name)
            if self.fields: # part is a field
                return part.getAll(self.numSensors)
            else: # part is a sub-component
                return part.getAll()
        elif self.fields is not None:
            return self.getFieldFromSensor(name, sensorIdx)
        else:
            return self.getComponentValuesFromSensor(name, sensorIdx)

    '''
    Return the sub-component with the specified name.
    Raise ArgParseError if self has no sub-component with the name 'name'.
    '''
    def getComponent(self, name):
        try:
            idx = getListIdxFromName(name, self.components)
        except ArgParseError as e:
            e.message = 'Component {} has no sub-component {}'.format(self.name, name)
            raise
        return self.components[idx]

    '''
    Get the indices belonging to a sub-component. The returned
    column indices are absolute values.
    Raise ArgParseError if self has no sub-component with the name 'name'.
    '''
    def getComponentValues(self, name):
        comp = self.getComponent(name)

        return comp.getAll()

    '''
    Get the indices belonging to a sub-component for the sensors that
    are specified by their indices. The returned
    column indices are absolute values.
    Raise ArgParseError if self has no sub-component with the name 'name'.
    '''
    def getComponentValuesFromSensor(self, name, sensorIdx):
        comp = self.getComponent(name)

        return comp.getFromSensor(sensorIdx)

    '''
    Return the field with the specified name.
    Raise ArgParseError if self has no field with the name 'name'.
    '''
    def getField(self, name):
        try:
            # index of the field in the field list
            idx = getListIdxFromName(name, self.fields)
        except ArgParseError as e:
            e.message = 'Component {} has no field {}'.format(self.name, name)
            raise

        return self.fields[idx]

    '''
    Get the indices belonging to the field for the specified sensor.
    The returned column indices are relative to the column indices of
    this component.
    Raise ArgParseError if self has no field with the name 'name'.
    '''
    def getFieldFromSensor(self, name, sensorIdx):
        field = self.getField(name)
        if sensorIdx < 0 or sensorIdx >= self.numSensors:
            raise ValueError('Sensor indices for component {}'.format(self.name)
                             + ' have to be between [0, {})'.format(self.numSensors))

        idxList = field.getFromSensor(sensorIdx)
        return [col + self.startIdx for col in idxList]

'''
A Field describes one or more values from one sensor (e.g. the x-field
is the first value of each sensor's value triplet). It can be used to
extract certain modalities from a data window, which means that the
indices of the corresponding columns in the window matrix are returned.
If only one value is wanted, 'end' can be omitted.
'''
class Field(object):
    def __init__(self, name, begin, end=None, valuesPerSensor=None):
        # name of the field, e.g. 'x'
        self.name = name
        # values per sensor, has to be the same as the surrounding component
        self.valuesPerSensor = valuesPerSensor
        # inclusive start index
        self.begin = begin
        # exlusive end index
        if end is not None:
            self.end = end
        else:
            self.end = begin + 1

        # assert that no field uses more values than a single sensor can provide
        if valuesPerSensor is not None:
            assert self.begin >= 0 and self.end <= valuesPerSensor

    def setValuesPerSensor(self, valuesPerSensor):
        assert self.begin >= 0 and self.end <= valuesPerSensor
        self.valuesPerSensor = valuesPerSensor

    '''
    Get the values corresponding to this field from the sensor with
    the index idx. The returned index list is relative to the start
    index of the surrounding component.
    '''
    def getFromSensor(self, idx):
        start = idx * self.valuesPerSensor
        return range(start + self.begin, start + self.end)

    '''
    Get the values corresponding to this field from all sensors whose
    index is in idxList. The returned index list is relative to the
    start index of the surrounding component.
    '''
    def getFromSensors(self, idxList):
        returnList = []
        for idx in idxList:
            returnList += self.getFromSensor(idx)
        return returnList

    '''
    Get the values corresponding to this field from all sensors The
    returned index list is relative to the start index of the
    surrounding component.
    '''
    def getAll(self, numSensors):
        return self.getFromSensors(range(numSensors))

# non-class functions

'''
Get the index of the sub-component or field with the specified name
in the list of components/fields. Uses assertions to make sure that
the list and an entry with the correct name exist.
If field is True self.fields is searched, otherwise self.components
is used.
Raise ArgParseError if there is no entry in liste with entry.name
'name'.
'''
def getListIdxFromName(name, liste):
    assert liste is not None
    fieldNames = [field.name for field in liste]
    # index of the sub-component/field in the component/field list
    try:
        return fieldNames.index(name)
    except ValueError as e:
        raise ArgParseError(e.message, fieldNames)
