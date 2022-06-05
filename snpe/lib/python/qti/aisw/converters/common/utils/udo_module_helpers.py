# ==============================================================================
#
#  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import sys
from six import with_metaclass

if sys.version_info[0] > 2:
    string_types = [str, bytes]
else:
    string_types = [unicode, str]


# ------------------------------------------------------------------------------
#   Udo Misc Helper Methods
# ------------------------------------------------------------------------------
def title_case(string):
    if not isoneofinstance(string, string_types):
        raise TypeError("Cannot change non string object to camel case")
    else:
        lower_case = string.lower()
        return lower_case.title()


def is_static_tensor(tensor):
    return tensor.static


def isoneofinstance(object, instances):
    if iter(instances):
        for instance in instances:
            if isinstance(object, instance):
                return True
        return False


def reverse_dict(orig_dict):
    return {v: k for k, v in orig_dict.items()}


def set_data_types_to_internal(data_types):
    if any(data_type == "ALL_VALID" for data_type in data_types):
        return list(SnpeUdoConstants.SNPE_UDO_DATATYPES.keys())
    else:
        return data_types


def check_all_equal(iterable):
    return not iterable or iterable.count(iterable[0]) == len(iterable)


# ------------------------------------------------------------------------------
#   Udo Module Helper Functions and Classes
# ------------------------------------------------------------------------------
def udo_property_static(name):
    attr_name = '_' + name

    @property
    def prop(self):
        return getattr(self, attr_name, None)

    @prop.setter
    def prop(self, value):
        if hasattr(self, attr_name):
            raise ValueError('Cannot reset this field')
        setattr(self, attr_name, value)

    return prop


def udo_property(name):
    attr_name = '_' + name

    @property
    def prop(self):
        return getattr(self, attr_name, list())

    @prop.setter
    def prop(self, value):
        raise ValueError('Cannot set this field')

    return prop


def udo_property_type(name, expected_type):
    attr_name = '_' + name

    @property
    def prop(self):
        return getattr(self, attr_name)

    @prop.deleter
    def prop(self):
        raise IndexError("Cannot delete this field")

    @prop.setter
    def prop(self, value):
        if not isinstance(value, expected_type):
            raise TypeError('{} must be a valid object of type {}'.format(name, expected_type))
        if hasattr(self, attr_name):
            raise AttributeError('Cannot set {} field once it has been initialized'.format(attr_name))
        setattr(self, attr_name, value)

    return prop


class SnpeUdoConstants(object):
    SNPE_UDO_CORETYPES = {'CPU': 'SNPE_UDO_CORETYPE_CPU',
                         'GPU': 'SNPE_UDO_CORETYPE_GPU',
                         'DSP': 'SNPE_UDO_CORETYPE_DSP'}

    SNPE_UDO_TENSOR_LAYOUT = {'NHWC': 'SNPE_UDO_LAYOUT_NHWC',
                             'NCHW': 'SNPE_UDO_LAYOUT_NCHW',
                             'NDCHW': 'SNPE_UDO_LAYOUT_NDCHW',
                             'GPU_OPTIMAL1': 'SNPE_UDO_LAYOUT_GPU_OPTIMAL1',
                             'GPU_OPTIMAL2': 'SNPE_UDO_LAYOUT_GPU_OPTIMAL2',
                             'DSP_OPTIMAL1': 'SNPE_UDO_LAYOUT_DSP_OPTIMAL1',
                             'DSP_OPTIMAL2': 'SNPE_UDO_LAYOUT_DSP_OPTIMAL2'}

    SNPE_UDO_DATATYPES = {'FLOAT_16': 'SNPE_UDO_DATATYPE_FLOAT_16',
                         'FLOAT_32': 'SNPE_UDO_DATATYPE_FLOAT_32',
                         'FIXED_4': 'SNPE_UDO_DATATYPE_FIXED_4',
                         'FIXED_8': 'SNPE_UDO_DATATYPE_FIXED_8',
                         'FIXED_16': 'SNPE_UDO_DATATYPE_FIXED_16',
                         'UINT_8': 'SNPE_UDO_DATATYPE_UINT_8',
                         'UINT_16': 'SNPE_UDO_DATATYPE_UINT_16',
                         'UINT_32': 'SNPE_UDO_DATATYPE_UINT_32',
                         'INT_32': 'SNPE_UDO_DATATYPE_INT_32',
                         'STRING': 'SNPE_UDO_DATATYPE_UINT_8'}

    SNPE_CALCULATION_TYPES = {'SNPE_UDO_CORETYPE_CPU': 'SNPE_UDO_DATATYPE_FLOAT_16 | SNPE_UDO_DATATYPE_FLOAT_32',
                             'SNPE_UDO_CORETYPE_GPU': 'SNPE_UDO_DATATYPE_FLOAT_16 | SNPE_UDO_DATATYPE_FLOAT_32',
                             'SNPE_UDO_CORETYPE_DSP': 'SNPE_UDO_DATATYPE_INT_8 | SNPE_UDO_DATATYPE_INT_16'}

    SNPE_UDO_QUANT_TYPES = {'TF': 'SNPE_UDO_QUANTIZATION_TF',
                           'SKIP': 'SNPE_UDO_QUANTIZATION_NONE',
                           'QMN': 'SNPE_UDO_QUANTIZATION_QMN'}

    snpe_udo_coretypes = reverse_dict(SNPE_UDO_CORETYPES)


class UdoDescriptor(object):
    def __init__(self, name=None, **kwargs):
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __get__(self, instance, owner):
        return instance.__dict__[str(self.name)]


class SnpeUdoValidator(UdoDescriptor):
    expected_type = type(None)
    _expected_values = dict()

    def __set__(self, instance, value):
        if isoneofinstance(value, [list, tuple]):
            if any(not isoneofinstance(x, self.expected_type) for x in value):
                raise TypeError('Each {} must be a valid object of type {}'.format(value, self.expected_type))
            if any(x not in self.expected_values for x in value):
                raise TypeError('Each {} must be one of the following values {}'.format(value, self.expected_values))
            if instance.__dict__.get(self.name, None):
                raise AttributeError('Cannot set this field once it has been initialized')
            super(SnpeUdoValidator, self).__set__(instance, [self._expected_values[x] for x in value]
            if value[0] in self.expected_values else [reverse_dict(self._expected_values)[x] for x in value])
        else:
            if not isoneofinstance(value, self.expected_type):
                raise TypeError('{} must be a valid object of type {}'.format(value, self.expected_type))
            if value not in self.expected_values and value not in reverse_dict(self._expected_values):
                raise ValueError('Expected one of {} instead got {}'.format(self.expected_values, value))
            if instance.__dict__.get(self.name, None):
                raise AttributeError('Cannot set this field once it has been initialized')
            super(SnpeUdoValidator, self).__set__(instance, self._expected_values[value]
            if value in self.expected_values else reverse_dict(self._expected_values)[value])

    @property
    def expected_values(self):
        return list(self._expected_values.keys())

    @expected_values.setter
    def expected_values(self, value):
        raise ValueError('Cannot set this field')


class SnpeUdoDataType(SnpeUdoValidator):
    expected_type = string_types
    _expected_values = SnpeUdoConstants.SNPE_UDO_DATATYPES


class SnpeUdoCoreTypes(SnpeUdoValidator):
    expected_type = string_types
    _expected_values = SnpeUdoConstants.SNPE_UDO_CORETYPES


class SnpeUdoTensorLayouts(SnpeUdoValidator):
    expected_type = string_types
    _expected_values = SnpeUdoConstants.SNPE_UDO_TENSOR_LAYOUT


class SnpeUdoQuantTypes(SnpeUdoValidator):
    expected_type = string_types
    _expected_values = SnpeUdoConstants.SNPE_UDO_QUANT_TYPES


class UdoMeta(type):
    def __new__(mcs, mcsname, bases, methods):
        for key, value in methods.items():
            if isinstance(value, UdoDescriptor):
                value.name = key
        return type.__new__(mcs, mcsname, bases, methods)


class UdoTensorInfo(with_metaclass(UdoMeta)):
    tensor_layout = SnpeUdoTensorLayouts()
    data_type = SnpeUdoDataType()
    quant_type = SnpeUdoQuantTypes()

    def __init__(self, **args):
        if args:
            self.from_dict(args)

    def from_dict(self, tensor_dict, name=''):
        self.name = tensor_dict.get('name', name)
        self.data_type = set_data_types_to_internal(tensor_dict.get("data_type", "FLOAT_32"))
        self.dims = tensor_dict.get('dims', [])
        self.static = tensor_dict.get('static', False)
        self.default_value = tensor_dict.get('default_value', None)
        self.data = tensor_dict.get('data', [])
        self.tensor_layout = tensor_dict.get('tensor_layout', "NHWC")
        self.quant_type = tensor_dict.get('quantization_mode', 'SKIP')
        self.quant_params = tensor_dict.get('quantization_params', {})

    def get(self, item, default=None):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            return default

    def __getitem__(self, name):
        return self.__dict__[name]

    @staticmethod
    def create_tensor_infos(operator_dict, type):

        tensor_infos = list()

        for tensor in operator_dict.get(str(type), list()):
            tensor_info = UdoTensorInfo()
            tensor_info.from_dict(tensor)
            tensor_infos.append(tensor_info)
            setattr(tensor_info, "repeated", tensor.get("repeated", False))

        return tensor_infos




