# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import collections
from abc import abstractmethod, ABCMeta
import numpy as np
import inspect
from qti.aisw.converters.common.utils.udo_module_helpers import *
from qti.aisw.converters.common.utils.converter_utils import log_debug, log_warning


# ------------------------------------------------------------------------------
#   Udo Module Core Classes
# ------------------------------------------------------------------------------
class UdoCollection(collections.MutableMapping):
    """
    Organizes a udo op based on its type into a mapping, whereby the key is the op_type and the
    values are a FIFO queue of all instances of that op_type that have been seen in an op_collection
    instance.
    """

    def __init__(self, **kwargs):
        super(UdoCollection, self).__init__()
        self.op_count = 0
        if kwargs:
            for name, value in kwargs:
                setattr(self, name, value)

    def __getitem__(self, name):
        if name not in self.__dict__:
            raise KeyError("Object has not been registered with a Udo Collection".format(name))
        return getattr(self, name)

    def __setitem__(self, op_type, udo_op_queue):
        """
        Sets an entry in a udo_collection, where the key is expected to be a Valid UdoOp name and
        the value is a udo_op_queue, which contains all udo_ops of a certain type in the order
        that they appear in the model when its nodes are traversed using <graph>.node.

        e.x MyUdoCollection[op_type] = [ArgmaxUdoOp_1, ArgMaxUdoOp_2.......]

        :param op_type: The name of the op
        :type udo_op_queue: list of udo ops
        :return:
        """
        if not isinstance(udo_op_queue, list):
            raise TypeError("Udo op queue argument must be a list. Got: {}".format(type(udo_op_queue)))
        if not all([op.op_type == op_type for op in udo_op_queue]):
            raise RuntimeError("Expected all provided Udo Ops to be of the same type: {}, instead got: {}",
                               op_type, [op.op_type != op_type for op in udo_op_queue])

        for op in udo_op_queue:
            if not isinstance(op, UdoOp):
                raise TypeError("Argument is not a valid Udo Op")
            elif op_type not in self.__dict__:
                setattr(self, op_type, [op])
                self.op_count += 1
            else:
                self.__dict__[op_type].append(op)

    def __delitem__(self, key):
        if hasattr(self, key):
            self.__dict__.pop(key)

    def __iter__(self):
        return iter(self)

    def __len__(self):
        return self.op_count

    def get_first_of(self, op_type):
        """
        Gets the first element of its internal queue for a given op_type, raises an IndexError if
        there are no ops left in the queue.

        :param op_type: The type of the op to be extracted
        :return:
        """
        udo_op_queue = getattr(self, op_type)
        udo_op = udo_op_queue[0]
        self.__dict__[op_type].pop(0)
        return udo_op

    def get_op_by_name(self, op_type, name):
        udo_op_queue = getattr(self, op_type)
        udo_op = [op for op in udo_op_queue if op.name == name]
        if not udo_op:
            raise LookupError("Udo op with name: {} was not found in the Udo Collection".format(name))
        return udo_op[0]


class UDOParam(object):
    param_obj = udo_property_type('param_obj', object)

    def __init__(self, name, param_type, param_obj):
        self.name = name
        self.param_type = param_type
        self.param_obj = param_obj


class UdoParamTypes(object):
    """
    Enum class that defines UdoParamTypes
    """
    TENSOR = 0
    SCALAR = 1
    STRING = 2


class UDOTensorParam(object):
    def __init__(self, data_type, data, tensor_info):
        self.data_type = data_type
        if type(data) is np.ndarray or data:
            self.data = np.asarray(data) # if array is already numpy then this does not create a new array
            self.dims = self.data.shape
        else:
            self.data = data
            self.dims = tensor_info['dims']
        self.static = tensor_info['static']
        self.rank = len(self.dims) if self.dims else 0
        self.tensor_layout, self.quant_type, self.quant_min, self.quant_max = self.get_tensor_param_info(tensor_info)
        self.default_value = tensor_info['default_value']

    @staticmethod
    def get_tensor_param_info(tensor_info):
        tensor_layout = tensor_info.get('tensor_layout', "NHWC")
        quant_type = tensor_info.get('quant_type', 'SKIP')  # should actually default to skip
        quant_params = tensor_info.get('quant_params', {})
        quant_min = 0
        quant_max = 0

        if quant_params and quant_type != "SKIP":
            quant_min = tensor_info['quant_params']['min']
            quant_max = tensor_info['quant_params']['max']

        return tensor_layout, quant_type, quant_min, quant_max

    def as_dict(self):
        attrs = dict()
        attrs['param_type'] = 'SNPE_UDO_PARAMTYPE_TENSOR'
        attrs['data_type'] = self.data_type
        attrs['data'] = self.data
        attrs['static'] = self.static
        attrs['dimensions'] = self.dims
        attrs['tensor_layout'] = self.tensor_layout
        attrs['quant_type'] = self.quant_type
        attrs['quant_min'] = self.quant_min
        attrs['quant_max'] = self.quant_max

        return attrs

    @staticmethod
    def from_tensor_info(tensor_info):
        self = UDOTensorParam(tensor_info['data_type'],
                             tensor_info['data'], tensor_info)
        self.__setattr__('name', tensor_info['name'])
        return self


class UDOScalarParam(object):
    def __init__(self, data_type, data):
        self.data_type = data_type
        self.data = data

    def as_dict(self):
        attrs = dict()
        attrs['param_type'] = 'SNPE_UDO_PARAMTYPE_SCALAR'
        attrs['data_type'] = self.data_type
        attrs['data'] = self.data
        return attrs


class UDOString(object):
    def __init__(self, string):
        self.string = string + '\0'  # null termination is necessary

    def as_dict(self):
        attrs = dict()
        attrs['param_type'] = 'SNPE_UDO_PARAMTYPE_STRING'
        attrs['data_type'] = 'SNPE_UDO_DATATYPE_UINT_8'
        attrs['string'] = self.string
        return attrs


class UdoOp(object):
    """
    Abstract class which describes a Udo Operation in terms of its inputs, outputs and parameters.
    The intended usage is to create an Op that can be consumed by converters.
    """
    __metaclass__ = ABCMeta
    methods = dict()
    inputs = udo_property_type('inputs', list)
    outputs = udo_property_type('outputs', list)

    def __init__(self, op_type, input_tensors, output_tensors, params=None, param_info=None, src_op=None,
                 infer_output_shapes=None, name=""):
        """
        This method initializes a UdoOp with args provided, and as well as other members which depend on the provided
        arguments.
        :param op_type: The type of the UdoOp
        :param input_tensors: A list of UdoTensorInfo or UdoTensor objects.
        :param output_tensors: A list of UdoTensorInfo or UdoTensor objects.
        :param params: A dictionary of params such that "name": UdoParam are the key-value pairs.
        :param param_info: An optional argument, which can define params as a list of UdoTensorInfos. Note that
                            initialization will attempt to call extract attrs on this argument.
        :param src_op: An optional argument, which is a framework node, or any object upon which a call
                        to extract attrs has a well-defined behavior.
        :param infer_output_shapes: An optional argument, which is expected to be an infer_output_shapes method
                                    that is callable with the same signature as the abstract method.
        :param name: Optional string indicating the name of the op
        """
        self.name = name
        self.op_type = op_type
        self.outputs = output_tensors
        self.output_dims = [tensor['dims'] for tensor in output_tensors]
        self.params = params if params else self.extract_attrs(src_op, param_info)
        self.inputs = self.set_static_tensor_to_param(input_tensors)
        self.axis_orders = dict()
        self.custom_infer_shape_method = self.check_and_return_method(infer_output_shapes)
        self.set_axis_orders(self.inputs)
        self.set_axis_orders(self.outputs)

    @classmethod
    @abstractmethod
    def extract_attrs(cls, src_op, param_info):
        """
        The intention of this method is to extract param_info from a framework src_op and return a dictionary of
        Udo Param objects, such that "attr_name": "UdoParam". This must be implemented, as it is called during
        initialization
        :param src_op: Framework src_op
        :param param_info: Parameter info
        :return: A dictionary of UdoParams
        """

    def validate(self, *args):
        """
        The intention of this method is to call validation on the input tensors and params provided in the src_op,
        in comparison with the input_tensor_info and param_info defined in a config spec, or included initialized with
        an instance of a UdoOp. It is optional to implement this method. The default behavior of this method is to
        return nothing.

        :param args: optional arguments
        :return: User defined behavior, or nothing if method is not overwritten.
        """

    @abstractmethod
    def infer_output_shapes(self, node, **kwargs):
        """
        This method recieves a framework node and returns the output shapes
        :param node:
        :param kwargs:
        :return: a list of lists which contain output dimensions for each output tensor
        """

    def set_output_dims(self, src_op, output_tensor_info, model):
        """
        Creates an output tensor from a tensor info object. An output tensor must have a valid dimension,
        or the shape of each output tensor must be determinable.
        :param src_op: The framework op as defined in the model
        :param output_tensor_info: A list of output tensor infos
        :param model: The framework model
        :return: An output tensor, which is a tensor info with a valid dimension field.
        """
        output_dims = [tensor_info['dims'] for tensor_info in output_tensor_info]
        if any(not dim for dim in output_dims):
            output_dims = self.infer_output_shapes(src_op, model=model, perform_shape_inference=True)
        for i, tensor_info in enumerate(output_tensor_info):
            tensor_info.dims = output_dims[i]
        return output_tensor_info

    def set_static_tensor_to_param(self, tensors):
        """
        Sets a static tensor to a param. This method is called by the base class, meaning instances of this class
        are expected to have static tensors become params. This method takes a single tensor, and changes it to a
        param object. Note that a static tensor must have a data field defined.
        :param tensors: The tensor to be made a param.
        """
        local_tensor = []
        for tensor_info in tensors:
            if tensor_info.static:
                log_debug('Static UDO input tensor found. Note this tensor will be ingested as a Udo param')
                self.params[tensor_info['name']] = UDOParam(tensor_info['name'], UdoParamTypes.TENSOR,
                                                            UDOTensorParam(tensor_info['data_type'],
                                                                           tensor_info['data'], tensor_info))
            else:
                local_tensor.append(tensor_info)

        return local_tensor

    def get_method(self, key):
        """
        Returns an overridden method to the user. The base class method can only return an overriden shape inference
        method. To add any other method to a Udo Op, either the base class must be subclassed, or a Customizable UdoOp
        must be used. See Customizable UdoOp in UdoHelper Module for more info.
        :param key: valid method key
        :exception: returns a KeyError if key is not known.
        """
        if key != "CUSTOM_SHAPE_INFERENCE":
            raise KeyError("Only the shape inference method can be overriden in the base class, use CustomizableUdoOp"
                           " from the udo helper module to override all methods ")
        else:
            return self.custom_infer_shape_method

    @staticmethod
    def check_and_return_method(method):
        """

        :param method: User provided method
        :raises AssertionError if method is not None or a valid method or free function

        """
        # if the method is None then there is no shape inference method provided
        # Otherwise it must be a valid free function or bound function(method)
        try:
            assert (inspect.isfunction(method) or inspect.ismethod(method) or method is None)
            return method
        except AssertionError as e:
            raise Exception('Received the following error: {} while calling Method: {]'.format(e, method.__name__))

    def set_axis_orders(self, tensors):
        """
         Returns the corresponding IR axis order from the tensor layouts defined by each tensor passed as argument.
        :param tensors: The list of UdoTensorInfo or UdoTensor objects
        """
        # first check all tensors have the same tensor layout,
        # otherwise implicit permute may cause issues.

        layouts = [tensor.tensor_layout for tensor in tensors]
        if not check_all_equal(layouts):
            log_warning(" Distinct tensor layouts for the same tensor type may introduce implicit "
                        "permutes for each tensor into the DLC during conversion. "
                        "The following tensors were found to be distinct: {}"
                        , zip([str(tensor.name) for tensor in tensors], layouts))

        for tensor in tensors:
            # need to preset this here, because we may need an implicit permute
            if tensor.tensor_layout == SnpeUdoConstants.SNPE_UDO_TENSOR_LAYOUT['NCHW']:
                self.axis_orders[tensor.name] = 'NCS'
            elif tensor.tensor_layout == SnpeUdoConstants.SNPE_UDO_TENSOR_LAYOUT['NHWC']:
                self.axis_orders[tensor.name] = 'NSC'
            elif tensor.tensor_layout in SnpeUdoConstants.SNPE_UDO_TENSOR_LAYOUT.values():
                # user has selected one of the runtime specific layouts, setting non-trivial
                # to avoid axis-tracking
                self.axis_orders[tensor.name] = 'NON-TRIVIAL'
            else:
                # let IR axis tracking determine it
                self.axis_orders[tensor.name] = 'NOT_YET_DEFINED'

    def as_dict(self):
        params = {param.name: param.param_obj.as_dict() for _, param in self.params.items()}
        inputs_as_tensor_param = list(map(UDOTensorParam.from_tensor_info, self.inputs))
        outputs_as_tensor_param = list(map(UDOTensorParam.from_tensor_info, self.outputs))
        inputs = collections.OrderedDict()
        inputs.update({param.name: param.as_dict() for param in inputs_as_tensor_param})
        outputs = collections.OrderedDict()
        outputs.update({param.name: param.as_dict() for param in outputs_as_tensor_param})

        return inputs, outputs, params
