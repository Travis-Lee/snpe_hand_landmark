# ==============================================================================
#
#  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from qti.aisw.converters.common.udo.snpe_udo_config import *
from qti.aisw.converters.common.udo.udo_core import *
from qti.aisw.converters.common.utils import code_to_message
import traceback
import sys
import copy

# global variables
udo_package_resolver = dict()


class UDOFactory(object):
    """
    The Factory object which manages all UdoOp creation and instantiation.
    """
    udo_collection = UdoCollection()

    def __init__(self):
        super(UDOFactory, self).__init__()

    @staticmethod
    def create_udo_op(name, inputs, outputs, params, **args):
        """
        This method creates a generic UdoOp. Note that the UdoOp is an abstract class, so the Op returned here
        is intended to be implemented.

        :param name: The name of the UdoOp object
        :param inputs: A list of inputs to the Op which must be of type UdoTensorInfo or UdoTensor.
        :param outputs: A list of outputs to the Op which must be of type UdoTensorInfo or UdoTensor.
        :param params: A list of params to the Op, which must be of type UdoParam
        :param args:   Optional arguments to the UdoOp constructor
        :return: A partially-defined UdoOp object
        """
        return UdoOp(name, input_tensors=inputs, output_tensors=outputs, params=params, **args)

    @classmethod
    def create_udo_ops_from_operator(cls, operator, model=None, converter_type='onnx', **kwargs):
        """
        Creates multiples ops from a single Operator object, using a list of src_ops in the model
        that match the operator spec.
        :param operator: The operator to be used
        :param model: The framework model
        :param converter_type: The given converter type
        :return:
        """
        nodes = cls.get_src_ops(operator.name, model, converter_type)
        resolved_udo_ops = []

        for node in nodes:
            resolved_udo_ops.append(cls.create_udo_op_from_operator(operator, node, model,
                                                                    converter_type, **kwargs))

        return resolved_udo_ops

    @classmethod
    def create_udo_op_from_operator(cls, operator, node=None, model=None, converter_type='onnx',
                                    **kwargs):
        """
        This method creates a UdoOp from an Operator object, based on the operator itself, the src_op node, the model
        containing said Op and the converter type.

        :param operator: An Operator object
        :param node:     The node as defined in the framework Model, needed if the model is not provided.
        :param model: The framework model, needed if the node is not provided.
        :param converter_type: The converter type to which the config containing the operator definition was provided.
        :return: A well-defined UdoOp based on the converter type selected, the default return value is a UdoOnnx Op.
        """
        input_tensor_info = copy.deepcopy(operator.input) # pass by value
        output_tensor_info = copy.deepcopy(operator.output)
        param_info = operator.scalar_param + operator.tensor_param
        if not node:
            raise RuntimeError(code_to_message.get_error_message("ERROR_CANNOT_CREATE_UDO_OP"))
        if str(converter_type).lower() == 'onnx':
            return cls.__create_onnx_udo_op(node, input_tensor_info, output_tensor_info, param_info, model)
        elif str(converter_type).lower() == 'caffe':
            return cls.__create_caffe_udo_op(node, input_tensor_info, output_tensor_info,
                                             param_info, caffe_net=kwargs['caffe_net'])
        elif str(converter_type).lower() == 'tf':
            return cls.__create_tf_udo_op(node, input_tensor_info, output_tensor_info, param_info, model)
        elif str(converter_type).lower() == 'caffe2':
            return cls.__create_caffe2_udo_op(node, input_tensor_info, output_tensor_info, param_info, model)

    @classmethod
    def get_src_ops(cls, name, model, converter_type):
        if str(converter_type).lower() == 'onnx':
            return cls.get_onnx_src_ops(name, model)
        elif str(converter_type).lower() == 'caffe':
            return cls.get_caffe_src_ops(name, model)
        elif str(converter_type).lower() == 'caffe2':
            raise NotImplementedError
        elif str(converter_type).lower() == 'tf':
            return cls.get_tf_src_ops(name, model)

    @staticmethod
    def update_tensor_infos_with_src_op_names(tensor_infos, src_op_names):
        """
        Changes the tensor infos that will be ingested by a UdoOp to match an actual instance
        of a src_op. Note that ordering of both iterables must match.
        :param tensor_infos: A list of input or output tensor infos
        :param src_op_names: A list of names for each input or output tensors
        :return:
        """
        new_tensor_infos = []
        num_of_repeated_tensors = len([tensor_info for tensor_info in tensor_infos if tensor_info.repeated])
        if num_of_repeated_tensors > 1:
            raise ValueError("There can be at most one repeated tensor in a UDO spec, found: {} repeated tensors"
                             .format(num_of_repeated_tensors))

        if tensor_infos[0].repeated:
            # find the tensor that is static, as that will not be replicated.
            # note that this requires all static tensors be placed after repeated tensors, which is a reasonable
            # assumption since the first input is almost always data.
            static_tensor_info = [tensor_info for tensor_info in tensor_infos if tensor_info.static]
            if static_tensor_info:
                variadic_tensor_info_len = len(src_op_names) - len(static_tensor_info)
            else:
                variadic_tensor_info_len = len(src_op_names)

            for j in range(variadic_tensor_info_len):
                # creates a copy of the tensor_info to be replicated, so that each new tensor_info in the list
                # will have a different address in memory
                __tensor_info = copy.deepcopy(tensor_infos[0])
                # once we duplicate a tensor it is no longer repeated
                __tensor_info.repeated = False
                new_tensor_infos.append(__tensor_info)

            new_tensor_infos = new_tensor_infos + static_tensor_info

            if len(new_tensor_infos) != len(src_op_names):
                raise Exception(code_to_message.get_error_message("ERROR_CANNOT_RESOLVE_VARIADIC_OPERATION")
                                (tensor_infos[0].name))
        else:
            new_tensor_infos = tensor_infos

        for i, name in enumerate(src_op_names):
                new_tensor_infos[i].name = name

        return new_tensor_infos

    @staticmethod
    def create_param_info(operator_dict):
        """
         This method returns a dictionary of UdoTensorInfos using a an operator-dictionary specification. This allows
         a user to create a list of param infos that can be passed to a UdoOp without creating an Operator object.

        :param operator_dict: A dictionary specification of the params, defined as:
                                scalar_params: {}, tensor_params: {}
        :return:
        """
        param_info = dict()
        if "scalar_params" in operator_dict:
            for param_info in operator_dict["scalar_params"]:
                param_info.update(param_info)

        if "tensor_params" in operator_dict:
            tensor_param_info = (UdoTensorInfo.create_tensor_infos(operator_dict, "tensor_params"))
            for param_info in tensor_param_info:
                param_info.update(param_info)

        return param_info

    @staticmethod
    def __create_onnx_udo_op(src_op, input_tensor_info, output_tensor_info, param_info, model=None):
        """
        :return: A well-defined Udo Onnx Op object
        """
        # TO-DO: there are weird dependencies in our converter where we import onnx
        #        in places where we don't need it.
        from qti.aisw.converters.onnx.udo_op_definition import UdoOnnxOp
        output_tensor_info = UDOFactory.update_tensor_infos_with_src_op_names(output_tensor_info, src_op.output)
        input_tensor_info = UDOFactory.update_tensor_infos_with_src_op_names(input_tensor_info, src_op.input)
        return UdoOnnxOp(src_op, input_tensor_info, output_tensor_info, param_info, model)

    @staticmethod
    def __create_caffe2_udo_op(src_op, input_tensor_info, output_tensor_info, param_info, model=None):
        """
        :return: TO-DO
        """
        raise NotImplementedError

    @staticmethod
    def __create_tf_udo_op(src_op, input_tensor_info, output_tensor_info, param_info, model=None):
        """
       :return: A well-defined UdoTfOp object
       """

        from qti.aisw.converters.tensorflow.layers.udo import UdoTfOp
        output_tensor_info = UDOFactory.update_tensor_infos_with_src_op_names(output_tensor_info,
                                                                              [output.name for output in
                                                                               src_op.outputs])

        input_tensor_names = [input.name for input in src_op.inputs]
        _param_info = copy.deepcopy(param_info)  # to avoid changing original param_info

        # loop is to support the case where an input is mis-classified as a param. This loop finds that param
        #  in src_op.op_def.input_args and designates it as static,
        #  implying that it has data which would be retrieved either from a constant/identity op.
        for i, name in enumerate(input_tensor_names):
            # if any tensor infos are repeated, the logic to identify mis-classified params would not be valid.
            # once a repeated tensor info is identified, the code below will repeat the first repeated tensor and
            # cause an early exit.
            if input_tensor_info[i].repeated:
                input_tensor_info[i:] = UDOFactory.update_tensor_infos_with_src_op_names(input_tensor_info[i:],
                                                                                        input_tensor_names[i:])
                break
            try:
                UDOFactory.update_tensor_infos_with_src_op_names([input_tensor_info[i]], [name])
            except IndexError:
                input_param = None
                for info in _param_info:
                    # iterates through the list of input args, and updates if the input index and the names match
                    # any input_arg
                    for j, input_arg in enumerate(src_op.op_def.input_arg):
                        if input_arg.name == info.name and i == j:
                            input_param = input_arg
                            info.name = name
                            info.static = True
                            break
                if not input_param:
                    raise RuntimeError("Input with name: {} is not registered in Udo operator config".format(name))
        return UdoTfOp(src_op, input_tensor_info, output_tensor_info, _param_info, model)

    @staticmethod
    def __create_caffe_udo_op(src_op, input_tensor_info, output_tensor_info, param_info, caffe_net=None):
        """
       :return: creates a well-defined Caffe Udo Op object
       """

        from qti.aisw.converters.caffe.udo_op_definition import UdoCaffeOp
        UDOFactory.update_tensor_infos_with_src_op_names(output_tensor_info, src_op.top)
        UDOFactory.update_tensor_infos_with_src_op_names(input_tensor_info, src_op.bottom)
        return UdoCaffeOp(src_op, input_tensor_info, output_tensor_info, param_info, caffe_net,
                          validate=True)

    @staticmethod
    def set_static_tensors_to_params(input_tensor_info, param_info):
        """
        Turns an input tensor with static data, such as a const, identity or initialized tensor into
         a param. It removes
        the tensor from the input tensor info and places it into the param info.

        :param input_tensor_info: A list of UdoTensorInfos
        :param param_info: A list of UdoTensorInfos
        :return:
        """
        for tensor_info in input_tensor_info:
            if tensor_info.get('static'):
                param_info.append(tensor_info)
                input_tensor_info.pop(tensor_info['name'])

    @staticmethod
    def get_onnx_src_ops(op_name, model):
        """
        Gets an onnx node from a model using its op_name

        :param op_name: The name of the onnx op
        :param model: The ModelProto object
        :return: the nodes if present, or a TypeError otherwise
        """
        nodes = []
        found = False
        for node in model.graph.node:
            if str(node.op_type).lower() == str(op_name).lower():
                nodes.append(node)
                found = True
        if not found:
            raise TypeError(code_to_message.get_error_message("ERROR_UDO_OP_NOT_FOUND")(op_name, model.graph.name))
        return nodes

    @staticmethod
    def get_tf_src_ops(op_type, model):
        """
        Gets a TF node from a model using its op_type

        :param op_type: The type of the TF op
        :param model: A TF converter ModelLoader object
        :return: the nodes if present, or a TypeError otherwise
        """
        nodes = []
        found = False
        for node in model.session.graph.get_operations():
            if str(node.type) == op_type:
                nodes.append(node)
                found = True
        if not found:
            log_error(code_to_message.get_error_message("ERROR_UDO_OP_NOT_FOUND")
                            (op_type, ""))
            raise TypeError
        return nodes

    @staticmethod
    def get_caffe_src_ops(op_type, model):
        """
        Gets a caffe node from a model using its op_type

        :param op_type: The type of the caffe op
        :param model: A repeated composite container of caffe objects
        :return: the nodes if present, or a TypeError otherwise
        """
        layers = []
        found = False
        for layer in model:
            if str(layer.type) == op_type:
                layers.append(layer)
                found = True
        if not found:
            log_error(code_to_message.get_error_message("ERROR_UDO_OP_NOT_FOUND")
                      (op_type, ""))
            raise TypeError
        return layers

    def parse_config(self, config_path, model=None, converter_type='onnx', **kwargs):
        """
         Parses a user provided json config into a udo op object. The config is expected to contain information
         about a user's operation as well as a package containing the op definition.
         See sample config in <examples> for more info. A UdoOp object is created from the parsed information and added
         to a UdoCollection object. Note that if no operator defined in the config spec, a udo op will not be created.

         :param config_path: The file path to the user's json config file
         :param model: The model containing the op(s) defined in the config spec.
         :param converter_type: The converter type from which the config was passed.
         """
        # Import config
        with open(config_path, 'r') as json_config:
            config_vars = json.load(json_config)

        for udo_package_name, udo_package_dict in config_vars.items():
            new_udo_package = UdoPackage(udo_package_dict['UDO_PACKAGE_NAME'])
            udo_package_info = UdoPackageInfo.from_dict(udo_package_dict)
            new_udo_package.add_package_info(udo_package_info)
            if model:
                for operator in udo_package_info.operators:
                    # Create UDO object and add to UDO collection
                    try:
                        udo_ops = self.create_udo_ops_from_operator(operator, model=model,
                                                                    converter_type=converter_type, **kwargs)
                        self.udo_collection[udo_ops[0].op_type] = udo_ops
                        for udo_op in udo_ops:
                            if udo_op.op_type in udo_package_resolver:
                                if udo_package_resolver[udo_op.op_type] != new_udo_package.name:
                                    raise ValueError("Attempted to register the same op with name:{} across"
                                                     " the two different packages:{} vs {}".
                                                     format(udo_op.op_type,
                                                            udo_package_resolver[udo_op.op_type], new_udo_package.name))
                            udo_package_resolver[udo_op.op_type] = new_udo_package.name
                    except TypeError:
                        continue
                    except Exception as e:
                        if not is_log_level_debug():
                            traceback.print_exc()
                            sys.exit(-1)
                        else:
                            log_debug(str(e))
