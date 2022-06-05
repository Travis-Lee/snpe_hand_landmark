# ==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.udo.udo_core import \
    UdoOp, UDOParam, UDOScalarParam, UDOTensorParam, UDOString, UdoParamTypes
from qti.aisw.converters.common.utils import code_to_message
import onnx

try:
    from onnx import version_converter
    version_converter_available = True
except ImportError:
    version_converter_available = False


class UdoOnnxOp(UdoOp):
    """
    A subclass of the UdoOp interface which implements framework specific methods defined in UdoOp. Calling this class
     requires that an Onnx module can be imported. Note that an instance of this class has further requirements than
     a generic UdoOp, in that the output dims must be determinable or provided. Additionally, the parameters must be
     extractable from the op. See UdoOp for all methods that will be called when a UdoOnnxOp is instantiated
    """

    def __init__(self, src_op, input_tensor_info, output_tensor_info, param_info, model, validate=False):
        if validate:
            self.validate(param_info, src_op)
        input_tensors = input_tensor_info
        output_tensors = self.set_output_dims(src_op, output_tensor_info, model)
        super(UdoOnnxOp, self).__init__(src_op.op_type, src_op=src_op, input_tensors=input_tensors,
                                        output_tensors=output_tensors, param_info=param_info)
        self.num_inputs = len(src_op.input)
        self.num_outputs = len(src_op.output)

    @classmethod
    def extract_attrs(cls, src_op, param_infos):
        """
        This method extracts attributes from the provided onnx src_op, using a list of param_infos that have been
        obtained from the operator spec.

        :param src_op: The onnx src_op
        :param param_infos: The list of parameter information which should be a list of UdoTensorInfo objects.
        :return: a dictionary of attributes, where the key is the attribute name and the value is a UdoParam object
        """
        attrs = dict()
        parm = None
        extract_tensor = onnx.numpy_helper.to_array
        for attr in src_op.attribute:

            name = attr.name
            result = [x for x in param_infos if name == x.name]
            param_info = result[0]
            code = attr.type
            data_type = param_info.data_type

            if code == 1:
                parm = UDOParam(name, UdoParamTypes.SCALAR, UDOScalarParam(data_type, attr.f))
            elif code == 2:
                parm = UDOParam(name, UdoParamTypes.SCALAR, UDOScalarParam(data_type, attr.i))
            elif code == 3:
                string = str(attr.s).decode('utf-8')
                parm = UDOParam(name, UdoParamTypes.STRING, UDOString(string))
            elif code == 4:
                parm = UDOParam(name, UdoParamTypes.TENSOR,
                                UDOTensorParam(data_type, extract_tensor(attr.t), param_info))
            elif code == 6:
                parm = UDOParam(name, UdoParamTypes.TENSOR, UDOTensorParam(data_type, list(attr.floats), param_info))
            elif code == 7:
                parm = UDOParam(name, UdoParamTypes.TENSOR, UDOTensorParam(data_type, list(attr.ints), param_info))
            elif code == 8:
                parm = UDOParam(name, UdoParamTypes.TENSOR, UDOTensorParam(data_type, list(attr.strings), param_info))
            elif code == 9:
                parm = UDOParam(name, UdoParamTypes.TENSOR,
                                UDOTensorParam(data_type, list(map(extract_tensor, attr.tensors)),
                                               param_info(name)))

            attrs[name] = parm

        return attrs

    def infer_output_shapes(self, node, model=None, perform_shape_inference=False):
        """
         This method infers the shape of an Onnx NodeProto's output tensors using the node itself, a user provided
         model containing the node and optionally Onnx's in-built shape inference function.

        :param node: The onnx NodeProto object
        :param model: A required field which should be an Onnx ModelProto object
        :param perform_shape_inference: if set to True, the method will call Onnx's shape inference method,
                                        otherwise, the method will assume that the value info contains shape information
                                        for all output tensors.
        :return: a list of lists which contains output dimensions for each output tensor in the Onnx NodeProto.
        """
        output_dims = []
        if not model:
            raise RuntimeError(code_to_message.get_error_message("ERROR_MODEL_NOT_VALID"))
        if perform_shape_inference:
            inferred_model = self.up_convert_infer_shapes(model)
        else:
            inferred_model = model

        # for each output in the node, this loop checks for a corresponding entry in the graph value info or
        # in the list of outputs. Note that Onnx's shape inference will return nothing if shape inference is not
        # available for the queried output so a value info must be verified as non-empty.If the output name is
        # present in either structure then the boolean "found" is set to True, otherwise, an Exception is raised.
        for j in range(len(node.output)):
            found = False

            for value_info in inferred_model.graph.output:
                if node.output[j] == value_info.name:
                    output_dims.append([int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim])
                    found = True
                    break

            if found:
                continue
            else:
                for value_info in inferred_model.graph.value_info:
                    if not value_info:
                        break
                    elif value_info.name == node.output[j]:
                        output_dims.append([int(dim.dim_value) for dim in value_info.type.tensor_type.shape.dim])
                        found = True
                        break

            if not found:
                raise Exception(code_to_message.get_error_message('ERROR_UDO_INFER_OUTPUT_SHAPES')(node.output[j]))

        return output_dims

    def validate(self, param_info, src_op):
        """
        This method calls validation on the input tensors and params provided in the src_op, in comparison with
        the input_tensor_info and param_info defined in the config spec. This method is only intended for use when
        an input tensor info and param info have been obtained from a config spec, prior to conversion of the model.

        :param input_tensor_info: A list of UdoTensorInfo objects which define tensor information for each input to the
                                  Op.
        :param param_info: A list of UdoTensorInfo objects which define parameter information for each param in the op.
        :param src_op: The Onnx src_op in the model, which is expected to be based off the param and input spec.
        :param model: The model containing the op.
        :raises An exception if validation fails on the input tensors or the params. See individual functions for
                the nature of the exception.
        """
        self.validate_params(src_op, param_info)

    @staticmethod
    def validate_params(src_op, param_info):
        """
        Validate params in the src_op with respect to param_infos defined in the config spec. Note that unlike tensors,
        params must be named in the config spec. If the param is not present in the op, a KeyError is raised. Likewise,
        if a param not provided in the config spec is included, the error is also raised.
        :param src_op: The onnx op containing the params
        :param param_info: The list of param information as defined in the config spec.
        :raises: a KeyError if the param is missing or an param is present in the op.
        """
        # TODO: add validation on param data-type which may be tricky.
        for param in param_info:
            if param.name not in (attr.name for attr in src_op.attribute) and not param.static:
                raise KeyError(code_to_message.get_error_message('ERROR_MISSING_UDO_ATTRIBUTE')(param.name,
                                                                                                src_op.op_type))
        for attr in src_op.attribute:
            if attr.name not in (param.name for param in param_info):
                raise KeyError(code_to_message.get_error_message("ERROR_UDO_ATTRIBUTE_NOT_SUPPORTED")
                                                                (attr.name, src_op.op_type,
                                                                [param.name for param in param_info]))

    def up_convert_infer_shapes(self, model):
        try:
            from onnx import shape_inference
        except ImportError:
            raise ImportError("Could not import Onnx shape inference module which is needed to infer output shapes")
        inferred_model = shape_inference.infer_shapes(model)
        if len(inferred_model.graph.value_info) + len(model.graph.input) < \
                len(self.get_all_tensors_in_model(model)):
            if version_converter_available:
                version_converted_model = version_converter.convert_version(model, 6)
                inferred_model = shape_inference.infer_shapes(version_converted_model)
            else:
                raise RuntimeError(
                    "Could not infer shapes for this model, as the opset version is too low. Expected > {}, "
                    "instead got > {}".format("6", model.opset_import))
        return inferred_model

    @staticmethod
    def get_all_tensors_in_model(model):
        tensors = set()
        for node in model.graph.node:
            list(map(tensors.add, [output for output in node.output]))
            list(map(tensors.add, [input for input in node.input]))
        return tensors
