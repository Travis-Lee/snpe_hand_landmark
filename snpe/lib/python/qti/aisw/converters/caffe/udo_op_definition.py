# ==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.udo.udo_core import UdoOp, UDOParam, UDOScalarParam, UDOTensorParam, \
    UDOString, UdoParamTypes
from qti.aisw.converters.common.utils import code_to_message
import numpy as np
import google.protobuf.pyext as proto
import caffe.proto.caffe_pb2 as caffe_pb2
from collections import OrderedDict


class UdoCaffeOp(UdoOp):
    """
    A subclass of the UdoOp interface which implements framework specific methods defined in UdoOp. Calling this class
     requires that a caffe module can be imported. Note that an instance of this class has further
     requirements than a generic UdoOp, in that the output dims must be determinable or provided. Additionally,
     the parameters must be extractable from the op. See UdoOp for all methods that will be called when a UdoCaffeOp is
     instantiated.
    """

    def __init__(self, src_op, input_tensor_info, output_tensor_info, param_info, model,
                 validate=False):
        if validate:
            self.validate(param_info, src_op, input_tensor_info)
        input_tensors = input_tensor_info
        output_tensors = self.set_output_dims(src_op, output_tensor_info, model)
        super(UdoCaffeOp, self).__init__(src_op.type, src_op=src_op, input_tensors=input_tensors,
                                         output_tensors=output_tensors, param_info=param_info,
                                         name=src_op.name)
        self.num_inputs = len(src_op.bottom)
        self.num_outputs = len(src_op.top)

    @classmethod
    def extract_attrs(cls, src_op, param_infos):
        """
        This method extracts attributes from the caffe src_op, using a list of param_infos that have been
        obtained from the operator spec.

        :param src_op: The caffe src_op
        :param param_infos: The list of parameter information which should be a list of UdoTensorInfo objects.
        :return: a dictionary of attributes, where the key is the attribute name and the value is a UdoParam object
        """
        attrs = OrderedDict()
        param_obj = cls.get_param_object(src_op)

        attr_value = None
        for param in param_infos:
            if hasattr(src_op, param.name):
                attr_value = getattr(src_op, param.name)
            elif param_obj and hasattr(param_obj, param.name):
                attr_value = getattr(param_obj, param.name)

                # attr value may be an protobuf scalar container or a filler parameter,
                # in both cases it is turned into a list. Note: the filler parameter
                # will be contained in the blobs object and filled during the udo translation.
                if isinstance(attr_value, proto._message.RepeatedScalarContainer):
                    attr_value = list(attr_value)
                    # if the attr_value is empty and the param is not static, then we don't
                    # bother adding it to the list of attributes as it there is no data to
                    # serialize.
                    if not attr_value and not param.static:
                        continue
                elif isinstance(attr_value, caffe_pb2.FillerParameter):
                    attr_value = []
            elif param.default_value:
                attr_value = param.default_value
            elif not param.static:
                raise RuntimeError('Could not extract parameter: {} from src_op: {} '.format(
                    param.name, src_op.name))

            if isinstance(attr_value, str):
                parm = UDOParam(param.name, UdoParamTypes.STRING, UDOString(attr_value))
            elif isinstance(attr_value, bool):
                parm = UDOParam(param.name, UdoParamTypes.SCALAR, UDOScalarParam(param.data_type,
                                                                             int(attr_value)))
            elif isinstance(attr_value, (list, tuple, np.ndarray)):
                parm = UDOParam(param.name, UdoParamTypes.TENSOR,
                                UDOTensorParam(param.data_type, attr_value, param))
            elif isinstance(attr_value, (int, float)):
                parm = UDOParam(param.name, UdoParamTypes.SCALAR, UDOScalarParam(param.data_type,
                                                                             attr_value))
            else:
                raise RuntimeError('Could not determine parameter type for: {} from src_op: {} '
                                   ''.format(param.name, src_op.name))

            attrs[param.name] = parm

        return attrs

    def infer_output_shapes(self, node, model=None, **kwargs):
        """
         This method infers the shape of a Caffe NodeProto's output tensors using the model and
         node information.

        :param node: The LayerParameter object
        :param model: The NetParameter object
        """
        output_dims = []
        if model:
            for top in node.top:

                if hasattr(model.blobs[top], 'shape'):
                    shape = model.blobs[top].shape
                elif hasattr(model.blobs[top], 'data'):
                    shape = model.blobs[top].data.shape
                else:
                    raise KeyError('Caffe blob:{} is missing shape parameter'.format(str(top)))

                output_dims.append(list([dim for dim in shape]))
        else:
            raise RuntimeError(code_to_message.get_error_message("ERROR_MODEL_NOT_VALID"))

        return output_dims

    def validate(self, param_info, src_op, input_tensor_info):
        """
        This method calls validation on the params provided in the src_op, in comparison with
        the input_tensor_info and param_info defined in the config spec. This method is only intended for use when
        an input tensor info and param info have been obtained from a config spec, prior to conversion of the model.

        :param input_tensor_info: A list of UdoTensorInfo objects which define tensor information for each input to the
                                  Op.
        :param param_info: A list of UdoTensorInfo objects which define parameter information for each param in the op.
        :param src_op: The Caffe src_op in the model, which is expected to be based off the param and input spec.
        :raises An exception if validation fails on the params. See individual functions for
                the nature of the exception.
        """
        self.validate_params(src_op, param_info, input_tensor_info)

    @staticmethod
    def validate_params(src_op, param_info, input_tensor_info):
        """
        Validate params in the src_op with respect to param_infos defined in the config spec. Note
        that unlike tensors, params must be named in the config spec. If the param is not known to
        the Udo op definition, a KeyError is raised. Likewise, if a param not provided in the
        config spec is included, the error is also raised.
        :param src_op: The onnx op containing the params
        :param param_info: The list of param information as defined in the config spec.
        :param input_tensor_info: The list of input tensors
        :raises: a KeyError if the param is missing or an param is present in the op.
        """
        param_obj = UdoCaffeOp.get_param_object(src_op)
        # TODO: add validation on param data-type which may be tricky.
        for param in param_info:
            # This checks that if an attribute present in the Udo Op description is also
            # present in the model. If the attribute is not present but is listed as either static:
            # meaning it will found in the inputs, or has a default value, then no error is raised.
            # Otherwise, a missing attribute error is raised.
            if param.name not in (attr[0].name for attr in param_obj.ListFields()) and not \
                    param.static and param.default_value is None:
                raise KeyError(
                    code_to_message.get_error_message('ERROR_MISSING_UDO_ATTRIBUTE')(param.name,
                                                                                     src_op.type))
        for attr in param_obj.ListFields():
            if attr[0].name not in (param.name for param in param_info):
                # for caffe, it is unclear if filler attributes are listed as attributes or inputs
                # we need to check that if an attribute is unknown, then we look in the list of
                # inputs
                if attr[0].name not in (tensor.name for tensor in input_tensor_info):
                    raise KeyError(
                        code_to_message.get_error_message("ERROR_UDO_ATTRIBUTE_NOT_SUPPORTED")
                        (attr.name, src_op.type,
                         [param.name for param in param_info]))

    @staticmethod
    def get_param_object(src_op):
        param_obj = None
        # identify parameter object, as caffe parameter are usually grouped as op_name_param
        # if we cannot find it that way, then look to see if any of the registered descriptors
        # matches the expected class name: op_typeParameter.
        if hasattr(src_op, (str(src_op.type).lower() + '_param')):
            param_obj = getattr(src_op, str(src_op.type).lower() + '_param')
        else:
            try:
                for potential_param in src_op.ListFields():
                    if hasattr(potential_param[1], 'DESCRIPTOR'):
                        if potential_param[1].DESCRIPTOR.name == str(src_op.type) + 'Parameter':
                            param_obj = potential_param[1]
                            break
                    else:
                        continue
            except:
                raise TypeError('Could not identify attributes from Caffe src_op:{} of '
                                'type: {}'.format(
                    src_op.name, src_op.type))
        return param_obj
