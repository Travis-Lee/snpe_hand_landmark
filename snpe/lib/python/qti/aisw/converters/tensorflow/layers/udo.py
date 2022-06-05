# ==============================================================================
#
#  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.udo.udo_factory import udo_package_resolver
from qti.aisw.converters.common.udo.udo_core import (
    UdoOp,
    UDOParam,
    UDOTensorParam,
    UDOScalarParam,
    UDOString,
    UdoParamTypes)
from qti.aisw.converters.common.utils import code_to_message, udo_module_helpers
from qti.aisw.converters.tensorflow.common import LayerBuilder, LayerDescriptor, LayerResolver
import numpy as np


class UdoTfOp(UdoOp):
    def __init__(self, src_op, input_tensor_info, output_tensor_info, param_info, model, validate=True):
        if validate:
            self.validate_params(src_op, param_info)
        output_tensor_info = self.set_output_dims(src_op, output_tensor_info, model)
        super(UdoTfOp, self).__init__(src_op.type, name=src_op.name, src_op=src_op, input_tensors=input_tensor_info,
                                      output_tensors=output_tensor_info, param_info=param_info)

    def infer_output_shapes(self, node, **kwargs):
        """
        Tensorflow output shapes can be either be present or variable, for which we can
        leverage already existing converter util methods.
        """
        output_dims = []
        for tensor in node.outputs:
            if not any(str(dim.value) == "?" or dim.value is None for dim in tensor.shape.dims):
                output_dims.append(tensor.get_shape().as_list())
            else:
                output_dims.append([])
        return output_dims

    @classmethod
    def extract_attrs(cls, src_op, param_infos):
        # check if attribute can be extracted
        attrs = dict()

        for param_info in param_infos:
            name = param_info.name
            if name in src_op.node_def.attr:
                attr_value = src_op.get_attr(name)
            elif param_info.static:
                attr_value = []
            else:
                raise KeyError(code_to_message.get_error_message('ERROR_MISSING_UDO_ATTRIBUTE')(param_info.name,
                                                                                                src_op.type))
            try:
                iterable = iter(attr_value)
            except TypeError as e:
                if type(attr_value) is bool:
                    attr_value = int(attr_value)
                    iterable = False
                else:
                    raise e

            if not iterable:
                attrs[name] = UDOParam(name, UdoParamTypes.SCALAR, UDOScalarParam(param_info.data_type, attr_value))
            else:
                if isinstance(attr_value, tuple(udo_module_helpers.string_types)):
                    if not isinstance(attr_value, str):
                        # assuming unicode or bytes and utf-8 encoding
                        attr_value = attr_value.decode('utf-8')
                    attrs[name] = UDOParam(name, UdoParamTypes.STRING, UDOString(attr_value))
                else:
                    attrs[name] = UDOParam(name, UdoParamTypes.TENSOR,
                                           UDOTensorParam(param_info.data_type, attr_value, param_info))
        return attrs

    def validate(self, param_info, src_op):
        """
        This method calls validation on the params provided in the src_op, in comparison with
        param_info defined in the config spec. This method is only intended for use when
        a param info have been obtained from a config spec, prior to conversion of the model.

        :param param_info: A list of UdoTensorInfo objects which define parameter information for each param in the op.
        :param src_op: The Onnx src_op in the model, which is expected to be based off the param and input spec.
        :raises An exception if validation fails on the params. See individual function for
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
        # TO-DO: add validation on param data-type which may be tricky.
        for param in param_info:
            if param.name not in src_op.node_def.attr and not param.static:
                raise KeyError(code_to_message.get_error_message('ERROR_MISSING_UDO_ATTRIBUTE')(param.name,
                                                                                                src_op.type))

        # some attributes are included simply to denote the type of an input
        type_attrs = [attr.name for attr in src_op.op_def.attr if attr.type == 'type']

        # some attributes only indicate the amount of expected inputs or outputs
        number_attr_string = "N"

        for attr in src_op.node_def.attr:
            if attr not in (param.name for param in param_info):
                if attr in type_attrs or str(attr) == number_attr_string:
                    continue
                raise KeyError(code_to_message.get_error_message("ERROR_UDO_ATTRIBUTE_NOT_SUPPORTED")
                               (attr, src_op.type,
                                [param.name for param in param_info]))


class UdoLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, udo_op_instance):
            super(UdoLayerResolver.Descriptor, self).__init__('udo', name, nodes)
            self.udo_op = udo_op_instance
            self.src_op = self.child_ops[-1]

    def __init__(self):
        self.sequence = None

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for graph_node in graph_matcher.graph:
            for node_type in graph_node.node_types:
                if node_type in graph_helper.op_collection:
                    original_tf_op = graph_node.original_node
                    target_udo_tf_op = graph_helper.op_collection.get_op_by_name(node_type, original_tf_op.name)
                    const_input_ops = set()
                    input_ops = []
                    consumable_types = ['Const', 'Identity', 'Variable', 'Fill']

                    consumable_inputs = [input for input in original_tf_op.inputs if input.op.type in consumable_types]
                    # resolve missing dims that require further evaluation
                    for i, dim in enumerate(target_udo_tf_op.output_dims):
                        if not dim:
                            tensor = original_tf_op.outputs[i]
                            output = graph_helper.evaluate_tensor_output(tensor)
                            assert all(shape for shape in output.shape)
                            target_udo_tf_op.output_dims[i] = list(output.shape)

                    for input in consumable_inputs:
                        candidate_input_ops = []
                        # The idea here is to first determine if the input to an op has a constant origin,
                        # which implies it may be part of an input sequence that should either be consumed
                        # as a series of child ops or resolved by the converter individually
                        # (as either ignored or a legitimate op). Note: Check_tensor_const_origin may take
                        # a while as it may propagate to the top-level node in the worst case.
                        if graph_helper.check_tensor_const_origin(input):
                            candidate_input_ops = graph_helper.get_consumable_input_sequence(input)

                        for op in candidate_input_ops:
                            if op.type not in consumable_types:
                                # Here we have determined that there is a input op which is part of an input sequence
                                # however the op type cannot be trivially consumed as a child op. Print a warning
                                # here to let the user know that the op may cause an error if it cannot be resolved.
                                print("WARNING: Cannot resolve non-const sequence of input ops as "
                                      "child operations of Udo Op. Converter will fail if Op: {} cannot be resolved"
                                      " independently or as part of another sequence.".
                                      format(op.type))
                            else:
                                const_input_ops.add(op)
                    input_ops.extend(const_input_ops)
                    input_ops.append(original_tf_op)  # always append the original op last
                    udo_descriptor = UdoLayerResolver.Descriptor(original_tf_op.name,
                                                                 input_ops,
                                                                 target_udo_tf_op)
                    descriptors.append(udo_descriptor)
        return descriptors

    def is_final_resolution(self):
        return True


class UdoLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        udo_op = descriptor.udo_op
        local_graph_helper = converter_context.graph_helper
        package_name = udo_package_resolver[udo_op.op_type]

        # Because descriptors may have been merged into the child ops, the original udo src op
        # is retrieved as a class member. All other ops are considered to be unconsumed input ops.
        src_op = descriptor.src_op
        unconsumed_input_ops = [op for op in descriptor.child_ops if op != src_op]

        for name, udo_param in udo_op.params.items():
            param = udo_param.param_obj
            if type(param) != UDOString and (type(param.data) is not np.ndarray and not param.data):
                if not param.static:
                    raise ValueError(code_to_message.get_error_message("ERROR_UDO_PARAM_NO_DATA")(name, udo_op.op_type))
                else:
                    tensor = local_graph_helper.get_tensor_by_name(udo_param.name)
                    tensor_idx = [i for i, input in enumerate(src_op.inputs) if input.name == udo_param.name]
                    consumed_op = [op for op in unconsumed_input_ops for output in op.outputs if
                                   output.name == tensor.name]
                    if consumed_op and tensor_idx:
                        try:
                            attr_name = getattr(src_op.op_def.input_arg[tensor_idx[0]], 'name')
                        except IndexError:
                            # try to split the string on the assumption that the tensor name will always
                            # include the param name before a ':'. This is purely an assumption based
                            # on tensorflow models seen so far.
                            attr_name = (str(tensor.name).split('/')[-1]).split(':')[0]

                            # make sure that name we have split exists in the input arg otherwise we give up
                            if not any(attr_name == input_arg.name for input_arg in src_op.op_def.input_arg):
                                raise LookupError(
                                    code_to_message.get_error_message("ERROR_CANNOT_INGEST_STATIC_UDO_INPUT")
                                    (str(name)))

                        output_tensor = local_graph_helper.evaluate_tensor_output(tensor)
                        udo_param.param_obj.data = output_tensor
                        if type(output_tensor) is np.ndarray:
                            udo_param.param_obj.dims = list(output_tensor.shape)
                            udo_param.param_obj.rank = len(output_tensor)
                            udo_param.name = attr_name
                        else:
                            if isinstance(output_tensor, tuple(udo_module_helpers.string_types)):
                                if not isinstance(output_tensor, str):
                                    # assuming unicode or bytes and utf-8 encoding
                                    output_tensor = output_tensor.decode('utf-8')
                                udo_op.params[name] = UDOParam(attr_name, UdoParamTypes.STRING, UDOString(output_tensor))
                            else:
                                udo_op.params[name] = UDOParam(attr_name, UdoParamTypes.SCALAR,
                                                               UDOScalarParam(param.data_type,
                                                                              output_tensor))

                    else:
                        raise LookupError(code_to_message.get_error_message("ERROR_CANNOT_INGEST_STATIC_UDO_INPUT")
                                          (name))

        inputs, outputs, params = udo_op.as_dict()
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_names = descriptor.output_names

        # since merging may have occurred, output names may have been updated.
        # we need to add the axis order for the new output to the udo_op axis orders dict
        new_output_names = [output_name for output_name in output_names if output_name not in udo_op.axis_orders.keys()]
        if new_output_names:
            for output_name in new_output_names:
                udo_op.axis_orders[output_name] = 'NOT_YET_DEFINED'
        ir_graph.add(op_adapter.UdoOp(name=descriptor.layer_name,
                                      package_name=package_name,
                                      output_dims=udo_op.output_dims,
                                      udo_type=src_op.type,
                                      axis_orders=udo_op.axis_orders,
                                      infer_shape_method=udo_op.get_method('CUSTOM_SHAPE_INFERENCE'),
                                      inputs=inputs,
                                      outputs=outputs,
                                      params=params), input_names, output_names)
