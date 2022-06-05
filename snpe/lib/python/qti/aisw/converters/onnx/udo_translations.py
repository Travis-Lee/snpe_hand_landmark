# ==============================================================================
#
#  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from .onnx_translations import *
from ..common.udo import udo_factory
from ..common.udo.udo_factory import UdoParamTypes
import numpy as np


# ------------------------------------------------------------------------------
#   Udo
# ------------------------------------------------------------------------------
class OnnxUdoTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.udo_op = None

    def extract_input_names(self, src_op, graph):
        return [str(input.name) for input in self.udo_op.inputs]

    def extract_output_names(self, src_op, graph):
        return [str(output.name) for output in self.udo_op.outputs]

    def extract_parameters(self, src_op, graph):
        udo_op = udo_factory.UDOFactory.udo_collection.get_first_of(src_op.op_type)
        package_name = udo_factory.udo_package_resolver[udo_op.op_type]
        self.udo_op = udo_op

        for name, udo_param in udo_op.params.items():
            param = udo_param.param_obj
            if udo_param.param_type == UdoParamTypes.TENSOR and \
                    (type(param.data) is not np.ndarray and not param.data):
                if not param.static:
                    raise ValueError(code_to_message.get_error_message("ERROR_UDO_PARAM_NO_DATA")(name, udo_op.op_type))
                elif graph.weights.has(name):
                    udo_param.param_obj.data = np.asarray(graph.weights.weight_map[str(name)].weights)
                    udo_param.param_obj.dims = udo_param.param_obj.data.shape
                    udo_param.param_obj.rank = len(udo_param.param_obj.data.shape)
                    graph.weights.weight_map[str(name)].consumed = True
                else:
                    raise LookupError(code_to_message.get_error_message("ERROR_CANNOT_INGEST_STATIC_UDO_INPUT")
                                      (str(name)))

        inputs, outputs, params = udo_op.as_dict()
        return op_adapter.UdoOp(name=src_op.name,
                                package_name=package_name,
                                output_dims=udo_op.output_dims,
                                udo_type=src_op.op_type,
                                axis_orders=udo_op.axis_orders,
                                infer_shape_method=udo_op.get_method('CUSTOM_SHAPE_INFERENCE'),
                                inputs=inputs,
                                outputs=outputs,
                                params=params)


OnnxTranslations.register_translation(OnnxUdoTranslation(),
                                      converter_type('udo', 'onnx'),
                                      op_adapter.UdoOp.TRANSLATION_KEY)
