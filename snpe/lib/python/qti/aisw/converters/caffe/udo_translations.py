# ==============================================================================
#
#  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from .caffe_base_translation import *
from ..common.udo import udo_factory
from ..common.udo.udo_core import UdoParamTypes
import numpy as np
from qti.aisw.converters.common.utils import code_to_message, converter_utils
from qti.aisw.converters.common.converter_ir import op_adapter


# ------------------------------------------------------------------------------
#   Udo
# ------------------------------------------------------------------------------
class CaffeUdoTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)
        self.udo_op = None

    def extract_input_names(self, src_op, graph):
        return [str(input.name) for input in self.udo_op.inputs]

    def extract_output_names(self, src_op, graph):
        return [str(output.name) for output in self.udo_op.outputs]

    def extract_parameters(self, src_op, graph):
        udo_op = udo_factory.UDOFactory.udo_collection.get_first_of(src_op.type)
        package_name = udo_factory.udo_package_resolver[udo_op.op_type]
        self.udo_op = udo_op
        static_idx = 0

        for name, udo_param in udo_op.params.items():
            param = udo_param.param_obj
            if udo_param.param_type == UdoParamTypes.TENSOR and \
                    (type(param.data) is not np.ndarray and not param.data):
                # if the parameter is not static, and does not have a known default value
                # then we error out here. If it has a default value, then it has been set in
                # extract_attrs
                if not param.static:
                    if param.default_value is None:
                        raise ValueError(code_to_message.get_error_message("ERROR_UDO_PARAM_NO_DATA")(name, udo_op.op_type))
                    continue
                else:
                    try:
                        udo_param.param_obj.data = np.ascontiguousarray(graph.weights.weights_map[
                                                                  src_op.name][static_idx].data)
                        udo_param.param_obj.dims = list(udo_param.param_obj.data.shape)
                        udo_param.param_obj.rank = len(udo_param.param_obj.data.shape)

                    except IndexError:
                        # Occasionally, not all filler parameters have a corresponding value in
                        # the weight map. If a static filler parameter is not found in the blobs
                        # object, then the user must provide a default value. If no default value
                        # is provided then an error is raised.
                        if param.default_value is not None:
                            udo_param.param_obj.data = np.asarray(param.default_value).astype(
                                np.float32)
                            udo_param.param_obj.dims = list(udo_param.param_obj.data.shape)
                            udo_param.param_obj.rank = len(udo_param.param_obj.data.shape)
                            # continue so that idx is not incremented
                            continue

                        raise IndexError(code_to_message.get_error_message(
                            "ERROR_CANNOT_INGEST_CAFFE_STATIC_UDO_INPUT")
                                          (str(name)))

                    # increment the index of the static parameters seen. Note this relies on
                    # the inputs/params being listed in order. Params must come before inputs.
                    static_idx = +1

        inputs, outputs, params = udo_op.as_dict()
        return op_adapter.UdoOp(name=src_op.name,
                                package_name=package_name,
                                output_dims=udo_op.output_dims,
                                udo_type=udo_op.op_type,
                                axis_orders=udo_op.axis_orders,
                                infer_shape_method=udo_op.get_method('CUSTOM_SHAPE_INFERENCE'),
                                inputs=inputs,
                                outputs=outputs,
                                params=params)


CaffeTranslations.register_translation(CaffeUdoTranslation(),
                                       converter_utils.converter_type('udo', 'caffe'),
                                       op_adapter.UdoOp.TRANSLATION_KEY)
