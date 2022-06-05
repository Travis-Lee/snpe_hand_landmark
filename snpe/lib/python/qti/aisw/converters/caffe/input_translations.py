# ==============================================================================
#
#  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


from .caffe_base_translation import CaffeTranslationBase, CaffeTranslations
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.utils.converter_utils import *


class CaffeInputTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)


CaffeTranslations.register_translation(CaffeInputTranslation(),
                                       converter_type('input', 'caffe'),
                                       converter_type('data', 'caffe'),
                                       op_adapter.InputOp.TRANSLATION_KEY)


class CaffeSubtractMeanTranslation(CaffeTranslationBase):
    def __init__(self):
        CaffeTranslationBase.__init__(self)

    def extract_parameters(self, layer, graph):
        transform_param = layer.transform_param
        return op_adapter.SubtractMeanOp(layer.name + "_subtract_mean",
                                         transform_param.mean_value)


CaffeTranslations.register_translation(CaffeSubtractMeanTranslation(),
                                       converter_type('subtract_mean', 'caffe'),
                                       op_adapter.SubtractMeanOp.TRANSLATION_KEY)
