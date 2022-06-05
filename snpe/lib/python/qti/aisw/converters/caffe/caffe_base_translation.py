# ==============================================================================
#
#  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.converter_ir import translation

# ------------------------------------------------------------------------------
#   CaffeTranslation
# ------------------------------------------------------------------------------
CaffeTranslations = translation.TranslationBank()


class CaffeTranslationBase(translation.ConversionTranslationBase):
    # method keys
    ADD_INPUT_PREPROCESSING_OP = "ADD_INPUT_PREPROCESSING_OP"
    def __init__(self):
        translation.ConversionTranslationBase.__init__(self)
        self.register_method(self.ADD_INPUT_PREPROCESSING_OP, self.add_input_preprocessing_op)

    def extract_parameters(self, src_op, graph):
        raise NotImplementedError("extract_parameters for {} not implemented ".format(str(self.__class__.__name__)))

    def extract_input_names(self, src_op, graph):
        return list(map(str, src_op.bottom))

    def extract_output_names(self, src_op, graph):
        return list(map(str, src_op.top))

    def add_input_preprocessing_op(self, src_op, graph):
        op = self.extract_parameters(src_op, graph)
        src_op.bottom.append(str(src_op.name))
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)
        graph.add(op, input_names, output_names)

