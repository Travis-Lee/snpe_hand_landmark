# =============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.utils.converter_utils import log_assert
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.translation_utils import IRPaddingStrategies, pads_symmetric, pads_righthanded


def validate_snpe_padding(node):
    supported_strategies = [IRPaddingStrategies.PADDING_SIZE_IMPLICIT_VALID,
                            IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_BEGIN,
                            IRPaddingStrategies.PADDING_SIZE_EXPLICIT,
                            IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR,
                            IRPaddingStrategies.PADDING_SIZE_EXPLICIT_RIGHTHANDED,
                            ]
    padding_size_strategy = node.op.padding_size_strategy
    log_assert(padding_size_strategy in supported_strategies,
               "Unsupported SNPE Padding Strategy {}".format(padding_size_strategy))

    # For explicit strategy SNPE only allows symmetric or right handed
    pads = [node.op.pady_before, node.op.padx_before, node.op.pady_after, node.op.padx_after]
    if padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_EXPLICIT or \
       padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR:
        log_assert(pads_symmetric(pads), code_to_message.get_error_message("ERROR_ASYMMETRIC_PADS_VALUES"))
    elif padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_EXPLICIT_RIGHTHANDED:
        log_assert(pads_righthanded(pads), code_to_message.get_error_message("ERROR_ASYMMETRIC_PADS_VALUES"))