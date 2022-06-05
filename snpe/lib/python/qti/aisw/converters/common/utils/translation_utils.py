# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from enum import Enum
import numpy as np
from math import ceil, floor
from .converter_utils import *
from . import code_to_message


# --------------------------------
# Common IR Enum for Padding Utils
# --------------------------------
class IRPaddingStrategies(Enum):
    """ Padding size strategies support in IR."""

    # No padding
    PADDING_SIZE_IMPLICIT_VALID = 0,
    # Pad input so that output spatial size matches input. In case of odd total
    # pad value across a spatial dimension, the extra padding is place at the end.
    PADDING_SIZE_IMPLICIT_SAME_BEGIN = 1,
    # Pad input so that output spatial size matches input. In case of odd total
    # pad value across a spatial dimension, the extra padding is place at the beginning.
    PADDING_SIZE_IMPLICIT_SAME_END = 2,
    # padding values are applied only to the right-hand side of the input and floor operation
    # is used to calculate output dims.
    PADDING_SIZE_EXPLICIT_RIGHTHANDED = 3,
    # padding values are explicitly specified by source framework and ceil operation is used
    # to calculate output dims
    PADDING_SIZE_EXPLICIT = 4,
    # same as explicit, but floor operation is used to calculate output dims
    PADDING_SIZE_EXPLICIT_FLOOR = 5,


def pads_symmetric(pads):
    num_dims = len(pads)//2
    for i in range(num_dims):
        if pads[i] != pads[i+num_dims]:
            return False
    return True


def pads_righthanded(pads):
    num_dims = len(pads)//2
    for i in range(num_dims):
        if pads[i] != 0:
            return False
    # don't call all zeros right-handed
    return not all(x == 0 for x in pads)


# ---------------------------------
# Utils for calculating output dims
# ---------------------------------
def calc_conv_output_dim(input_size, filter_size, pad_before, pad_after, stride, dilation, padding_size_strategy):
    kernel_extent = filter_size + ((filter_size - 1) * (dilation - 1))
    full_size = float(pad_before + pad_after) + input_size - kernel_extent

    if padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_VALID:
        output_dim = ceil(float(input_size - int(kernel_extent) + 1) / float(stride))
    elif padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_BEGIN \
            or padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_END:
        output_dim = ceil(float(input_size) / float(stride))
    elif padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR:
        output_dim = 1 + floor(full_size/float(stride))
    else:  # EXPLICIT or UNDEFINED
        output_dim = 1 + ceil(full_size / float(stride))

    return int(output_dim)


def calc_deconv_output_dim(input_size, filter_size, pad_before, pad_after, stride, padding_size_strategy):
    if padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_VALID:
        output_dim = input_size * stride + max(filter_size - stride, 0)
    elif padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_BEGIN \
            or padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_END:
        output_dim = input_size * stride
    else:  # EXPLICIT, EXPLICIT_FLOOR or UNDEFINED
        output_dim = stride * (input_size - 1) - (pad_before + pad_after) + filter_size

    return int(output_dim)


def calc_pool_output_dim(input_size, pool_size, pad_before, pad_after, stride, padding_size_strategy):
    padding = -(pad_before + pad_after)
    full_size = float(input_size - padding - pool_size)

    if padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_VALID:
        output_dim = ceil((1 + full_size) / stride)
    elif padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_BEGIN \
            or padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_END:
        output_dim = ceil((float(input_size) / stride))
    elif padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR:
        output_dim = 1 + floor(full_size/stride)
    elif padding_size_strategy == IRPaddingStrategies.PADDING_SIZE_EXPLICIT_RIGHTHANDED:
        full_size = float(input_size - padding - pool_size)
        output_dim = 1 + floor(full_size / stride)
    else:  # EXPLICIT or UNDEFINED
        output_dim = 1 + ceil(full_size / stride)

    if (output_dim - 1) * stride + padding >= input_size:
        # don't start a pool beyond the border of the image
        log_debug(code_to_message.get_debugging_message("DEBUG_OUTPUT_DIM_BEYOND_BORDER")(output_dim, output_dim - 1))

        output_dim -= 1

    return int(output_dim)


# This is used for cases where one of the input shapes is of 1 dimension
# and we don't know how it aligns with the other inputs
# e.g. (16) & (1,16,200,200)
def get_broadcasted_shape(input_shapes):
    def rank(in_shape):
        return len(in_shape)
    output_shape = input_shapes[0]
    output_rank = rank(input_shapes[0])
    for shape in input_shapes[1:]:
        if rank(shape) > output_rank:
            output_shape = shape
            output_rank = rank(shape)
    return output_shape


# This is used for general broadcasting where either shape
# can contribute to the final shape
# e.g. (1,3,1) & (1,1,4)
def get_max_broadcasted_shape(input_shapes):
    temp_tensors = [np.ones(shape) for shape in input_shapes]
    broadcast_sum = temp_tensors[0]
    for tensor in temp_tensors[1:]:
        try:
            broadcast_sum = broadcast_sum + tensor
        except ValueError as e:
            log_error(
                "ValueError: Tensors cannot be broadcast together with shapes {}".format(input_shapes))
            raise e
    return list(np.shape(broadcast_sum))


# Remove the following function once Caffe/Onnx use op.infer_shape
def get_conv_output_shape(ir_op, input_shapes):
    input_height = input_shapes[0][2]
    input_width = input_shapes[0][3]

    output_height = calc_conv_output_dim(input_height,
                                         ir_op.weights.shape[2],
                                         ir_op.pady,
                                         ir_op.stridey,
                                         ir_op.dilationy,
                                         ir_op.padding_size_strategy)
    output_width = calc_conv_output_dim(input_width,
                                        ir_op.weights.shape[3],
                                        ir_op.padx,
                                        ir_op.stridex,
                                        ir_op.dilationx,
                                        ir_op.padding_size_strategy)
    output_depth = ir_op.bias.shape[0]
    batch = input_shapes[0][0]
    output_shape = [batch, output_depth, output_height, output_width]
    log_debug(code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(ir_op.name, output_shape))
    return [output_shape]


# Remove the following function once Caffe/Onnx use op.infer_shape
def get_deconv_output_shape(ir_op, input_shapes):
    input_shape = input_shapes[0]
    if ir_op.output_height == 0:
        # calculate according to provided formula
        input_height = input_shape[2]
        input_width = input_shape[3]

        output_height = calc_deconv_output_dim(input_height,
                                               ir_op.weights.shape[2],
                                               ir_op.stride,
                                               ir_op.pady)
        ir_op['output_height'] = output_height

        output_width = calc_deconv_output_dim(input_width,
                                              ir_op.weights.shape[3],
                                              ir_op.stride,
                                              ir_op.padx)
        ir_op['output_width'] = output_width
    else:
        output_height = ir_op.output_height
        output_width = ir_op.output_width

    output_depth = ir_op.bias.shape[0]
    batch = input_shapes[0][0]
    output_shape = [batch, output_depth, output_height, output_width]
    log_debug(code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(ir_op.name, output_shape))
    return [output_shape]


# Remove the following function once Caffe/Onnx use op.infer_shape
def get_pool_output_shape(ir_op, input_shapes):
    input_shape = input_shapes[0]
    input_height = input_shape[2]
    input_width = input_shape[3]
    output_height = calc_pool_output_dim(input_height,
                                         ir_op.size_y,
                                         ir_op.pad_y,
                                         ir_op.stride_y,
                                         ir_op.padding_size_strategy)
    output_width = calc_pool_output_dim(input_width,
                                        ir_op.size_x,
                                        ir_op.pad_x,
                                        ir_op.stride_x,
                                        ir_op.padding_size_strategy)

    output_shape = input_shape[0:2] + [output_height, output_width]
    log_debug(code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(ir_op.name, output_shape))
    return [output_shape]


# ------------------------------
# Util used for common squashing
# ------------------------------
def squash_nodes_into_previous(graph, matched_node_list, msg):
    """
     Squashes a nodes weights and biases (which are determined only if the previous op has weights and biases)
     arithmetically by adding the two biases or multiplying the weights. Intended use is for elementwise ops
     that follow batchnorm, FC or convolution.
    :param graph: The IROpGraph object
    :param matched_node_list: the list of nodes that are elementwise ops, have a constant input, and are preceded by a
                              batchnorm ,FC or convolution.
    :param msg: The debug message to be printed.

    """
    # TODO: Remove function once ElementWise and ElementWiseBinary merges and move scale to eltwise_squash function
    for node_tuple in matched_node_list:

        # collect previous and current op information
        node = node_tuple[0]
        input_buffer = graph.get_input_buffers(node)[0]
        prev = input_buffer.producer

        # we need separate conditionals since the arithmetic is different for addition/subtraction
        # vs multiplication/division
        if node.op.type == 'elementwise_sum' or node.op.type == 'elementwise_sub':
            scale_bias = node.op.bias
            prev.op.bias += scale_bias
        elif node.op.type == 'elementwise_div' or node.op.type == 'elementwise_product':
            scale_weights = node.op.weights
            prev.op.weights *= scale_weights
            prev.op.bias = (prev.op.bias * scale_weights)
        elif node.op.type == "scale":
            scale_weights = node.op.weights
            scale_bias = node.op.bias
            prev.op.weights *= scale_weights
            prev.op.bias = (prev.op.bias * scale_weights) + scale_bias
        else:
            continue

        graph.squash(node, input_buffer.name)
        log_debug2(code_to_message.get_debugging_message(msg)(node.op.name,
                                                              prev.op.name,
                                                              prev.op.type))


def validate_eltwise_pattern(graph, nodes_tuple, mode):
    """
    Common function to validate if pattern is squashable
    :param graph: the IROpGraph
    :param nodes_tuple: the matched list of nodes
    :param mode: either bias or weight. Use to determine if squashing is
                 eltwise[add|sub] or eltwise[prod|div] respectively.
    :return:
    """

    node = nodes_tuple[0]
    nn_op = None
    const_op = None
    for name_ in node.input_names:
        op_ = graph.get_buffer(name_).producer.op
        # verify that one of the inputs is constant and the other input is produced by nn_type op(BN, FC, Conv/Deconv)
        if op_.type == "constant":
            const_op = op_
        elif (mode == "weights" and hasattr(op_, "weights") and hasattr(op_, "bias")) or \
                (mode == "bias" and hasattr(op_, "bias")):
            if len(graph.get_buffer(name_).consumers) != 1:
                # Unable to squash into nn_op which has more than one consumer
                return False
            nn_op = op_

    # For mode:weights, Only valid to squash if the nn_op weights and bias are broadcastable with const_op
    # For mode:bias, Only valid if the nn_op has a bias with the same shape as const_op
    if (nn_op is not None and const_op is not None) and \
            ((mode == "weights" and broadcastable(nn_op.weights.shape, const_op.tensor.shape) and
              broadcastable(nn_op.bias.shape, const_op.tensor.shape)) or
                (mode == "bias" and broadcastable(nn_op.bias.shape, const_op.tensor.shape))):
        constshape = np.prod(const_op.tensor.shape)
        nnshape = np.prod(nn_op.bias.shape)
        if constshape <= nnshape:
            return True
    return False


def squash_eltwisebinary_to_nn_node(graph, matched_node_list):
    """
    Squashes eltwise ops with inputs of a Constant Node and NN node(NN:batchnorm, FC or conv/deconv)
    :param graph: The IROpGraph object
    :param matched_node_list: the list of nodes that are elementwise ops, have a constant input, and are preceded by a
                              batchnorm ,FC or convolution.
    """

    for node_tuple in matched_node_list:
        # collect previous and current op information
        node = node_tuple[0]
        nn_buf = None
        nn_op = None
        const_buf = None
        const_op = None
        for name in node.input_names:
            input_buf = graph.get_buffer(name)
            input_op = graph.get_producer_op(name)
            if hasattr(input_op, "weights") and \
                    hasattr(input_op, "bias"):
                nn_buf = input_buf
                nn_op = input_op
            elif input_op.type == "constant":
                const_buf = input_buf
                const_op = input_op

        # we need separate conditionals since the arithmetic is different for addition/subtraction
        # vs multiplication/division
        if node.op.type in ['elementwise_binary_sum','elementwise_sum','elementwise_binary_sub', 'elementwise_sub']:
            scale_bias = const_op.tensor
            nn_op.bias += scale_bias
        elif node.op.type in ['elementwise_binary_div' ,'elementwise_binary_product']:
            scale_weights = const_op.tensor
            nn_op.weights *= scale_weights
            nn_op.bias = (nn_op.bias * scale_weights)
        else:
            raise ValueError(code_to_message.get_debugging_message("ERROR_ELEMENTWISEBINARY_SQUASH")(node.op.type))

        graph.squash(node, nn_buf.name)
        log_debug2(code_to_message.get_debugging_message("DEBUG_ELEMENTWISEBINARY_SQUASH")(node.op.name,
                                                                                           node.op.type,
                                                                                           nn_op.name,
                                                                                           nn_op.type))


# -----------------------------
# Util for common chain replace
# -----------------------------
def chain_matched_eltwise_ops(graph, matched_node_list, op_class):
    """
    Replaces matched elementwise op nodes that have > 2 inputs with a chain of binary elementwise nodes
    :param graph: The IROpGraph object
    :param matched_node_list: The list of nodes that are product or sum elementwise ops with > 2 inputs
    :param op_class: A reference to the op_adapter class for the elementwise operation we're replacing
    """
    for nodes_tuple in matched_node_list:
        old_node = nodes_tuple[0]
        old_node_name = old_node.op.name
        old_node_inputs = old_node.input_names
        current_idx = graph.list_nodes().index(old_node)
        old_node_output_buffers = graph.get_output_buffers(old_node)
        old_node_coeffs = old_node.op.coeffs if hasattr(old_node.op, "coeffs") else None

        log_debug(code_to_message.get_debugging_message("DEBUG_ELEMENTWISEBINARY_CHAIN")
                  (old_node_name, old_node.op.type, old_node_inputs))

        # Remove the existing node with > 2 inputs
        graph.prune(old_node, force_remove=True)

        # Construct the names of nodes and input/output buffers in chain
        new_nodes_names = ["%s_chain_0" % old_node_name]
        new_nodes_input_buf_names = [[old_node_inputs[0], old_node_inputs[1]]]
        new_nodes_coeffs = [[old_node_coeffs[0], old_node_coeffs[1]]] if old_node_coeffs else [[]]
        for i in range(1, len(old_node_inputs) - 1):
            new_nodes_input_buf_names.append([new_nodes_names[i-1], old_node_inputs[i+1]])
            new_nodes_coeffs.append([1.0, old_node_coeffs[i+1]]) if old_node_coeffs else new_nodes_coeffs.append([])
            new_nodes_names.append("%s_chain_%d" % (old_node_name, i))

        # Reusing new_nodes_names as new_nodes_output_buf_names for readability
        new_nodes_output_buf_names = new_nodes_names
        # Add constructed nodes into op_graph
        for i in range(len(new_nodes_names)):
            new_op = op_class(new_nodes_names[i], coeffs=new_nodes_coeffs[i])
            graph.add(new_op, new_nodes_input_buf_names[i], new_nodes_output_buf_names[i], idx=current_idx)
            current_idx += 1

        # Set input buffers of original sequence's consumers to the output buffer of last producer in new chain
        for i in range(len(old_node_output_buffers)):
            consumers = old_node_output_buffers[i].consumers
            for consumer in consumers:
                consumer.input_names.append(new_nodes_names[-1])


# -------------------------------------------------------
# Util used for mapping framework activation to snpe enum
# -------------------------------------------------------
def extract_activation(activation):
    acts = {'RELU': "NEURON_RELU",
            'TANH': "NEURON_TANH",
            'SIGMOID': "NEURON_LOGISTIC",
            'ELU': "NEURON_ELU"}
    try:
        return acts[str(activation).upper()]
    except KeyError:
        raise ValueError(code_to_message.get_error_message("ERROR_ACTIVATION_FUNCTION_UNSUPPORTED")(activation))


# -------------------------------------------------------
# General
# -------------------------------------------------------
def expand_to_rank(shape, rank):
    """
    :type shape: list[int]
    :type rank: int
    :rtype: list[int]
    """
    result = shape[:]
    while len(result) < rank:
        result.insert(0, 1)
    return result


def to_list(val):
    if not val:
        return []
    if type(val) != list:
        return [val]
    return val


def broadcastable(shape1, shape2):
    """
    Checks if two shapes are can be broadcast into one another in the numpy sense.
    :param shape1: Shape of the data1
    :param shape2: Shape of the data2
    :return: boolean if broadcast is possible otherwise false
    """

    # loop backwards on both shapes and validate each index for broadcasting.
    # Eg: for [4,11,1,9] with [8,9], we only need to validate 8 and 9.
    for shape_idx1, shape_idx2 in zip(shape1[::-1], shape2[::-1]):
        if shape_idx1 != 1 and shape_idx2 != 1 and shape_idx1 != shape_idx2:
            return False
    return True


def compare_values(val1, val2, rtol=1.e-5, atol=1.e-8):
    """
    :param val1: type: (str, float, int, ndarray)
    :param val2: type: (str, float, int, ndarray)
    :param rtol: type: float The relative tolerance parameter to use if vals are numeric.
    :param atol: type: float The absolute tolerance parameter to use if vals are numeric.
    :return:
    """
    if type(val1) != type(val2):
        return False
    if type(val1 != val2) is np.ndarray:
        # Check if any value in arrays are different. Need shape check first since numpy.allclose
        # broadcasts if shapes are not equal
        return val1.shape == val2.shape and np.allclose(val1, val2, rtol=rtol, atol=atol)
    else:
        if type(val1) == float and type(val2) == float:
            # do tolerance comparison for floats
            # TODO: use python built-in isclose function once we move to python 3.5
            return abs(val2 - val1) < max(rtol * max(abs(val1), abs(val2)), atol)
        return val1 == val2


