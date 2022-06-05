# =============================================================================
#
#  Copyright (c) 2015-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import numpy as np

from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.translation_utils import IRPaddingStrategies
from qti.aisw.converters.common.converter_ir.op_adapter import ConvolutionOp, ResizeOp, PadOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    ConverterRepeatableSequenceTreeNode,
    NonConsumableConverterSequenceNode
)
from qti.aisw.converters.tensorflow.layers.ignored_patterns import IgnoredLayersResolver
from qti.aisw.converters.tensorflow.layers.batchnorm import BatchNormLayerResolver
from qti.aisw.converters.tensorflow.layers.crop import CropLayerResolver
from qti.aisw.converters.tensorflow.layers.pad import PadLayerResolver, PadLayerBuilder
from qti.aisw.converters.tensorflow.layers.resize import ResizeBilinearLayerResolver, ResizeLayerBuilder
from qti.aisw.converters.tensorflow.util import ConverterError
from qti.aisw.converters.tensorflow.util import GraphHelper
from qti.aisw.converters.tensorflow.util import OperationNotFoundError
from qti.aisw.converters.tensorflow.util import TensorNotFoundError


class ConvolutionLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_STRIDES = 'strides'
    TF_ATTRIBUTE_PADDING = 'padding'
    TF_ATTRIBUTE_EXPLICIT_PADDING = 'explicit_paddings'

    def get_spatial_padding(self, conv_op):
        try:
            paddings = conv_op.get_attr(self.TF_ATTRIBUTE_EXPLICIT_PADDING)
        except ValueError:
            return [[0, 0], [0, 0]]

        spatial_padding = []
        for i in range(1, (len(paddings) - 1)):  # for NHWC, get HW
            for j in range(len(paddings[0])):
                spatial_padding.append([paddings[i][j]])

        return spatial_padding

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, conv_op, bias_op, output_op,
                     strides, padding_size_strategy,
                     weights, biases, output_names=None, explicit_pads=list([[0, 0], [0, 0]]),
                     resize_desc=None, pad_desc=None):
            super(ConvolutionLayerResolver.Descriptor, self).__init__('Convolution', name, nodes,
                                                                      output_names=output_names)
            self.conv_op = conv_op
            self.bias_op = bias_op
            self.strides = strides
            self.padding_size_strategy = padding_size_strategy
            self.explicit_pads = explicit_pads  # Per tf docs: [[top, bottom], [left, right]]
            self.weights = weights
            self.biases = biases
            self.dilationX = 1
            self.dilationY = 1
            self.groups = len([op for op in nodes if op.type == 'Conv2D' or op.type == 'FusedResizeAndPadConv2D'])
            self.output_op = output_op
            self.input_ops = [conv_op]
            # FusedResizeAndPadConv2D Op descriptors that will be used later in building IR graph
            # since no support for the fused Op
            self.resize_desc = resize_desc
            self.pad_desc = pad_desc

        def is_input_op(self, op):
            return op in self.input_ops

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Conv2D', 'FusedResizeAndPadConv2D'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            conv_op = match['root']
            output_op = conv_op
            strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding_size_strategy = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            pads = self.get_spatial_padding(conv_op)
            weights = self.get_weights(graph_helper, conv_op)
            consumed_nodes = list(match.consumed_nodes)
            output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in self.sequence.output_nodes]
            conv_output_ops = graph_helper.get_op_outputs(conv_op)
            bias_op, biases = self.get_conv_bias(graph_helper, conv_op, conv_output_ops)
            if bias_op is not None and biases is not None:
                output_op_nodes_names = [str(bias_op.outputs[0].name)]
                consumed_nodes.append(bias_op)
            else:
                bias_op = None
                biases = np.zeros(weights.shape[-1], dtype=np.float32)

            # Add resize and pad if conv is fused
            resize_desc, pad_desc = self.get_resize_pad_desc(graph_helper, conv_op)
            descriptor = ConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes,
                                                             conv_op, bias_op, output_op,
                                                             strides, padding_size_strategy, weights, biases,
                                                             explicit_pads=pads,
                                                             output_names=output_op_nodes_names,
                                                             resize_desc=resize_desc,
                                                             pad_desc=pad_desc)
            descriptors.append(descriptor)
        return descriptors

    def get_biases(self, graph_helper, conv_op, bias_op):
        _, biases_tensor = GraphHelper.get_op_input_tensors(bias_op, ('?', '?'))
        if biases_tensor.op.type not in ['Identity', 'Const'] and \
                not graph_helper.check_tensor_const_origin(biases_tensor):
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONV_RESOLVE_BIAS')(conv_op.name))
        biases = graph_helper.evaluate_tensor_output(biases_tensor)
        return biases

    def get_conv_bias(self, graph_helper, input_op, conv_output_ops):
        bias_op = None
        biases = None
        try:
            bias_op = GraphHelper.filter_single_op_by_type(conv_output_ops, 'BiasAdd')
            biases = self.get_biases(graph_helper, input_op, bias_op)

        except OperationNotFoundError:
            pass

        if bias_op is None:
            try:
                bias_op = GraphHelper.filter_single_op_by_type(conv_output_ops, 'Add')
                biases = self.get_biases(graph_helper, input_op, bias_op)
            except (OperationNotFoundError, ConverterError):
                bias_op = None
                biases = None

        if biases is not None and len(biases.shape) != 1:
            bias_op = None
            biases = None

        return bias_op, biases

    def get_weights(self, graph_helper, conv_op):
        try:
            _, weights_tensor = GraphHelper.get_op_input_tensors(conv_op, ('?', '?'))
        except TensorNotFoundError:
            _, _, _, weights_tensor = GraphHelper.get_op_input_tensors(conv_op, ('?', '?', '?', '?'))
        if weights_tensor.op.type not in ['Identity', 'Const', 'Split', 'FakeQuantWithMinMaxVars'] and \
                not graph_helper.check_tensor_const_origin(weights_tensor):
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONV_RESOLVE_WEIGHTS')(conv_op.name))
        weights = graph_helper.evaluate_tensor_output(weights_tensor)
        return weights

    def get_resize_pad_desc(self, graph_helper, conv_op):
        resize_desc = None
        pad_desc = None
        try:
            conv_input, resize_size, pad, _ = GraphHelper.get_op_input_tensors(conv_op, ('?', '?', '?', '?'))
            # input.outputs[0].name = resize_size.name
            input_tensor_shape = graph_helper.get_op_output_shape(conv_input)
            mul_const = graph_helper.evaluate_tensor_output(resize_size)
            if type(mul_const) is np.ndarray:
                mul_const = mul_const.squeeze().shape  # get the actual scale values for height and width
            if len(mul_const) < 2:
                mul_const = [0, 0]
            resize_desc = ResizeBilinearLayerResolver.Descriptor(str(resize_size.name),
                                                                 [resize_size.op],
                                                                 input_tensor_shape,
                                                                 resize_size,
                                                                 conv_op.get_attr('resize_align_corners'),
                                                                 mul_const,
                                                                 output_names=[str(resize_size.op.outputs[0].name)])
            mode = conv_op.get_attr('mode')
            if mode.decode() == "REFLECT":
                mode = "PADDING_REFLECT"
            elif mode.decode() == "SYMMETRIC":
                mode = "PADDING_SYMMETRIC"
            else:
                raise ConverterError(code_to_message.get_error_message("ERROR_TF_PAD_MODE_UNKNOWN")
                                     (mode.decode()))
            pad_desc = PadLayerResolver.Descriptor(str(pad.name),
                                                   [pad.op],
                                                   graph_helper.evaluate_tensor_output(pad),
                                                   mode,
                                                   0.0, # Only MirrorPad is matched, so constant val 0
                                                   output_names=[str(pad.op.outputs[0].name)])

        except TensorNotFoundError:
            pass
        return resize_desc, pad_desc


class DilatedConvolutionLayerResolver(ConvolutionLayerResolver, object):
    class Descriptor(ConvolutionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(DilatedConvolutionLayerResolver, self).__init__()
        self.graph_sequence = GraphSequence([
            NonConsumableConverterSequenceNode('space_to_batch', ['SpaceToBatchND']),
            NonConsumableConverterSequenceNode('inputs', ['?']),
            NonConsumableConverterSequenceNode('dilation_sizes', ['?']),
            NonConsumableConverterSequenceNode('paddings', ['?']),
            ConverterSequenceNode('conv_op', ['Conv2D']),
            NonConsumableConverterSequenceNode('kernel', ['?']),
            NonConsumableConverterSequenceNode('fake_quant', ['FakeQuantWithMinMaxVars']),
            NonConsumableConverterSequenceNode('min', ['?']),
            NonConsumableConverterSequenceNode('max', ['?']),
            NonConsumableConverterSequenceNode('batch_to_space', ['BatchToSpaceND']),
            NonConsumableConverterSequenceNode('block_shape_out', ['?']),
            NonConsumableConverterSequenceNode('crops', ['?'])]
        )
        self.graph_sequence.set_inputs('space_to_batch', ['inputs', 'dilation_sizes', 'paddings'])
        self.graph_sequence.set_inputs('conv_op', ['space_to_batch', 'kernel'])

        self.__symmetry_pad = False

    def __if_symmetry_pad(self, paddings_tensor, crops_tensor):
        actual_padding_sizes = [[paddings_tensor[i][j] - crops_tensor[i][j] for j in range(len(paddings_tensor[0]))] for i in range(len(paddings_tensor))]

        for index in range(len(actual_padding_sizes)):
            assert len(actual_padding_sizes[index]) == 2
            if (actual_padding_sizes[index][1] - actual_padding_sizes[index][0]) > 0:
                return False
        self.__symmetry_pad = True
        return True

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = []
        # match sequence with/without quantization node
        # no fake-quant
        self.graph_sequence.set_inputs('batch_to_space', ['conv_op', 'block_shape_out', 'crops'])
        self.graph_sequence.set_outputs(['batch_to_space'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        # with fake-quant
        self.graph_sequence.clear_inputs_for_nodes(['batch_to_space'])
        self.graph_sequence.clear_outputs()
        self.graph_sequence.set_inputs('fake_quant', ["conv_op", "min", "max"])
        self.graph_sequence.set_inputs('batch_to_space', ['fake_quant', 'block_shape_out', 'crops'])
        self.graph_sequence.set_outputs(['batch_to_space'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            conv_op = match['conv_op']
            output_op = conv_op
            strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding_size_strategy = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            pads = self.get_spatial_padding(conv_op)
            weights = self.get_weights(graph_helper, conv_op)
            consumed_nodes = match.consumed_nodes
            input_op = conv_op
            conv_output_names = [str(conv_op.outputs[0].name)]
            sequence_output_names = [str(match[node.identifier].outputs[0].name) for node in
                                     self.graph_sequence.output_nodes]

            dilation_sizes = match['dilation_sizes']
            dilation_sizes = graph_helper.evaluate_tensor_output(dilation_sizes.outputs[0])
            if np.shape(dilation_sizes) != (2,):
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONV_RESOLVE_DILATION')(conv_op.name))

            space_to_batch_op = match['space_to_batch']
            paddings_op = match['paddings']
            paddings_tensor = graph_helper.evaluate_tensor_output(paddings_op.outputs[0])
            if np.shape(paddings_tensor) != (2,2):
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONV_RESOLVE_PADDING')(np.shape(paddings_tensor), conv_op.name))

            batch_to_space_op = match['batch_to_space']
            conv_output_ops = graph_helper.get_op_outputs(batch_to_space_op)

            crop_op = match['crops']
            crops_tensor = graph_helper.evaluate_tensor_output(crop_op.outputs[0])
            if np.shape(crops_tensor) != (2,2):
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONV_RESOLVE_CROP')(np.shape(paddings_tensor), conv_op.name))

            if paddings_tensor.any() and not np.array_equal(paddings_tensor, crops_tensor) \
                    and not self.__if_symmetry_pad(paddings_tensor, crops_tensor):
                # Reshape the padding tensor to be 4D, pad it with 1 before and after
                paddings_tensor = np.pad(paddings_tensor, ((1, 1), (0, 0)), 'constant')
                pad_descriptor = PadLayerResolver.Descriptor(
                    str(space_to_batch_op.name),
                    [match['space_to_batch'], match['dilation_sizes'], match['paddings']],
                    paddings_tensor,
                    "PADDING_CONSTANT",
                    0.0,
                    output_names=[str(space_to_batch_op.outputs[0].name)])
                descriptors.append(pad_descriptor)
            else:
                if self.__symmetry_pad:
                    padding_size_strategy = b'SYMMETRY'
                consumed_nodes.extend([space_to_batch_op, paddings_op, match['dilation_sizes']])
                input_op = space_to_batch_op

            crop_descriptor = None
            output_names = conv_output_names
            if crops_tensor.any() and not np.array_equal(paddings_tensor, crops_tensor) and not self.__symmetry_pad:
                crops_tensor = np.pad(crops_tensor, ((1, 1), (0, 0)), 'constant')
                offsets = crops_tensor[:, 0]
                counts = np.empty(shape=(0,0))
                size = np.array(graph_helper.get_op_output_shape(match['batch_to_space']), dtype=np.int32)
                crop_descriptor = CropLayerResolver.Descriptor(
                    str(match['batch_to_space'].name),
                    [match['batch_to_space'], match['block_shape_out'], match['crops']],
                    offsets,
                    counts,
                    size,
                    output_names=[str(match['batch_to_space'].outputs[0].name)])
                descriptors.append(crop_descriptor)
            else:
                consumed_nodes.extend([match['block_shape_out'], crop_op, batch_to_space_op])
                output_names = sequence_output_names
                output_op = batch_to_space_op

            bias_op, biases = self.get_conv_bias(graph_helper, batch_to_space_op, conv_output_ops)
            if bias_op is not None and biases is not None:
                if crop_descriptor:
                    # Conv -> Crop -> Bias
                    # Because Crop is between Conv and Bias, if we append bias node to consumed_nodes,
                    # Crop descriptor will be disconnected and filtered out from the graph.
                    # Hence, let the Conv Descriptor absorb the Bias tensor mathematically and we will
                    # create an IgnoredLayerResolver for the bias node.
                    bias_desc = IgnoredLayersResolver.Descriptor(str(bias_op.outputs[0].name), [bias_op])
                    descriptors.append(bias_desc)
                else:
                    # Bias op can be merged into Convolution Op. Update the consumed nodes and output names
                    consumed_nodes.append(bias_op)
                    output_names = [str(bias_op.outputs[0].name)]
            else:
                bias_op = None
                biases = np.zeros(weights.shape[-1], dtype=np.float32)

            d = ConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes,
                                                    conv_op, bias_op, output_op,
                                                    strides, padding_size_strategy, weights,
                                                    biases, explicit_pads=pads, output_names=output_names)

            d.dilationY = int(dilation_sizes[0])
            d.dilationX = int(dilation_sizes[1])
            d.input_ops = [input_op]
            descriptors.append(d)

        return descriptors


class DepthwiseConvolutionLayerResolver(ConvolutionLayerResolver, object):

    def __init__(self):
        super(DepthwiseConvolutionLayerResolver, self).__init__()
        self.graph_sequence_with_bias = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            NonConsumableConverterSequenceNode('filter', ['?']),
            ConverterSequenceNode('conv', ['DepthwiseConv2dNative']),
            ConverterSequenceNode('bias', ['BiasAdd', 'Add']),
            NonConsumableConverterSequenceNode('other', ['?'])
        ])
        self.graph_sequence_with_bias.set_inputs('conv', ['input', 'filter'])
        self.graph_sequence_with_bias.set_inputs('bias', ['conv', 'other'])
        self.graph_sequence_with_bias.set_outputs(['bias'])

        self.graph_sequence = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            NonConsumableConverterSequenceNode('filter', ['?']),
            ConverterSequenceNode('conv', ['DepthwiseConv2dNative'])
        ])
        self.graph_sequence.set_inputs('conv', ['input', 'filter'])
        self.graph_sequence.set_outputs(['conv'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.graph_sequence)
        matches += graph_matcher.match_sequence(self.graph_sequence_with_bias)
        descriptors = []
        for match in matches:
            self._resolve_from_match(descriptors, graph_helper, match)
        return descriptors

    def _resolve_from_match(self, descriptors, graph_helper, match):
        conv_op = match['conv']
        output_op = conv_op
        strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
        padding = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
        pads = self.get_spatial_padding(conv_op)
        weights = self.get_weights(graph_helper, conv_op)
        shape = weights.shape
        weights = np.reshape(weights, (shape[0], shape[1], 1, -1))

        if 'bias' in match:
            biases = self.get_biases(graph_helper, conv_op, match['bias'])
        else:
            biases = np.zeros(np.shape(weights)[-1], dtype=np.float32)

        filter_op = match['filter']
        _, _, filter_consumed_nodes = graph_helper.get_static_data_info(conv_op, filter_op.outputs[0])

        consumed_nodes = filter_consumed_nodes
        consumed_nodes.extend(match.consumed_nodes)

        d = ConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes,
                                                conv_op, None, output_op,
                                                strides, padding, weights, biases, explicit_pads=pads)
        input_tensor, _ = GraphHelper.get_op_input_tensors(conv_op, ('?', '?'))
        d.groups = graph_helper.get_op_output_shape(input_tensor)[-1]
        descriptors.append(d)


class DilatedDepthwiseConvolutionLayerResolver(ConvolutionLayerResolver, object):
    class Descriptor(ConvolutionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(DilatedDepthwiseConvolutionLayerResolver, self).__init__()
        self.graph_sequence = GraphSequence([
            NonConsumableConverterSequenceNode('space_to_batch', ['SpaceToBatchND']),
            NonConsumableConverterSequenceNode('inputs', ['?']),
            NonConsumableConverterSequenceNode('dilation_sizes', ['?']),
            NonConsumableConverterSequenceNode('paddings', ['?']),
            ConverterSequenceNode('conv_op', ['DepthwiseConv2dNative']),
            NonConsumableConverterSequenceNode('kernel', ['?']),
            NonConsumableConverterSequenceNode('fake_quant', ['FakeQuantWithMinMaxVars']),
            NonConsumableConverterSequenceNode('min', ['?']),
            NonConsumableConverterSequenceNode('max', ['?']),
            NonConsumableConverterSequenceNode('biasAdd', ['BiasAdd']),
            NonConsumableConverterSequenceNode('bias', ['?']),
            NonConsumableConverterSequenceNode('batch_to_space', ['BatchToSpaceND']),  # output
            NonConsumableConverterSequenceNode('block_shape_out', ['?']),
            NonConsumableConverterSequenceNode('crops', ['?'])
        ])
        self.graph_sequence.set_inputs('space_to_batch', ['inputs', 'dilation_sizes', 'paddings'])
        self.graph_sequence.set_inputs('conv_op', ['space_to_batch', 'kernel'])

        self.__symmetry_pad = False

    def __if_symmetry_pad(self, paddings_tensor, crops_tensor):
        actual_padding_sizes = [[paddings_tensor[i][j] - crops_tensor[i][j] for j in range(len(paddings_tensor[0]))] for i in range(len(paddings_tensor))]

        for index in range(len(actual_padding_sizes)):
            assert len(actual_padding_sizes[index]) == 2
            if abs(actual_padding_sizes[index][1] - actual_padding_sizes[index][0]) > 0:
                return False
        self.__symmetry_pad = True
        return True

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = []
        # match sequence with/without quantization node
        # no fake-quant
        self.graph_sequence.set_inputs('batch_to_space', ['conv_op', 'block_shape_out', 'crops'])
        self.graph_sequence.set_outputs(['batch_to_space'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        # with fake-quant
        self.graph_sequence.clear_inputs_for_nodes(['batch_to_space'])
        self.graph_sequence.clear_outputs()
        self.graph_sequence.set_inputs('fake_quant', ["conv_op", "min", "max"])
        self.graph_sequence.set_inputs('batch_to_space', ['fake_quant', 'block_shape_out', 'crops'])
        self.graph_sequence.set_outputs(['batch_to_space'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        # SpaceToBatchND->Depthwise Conv->BiasAdd->BatchToSpaceND
        self.graph_sequence.clear_inputs_for_nodes(['batch_to_space'])
        self.graph_sequence.clear_outputs()
        self.graph_sequence.set_inputs('biasAdd', ['bias', 'conv_op'])
        self.graph_sequence.set_inputs('batch_to_space', ['biasAdd', 'block_shape_out', 'crops'])
        self.graph_sequence.set_outputs(['batch_to_space'])
        matches.extend(graph_matcher.match_sequence(self.graph_sequence))

        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            conv_op = match['conv_op']
            output_op = conv_op
            strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            pads = self.get_spatial_padding(conv_op)
            weights = self.get_weights(graph_helper, conv_op)
            if weights.shape[3] != 1:
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONV_RESOLVE_WEIGHTS_SHAPE')(conv_op.name))
            weights = np.transpose(weights, [0, 1, 3, 2])
            consumed_nodes = match.consumed_nodes
            output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in
                                     self.graph_sequence.output_nodes]
            batch_to_space_op = match['batch_to_space']
            conv_output_ops = graph_helper.get_op_outputs(batch_to_space_op)

            dilation_sizes = match['dilation_sizes']
            dilation_sizes = graph_helper.evaluate_tensor_output(dilation_sizes.outputs[0])
            if np.shape(dilation_sizes) != (2,):
                raise ConverterError(code_to_message.get_error_message('ERROR_TF_CONV_RESOLVE_DILATION')(conv_op.name))

            space_to_batch_op = match['space_to_batch']
            paddings_op = match['paddings']
            paddings_tensor = graph_helper.evaluate_tensor_output(paddings_op.outputs[0])
            input_op = conv_op

            batch_to_space_op = match['batch_to_space']
            crop_op = match['crops']
            crops_tensor = graph_helper.evaluate_tensor_output(crop_op.outputs[0])
            output_names = [str(conv_op.outputs[0].name)]

            if paddings_tensor.any() and not np.array_equal(paddings_tensor, crops_tensor) \
                    and not self.__if_symmetry_pad(paddings_tensor, crops_tensor):
                # Reshape the padding tensor to be 4D, pad it with 1 before and after
                paddings_tensor = np.pad(paddings_tensor, ((1, 1), (0, 0)), 'constant')
                pad_descriptor = PadLayerResolver.Descriptor(
                    str(space_to_batch_op.name),
                    [match['space_to_batch'], match['dilation_sizes'], match['paddings']],
                    paddings_tensor,
                    "PADDING_CONSTANT",
                    0.0,
                    output_names=[str(space_to_batch_op.outputs[0].name)])
                descriptors.append(pad_descriptor)
            else:
                if self.__symmetry_pad:
                    padding = b'SYMMETRY'
                consumed_nodes.extend([space_to_batch_op, paddings_op, match['dilation_sizes']])
                input_op = space_to_batch_op

            crop_descriptor = None
            if crops_tensor.any() and not np.array_equal(paddings_tensor, crops_tensor) and not self.__symmetry_pad:
                crops_tensor = np.pad(crops_tensor, ((1, 1), (0, 0)), 'constant')
                offsets = crops_tensor[:, 0]
                counts = np.empty(shape=(0,0))
                size = np.array(graph_helper.get_op_output_shape(match['batch_to_space']), dtype=np.int32)
                crop_descriptor = CropLayerResolver.Descriptor(
                    str(match['batch_to_space'].name),
                    [match['batch_to_space'], match['block_shape_out'], match['crops']],
                    offsets,
                    counts,
                    size,
                    output_names=[str(match['batch_to_space'].outputs[0].name)])
                descriptors.append(crop_descriptor)
            else:
                consumed_nodes.extend([match['block_shape_out'], crop_op, batch_to_space_op])
                output_names = output_op_nodes_names
                output_op = batch_to_space_op

            if ('biasAdd' in match):
                conv_output_ops.append(match['biasAdd'])
            bias_op, biases = self.get_conv_bias(graph_helper, batch_to_space_op, conv_output_ops)
            if bias_op is not None and biases is not None:
                if crop_descriptor:
                    # Conv -> Crop -> Bias
                    # Because Crop is between Conv and Bias, if we append bias node to consumed_nodes,
                    # Crop descriptor will be disconnected and filtered out from the graph.
                    # Hence, let the Conv Descriptor absorb the Bias tensor mathematically and we will
                    # create an IgnoredLayerResolver for the bias node.
                    bias_desc = IgnoredLayersResolver.Descriptor(str(bias_op.outputs[0].name), [bias_op])
                    descriptors.append(bias_desc)
                else:
                    # Bias op can be merged into Convolution Op. Update the consumed nodes and output names
                    consumed_nodes.append(bias_op)
                    output_names = [str(bias_op.outputs[0].name)]
            else:
                bias_op = None
                biases = np.zeros(weights.shape[-1], dtype=np.float32)

            d = ConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes,
                                                    conv_op, bias_op, output_op, strides, padding, weights,
                                                    biases, explicit_pads=pads,
                                                    output_names=output_names)

            d.groups = graph_helper.get_op_output_shape(space_to_batch_op)[-1]
            d.dilationY = int(dilation_sizes[0])
            d.dilationX = int(dilation_sizes[1])
            d.input_ops = [input_op]
            descriptors.append(d)

        return descriptors


class ConvolutionLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConvolutionLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_names(converter_context, descriptor, input_descriptors)[0]
        input_dims = converter_context.get_input_layer_output_shape_for(descriptor.input_ops[0])

        output_dims = converter_context.graph_helper.get_op_output_shape(descriptor.output_op)

        # First build resize -> pad sequence if Conv is fused
        if descriptor.resize_desc is not None and descriptor.pad_desc is not None:
            resize_desc = descriptor.resize_desc
            pad_desc = descriptor.pad_desc
            resize_output_shape = [input_dims[0],
                                   *converter_context.graph_helper.evaluate_tensor_output(resize_desc.resize_op),
                                   input_dims[-1]]
            ir_graph.add(ResizeOp(resize_desc.output_names[0],
                                  resize_output_shape,
                                  pad_value=0.0,
                                  maintain_aspect_ratio=False,
                                  resize_mode=resize_desc.resize_mode,
                                  scale_height=resize_desc.mul_const[0],
                                  scale_width=resize_desc.mul_const[1],
                                  align_corners=resize_desc.align_corners),
                         input_names=input_descriptors[0].output_names[0],
                         output_names=resize_desc.output_names[0])

            ir_graph.add(PadOp(pad_desc.layer_name,
                               pads=pad_desc.paddings.tolist(),
                               mode=pad_desc.mode,
                               constant_value=float(pad_desc.constant_values)),
                         input_names=resize_desc.output_names[0],
                         output_names=pad_desc.output_names[0])

            input_name = descriptor.pad_desc.output_names[0]
            input_dims = resize_output_shape

        pads, ir_padding_strategy = ConvolutionLayerBuilder.calculate_padding_size(input_size=input_dims[-3:-1],
                                                                                   output_size=output_dims[-3:-1],
                                                                                   strides=descriptor.strides[1:3],
                                                                                   padding_size_strategy=descriptor.padding_size_strategy,
                                                                                   explicit_pads=descriptor.explicit_pads,
                                                                                   filter_dims=descriptor.weights.shape,
                                                                                   dilation=[descriptor.dilationY,
                                                                                             descriptor.dilationX])

        # using layer name for output buffer name
        return ir_graph.add(ConvolutionOp(name=descriptor.layer_name,
                                          weights=descriptor.weights,
                                          bias=descriptor.biases,
                                          pady_before=pads[0][0],
                                          pady_after=pads[0][1],
                                          padx_before=pads[1][0],
                                          padx_after=pads[1][1],
                                          padding_mode="PADDING_ZERO",
                                          padding_size_strategy=ir_padding_strategy,
                                          stridex=int(descriptor.strides[2]),
                                          stridey=int(descriptor.strides[1]),
                                          dilationx=descriptor.dilationX,
                                          dilationy=descriptor.dilationY,
                                          groups=descriptor.groups),
                            input_name,
                            descriptor.output_names[0])

    @classmethod
    def calculate_padding_size(cls, input_size, output_size, strides, padding_size_strategy,
                               filter_dims, dilation, explicit_pads=list([[0, 0], [0, 0]])):

        if padding_size_strategy.decode() in ["SAME", "SYMMETRY"]:
            filter_h = filter_dims[0] + (filter_dims[0] - 1) * (dilation[0] - 1)
            filter_w = filter_dims[1] + (filter_dims[1] - 1) * (dilation[1] - 1)
            pad_y = max(((output_size[0] - 1) * strides[0] + filter_h - input_size[0]), 0)
            pad_x = max(((output_size[1] - 1) * strides[1] + filter_w - input_size[1]), 0)
            # We divide by two and truncate if odd padding given the runtime will
            # take care of Implicit Asymmetry
            pad_y = int(pad_y // 2)
            pad_x = int(pad_x // 2)
            pads = [[pad_y, pad_y], [pad_x, pad_x]]

            if padding_size_strategy.decode() == 'SAME':
                # i.e for odd padding, add the extra padding at the end
                ir_padding_strategy = IRPaddingStrategies.PADDING_SIZE_IMPLICIT_SAME_BEGIN
            else:
                ir_padding_strategy = IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR
        elif padding_size_strategy.decode() == "EXPLICIT":
            pads = explicit_pads
            ir_padding_strategy = IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR
        elif padding_size_strategy.decode() == 'VALID':
            pads = [[0, 0], [0, 0]]
            ir_padding_strategy = IRPaddingStrategies.PADDING_SIZE_IMPLICIT_VALID
        else:
            raise ValueError("Unsupported TF padding strategy {}".format(padding_size_strategy.decode()))

        return pads, ir_padding_strategy

    def transform_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):

        # add folded bn params
        filtered_input_descriptors = [d for d in input_descriptors if isinstance(d, BatchNormLayerResolver.Descriptor)
                                      and d.bn_folded]

        if len(filtered_input_descriptors) == 1:
            if not filtered_input_descriptors[0].pre_calculated:
                # TODO: remove adding the quantization params here when tf_to_ir starts using op_graph_optimizations
                ir_graph.add_quantization_params(descriptor.layer_name,
                                                 bn_params={"gamma": filtered_input_descriptors[0].scale,
                                                            "beta": filtered_input_descriptors[0].beta})
            # Note: no need to multiply the weights and bias in this case since these 2 will have resolved as const
            #       inputs and already are weights/bias to the convolution op. (see above in get_weights/get_biases)
            converter_context.merge_descriptors(filtered_input_descriptors[0], descriptor)
        else:
            # fold Batchnorm as applicable and add bn params
            filtered_output_descriptors = [d for d in output_descriptors if isinstance(d, BatchNormLayerResolver.Descriptor)]
            if filtered_output_descriptors == output_descriptors and len(output_descriptors) == 1:
                descriptor.weights = descriptor.weights * filtered_output_descriptors[0].weights
                descriptor.biases = (descriptor.biases * filtered_output_descriptors[0].weights) + filtered_output_descriptors[0].biases
                # TODO: remove adding the quantization params here when tf_to_ir is complete
                if not filtered_output_descriptors[0].pre_calculated:
                    ir_graph.add_quantization_params(descriptor.layer_name,
                                                     bn_params={"gamma": filtered_output_descriptors[0].scale,
                                                                "beta": filtered_output_descriptors[0].beta})
                ir_graph.merge_quantization_params(output_descriptors[0].layer_name, descriptor.layer_name,
                                                   descriptor.output_names[0], output_descriptors[0].output_names[0])
                descriptor.output_names = output_descriptors[0].output_names
                converter_context.merge_descriptors(output_descriptors[0], descriptor)


class GroupedConvolutionLayerResolver(ConvolutionLayerResolver, object):
    class Descriptor(ConvolutionLayerResolver.Descriptor):
        pass

    def __init__(self):
        super(GroupedConvolutionLayerResolver, self).__init__()

        # grouped convolution with split
        tree_output_node = ConverterSequenceNode('conv_op', ['Conv2D'])
        self.sequence = GraphSequence([
            ConverterSequenceNode('a', ['Split']),
            ConverterSequenceNode('b', ['Split']),
            ConverterRepeatableSequenceTreeNode('repeatable_graph', tree_output_node, tree_output_node),
            ConverterSequenceNode('concat_op', ['Concat']),
            ConverterSequenceNode('weights', ['Identity', 'Const']),
            NonConsumableConverterSequenceNode('inputs', ['?']),
            NonConsumableConverterSequenceNode('concat_dim', ['Const']),
            NonConsumableConverterSequenceNode('split_dim1', ['Const']),
            ConverterSequenceNode('split_dim2', ['Const'])
        ])
        self.sequence.set_inputs('a', ['inputs', 'split_dim1'])
        self.sequence.set_inputs('b', ['weights', 'split_dim2'])
        self.sequence.set_inputs('repeatable_graph', ['a', 'b'])
        self.sequence.set_inputs('concat_op', ['repeatable_graph', 'concat_dim'])
        self.sequence.set_outputs(['concat_op'])

        # grouped convolution with strided slice
        repeatable_sequence = GraphSequence([
            ConverterSequenceNode('ss', ['StridedSlice']),
            ConverterSequenceNode('ss_begin', ['Const']),
            ConverterSequenceNode('ss_end', ['Const']),
            ConverterSequenceNode('ss_strides', ['Const']),
            ConverterSequenceNode('conv', ['Conv2D']),
            ConverterSequenceNode('bias', ['BiasAdd']),
            ConverterSequenceNode('weights', ['Identity', 'Const']),
            ConverterSequenceNode('biases', ['Identity', 'Const'])
        ])
        repeatable_sequence.set_inputs('ss', ['ss_begin', 'ss_end', 'ss_strides'])
        repeatable_sequence.set_inputs('conv', ['ss', 'weights'])
        repeatable_sequence.set_inputs('bias', ['biases', 'conv'])
        repeatable_sequence.set_outputs(['bias'])

        self.sequence_with_strided_slice = GraphSequence([
            ConverterRepeatableSequenceTreeNode('repeatable_graph',
                                                tree_output_node=repeatable_sequence['bias'],
                                                tree_input_node=repeatable_sequence['ss']),
            ConverterSequenceNode('concat', ['Concat', 'ConcatV2']),
            ConverterSequenceNode('axis', ['Const']),
            NonConsumableConverterSequenceNode('input', ['?'])
        ])
        self.sequence_with_strided_slice.set_inputs('repeatable_graph', ['input'])
        self.sequence_with_strided_slice.set_inputs('concat', ['repeatable_graph', 'axis'])
        self.sequence_with_strided_slice.set_outputs(['concat'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for match in graph_matcher.match_sequence(self.sequence):
            conv_op = match['conv_op_1']
            output_op = conv_op
            strides = conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES)
            padding = conv_op.get_attr(self.TF_ATTRIBUTE_PADDING)
            pads = self.get_spatial_padding(conv_op)
            weights = match['weights']
            consumed_nodes = match.consumed_nodes
            output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in
                                     self.sequence.output_nodes]
            concat_op = match['concat_op']
            concat_op_output_ops = graph_helper.get_op_outputs(concat_op)
            bias_op, biases = self.get_conv_bias(graph_helper, concat_op, concat_op_output_ops)
            if bias_op is not None and biases is not None:
                output_op_nodes_names = [str(bias_op.outputs[0].name)]
                consumed_nodes.append(bias_op)
            else:
                bias_op = None
                biases = np.zeros(weights.outputs[0].get_shape()[-1], dtype=np.float32)

            weights = graph_helper.evaluate_tensor_output(weights.outputs[0])
            descriptor = ConvolutionLayerResolver.Descriptor(str(conv_op.name), consumed_nodes,
                                                             conv_op, bias_op, output_op,
                                                             strides, padding, weights, biases, explicit_pads=pads,
                                                             output_names=output_op_nodes_names)
            descriptor.input_ops = [match['a'], match['b']]
            descriptors.append(descriptor)

        for match in graph_matcher.match_sequence(self.sequence_with_strided_slice):
            if not match.consumed_nodes:
                continue
            input_op = match['input']
            concat_op = match['concat']
            axis_op = match['axis']
            conv_ops = self._get_repeatable_op_by_id(match, 'conv')
            weight_ops = self._get_repeatable_op_by_id(match, 'weights')
            bias_ops = self._get_repeatable_op_by_id(match, 'biases')
            bias_add_ops = self._get_repeatable_op_by_id(match, 'bias')
            ss_ops = self._get_repeatable_op_by_id(match, 'ss')

            input_shape = graph_helper.get_op_output_shape(input_op)
            weight_shapes = [graph_helper.get_op_output_shape(weight_op) for weight_op in weight_ops]

            ss_strides = [graph_helper.evaluate_tensor_output(ss_strides_op.outputs[0]).tolist()
                          for ss_strides_op in self._get_repeatable_op_by_id(match, 'ss_strides')]
            ss_begins = [graph_helper.evaluate_tensor_output(ss_begin_op.outputs[0]).tolist()
                         for ss_begin_op in self._get_repeatable_op_by_id(match, 'ss_begin')]
            ss_ends = [graph_helper.evaluate_tensor_output(ss_end_op.outputs[0]).tolist()
                       for ss_end_op in self._get_repeatable_op_by_id(match, 'ss_end')]

            bias_add_shapes = [graph_helper.get_op_output_shape(bias_add_op) for bias_add_op in bias_add_ops]

            strides = [conv_op.get_attr(self.TF_ATTRIBUTE_STRIDES) for conv_op in conv_ops]
            paddings = [conv_op.get_attr(self.TF_ATTRIBUTE_PADDING) for conv_op in conv_ops]
            pads = [self.get_spatial_padding(conv_op) for conv_op in conv_ops]

            ss_shapes = [graph_helper.get_op_output_shape(ss_op.outputs[0])
                         for ss_op in ss_ops]

            num_groups = len(conv_ops)

            axis = graph_helper.evaluate_tensor_output(axis_op.outputs[0])

            is_grouped_convolution = True
            is_grouped_convolution &= self._elements_are_same(bias_add_shapes)
            is_grouped_convolution &= self._elements_are_same(weight_shapes)
            is_grouped_convolution &= self._elements_are_same(strides)
            is_grouped_convolution &= self._elements_are_same(paddings)
            is_grouped_convolution &= self._elements_are_same(ss_shapes)
            is_grouped_convolution &= self._elements_are_same(ss_strides)
            is_grouped_convolution &= not self._elements_are_same(ss_begins)
            is_grouped_convolution &= not self._elements_are_same(ss_ends)
            # stride slices must evenly divide the last dimension of input to number of groups
            is_grouped_convolution &= ss_shapes[0][-1] * num_groups == input_shape[-1]
            # strides must be all ones at all dimensions
            is_grouped_convolution &= ss_strides[0] == [1] * len(ss_strides[0])
            # concat must be on the last axis in grouped convolution
            is_grouped_convolution &= axis == -1 or axis == (len(bias_add_shapes[0]) - 1)

            if not is_grouped_convolution:
                logging.getLogger().warning(code_to_message.get_error_message('WARNING_TF_GROUP_CONV_RESOLVE'))
                continue

            weight_tensors = [graph_helper.evaluate_tensor_output(weight_op.outputs[0])
                              for weight_op in weight_ops]
            weights = np.concatenate(weight_tensors, axis=-1)

            bias_tensors = [graph_helper.evaluate_tensor_output(bias_op.outputs[0])
                            for bias_op in bias_ops]
            biases = np.concatenate(bias_tensors, axis=-1)

            descriptor = ConvolutionLayerResolver.Descriptor(
                str(concat_op.name), match.consumed_nodes, conv_ops[0], None, conv_ops[0],
                strides[0], paddings[0], weights, biases, explicit_pads=pads[0],
                output_names=[str(concat_op.outputs[0].name)])
            descriptor.input_ops = ss_ops
            descriptor.output_op = concat_op
            descriptors.append(descriptor)

        return descriptors

    @classmethod
    def _get_repeatable_op_by_id(cls, match, name):
        ops = []
        indexed_id = name + '_{}'
        i = 1
        while indexed_id.format(i) in match:
            ops.append(match[indexed_id.format(i)])
            i += 1
        return ops

    @classmethod
    def _elements_are_same(cls, array):
        return all([element == array[0] for element in array])
