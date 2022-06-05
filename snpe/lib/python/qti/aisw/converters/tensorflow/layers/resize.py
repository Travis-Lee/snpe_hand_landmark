# =============================================================================
#
#  Copyright (c) 2015-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import numpy as np

from qti.aisw.converters.common.converter_ir.op_adapter import ResizeOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.util import GraphHelper
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)


class ResizeBilinearLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_ALIGN_CORNERS = 'align_corners'

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, input_tensor_shape, resize_op,
                     align_corners=False, mul_const = [0, 0], output_names=None):
            super(ResizeBilinearLayerResolver.Descriptor, self).__init__('Resize', name, nodes, output_names=output_names)
            self.align_corners = align_corners
            self.input_tensor_shape = input_tensor_shape
            self.resize_mode = "bilinear"
            self.resize_op = resize_op
            self.mul_const = mul_const

        def is_input_tensor(self, op, tensor):
            if op == self.resize_op and tensor != self.resize_op.inputs[0]:
                return False
            return True

    def __init__(self):
        sequence_resize = GraphSequence([ConverterSequenceNode('root', ['ResizeBilinear'])])
        sequence_resize.set_outputs(['root'])

        sequence_shape_stridedslice_resize = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('shape', ['Shape']),
            ConverterSequenceNode('stridedSlice', ['StridedSlice']),
            ConverterSequenceNode('mul', ['Mul']),
            ConverterSequenceNode('const_stridedSlice_1', ['?']),
            ConverterSequenceNode('const_stridedSlice_2', ['?']),
            ConverterSequenceNode('const_stridedSlice_3', ['?']),
            ConverterSequenceNode('mul_const', ['?']),
            ConverterSequenceNode('root', ['ResizeBilinear'])])

        sequence_shape_stridedslice_resize.set_inputs('shape', ['input'])
        sequence_shape_stridedslice_resize.set_inputs('stridedSlice', ['shape',
                                                                       'const_stridedSlice_1',
                                                                       'const_stridedSlice_2',
                                                                       'const_stridedSlice_3'])
        sequence_shape_stridedslice_resize.set_inputs('mul', ['stridedSlice', 'mul_const'])
        sequence_shape_stridedslice_resize.set_inputs('root', ['mul', 'input'])
        sequence_shape_stridedslice_resize.set_outputs(['root'])

        sequence_resize_for_decode = GraphSequence([
            ConverterSequenceNode('shape', ['Shape']),
            ConverterSequenceNode('strided_slice_1', ['StridedSlice']),
            ConverterSequenceNode('strided_slice', ['StridedSlice']),
            ConverterSequenceNode('mul_1', ['Mul']),
            ConverterSequenceNode('mul', ['Mul']),
            ConverterSequenceNode('size', ['Pack']),
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('root', ['ResizeBilinear']),
            ConverterSequenceNode('stub_8', ['?']),
            ConverterSequenceNode('stub_9', ['?']),
            ConverterSequenceNode('stub_10', ['?']),
            ConverterSequenceNode('stub_11', ['?']),
            ConverterSequenceNode('stub_12', ['?']),
            ConverterSequenceNode('stub_13', ['?']),
            ConverterSequenceNode('stub_14', ['?']),
            ConverterSequenceNode('stub_15', ['?']),
        ])
        sequence_resize_for_decode.set_inputs('mul_1', ['strided_slice_1', 'stub_14'])
        sequence_resize_for_decode.set_inputs('root', ['input', 'size'])
        sequence_resize_for_decode.set_inputs('shape', ['input'])
        sequence_resize_for_decode.set_inputs('mul', ['strided_slice', 'stub_15'])
        sequence_resize_for_decode.set_inputs('strided_slice', ['shape', 'stub_11', 'stub_12', 'stub_13'])
        sequence_resize_for_decode.set_inputs('size', ['mul', 'mul_1'])
        sequence_resize_for_decode.set_inputs('strided_slice_1', ['shape', 'stub_8', 'stub_9', 'stub_10'])
        sequence_resize_for_decode.set_outputs(['root'])

        self.sequences = [sequence_resize, sequence_shape_stridedslice_resize, sequence_resize_for_decode]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                resize_op = match['root']
                align_corners_bool = resize_op.get_attr(self.TF_ATTRIBUTE_ALIGN_CORNERS)
                input_tensor, _ = GraphHelper.get_op_input_tensors(resize_op, ('?', '?'))
                input_tensor_shape = graph_helper.get_op_output_shape(input_tensor)
                consumed_nodes = match.consumed_nodes
                mul_const = [0, 0]

                if('mul_const' in match):
                    mul_const_op = match['mul_const']
                    mul_const = graph_helper.evaluate_tensor_output(mul_const_op.outputs[0])
                    if(len(mul_const) < 2):
                        mul_const = [0, 0]

                descriptors.append(
                    ResizeBilinearLayerResolver.Descriptor(str(resize_op.name),
                                                           consumed_nodes,
                                                           input_tensor_shape,
                                                           resize_op,
                                                           align_corners_bool,
                                                           mul_const,
                                                           output_names=[str(resize_op.outputs[0].name)]))

        return descriptors


class ResizeNearestNeighborLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_ALIGN_CORNERS = 'align_corners'

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, input_tensor_shape, resize_op, align_corners=False, mul_const = [0, 0]):
            super(ResizeNearestNeighborLayerResolver.Descriptor, self).__init__('ResizeNearestNeighbor', name, nodes)
            self.align_corners = align_corners
            self.input_tensor_shape = input_tensor_shape
            self.resize_mode = "nearest"
            self.resize_op = resize_op
            self.mul_const = mul_const

    def __init__(self):
        sequence_resize = GraphSequence([ConverterSequenceNode('root', ['ResizeNearestNeighbor'])])
        sequence_resize.set_outputs(['root'])

        sequence_shape_stridedslice_resize = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('shape', ['Shape']),
            ConverterSequenceNode('stridedSlice', ['StridedSlice']),
            ConverterSequenceNode('mul', ['Mul']),
            ConverterSequenceNode('const_stridedSlice_1', ['?']),
            ConverterSequenceNode('const_stridedSlice_2', ['?']),
            ConverterSequenceNode('const_stridedSlice_3', ['?']),
            ConverterSequenceNode('mul_const', ['?']),
            ConverterSequenceNode('root', ['ResizeNearestNeighbor'])])

        sequence_shape_stridedslice_resize.set_inputs('shape', ['input'])
        sequence_shape_stridedslice_resize.set_inputs('stridedSlice', ['shape',
                                                                       'const_stridedSlice_1',
                                                                       'const_stridedSlice_2',
                                                                       'const_stridedSlice_3'])
        sequence_shape_stridedslice_resize.set_inputs('mul', ['stridedSlice', 'mul_const'])
        sequence_shape_stridedslice_resize.set_inputs('root', ['mul', 'input'])
        sequence_shape_stridedslice_resize.set_outputs(['root'])

        # sequence for nearest neighbour resize without using tf resize op. Eg: seen in Mobilenetv1-FPN-SSD
        sequence_reshape_mul_resize = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('Shape', ['Shape']),
            ConverterSequenceNode('Reshape/shape', ['Pack']),
            ConverterSequenceNode('strided_slice', ['StridedSlice']),
            ConverterSequenceNode('Reshape', ['Reshape']),
            ConverterSequenceNode('Reshape_1/shape', ['Pack']),
            ConverterSequenceNode('scale_mul', ['Mul']),
            ConverterSequenceNode('root', ['Reshape']),  # root here is the reshape layer for getting back
                                                         # to input shape
            ConverterSequenceNode('stub_1', ['?']),
            ConverterSequenceNode('stub_2', ['?']),
            ConverterSequenceNode('stub_3', ['?']),
            ConverterSequenceNode('stub_4', ['?']),
            ConverterSequenceNode('stub_5', ['?']),
            ConverterSequenceNode('stub_6', ['?']),
            ConverterSequenceNode('stub_7', ['?']),
            ConverterSequenceNode('stub_8', ['?']),
            ConverterSequenceNode('mul_const', ['?']),
            ConverterSequenceNode('stub_10', ['?']),
            ConverterSequenceNode('stub_11', ['?']),
            ConverterSequenceNode('stub_12', ['?'])
            ])
        sequence_reshape_mul_resize.set_inputs('Shape', ['input'])
        sequence_reshape_mul_resize.set_inputs('strided_slice', ['Shape', 'stub_1', 'stub_2', 'stub_3'])
        sequence_reshape_mul_resize.set_inputs('Reshape/shape', ['strided_slice', 'stub_4', 'stub_5', 'stub_6', 'stub_7', 'stub_8'])
        sequence_reshape_mul_resize.set_inputs('Reshape', ['input','Reshape/shape'])
        sequence_reshape_mul_resize.set_inputs('scale_mul', ['Reshape', 'mul_const'])
        sequence_reshape_mul_resize.set_inputs('Reshape_1/shape', ['strided_slice', 'stub_10', 'stub_11', 'stub_12'])
        sequence_reshape_mul_resize.set_inputs('root', ['scale_mul', 'Reshape_1/shape'])
        sequence_reshape_mul_resize.set_outputs(['root'])

        self.sequences = [sequence_resize, sequence_shape_stridedslice_resize, sequence_reshape_mul_resize]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                resize_op = match['root']
                align_corners_bool = False
                input_tensor = None
                try:
                    # Model where resize is done without calling tf resize nearest neighbor by just
                    # using reshape slice and mul. Hence have a default of False for align corners
                    align_corners_bool = resize_op.get_attr(self.TF_ATTRIBUTE_ALIGN_CORNERS)
                    input_tensor, _ = GraphHelper.get_op_input_tensors(resize_op, ('?', '?'))
                except ValueError:
                    pass
                # if a tf resize op is not used, input should be the first input to the pattern matching
                if input_tensor is None:
                    input_tensor = match['input']
                input_tensor_shape = graph_helper.get_op_output_shape(input_tensor)
                consumed_nodes = match.consumed_nodes
                mul_const = [0, 0]

                if 'mul_const' in match:
                    mul_const_op = match['mul_const']
                    mul_const = graph_helper.evaluate_tensor_output(mul_const_op.outputs[0])
                    if type(mul_const) is np.ndarray:
                        mul_const = mul_const.squeeze().shape  # get the actual scale values for height and width
                    if len(mul_const) < 2:
                        mul_const = [0, 0]

                descriptors.append(
                    ResizeNearestNeighborLayerResolver.Descriptor(str(resize_op.name),
                                                                  consumed_nodes,
                                                                  input_tensor_shape,
                                                                  resize_op,
                                                                  align_corners_bool,
                                                                  mul_const))

        return descriptors


class ResizeLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReshapeLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_shape = converter_context.graph_helper.get_op_output_shape(descriptor.resize_op)
        output_shape = output_shape[-4:] if len(output_shape) > 4 else output_shape
        return ir_graph.add(ResizeOp(descriptor.output_names[0],
                                     output_shape,
                                     pad_value=0.0,
                                     maintain_aspect_ratio=False,
                                     resize_mode=descriptor.resize_mode,
                                     scale_height=descriptor.mul_const[0],
                                     scale_width=descriptor.mul_const[1],
                                     align_corners=descriptor.align_corners),
                            input_names=input_name,
                            output_names=descriptor.output_names[0])
