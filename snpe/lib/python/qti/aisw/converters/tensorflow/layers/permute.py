# =============================================================================
#
#  Copyright (c) 2018-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.op_adapter import PermuteOp
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
from qti.aisw.converters.tensorflow.util import ConverterError


class PermuteLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, order, output_names=None):
            super(PermuteLayerResolver.Descriptor, self).__init__('Permute', name, nodes, output_names=output_names)
            self.order = order

    def __init__(self):
        self.sequence_with_explicit_order = GraphSequence([
            ConverterSequenceNode('root', ['Transpose']),
            ConverterSequenceNode('order', ['Const']),
            NonConsumableConverterSequenceNode('input', ['?']),
        ])
        self.sequence_with_explicit_order.set_inputs('root', ['input', 'order'])
        self.sequence_with_explicit_order.set_outputs(['root'])

        self.sequence_with_implicit_order = GraphSequence([
            ConverterSequenceNode('root', ['Transpose']),
            ConverterSequenceNode('order', ['Sub']),
            ConverterSequenceNode('a', ['Sub']),
            ConverterSequenceNode('b', ['Const']),
            ConverterSequenceNode('c', ['Range']),
            ConverterSequenceNode('d', ['Const']),
            ConverterSequenceNode('e', ['Const']),
            ConverterSequenceNode('f', ['Rank']),
            NonConsumableConverterSequenceNode('input', ['?'])
        ])

        self.sequence_with_implicit_order.set_inputs('root', ['input', 'order'])
        self.sequence_with_implicit_order.set_inputs('order', ['a', 'c'])
        self.sequence_with_implicit_order.set_inputs('a', ['b', 'f'])
        self.sequence_with_implicit_order.set_inputs('c', ['d', 'e', 'f'])
        self.sequence_with_implicit_order.set_inputs('f', ['input'])
        self.sequence_with_implicit_order.set_outputs(['root'])

        self.sequences = [self.sequence_with_explicit_order, self.sequence_with_implicit_order]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            for match in graph_matcher.match_sequence(sequence):
                transpose_op = match['root']
                input_op = match['input']
                order_op = match['order']

                order_tensor = graph_helper.evaluate_tensor_output(order_op.outputs[0])

                input_shape = graph_helper.get_op_output_shape(input_op)
                order_shape = graph_helper.get_op_output_shape(order_op)

                input_rank = len(input_shape)
                order_rank = len(order_shape)
                try:
                    assert order_rank == 1
                    for d in range(input_rank):
                        assert d in order_tensor
                except AssertionError:
                    raise ConverterError(code_to_message.get_error_message(
                        'ERROR_TF_PERMUTE_INVALID_ORDER_TENSOR')(str(order_tensor)))

                consumed_nodes = match.consumed_nodes
                permute_descriptor = PermuteLayerResolver.Descriptor(
                    str(transpose_op.name), consumed_nodes, order_tensor,
                    output_names=[str(transpose_op.outputs[0].name)])
                descriptors.extend([permute_descriptor])

        return descriptors


class PermuteLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: PermuteLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]

        # Check for known permute orders to set the axis format correctly
        order_list = descriptor.order.tolist()
        if order_list == [0, 2, 3, 1]:
            axis_format = AxisTracker.AxisFormat.NSC
        elif order_list == [0, 3, 1, 2]:
            axis_format = AxisTracker.AxisFormat.NCS
        elif order_list == [1, 0, 2]:
            if ir_graph.get_buffer(input_name).get_axis_format() == AxisTracker.AxisFormat.BTF:
                axis_format = AxisTracker.AxisFormat.TBF
            elif ir_graph.get_buffer(input_name).get_axis_format() == AxisTracker.AxisFormat.TBF:
                axis_format = AxisTracker.AxisFormat.BTF
            else:
                axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        else:
            axis_format = AxisTracker.AxisFormat.NONTRIVIAL

        return ir_graph.add(PermuteOp(name=descriptor.layer_name,
                                      order=order_list),
                            input_names=input_name,
                            output_names=output_name,
                            axis_formats=[axis_format])
