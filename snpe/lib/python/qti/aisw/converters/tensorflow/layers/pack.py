# =============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import sys

from qti.aisw.converters.common.converter_ir.op_adapter import PackOp, UnpackOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerBuilder, LayerResolver
from qti.aisw.converters.tensorflow.util import ConverterError
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)


class PackLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_AXIS = 'axis'

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, axis, pack_op, output_names=None):
            super(PackLayerResolver.Descriptor, self).__init__('Pack', name, nodes, output_names=output_names)
            self.axis = axis
            self.pack_op = pack_op

    def __init__(self):
        self.sequence = GraphSequence([
            ConverterSequenceNode('root', ['Pack'])
        ])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            pack_op = match['root']
            consumed_nodes = match.consumed_nodes
            axis = int(pack_op.get_attr(self.TF_ATTRIBUTE_AXIS))
            pack_inputs = [tensor for tensor in pack_op.inputs]
            max_shape = 0
            for t in pack_inputs:
                shape = graph_helper.get_op_output_shape(t.op)
                if len(shape) > max_shape:
                    max_shape = len(shape)
            if axis < 0:
                axis += (max_shape + 1)
            pack_descriptor = PackLayerResolver.Descriptor(str(pack_op.name), consumed_nodes,
                                                           axis, pack_op, [pack_op.outputs[0].name])

            descriptors.append(pack_descriptor)
        return descriptors


class PackLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConcatLayerResolver.Descriptor
        :rtype: int
        """
        input_names = self.get_input_names(converter_context, descriptor, input_descriptors)
        output_shape = converter_context.graph_helper.get_op_output_shape(descriptor.pack_op)
        output_shape = output_shape[-4:] if len(output_shape) > 4 else output_shape
        return ir_graph.add(PackOp(descriptor.layer_name,
                                   output_dim=output_shape,
                                   axis=descriptor.axis),
                            input_names=input_names,
                            output_names=descriptor.output_names[0])


class UnPackLayerResolver(LayerResolver, object):
    TF_ATTRIBUTE_AXIS = 'axis'
    TF_ATTRIBUTE_NUM = 'num'

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, axis, num, unpack_op, output_names=None):
            super(UnPackLayerResolver.Descriptor, self).__init__('Unpack', name, nodes, output_names=output_names)
            self.axis = axis
            self.num = num
            self.unpack_op = unpack_op

    def __init__(self):
        self.sequence = GraphSequence([
            ConverterSequenceNode('root', ['Unpack'])
        ])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            unpack_op = match['root']
            consumed_nodes = match.consumed_nodes
            axis = int(unpack_op.get_attr(self.TF_ATTRIBUTE_AXIS))
            num = unpack_op.get_attr(self.TF_ATTRIBUTE_NUM)

            unpack_inputs = [tensor for tensor in unpack_op.inputs]
            max_shape = 0
            for t in unpack_inputs:
                shape = graph_helper.get_op_output_shape(t.op)
                if len(shape) > max_shape:
                    max_shape = len(shape)
            if axis < 0:
                axis += max_shape
            output_names = []
            for i in range(num):
                output_names.append(unpack_op.outputs[i].name)

            unpack_descriptor = UnPackLayerResolver.Descriptor(str(unpack_op.name), consumed_nodes,
                                                                axis, num, unpack_op, output_names)

            descriptors.append(unpack_descriptor)
        return descriptors


class UnpackLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ConcatLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors[:1])
        output_shape = converter_context.graph_helper.get_op_output_shape(descriptor.unpack_op)
        output_shape = output_shape[-4:] if len(output_shape) > 4 else output_shape
        return ir_graph.add(UnpackOp(descriptor.layer_name,
                                     output_dim=output_shape,
                                     axis=descriptor.axis,
                                     num=descriptor.num),
                            input_names=input_name,
                            output_names=descriptor.output_names)
