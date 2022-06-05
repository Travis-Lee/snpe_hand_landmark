# =============================================================================
#
#  Copyright (c) 2015-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
from qti.aisw.converters.common.converter_ir.op_adapter import ConstantOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.util import GraphHelper
from qti.aisw.converters.tensorflow.graph_matcher import(
    ConverterSequenceNode,
    GraphSequence
)


class FillLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, shape, scalar):
            super(FillLayerResolver.Descriptor, self).__init__('Fill', name, nodes)
            self.shape = shape
            self.scalar = scalar

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Fill'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            fill_op = match['root']
            consumed_nodes = match.consumed_nodes
            shape_tensor, scalar_tensor = GraphHelper.get_op_input_tensors(fill_op, ('?', 'Const'))
            shape = graph_helper.evaluate_tensor_output(shape_tensor).tolist()
            while len(shape) > 4:
                shape = shape[1:]

            while len(shape) < 4:
                shape = [1] + shape
            scalar = graph_helper.evaluate_tensor_output(scalar_tensor)

            d = FillLayerResolver.Descriptor(str(fill_op.name), consumed_nodes, shape, scalar)
            descriptors.append(d)

        return descriptors


class FillLayerBuilder(LayerBuilder):

    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: FillLayerResolver.Descriptor
        :rtype: int
        """
        tensor = np.zeros(descriptor.shape, dtype=np.float32)
        tensor[...] = descriptor.scalar
        return ir_graph.add(ConstantOp(descriptor.output_names[0],
                                       tensor),
                            [],
                            descriptor.output_names)
