# =============================================================================
#
#  Copyright (c) 2018-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import numpy as np

from qti.aisw.converters.common.converter_ir.op_adapter import PowerOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class PowLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, power, scale, shift, output_names=None):
            super(PowLayerResolver.Descriptor, self).__init__('Pow', name, nodes, output_names=output_names)
            self.power = power
            self.scale = scale
            self.shift = shift

    def __init__(self):
        sequence_scalar_pow = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('pow', ['Pow']),
            NonConsumableConverterSequenceNode('const', ['Const'])
        ])
        sequence_scalar_pow.set_inputs('pow', ['input', 'const'])
        sequence_scalar_pow.set_outputs(['pow'])

        self.sequences = [sequence_scalar_pow]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            for match in graph_matcher.match_sequence(sequence):
                pow_op = match['pow']
                const_op = match['const']
                const_tensor, const_tensor_shape, const_consumed_nodes = \
                    graph_helper.get_static_data_info(pow_op, const_op.outputs[0])

                if type(const_tensor) is not np.ndarray and not const_tensor_shape:
                    const_tensor = np.asarray(const_tensor)
                    const_tensor.shape = [1]

                pow_descriptor = PowLayerResolver.Descriptor(
                    str(pow_op.name), const_consumed_nodes + match.consumed_nodes, const_tensor, 1, 0,
                    output_names=[str(pow_op.outputs[0].name)])
                descriptors.append(pow_descriptor)

        return descriptors


class PowLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: PowLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(PowerOp(name=descriptor.layer_name,
                                    power=descriptor.power,
                                    scale=descriptor.scale,
                                    shift=descriptor.shift),
                            input_names=input_name,
                            output_names=output_name)
