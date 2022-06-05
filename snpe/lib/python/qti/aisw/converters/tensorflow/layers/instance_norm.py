# =============================================================================
#
#  Copyright (c) 2018-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np

from qti.aisw.converters.common.converter_ir.op_adapter import BatchnormOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)


class InstanceNormLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, shape, epsilon):
            super(InstanceNormLayerResolver.Descriptor, self).__init__('InstanceNorm', name, operations)
            self.shape = shape
            # SNPE runtime algo is y = x * WEIGHT / rms + BIAS
            # While L2 Normalization is y = x / rms
            # That requires WEIGHT = 1.0 and BIAS = 0.0 to mimic L2 Norm in SNPE
            # Shape of weights/biases should be same as the last dimension of input.
            self.weights = np.ones(shape[-1])
            self.biases = np.zeros(shape[-1])
            self.epsilon = epsilon

    def __init__(self):
        self.sequence1 = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('StopGradient', ['StopGradient']),
            ConverterSequenceNode('SquaredDifference', ['SquaredDifference']),
            ConverterSequenceNode('variance', ['Mean']),
            ConverterSequenceNode('add', ['Add']),
            ConverterSequenceNode('mean', ['Mean']),
            NonConsumableConverterSequenceNode('gamma', ['Identity']),
            ConverterSequenceNode('Rsqrt', ['Rsqrt']),
            ConverterSequenceNode('mul_2', ['Mul']),
            NonConsumableConverterSequenceNode('beta', ['Identity']),
            ConverterSequenceNode('mul', ['Mul']),
            ConverterSequenceNode('sub', ['Sub']),
            ConverterSequenceNode('mul_1', ['Mul']),
            ConverterSequenceNode('add_1', ['Add']),
            NonConsumableConverterSequenceNode('epsilon', ['?']),
            NonConsumableConverterSequenceNode('stub_14', ['?']),
            NonConsumableConverterSequenceNode('stub_15', ['?']),
            NonConsumableConverterSequenceNode('stub_16', ['?']),
            NonConsumableConverterSequenceNode('stub_17', ['?']),
        ])
        self.sequence1.set_inputs('variance', ['SquaredDifference', 'stub_14'])
        self.sequence1.set_inputs('StopGradient', ['mean'])
        self.sequence1.set_inputs('add', ['variance', 'epsilon'])
        self.sequence1.set_inputs('sub', ['beta', 'mul_2'])
        self.sequence1.set_inputs('mean', ['input', 'stub_15'])
        self.sequence1.set_inputs('gamma', ['stub_16'])
        self.sequence1.set_inputs('mul_2', ['mean', 'mul'])
        self.sequence1.set_inputs('Rsqrt', ['add'])
        self.sequence1.set_inputs('beta', ['stub_17'])
        self.sequence1.set_inputs('mul', ['Rsqrt', 'gamma'])
        self.sequence1.set_inputs('add_1', ['mul_1', 'sub'])
        self.sequence1.set_inputs('mul_1', ['input', 'mul'])
        self.sequence1.set_inputs('SquaredDifference', ['input', 'StopGradient'])
        self.sequence1.set_outputs(['add_1'])

        self.sequence2 = GraphSequence([
            NonConsumableConverterSequenceNode('input', ['?']),
            ConverterSequenceNode('mean', ['?']),
            ConverterSequenceNode('StopGradient', ['StopGradient']),
            ConverterSequenceNode('sub', ['Sub']),
            ConverterSequenceNode('SquaredDifference', ['SquaredDifference']),
            ConverterSequenceNode('variance', ['Mean']),
            ConverterSequenceNode('add', ['Add']),
            ConverterSequenceNode('sqrt', ['Sqrt']),
            ConverterSequenceNode('real_div', ['RealDiv']),

            ConverterSequenceNode('epsilon', ['?']),
            NonConsumableConverterSequenceNode('mean_reduction_indices_1', ['?']),
            NonConsumableConverterSequenceNode('mean_reduction_indices_2', ['?']),
            NonConsumableConverterSequenceNode('variance_reduction_indices', ['?']),
            NonConsumableConverterSequenceNode('stub_1', ['?'])
        ])
        self.sequence2.set_inputs('mean', ['input', 'mean_reduction_indices_1'])
        self.sequence2.set_inputs('sub', ['stub_1', 'mean'])
        self.sequence2.set_inputs('StopGradient', ['mean'])
        self.sequence2.set_inputs('SquaredDifference', ['input', 'StopGradient'])
        self.sequence2.set_inputs('variance', ['SquaredDifference', 'mean_reduction_indices_2'])
        self.sequence2.set_inputs('add', ['variance', 'epsilon'])
        self.sequence2.set_inputs('sqrt', ['add'])
        self.sequence2.set_inputs('real_div', ['sub', 'sqrt'])
        self.sequence2.set_outputs(['real_div'])

        self.sequences = [self.sequence1, self.sequence2]

    def is_final_resolution(self):
        return True

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                bn_op = match['SquaredDifference']
                input_op = match['input']

                shape = graph_helper.get_op_output_shape(input_op)
                add_op = match['epsilon']
                consumed_nodes = match.consumed_nodes
                potential_descriptors.append(InstanceNormLayerResolver.Descriptor(str(bn_op.name),
                                                                                  consumed_nodes,
                                                                                  shape=shape,
                                                                                  epsilon=add_op.get_attr('value').float_val[0]))
        return potential_descriptors


class InstanceNormLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: InstanceNormLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)

        return ir_graph.add(BatchnormOp(descriptor.layer_name,
                                        descriptor.weights,
                                        descriptor.biases,
                                        compute_statistics=True,
                                        use_mu_sigma=True,
                                        across_spatial=True,
                                        epsilon=descriptor.epsilon),
                            input_names=input_name,
                            output_names=descriptor.output_names[0])
