# =============================================================================
#
#  Copyright (c) 2018-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.common.converter_ir.op_adapter import ElementwiseUnarySqrtOp, ElementwiseUnaryAbsOp, \
    ElementwiseUnaryFloorOp, ElementwiseUnaryExpOp
from abc import ABCMeta
from abc import abstractmethod
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class EltWiseUnaryLayerResolver(LayerResolver, object):
    __metaclass__ = ABCMeta

    def __init__(self, layer_type, op_type, descriptor_class):
        super(EltWiseUnaryLayerResolver, self).__init__()
        self._layer_type = layer_type
        self._op_type = op_type
        self._descriptor_class = descriptor_class

        self.sequence = GraphSequence([
            ConverterSequenceNode('root', [self._op_type]),
            NonConsumableConverterSequenceNode('input1', ['?']),
        ])
        self.sequence.set_inputs('root', ['input1'])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        non_const_input_sequences = [self.sequence]
        for sequence in non_const_input_sequences:
            for match in graph_matcher.match_sequence(sequence):
                eltwise_op = match['root']
                descriptor = self._descriptor_class(self._layer_type, str(eltwise_op.name), match.consumed_nodes)
                descriptors.append(descriptor)

        return descriptors


class EltWiseUnaryLayerBuilder(LayerBuilder):
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        pass


class EltWiseUnarySqrtLayerResolver(EltWiseUnaryLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseUnarySqrtLayerResolver, self).__init__('ElementWiseUnarySqrt', 'Sqrt',
                                                            EltWiseUnarySqrtLayerResolver.Descriptor)


class EltWiseUnarySqrtLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseUnarySqrtLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(ElementwiseUnarySqrtOp(descriptor.layer_name),
                            input_name,
                            output_name)


class EltWiseUnaryAbsLayerResolver(EltWiseUnaryLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseUnaryAbsLayerResolver, self).__init__('ElementWiseUnaryAbs', 'Abs',
                                                           EltWiseUnaryAbsLayerResolver.Descriptor)


class EltWiseUnaryAbsLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseUnaryAbsLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(ElementwiseUnaryAbsOp(descriptor.layer_name),
                            input_name,
                            output_name)


class EltWiseUnaryFloorLayerResolver(EltWiseUnaryLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseUnaryFloorLayerResolver, self).__init__('ElementWiseUnaryFloor', 'Floor',
                                                             EltWiseUnaryFloorLayerResolver.Descriptor)


class EltWiseUnaryFloorLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseDivLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(ElementwiseUnaryFloorOp(descriptor.layer_name),
                            input_name,
                            output_name)


class EltWiseUnaryExpLayerResolver(EltWiseUnaryLayerResolver):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        super(EltWiseUnaryExpLayerResolver, self).__init__('ElementWiseUnaryExp', 'Exp',
                                                             EltWiseUnaryExpLayerResolver.Descriptor)


class EltWiseUnaryExpLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EltWiseUnaryExpLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return ir_graph.add(ElementwiseUnaryExpOp(descriptor.layer_name),
                            input_name,
                            output_name)