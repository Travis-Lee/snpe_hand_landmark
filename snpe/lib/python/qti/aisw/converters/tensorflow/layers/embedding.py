# =============================================================================
#
#  Copyright (c) 2015-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.utils.code_to_message import get_error_message
from qti.aisw.converters.common.converter_ir.op_adapter import EmbeddingOp
from qti.aisw.converters.common.converter_ir.op_adapter import GatherOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerBuilder, LayerResolver
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
from abc import ABCMeta
from abc import abstractmethod
from qti.aisw.converters.tensorflow.layers.constant import ConstantLayerResolver
from qti.aisw.converters.tensorflow.util import GraphHelper
from qti.aisw.converters.tensorflow.util import TensorNotFoundError, ConverterError


class GatherBaseResolver(LayerResolver, object):
    __metaclass__ = ABCMeta

    class Descriptor(LayerDescriptor):
        def __init__(self, layer_type, name, nodes, input_names, output_names=None):
            super(GatherBaseResolver.Descriptor, self).__init__(layer_type, name, nodes, output_names=output_names)
            self.input_names = input_names

    def __init__(self, layer_type, descriptor_class):
        self._layer_type = layer_type
        self._descriptor_class = descriptor_class
        sequence_1 = GraphSequence([
            ConverterSequenceNode('root', ['GatherV2']),
            NonConsumableConverterSequenceNode('params', ['?']),
            NonConsumableConverterSequenceNode('axis', ['?']),
            NonConsumableConverterSequenceNode('indices', ['?'])
        ])
        sequence_1.set_inputs('root', ['params', 'axis', 'indices'])
        sequence_1.set_outputs(['root'])

        # Filter seqs 2
        sequence_2 = GraphSequence([
            ConverterSequenceNode('root', ['Gather']),
            NonConsumableConverterSequenceNode('params', ['?']),
            NonConsumableConverterSequenceNode('indices', ['?'])
        ])
        sequence_2.set_inputs('root', ['params', 'indices'])
        sequence_2.set_outputs(['root'])

        self.sequences = [sequence_1, sequence_2]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            for match in graph_matcher.match_sequence(sequence):
                gather_op = match['root']
                consumed_nodes = match.consumed_nodes
                indices, params, axis = self.get_tensors(graph_helper, gather_op)
                params, const_params_consumed_ops = graph_helper.get_none_identity_input(params)
                indices, const_indices_consumed_ops = graph_helper.get_none_identity_input(indices)
                input_names = []
                if self._layer_type == 'Gather':
                    # at this point, we depend on the name of gather op to determine the real type for the op,
                    # i.e. embedding or gather, this is possible only if name attribute is not set for
                    # tf.nn.embedding_lookup or tf.gather
                    if gather_op.name == 'embedding_lookup' or gather_op.name.endswith("embedding_lookup"):
                        continue
                    # for gather, params come firstly.
                    input_names.extend([GraphHelper.indexed_tensor_name(params.op.name),
                                        GraphHelper.indexed_tensor_name(indices.op.name)])
                    descriptor = self._descriptor_class(str(gather_op.name), consumed_nodes,
                                                        input_names, axis, [gather_op.outputs[0].name])
                else:
                    if gather_op.name == 'GatherV2' or gather_op.name.endswith("GatherV2") \
                            or gather_op.name.endswith("Gather"):
                        continue
                    # for embedding, indices come firstly.
                    # Also, currently we only support params length with 1, the case which has the simplest pattern of
                    # nodes. For params length > 1, new sequence is needed.
                    input_names.extend([GraphHelper.indexed_tensor_name(indices.op.name),
                                        GraphHelper.indexed_tensor_name(params.op.name)])
                    output_dim = graph_helper.get_op_output_shape(gather_op)
                    descriptor = self._descriptor_class(str(gather_op.name), consumed_nodes,
                                                        input_names, output_dim)

                descriptors.append(descriptor)

                if indices.op.type == 'Const':
                    const_indices_shape = GraphHelper.get_tensor_output_shape(indices)
                    const_indices_val = graph_helper.evaluate_tensor_output(indices)
                    const_indices_descriptor = ConstantLayerResolver.Descriptor(str(indices.op.name),
                                                                                const_indices_consumed_ops,
                                                                                const_indices_val, const_indices_shape,
                                                                                descriptor)
                    descriptors.append(const_indices_descriptor)

                if params.op.type == 'Const':
                    const_shape = GraphHelper.get_tensor_output_shape(params)
                    const_val = graph_helper.evaluate_tensor_output(params)
                    const_descriptor = ConstantLayerResolver.Descriptor(str(params.op.name),
                                                                        const_params_consumed_ops,
                                                                        const_val, const_shape,
                                                                        descriptor)
                    descriptors.append(const_descriptor)
        return descriptors

    @classmethod
    def get_tensors(cls, graph_helper, gather_op):
        try:
            indices, params, axis = GraphHelper.get_op_input_tensors(gather_op, ('?', '?', 'Const'))
        except TensorNotFoundError:
            indices, params = GraphHelper.get_op_input_tensors(gather_op, ('?', '?'))
            axis = 0
        indices_shape = graph_helper.get_op_output_shape(indices.op)
        params_shape = graph_helper.get_op_output_shape(params.op)
        if not isinstance(axis, int):
            if len(GraphHelper.get_tensor_output_shape(axis)) == 0:
                pass
            elif len(indices_shape) == 0 and indices.op.type == 'Const':
                indices, axis = axis, indices
            elif len(params_shape) == 0 and params.op.type == 'Const':
                params, axis = axis, params
            axis = graph_helper.evaluate_tensor_output(axis)

        if (indices.dtype.name == 'float32' or indices.dtype.name == 'float64') \
                and (params.dtype.name == 'int32' or params.dtype.name == 'int64'):
            indices, params = params, indices
        return indices, params, axis


class GatherBaseBuilder(LayerBuilder):
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        pass


class GatherLayerResolver(GatherBaseResolver):
    class Descriptor(GatherBaseResolver.Descriptor):
        def __init__(self, name, nodes, input_names, axis, output_names=None):
            super(GatherLayerResolver.Descriptor, self).__init__('Gather', name, nodes, input_names, output_names)
            self.axis = axis

    def __init__(self):
        super(GatherLayerResolver, self).__init__('Gather', GatherLayerResolver.Descriptor)


class GatherLayerBuilder(GatherBaseBuilder):
    def transform_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        const_indices = [d for d in input_descriptors if (d.layer_name+":0") == descriptor.input_names[1] and isinstance(d, ConstantLayerResolver.Descriptor)]
        if len(const_indices) != 0:
            for d in const_indices:
                d.set_quantizable(False)

    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EmbeddingLayerResolver.Descriptor
        :rtype: int
        """
        output_name = descriptor.output_names[0]
        return ir_graph.add(GatherOp(descriptor.layer_name,
                                     axis=descriptor.axis),
                            descriptor.input_names,
                            output_name)


class EmbeddingLayerResolver(GatherBaseResolver):
    class Descriptor(GatherBaseResolver.Descriptor):
        def __init__(self, name, nodes, input_names, output_dim):
            super(EmbeddingLayerResolver.Descriptor, self).__init__('Embedding', name, nodes, input_names)
            self.output_dim = output_dim

    def __init__(self):
        super(EmbeddingLayerResolver, self).__init__('Embedding', EmbeddingLayerResolver.Descriptor)


class EmbeddingLayerBuilder(GatherBaseBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: EmbeddingLayerResolver.Descriptor
        :rtype: int
        """
        if len(input_descriptors) != 2:
            raise ConverterError(get_error_message('ERROR_TF_EMBEDDING_REQUIRES_2_INPUTS'))

        output_name = descriptor.output_names[0]
        return ir_graph.add(EmbeddingOp(descriptor.layer_name,
                                        output_dim=descriptor.output_dim),
                            input_names=descriptor.input_names,
                            output_names=output_name)

