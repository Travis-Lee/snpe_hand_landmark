# =============================================================================
#
#  Copyright (c) 2015-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.converters.common.converter_ir.op_adapter import ReshapeOp
from qti.aisw.converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from qti.aisw.converters.tensorflow.layers.fullyconnected import FullyConnectedLayerResolver
from qti.aisw.converters.tensorflow.layers.constant import ConstantLayerResolver
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)
from functools import reduce
import numpy as np

class ReshapeLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, reshape_op):
            super(ReshapeLayerResolver.Descriptor, self).__init__('Reshape', name, nodes)
            self.reshape_op = reshape_op

    def __init__(self):
        sequence_reshape = GraphSequence([ConverterSequenceNode('root', ['Reshape', 'Squeeze', 'ExpandDims'])])
        sequence_reshape.set_outputs(['root'])

        sequence2 = GraphSequence([
            ConverterSequenceNode('slice', ['Slice']),
            NonConsumableConverterSequenceNode('input', ['Identity', 'Const']),
            NonConsumableConverterSequenceNode('offsets', ['Identity', 'Const']),
            NonConsumableConverterSequenceNode('size', ['Identity', 'Const']),
            NonConsumableConverterSequenceNode('shape', ['Identity', 'Const']),
            ConverterSequenceNode('root', ['Reshape']),
        ])
        sequence2.set_inputs('slice', ['input', 'offsets', 'size'])
        sequence2.set_inputs('root', ['slice', 'shape'])
        sequence2.set_outputs(['root'])

        self.sequences = [sequence_reshape, sequence2]

    @staticmethod
    def _is_const_reshape_input(graph_helper, input_tensor, tensor):
        """ This function determines if the data (non-shape) input to Reshape is Const """

        if input_tensor.op.type != 'Const':
            return False

        input_tensor_value = graph_helper.evaluate_tensor_output(input_tensor).tolist()
        tensor_shape = graph_helper.get_op_output_shape(tensor)
        # its a shape input if -1 or 0 in value or if the output of reshape is the same as the value of the
        # the shape tensor
        if tensor_shape == input_tensor_value or ((np.shape(tensor_shape) == np.shape(input_tensor_value)) and
                                                  ((-1 in input_tensor_value) or
                                                  (0 in input_tensor_value))):
            return False

        return True

    @staticmethod
    def _is_shape_sequence(graph_helper, input_tensor, output_tensor):
        """ This function determines if the shape input to Reshape is a sequence instead of just Const value
            This sequence will be used to create a Constant descriptor
         """

        if input_tensor.op.type == 'Const':
            return False

        input_tensor_value = graph_helper.evaluate_tensor_output(input_tensor).tolist()
        tensor_shape = graph_helper.get_op_output_shape(output_tensor)
        # its a shape input if -1 or 0 in value or if the output of reshape is the same as the value of the
        # the shape tensor
        if tensor_shape == input_tensor_value or ((np.shape(tensor_shape) == np.shape(input_tensor_value)) and
                                                  ((-1 in input_tensor_value) or
                                                  (0 in input_tensor_value))):
            return True

        return False

    @staticmethod
    def _get_const_descriptor(graph_helper, input_tensor, consumed_nodes, consumer_descriptor):
        const_value = graph_helper.evaluate_tensor_output(input_tensor)
        const_shape = graph_helper.get_op_output_shape(input_tensor)
        return ConstantLayerResolver.Descriptor(str(input_tensor.op.name),
                                                consumed_nodes,
                                                const_value,
                                                const_shape,
                                                consumer_descriptor)

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:

                reshape_op = match['root']
                consumed_nodes = match.consumed_nodes
                slice_op = None

                try:
                    slice_op = match['slice']
                except:
                    pass

                if slice_op is None:
                    reshape_descriptor = ReshapeLayerResolver.Descriptor(str(reshape_op.name),
                                                                     consumed_nodes,
                                                                     reshape_op)
                    if reshape_op.type == "Reshape":
                        for reshape_input in reshape_op.inputs:
                            # Create constant descriptor is the non-shape input to reshape is Static. This will later
                            # be used in transform layer to squash reshape. Note: we are not squashing here since we
                            # do not have the output descriptor yet.
                            # e.g. Constant -> Identity -> Reshape
                            input_, consumed_nodes = graph_helper.get_none_identity_input(reshape_input)
                            if self._is_const_reshape_input(graph_helper, input_, reshape_op.outputs[0]):
                                descriptors.append(self._get_const_descriptor(graph_helper, input_, consumed_nodes,
                                                                              reshape_descriptor))
                            # Create constant descriptor for Shape input to Reshape since we only support static Shape
                            elif self._is_shape_sequence(graph_helper, reshape_input, reshape_op.outputs[0]):
                                _, _, consumed_nodes = graph_helper.get_static_data_info(reshape_op, reshape_input)
                                if consumed_nodes:
                                    # Add the shape sequence nodes at the beginning and keep the original sequence
                                    # nodes at the end. This way the output name for the reshape descriptor will
                                    # be based on the root node (child_ops[-1)
                                    const_desc = self._get_const_descriptor(graph_helper,
                                                                            reshape_input,
                                                                            consumed_nodes,
                                                                            reshape_descriptor)
                                    const_desc.set_ignored(True)
                                    descriptors.append(const_desc)

                    descriptors.append(reshape_descriptor)
                else:
                    input_value = graph_helper.evaluate_tensor_output(reshape_op.outputs[0])
                    input_shape = graph_helper.get_op_output_shape(reshape_op)
                    const_desc = ConstantLayerResolver.Descriptor(reshape_op.name, consumed_nodes, input_value, input_shape,None)
                    descriptors.append(const_desc)


        return descriptors


class ReshapeLayerBuilder(LayerBuilder):
    def build_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type ir_graph: converters.common.converter_ir.op_graph.IROpGraph
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReshapeLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors[:1])
        output_shape = converter_context.graph_helper.get_op_output_shape(descriptor.reshape_op)
        output_shape = output_shape[-4:] if len(output_shape) > 4 else output_shape
        return ir_graph.add(ReshapeOp(descriptor.output_names[0],
                                      output_shape),
                            input_names=input_name,
                            output_names=descriptor.output_names[0])

    def transform_layer(self, ir_graph, converter_context, descriptor, input_descriptors, output_descriptors):
        fc_outputs = [d for d in output_descriptors if isinstance(d, FullyConnectedLayerResolver.Descriptor)]
        if len(output_descriptors) == 1 and fc_outputs == output_descriptors:
            # Only make this optimization if the batch dimension is maintained through the reshape.
            # If not maintained, then reshape op is needed.
            non_ignored_inputs = [d for d in input_descriptors if not d.is_ignored]
            if len(non_ignored_inputs) == 1:
                tensors = converter_context.get_output_tensors_between(non_ignored_inputs[0],
                                                                       descriptor)
                input_batch = converter_context.graph_helper.get_op_output_shape(tensors[0].op)[0]
                output_batch = converter_context.graph_helper.get_op_output_shape(
                    descriptor.reshape_op)[0]
                if input_batch == output_batch:
                    converter_context.merge_descriptors(descriptor, fc_outputs[0])
                    return

        non_ignored_inputs = [d for d in input_descriptors if not d.is_ignored]
        if len(non_ignored_inputs) == 1:
            tensors = converter_context.get_output_tensors_between(non_ignored_inputs[0], descriptor)
            input_shape = converter_context.graph_helper.get_op_output_shape(tensors[0].op)
            output_shape = converter_context.graph_helper.get_op_output_shape(descriptor.child_ops[0])
            if input_shape == output_shape:
                # Input and output of Reshape is same, hence it has no effect on the output and can be removed.
                # Merge the reshape descriptor into the input descriptor
                converter_context.merge_descriptors(descriptor, non_ignored_inputs[0])
            elif isinstance(non_ignored_inputs[0], ConstantLayerResolver.Descriptor):
                const_desc = non_ignored_inputs[0]
                const_desc.value = np.reshape(const_desc.value, output_shape)
                const_desc.shape = output_shape
                converter_context.merge_descriptors(descriptor, const_desc)
        elif len(non_ignored_inputs) == 0:
            # only set descriptor as ignored if there no inputs to ignored op
            is_input_independent = [len(d.child_ops[0].inputs) == 0 for d in input_descriptors if d.child_ops]
            if is_input_independent and all(is_input_independent):
                descriptor.set_ignored(True)
