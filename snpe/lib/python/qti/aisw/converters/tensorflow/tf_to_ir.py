# =============================================================================
#
#  Copyright (c) 2016-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from collections import OrderedDict
import sys
import traceback

import qti.aisw.converters.common.utils.code_to_message as code_to_message
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils.converter_base import ConverterBase, CustomOpBackend
from qti.aisw.converters.common.converter_ir import op_graph_optimizations
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrders

from qti.aisw.converters.tensorflow.loader import ModelLoader
import qti.aisw.converters.tensorflow.layers as layers
from qti.aisw.converters.tensorflow.common import InputLayerDescriptor
from qti.aisw.converters.tensorflow.common import LayerDescriptor
from qti.aisw.converters.tensorflow.graph_matcher import (
    GraphMatcher,
    TFGraphBuilder
)
from qti.aisw.converters.tensorflow.layers.ignored_patterns import IgnoredLayersResolver
from qti.aisw.converters.tensorflow.util import (
    ConverterError,
    GraphHelper,
    uniques
)


class TopologyResolver(object):

    class Topology(object):
        def __init__(self):
            self.inputs = []
            self.outputs = []

    def __init__(self):
        self._descriptor_topology_map = dict()
        """
        :type: dict(LayerDescriptor,TopologyResolver.Topology)
        """
        self._descriptor_ops_map = dict()
        """
        :type: dict(tensorflow.Operation,LayerDescriptor)
        """

    @property
    def descriptor_ops_map(self):
        """ :rtype: descriptor_ops_map """
        return self._descriptor_ops_map

    def resolve_topology(self, descriptors):
        """
        :type descriptors: list(LayerDescriptor)
        :rtype: list(LayerDescriptor)
        """
        self._descriptor_topology_map.clear()
        self._descriptor_ops_map.clear()

        for d in descriptors:
            self._descriptor_topology_map[d.layer_name] = TopologyResolver.Topology()
            for op in d.child_ops:
                self._descriptor_ops_map[op] = d

        for d in descriptors:
            topology = self._descriptor_topology_map[d.layer_name]
            inputs = self._get_input_layers_for(d)
            for i in inputs:
                input_topology = self._descriptor_topology_map[i.layer_name]
                input_topology.outputs.append(d)
            topology.inputs.extend(inputs)

    def get_input_layers_for(self, descriptor):
        return self._descriptor_topology_map[descriptor.layer_name].inputs

    def get_output_layers_for(self, descriptor):
        return self._descriptor_topology_map[descriptor.layer_name].outputs

    def sort_descriptors_in_execution_order(self, _descriptors, _input_descriptors):
        """
        :type _descriptors: list(LayerDescriptor)
        :type _input_descriptors: list(LayerDescriptor)
        :rtype: list(LayerDescriptor)
        """
        sorted_descriptors = []
        queue = list(_input_descriptors)
        visited = set()
        ready = set()
        while len(queue) > 0:
            head = queue.pop(0)
            visited.add(head)
            input_descriptors = self.get_input_layers_for(head)
            if all(i in ready for i in input_descriptors):
                if head in ready:
                    continue

                sorted_descriptors.append(head)
                ready.add(head)

                for o in self.get_output_layers_for(head):
                    if o not in _descriptors or o in visited:
                        continue
                    queue.append(o)
            else:
                for i in input_descriptors:
                    if i in ready or i in visited:
                        continue
                    queue.append(i)
                queue.append(head)

        return sorted_descriptors[len(_input_descriptors):]

    def _get_input_layers_for(self, descriptor):
        """
        :type descriptor: LayerDescriptor
        :rtype: list[LayerDescriptor]
        """
        predecessors = []
        descriptor_input_ops = [op for op in descriptor.child_ops if descriptor.is_input_op(op)]
        for o in descriptor_input_ops:
            q = [t.op for t in o.inputs if descriptor.is_input_tensor(o, t)]
            visited = set()
            while len(q) > 0:
                next_op = q.pop(0)
                if next_op in visited:
                    continue
                visited.add(next_op)

                d = self._descriptor_ops_map.get(next_op, None)
                if d is None:
                    continue

                if d == descriptor:
                    if descriptor.is_input_op(next_op):
                        q = [t.op for t in next_op.inputs if descriptor.is_input_tensor(next_op, t)] + q
                    else:
                        continue
                elif d.is_ignored:
                    q = [t.op for t in next_op.inputs if descriptor.is_input_tensor(next_op, t)] + q
                else:
                    predecessors.append(d)
        return uniques(predecessors)


class ConverterContext(object):
    def __init__(self, converter_model, graph_helper, topology_resolver):
        """
        This class contains state information pertaining a model during conversion.
        It is shared with LayerBuilder instances in order to retrieve layer connectivity, etc.
        :type converter_model: converters.tensorflow.loader.Model
        :type graph_helper: converters.tensorflow.util.GraphHelper
        :type topology_resolver: converters.tensorflow.converter.TopologyResolver
        """
        super(ConverterContext, self).__init__()
        self.__converter_model = converter_model
        self.__graph_helper = graph_helper
        self._topology_resolver = topology_resolver

    @property
    def session(self):
        """ :rtype: tensorflow.Session """
        return self.__converter_model.session

    @property
    def graph(self):
        """ :rtype tensorflow.Graph """
        return self.session.graph

    @property
    def inputs(self):
        """ :rtype: list[converters.tensorflow.loader.Model.Input] """
        return self.__converter_model.inputs

    @property
    def graph_helper(self):
        """ :rtype: converters.tensorflow.util.GraphHelper """
        return self.__graph_helper

    @property
    def topology_resolver(self):
        """ :rtype: converters.tensorflow.converter.TopologyResolver """
        return self._topology_resolver

    def replace_layer_input_with(self, descriptor, old, new):
        inputs = self._topology_resolver.get_input_layers_for(descriptor)
        inputs.remove(old)
        inputs.extend(new)

    def get_input_layer_output_shape_for(self, operation):
        """
        :type operation: tensorflow.Operation
        :rtype: [int]
        """
        output_op = self._get_input_layer_output_op_for(operation)
        return self.__graph_helper.get_op_output_shape(output_op)

    def get_output_tensors_between(self, descriptor_from, descriptor_to):
        """
        :type descriptor_from: LayerDescriptor
        :type descriptor_to: LayerDescriptor
        :rtype: list[tensorflow.Tensor]
        """
        tensors = []
        for o in descriptor_to.child_ops:
            ts = self._get_input_layers_output_tensors_for(o)
            for t in ts:
                d = self._topology_resolver.descriptor_ops_map.get(t.op, None)
                if d == descriptor_from:
                    tensors.append(t)
        return uniques(tensors)

    @classmethod
    def merge_descriptors(cls, source, destination):
        destination.child_ops.extend(source.child_ops)
        source.child_ops = []
        source.set_ignored(True)

    def _get_input_layers_output_tensors_for(self, operation):
        """
        :type operation: tensorflow.Operation
        :rtype: list[tensorflow.Tensor]
        """
        descriptor = self._topology_resolver.descriptor_ops_map.get(operation, None)
        if descriptor is None:
            raise ConverterError('Unable to find input layer for operation not in layer.')

        output_tensors = []

        input_descriptors = self._topology_resolver.get_input_layers_for(descriptor)
        input_descriptors_outputs = [o for d in input_descriptors for o in d.child_ops if d.is_output_op(o)]

        visited = set()
        op_queue = [operation]
        while len(op_queue) > 0:
            next_op = op_queue.pop(0)
            visited.add(next_op)
            for input_tensor in next_op.inputs:
                input_op = input_tensor.op
                if input_op in input_descriptors_outputs:
                    output_tensors.append(input_tensor)
                elif input_op not in visited:
                    op_queue.insert(0, input_op)

        return uniques(output_tensors)

    def _get_input_layer_output_op_for(self, operation):
        """
        :type operation: tensorflow.Operation
        :rtype: tensorflow.Operation
        """
        input_tensors = self._get_input_layers_output_tensors_for(operation)
        ops = uniques([t.op for t in input_tensors])
        if len(ops) == 0:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_INPUT_OPERATION_NOT_FOUND')(operation.name))
        if len(ops) != 1:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_EXPECTED_SINGLE_OUTPUT_FROM_PREVIOUS_LAYER'))
        return ops[0]


class TFConverter(ConverterBase):
    class ArgParser(ConverterBase.ArgParser):
        def __init__(self, backend_framework, **kwargs):
            super(TFConverter.ArgParser, self).__init__("Tensorflow", backend_framework,
                                                        conflict_handler='resolve',
                                                        **kwargs)
            # add command-line options custom to tensorflow converter
            self.parser.add_required_argument('-d', '--input_dim', nargs=2, action='append',
                                              metavar=('INPUT_NAME', 'INPUT_DIM'),
                                              help="The names and dimensions of the network input layers specified "
                                                   "in the format [input_name comma-separated-dimensions], "
                                                   "for example: \n"
                                                   "    'data' 1,224,224,3\n"
                                                   "Note that the quotes should always be included in order to handle"
                                                   "special characters, spaces, etc. \n"
                                                   "For multiple inputs specify multiple --input_dim on the command "
                                                   "line like: \n"
                                                   "    --input_dim 'data1' 1,224,224,3 --input_dim 'data2' 1,50,100,3")
            self.parser.add_required_argument('--out_node', type=str, action='append',
                                              help="Name of the graph\'s output nodes. Multiple output nodes "
                                                   "should be provided separately like:\n"
                                                   "    --out_node out_1 --out_node out_2")
            self.parser.add_optional_argument("--allow_unconsumed_nodes", action="store_true",
                                              help="Uses a relaxed graph node to layer mapping algorithm which may not "
                                                   "use all graph nodes during conversion while retaining structural "
                                                   "integrity. WARNING: This flag is unused and is now the default "
                                                   "behavior. The option will be deprecated in future versions.",
                                              default=False)
            self.parser.add_optional_argument("--show_unconsumed_nodes", action="store_true",
                                              help="Displays a list of unconsumed nodes, if there any are found. Nodes"
                                                   "which are unconsumed do not violate the structural fidelity of the"
                                                   "generated graph.",
                                              default=False)

    def __init__(self, session, args):
        """
        :type session: tf.Session
        :type args: argparse.Namespace
        """
        super(TFConverter, self).__init__(args, axis_order=AxisOrders.TF)
        self._show_unconsumed_nodes = args.show_unconsumed_nodes
        self._input_descriptors = []

        if args.allow_unconsumed_nodes:
            log_warning(code_to_message.get_warning_message('WARNING_TF_ALLOW_UNCONSUMED_OPTION_USED'))

        input_model_path = args.input_network
        (in_nodes, in_dims) = list(zip(*args.input_dim))
        in_types = [type_ for _, type_ in args.input_type]
        self._in_types = dict(args.input_type)
        loader = ModelLoader(session, input_model_path, in_nodes,
                             in_dims, in_types, self.output_nodes)
        self._model = loader.get_model()
        self._ops = self._resolve_graph_operations_from_model(self._model)
        self._graph_helper = GraphHelper(self._model.session, self._model, self._ops)
        self._topology_resolver = TopologyResolver()
        self._context = ConverterContext(self._model, self._graph_helper, self._topology_resolver)

    def convert(self):
        """
        :rtype: IROpGraph
        """
        log_info(code_to_message.get_progress_message('INFO_ALL_BUILDING_NETWORK'))
        if self.custom_op_backend is CustomOpBackend.SNPE_UDO:
            self.populate_udo_collection(self._model, 'tf')
            self._graph_helper.op_collection = self.udo_factory.op_collection
        else:
            self.populate_custom_op_collection(self._model, 'tf')
            self._graph_helper.op_collection = self.custom_op_factory.op_collection
        self._convert_input_layers()
        self._convert_layers()
        return self.graph

    def ir_optimize(self, graph, **kwargs):
        # apply graph transformations
        op_graph_optimizations.apply_graph_optimizations(graph, **kwargs)
        return graph

    def _convert_input_layers(self):
        """
        :rtype: None
        """
        for model_input in self._context.inputs:
            input_operation = self._context.graph.get_operation_by_name(model_input.name)
            shape = self._graph_helper.get_op_output_shape(input_operation)
            if None in shape:
                message = code_to_message.get_error_message('ERROR_TF_UNABLE_TO_RESOLVE_GRAPH_INPUT_DIMS')
                raise ConverterError(message(model_input.name))
            if model_input.shape != shape:
                message = code_to_message.get_error_message('ERROR_TF_UNEXPECTED_INPUT_SHAPE')
                raise ConverterError(message(model_input.shape, shape))

            log_debug(code_to_message.get_progress_message('INFO_TF_BUILDING_INPUT_LAYER')(input_operation.name, shape))

            layer_name = str(input_operation.outputs[0].name)
            layer_type = self._in_types.get(input_operation.name, None)

            descriptor = InputLayerDescriptor(layer_name, [input_operation])
            self._input_descriptors.append(descriptor)
            self._ops.remove(input_operation)

            self.graph.add_input(layer_name, shape, input_type=layer_type)

    def _convert_layers(self):
        """
        :rtype: None
        """
        graph_ops = list(self._ops)
        descriptors = self._resolve_descriptors_from_nodes(graph_ops)
        descriptors = self._resolve_hierarchical_resolution_conflicts(descriptors)
        original_descriptors = descriptors

        self._topology_resolver.resolve_topology(self._input_descriptors + descriptors)
        descriptors = self._topology_resolver.sort_descriptors_in_execution_order(descriptors, self._input_descriptors)
        descriptors = self._filter_disconnected_descriptors(descriptors)

        self._assert_all_ops_supported(original_descriptors, graph_ops)
        self._assert_all_descriptors_consumed(descriptors, original_descriptors)

        self._transform_descriptors(descriptors)
        self._topology_resolver.resolve_topology(self._input_descriptors + descriptors)
        descriptors = [d for d in descriptors if not d.is_ignored]

        self._create_layers(descriptors)

    def _assert_all_ops_supported(self, descriptors, graph_ops):
        graph_ops = self._filter_unconsumed_ops(descriptors, graph_ops)

        def is_parameter_op(o):
            return o.type in ['Const', 'Identity', 'Variable']

        remaining_ops = [op for op in graph_ops if not is_parameter_op(op)]
        if len(remaining_ops) > 0:
            if not self._show_unconsumed_nodes:
                log_warning(code_to_message.get_warning_message('WARNING_UNSUPPORTED_OPS_FOUND'))
            else:
                for op in remaining_ops:
                    log_warning(code_to_message.get_warning_message('WARNING_TF_OP_NOT_SUPPORTED')(op.name, op.type))

    def _assert_all_descriptors_consumed(self, descriptors, original_descriptors):
        unconsumed_descriptors = list(set(original_descriptors) - set(descriptors))
        unconsumed_descriptors = unconsumed_descriptors + [d for d in descriptors if d.is_ignored]
        unconsumed_descriptors = [d for d in unconsumed_descriptors if d.layer_type not in ['Constant', 'IgnoredLayer']]
        if len(unconsumed_descriptors) > 0:
            if not self._show_unconsumed_nodes:
                log_warning(code_to_message.get_warning_message('WARNING_UNCONSUMED_LAYERS'))
            else:
                for d in unconsumed_descriptors:
                    log_warning(code_to_message.get_warning_message('WARNING_TF_LAYER_NOT_CONSUMED')(d.layer_name,
                                                                                                     d.layer_type))

    def _filter_disconnected_descriptors(self, descriptors):
        output_descriptors = [descriptor for op, descriptor in list(self._topology_resolver.descriptor_ops_map.items()) if
                              op.name in self._model.out_nodes_names]
        descriptors_queue = list(set(output_descriptors))
        result = list(output_descriptors)
        while len(descriptors_queue) > 0:
            current_descriptor = descriptors_queue.pop(0)
            inputs = self._topology_resolver.get_input_layers_for(current_descriptor)
            for input_descriptor in inputs:
                if input_descriptor in descriptors and input_descriptor not in result:
                    descriptors_queue.append(input_descriptor)
                    result.append(input_descriptor)
        descriptors_to_ignore = set(descriptors) - set(result)
        for descriptor in descriptors:
            if descriptor in descriptors_to_ignore:
                descriptor.set_ignored(True)
        return descriptors

    def _create_layers(self, descriptors):
        for descriptor in descriptors:
            layer_builder = self._create_layer_builder(descriptor)
            self._create_layer(layer_builder, descriptor)

    def _transform_descriptors(self, descriptors):
        for descriptor in descriptors:
            layer_builder = self._create_layer_builder(descriptor)
            inputs = self._topology_resolver.get_input_layers_for(descriptor)
            outputs = self._topology_resolver.get_output_layers_for(descriptor)
            layer_builder.transform_layer(self.graph, self._context, descriptor, inputs, outputs)

    def _resolve_hierarchical_resolution_conflicts(self, descriptors):
        """
        :type descriptors: list(LayerDescriptor)
        :rtype: list(LayerDescriptor)
        """
        input_ops = set([o for d in self._input_descriptors for o in d.child_ops])
        op_to_descriptor = OrderedDict()
        for d in descriptors:
            for o in d.child_ops:
                if o in input_ops and len(d.child_ops) == 1:
                    continue

                current_descriptor = op_to_descriptor.get(o, None)
                if current_descriptor:
                    if (len(d.child_ops) > len(current_descriptor.child_ops)) or \
                            (len(d.child_ops) == len(current_descriptor.child_ops) and
                             isinstance(current_descriptor, IgnoredLayersResolver.Descriptor)):
                        op_to_descriptor[o] = d
                        for op, descriptor in list(op_to_descriptor.items()):
                            if descriptor == current_descriptor:
                                if o in op_to_descriptor[op].child_ops:
                                    op_to_descriptor[op].child_ops.remove(o)
                                    op_to_descriptor[op].set_ignored(True)
                                    op_to_descriptor[op].layer_name += '_ignored'
                    else:
                        break
                else:
                    op_to_descriptor[o] = d
        return uniques(list(op_to_descriptor.values()))

    @classmethod
    def _filter_unconsumed_ops(cls, descriptors, ops):
        consumed_ops = set()
        for d in descriptors:
            for o in d.child_ops:
                consumed_ops.add(o)
        return list(set(ops) - consumed_ops)

    @classmethod
    def _remove_descriptors_with_removed_ops(cls, _descriptors, ops):
        descriptors = []
        for descriptor in _descriptors:
            do_filter = False
            for op in descriptor.child_ops:
                if op not in ops:
                    do_filter = True
                    break
            if not do_filter:
                descriptors.append(descriptor)
        return descriptors

    def _resolve_descriptors_from_nodes(self, ops):
        """
        :type nodes: list(tf.Operations)
        :rtype: list(LayerDescriptor)
        """
        descriptors = []
        resolvers = self._create_layer_resolvers()

        constructor = TFGraphBuilder(ops)
        constructor.link_nodes()

        graph_matcher = GraphMatcher(constructor.nodes)

        for resolver in resolvers:
            resolved_descriptors = resolver.resolve_layer(graph_matcher, self._graph_helper)
            if len(resolved_descriptors) == 0:
                continue

            resolved_descriptors = self._remove_descriptors_with_removed_ops(resolved_descriptors, ops)

            if resolver.is_final_resolution():
                ops_to_remove = [n for d in resolved_descriptors for n in d.child_ops]
                constructor = TFGraphBuilder([o for o in ops if o not in ops_to_remove])
                constructor.link_nodes()
                graph_matcher = GraphMatcher(constructor.nodes)
            descriptors.extend(resolved_descriptors)
        return descriptors

    def _create_layer_resolvers(self):
        resolvers = list()
        # TODO: Remove after Custom Op API is used by all backends
        if self.config_paths:
            # Udo Layers are placed first to ensure priority resolution
            resolvers.append(layers.UdoLayerResolver())
            # Custom Layers are placed first to ensure priority resolution
            if self.custom_op_backend is not CustomOpBackend.SNPE_UDO:
                resolvers.append(layers.CustomLayerResolver())
        for resolver_class in layers.layer_resolvers:
            resolvers.append(resolver_class())
        return resolvers

    def _create_layer(self, layer_builder, descriptor):
        """
        :type descriptor: converters.tensorflow.common.LayerDescriptor
        :rtype: None
        """
        log_debug(code_to_message.get_progress_message('INFO_ALL_BUILDING_LAYER_W_NODES')(
            descriptor.layer_type, [op.name for op in descriptor.child_ops]))

        inputs = self._topology_resolver.get_input_layers_for(descriptor)
        outputs = self._topology_resolver.get_output_layers_for(descriptor)
        try:
            layer_builder.build_layer(self.graph, self._context, descriptor, inputs, outputs)
        except Exception as e:
            log_error("%".format(layer_builder.__class__.__name__))
            traceback.print_exc(file=sys.stdout)
            raise ConverterError("build_layer failed with Exception %s in layer %s" %
                                 (e,  layer_builder.__class__.__name__))

    @classmethod
    def _create_layer_builder(cls, descriptor):
        builder_class = layers.layer_builders.get(type(descriptor), None)
        if builder_class is None:
            raise ConverterError(code_to_message.get_error_message('ERROR_TF_NO_INPUT_TO_CREATE_LAYER')(type(descriptor)))
        return builder_class()

    @classmethod
    def _resolve_graph_operations_from_model(cls, model):
        """
        :type model: converters.tensorflow.loader.Model
        :rtype: list[tensorflow.Operation]
        """
        operations_map = dict()
        for op in model.session.graph.get_operations():
            operations_map[str(op.name)] = op

        input_ops = set()
        for i in model.inputs:
            input_ops.add(operations_map[i.name])

        all_ops_in_paths = set()
        for output_op_name in model.out_nodes_names:
            queue = [operations_map[output_op_name]]
            visited = set()
            while len(queue) > 0:
                head = queue.pop(0)
                if head in visited:
                    continue
                visited.add(head)

                if head in input_ops:
                    continue

                for t in head.inputs:
                    queue.append(t.op)

            all_ops_in_paths.update(visited)

        return list(all_ops_in_paths)