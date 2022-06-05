# ==============================================================================
#
#  Copyright (c) 2018-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import inspect
import json
import numpy as np

from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils import code_to_message, translation_utils


class InputOpNode(object):
    def __init__(self, name, shape, axis_format=None):
        self.name = name
        self.shape = shape
        self.axis_format = axis_format


class OpNode(object):
    def __init__(self, op, input_names, output_names, axis_order=None):
        self.op = op
        self.input_names = input_names
        self.output_names = output_names
        self.axis_order = axis_order

    def is_equal(self, other_node):
        if not isinstance(other_node, self.__class__):
            return False
        node_vars = dict(self.__dict__)
        other_node_vars = dict(other_node.__dict__)
        for var in list(node_vars.keys()):
            if node_vars[var] != other_node_vars[var]:
                return False
        return True

    def encode(self, graph):
        data = {"name": self.op.name, "type": self.op.type, "inputs": self.input_names}
        output_dict = {}
        for name in self.output_names:
            output_dict[name] = list(graph.get_buffer(name).shape)
        data["outputs"] = output_dict
        return data


class Buffer(object):
    def __init__(self, name, shape, producer, axis_format=None):
        if not isinstance(shape, list):
            raise TypeError("Shape %s needs to be a list" % shape)
        self.name = name
        self.producer = producer
        self.consumers = set()
        self.shape = shape
        if axis_format is None:
            self.axis_format = AxisTracker.AxisFormat.NOT_YET_DEFINED
        else:
            self.axis_format = axis_format

    def rank(self):
        return len(self.shape)

    def get_buf_dims(self):
        return self.shape

    def get_axis_format(self):
        return self.axis_format

    def populate_axis_format(self, axis_order):
        self.axis_format = axis_order.get_axis_format(self.rank())

    def get_axis_annotations(self):
        """Translate AxisFormat enum to modeltools axis order list"""
        if self.axis_format == 'NSC':
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.CHANNEL,
                    AxisTracker.AxisAnnotations.HEIGHT, AxisTracker.AxisAnnotations.WIDTH]
        if self.axis_format == 'NCS':
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.HEIGHT,
                    AxisTracker.AxisAnnotations.WIDTH, AxisTracker.AxisAnnotations.CHANNEL]
        elif self.axis_format == 'FEATURE':
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.FEATURE]
        elif self.axis_format == 'BTF':
            return [AxisTracker.AxisAnnotations.BATCH, AxisTracker.AxisAnnotations.TIME,
                    AxisTracker.AxisAnnotations.FEATURE]
        elif self.axis_format == 'TBF':
            return [AxisTracker.AxisAnnotations.TIME, AxisTracker.AxisAnnotations.BATCH,
                    AxisTracker.AxisAnnotations.FEATURE]
        elif self.axis_format == 'NONTRIVIAL':
            return [AxisTracker.AxisAnnotations.NONTRIVIAL]
        elif self.axis_format == 'ANY':
            return [AxisTracker.AxisAnnotations.ANY for _ in range(len(self.shape))]
        else:
            raise ValueError("Encountered unexpected axis format for get_axis_order: %s" % self.axis_format)


class BufferCriteria(object):
    """
    Class(enum) to use for setting buffer criteria on inputs/outputs for validating matched node sequences
    """
    # to be used for individual buffers
    ALL = "ALL"  # all the buffer(s) must be this same expected op_type
    ANY = "ANY"  # There can be one or more of this op_type as buffer(s)
    NONE = "NONE"  # None of the buffer(s) should be of this type

    # to be used for set of buffers
    MATCH_NUM_BUFS = "MATCH_NUM_BUFS"  # the expected number of buffers must be same length as matched buffers
    FLEXIBLE_NUM_BUFS = "FLEXIBLE_NUM_BUFS"  # the expected number of buffers doesnt need to be equal to matched buffers


class InputType(object):
    """
    Contains supported input types. This will be used by DSP to determine quantization
    """
    IMAGE = "image"  # input is float between 0-255 and the input's mean is 0.0f and the input's max is 255.0f
    DEFAULT = "default"  # pass the input as floats to the dsp directly and the DSP will quantize it
    OPAQUE = "opaque"  # assumes input is float because the consumer layer(i.e next layer) requires it as float,
    # therefore it won't be quantized by DSP

    @classmethod
    def get_supported_types(cls):
        return [cls.IMAGE, cls.DEFAULT, cls.OPAQUE]

    @classmethod
    def is_valid_type(cls, input_type):
        return input_type in cls.get_supported_types()


class InputEncodings(object):
    """
    Contains supported input encodings
    """
    BGR = "bgr"
    RGB = "rgb"
    RGBA = "rgba"
    ARGB32 = "argb32"
    NV21 = "nv21"
    TIME_SERIES = "time_series"
    OTHER = "other"

    @classmethod
    def get_supported_encodings(cls):
        return [cls.BGR, cls.RGB, cls.RGBA, cls.ARGB32, cls.NV21, cls.TIME_SERIES, cls.OTHER]

    @classmethod
    def is_valid_encoding(cls, input_encoding):
        return input_encoding in cls.get_supported_encodings()


class QuantParams(object):
    """
    Contains supported quantization params
    """
    BN_PARAMS = "bn_params"
    OUTPUT_ENCODINGS = "output_encodings"
    PARAM_ENCODINGS = "param_encodings"

    @classmethod
    def get_supported_quant_params(cls):
        return [cls.BN_PARAMS, cls.OUTPUT_ENCODINGS, cls.PARAM_ENCODINGS]

    @classmethod
    def is_valid_quant_param(cls, param):
        return param in cls.get_supported_quant_params()


class IROpGraph(object):
    def __init__(self,
                 naming_policy,
                 shape_inference_policy,
                 input_types,
                 input_encodings,
                 src_axis_order,
                 output_nodes=[]):
        self.naming_policy = naming_policy
        self.shape_inference_policy = shape_inference_policy
        self.src_axis_order = src_axis_order
        self.inputs_type_dict = self._create_input_types_dict(input_types)
        self.inputs_encoding_dict = self._create_input_encodings_dict(input_encodings)
        self.nodes_by_name = {}
        self.nodes_in_order = []
        self.buffers = {}
        self.output_nodes = output_nodes
        self.quantization_params = {}

    def __iter__(self):
        return iter(self.nodes_in_order)

    def dump_json(self, filename):
        graph = self

        class Encoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, OpNode):
                    return obj.encode(graph)
                elif isinstance(obj, np.int32):
                    return int(obj)
                else:
                    # Let the base class default method raise the TypeError
                    return json.JSONEncoder.default(self, obj)

        if filename[-5:] != ".json":
            filename = filename + ".json"
        with open(filename, "w") as f:
            json.dump(self.nodes_by_name, f, cls=Encoder,
                      indent=4, separators=(',', ': '))

    @staticmethod
    def _create_input_types_dict(input_types):
        log_assert(all(InputType.is_valid_type(type_) for _, type_ in input_types),
                   code_to_message.get_error_message("ERROR_UNSUPPORTED_INPUT_TYPE")(InputType.get_supported_types()))
        return {input_name: input_type for input_name, input_type in input_types}

    @staticmethod
    def _create_input_encodings_dict(input_encodings):
        log_assert(all(InputEncodings.is_valid_encoding(encoding) for _, encoding in input_encodings),
                   code_to_message.get_error_message("ERROR_UNSUPPORTED_INPUT_ENCODING")
                   (InputEncodings.get_supported_encodings()))
        return {input_name: input_encoding for input_name, input_encoding in input_encodings}

    def get_input_type(self, input_name):
        # use input_type: default as the default for all inputs
        return self.inputs_type_dict.get(input_name, InputType.DEFAULT)

    def get_input_encoding(self, input_name):
        # use input_encoding: bgr as the default for all inputs
        return self.inputs_encoding_dict.get(input_name, InputEncodings.BGR)

    def add_quantization_params(self, op_name, **kwargs):
        """
        Adds quantization params to an IR graph object for a given op_name. The dictionary provided
        is expected to contain one/all of output_encodings, param_encodings or bn_params as a key(s).

        :param op_name: The name of the op whose quantization params will be added.
        :param kwargs: The dictionary containing the output encodings, param encodings and bn_params for that op
        :raises: An assertion error if the quantization params are not valid.
        """
        log_assert(all(QuantParams.is_valid_quant_param(param) for param, _ in kwargs.items()),
                   code_to_message.get_error_message("ERROR_UNSUPPORTED_QUANT_PARAM")
                   (QuantParams.get_supported_quant_params()))

        bn_params = {}
        output_encodings = []
        param_encodings = []
        if op_name in self.quantization_params:
            # update defaults to existing params if we are updating a layer
            bn_params = self.quantization_params[op_name][QuantParams.BN_PARAMS]
            output_encodings = self.quantization_params[op_name][QuantParams.OUTPUT_ENCODINGS]
            param_encodings = self.quantization_params[op_name][QuantParams.PARAM_ENCODINGS]

        # extend existing lists(if exist) for encodings
        output_encodings.extend(translation_utils.to_list(kwargs.get(QuantParams.OUTPUT_ENCODINGS, [])))
        param_encodings.extend(translation_utils.to_list(kwargs.get(QuantParams.PARAM_ENCODINGS, [])))
        self.quantization_params.update({
            op_name: {
                QuantParams.BN_PARAMS: kwargs.get(QuantParams.BN_PARAMS, bn_params),
                QuantParams.OUTPUT_ENCODINGS: output_encodings,
                QuantParams.PARAM_ENCODINGS: param_encodings
            }
        })

    def get_layer_quantization_param(self, layer_name):
        log_assert(layer_name in self.quantization_params,
                   code_to_message.get_error_message("ERROR_LAYER_NOT_FOUND_IN_QUANT_PARAM")(layer_name))
        return self.quantization_params[layer_name]

    def merge_quantization_params(self, source_op_name, destination_op_name, pre_merge_dest_tensor_name,
                                  post_merge_dest_tensor_name):
        """
        Merges the output encodings for source op to destination
        :param source_op_name: the layer/op name for the source Op
        :param destination_op_name: the layer/op name for the destination Op
        :param pre_merge_dest_tensor_name: the output tensor name for the destination op before merging source op
        :param post_merge_dest_tensor_name: the output tensor name for the destination op after merging source op
        """
        if source_op_name in self.quantization_params:
            # verify this is proper merging. i.e. source op with more than one output is not allowed
            log_assert(len(self.quantization_params[source_op_name][QuantParams.OUTPUT_ENCODINGS]) == 1,
                       code_to_message.get_error_message("ERROR_UNABLE_TO_MERGE_QUANT_PARAMS")(source_op_name,
                                                                                               destination_op_name))
            source_output_encodings = self.quantization_params[source_op_name][QuantParams.OUTPUT_ENCODINGS]

            dest_bn_params = {}
            dest_output_encodings = []
            dest_param_encodings = []
            if destination_op_name in self.quantization_params:
                dest_bn_params = self.quantization_params[destination_op_name][QuantParams.BN_PARAMS]
                dest_param_encodings = self.quantization_params[destination_op_name][QuantParams.PARAM_ENCODINGS]
                dest_output_encodings = self.quantization_params[destination_op_name][QuantParams.OUTPUT_ENCODINGS]
                # remove the entry for the destination output_tensor's encoding as that will be replaced with the
                # source op's output encoding
                for i, encodings in enumerate(dest_output_encodings):
                    if pre_merge_dest_tensor_name == encodings["name"]:
                        del self.quantization_params[destination_op_name][QuantParams.OUTPUT_ENCODINGS][i]

            # Note: only need output encoding from source since the weights/bias will be merged into destination op
            source_output_encodings[0]["name"] = post_merge_dest_tensor_name  # replace output tensor name
            dest_output_encodings.extend(source_output_encodings)
            self.quantization_params.update({
                destination_op_name: {
                    QuantParams.BN_PARAMS: dest_bn_params,
                    QuantParams.OUTPUT_ENCODINGS: dest_output_encodings,
                    QuantParams.PARAM_ENCODINGS: dest_param_encodings
                }
            })

            # remove quantization entry for source op as the op will be merged
            del self.quantization_params[source_op_name]

    def __insert_node(self, node, output_shapes, axis_formats=None, idx=-1):
        """Insert a node into the graph's internal data structures.

        node: Node to be inserted
        output_shapes: shapes of the node's output buffers, which must be created.
        axis_formats: List of axis_format to override for each output Buffer
        idx: index in nodes_in_order at which to insert. By default, appends to
             the list.
        """
        for i, (name, shape) in enumerate(zip(node.output_names, output_shapes)):
            if axis_formats is None:
                buf = Buffer(name, shape, node)
                node.op.populate_axis_format(buf, self.src_axis_order)
            else:
                buf = Buffer(name, shape, node, axis_formats[i])
            self.buffers[name] = buf
            log_debug1("Added buffer named {0} of shape {1}", name, shape)

        for name in node.input_names:
            self.buffers[name].consumers.add(node)

        if node in self.nodes_in_order:
            raise IndexError("Node by name {} already exists at index {}".format(
                node.op.name, self.nodes_in_order.index(node)))
        self.nodes_by_name[node.op.name] = node
        if idx == -1:
            self.nodes_in_order.append(node)
        else:
            self.nodes_in_order.insert(idx, node)

    def add(self, op, input_names, output_names, axis_formats=None, idx=-1):
        """
        Adds op to graph by creating a node and corresponding buffer, as well as update
        input and output buffers for node.
        :param op: an operation from op_adapter class
        :param input_names: inputs to node. (This will be the buffer input names)
        :param output_names: output buffer names of node
        :param axis_formats: axis format of output buffers
        :param idx: index in nodes_in_order at which to insert. By default, appends to the list.
        :return: The created node for op.
        """
        op.name = self.naming_policy.get_op_name(op)

        if not isinstance(input_names, list):
            input_names = [input_names]
        input_names = self.naming_policy.get_input_names(op, input_names)

        input_shapes = []
        for name in input_names:
            if name not in self.buffers:
                raise KeyError("Graph has no buffer %s, referred to as input for %s" % (name, op.name))
            input_shapes.append(self.buffers[name].shape)

        if not isinstance(output_names, list):
            output_names = [output_names]
        output_names = self.naming_policy.get_output_names(op, output_names)

        # TODO: Move output_shapes, input_shapes inside OpNode.
        node = OpNode(op, input_names, output_names, self.src_axis_order)
        log_debug("Added OpNode with name {0}, in_names {1}, out_names {2}".format(op.name, input_names, output_names))
        try:
            output_shapes = op.infer_shape(input_shapes, len(output_names), self.src_axis_order)
        except NotImplementedError as e:
            if self.shape_inference_policy:
                try:
                    output_shapes = self.shape_inference_policy.infer_shape(op, input_shapes)
                except KeyError as e:
                    log_error("Node %s: %s", op.name, e)
                    raise e
            else:
                log_error("Node %s: %s", op.name, e)
                raise e

        if len(output_shapes) != len(output_names):
            raise ValueError("Op %s: produced %d output shapes, but have %d outputs" % (op.name, len(output_shapes),
                                                                                        len(output_names)))

        # at this point everything should be error free, so it's fine to actually
        # touch the data structures
        self.__insert_node(node, output_shapes, axis_formats, idx=idx)

        # return the added node
        return node

    def replace(self, old_op, new_op):
        old_node = self.nodes_by_name[old_op.name]
        input_buffers = self.get_input_buffers(old_node)
        output_buffers = self.get_output_buffers(old_node)
        input_names = [buf.name for buf in input_buffers]
        output_names = [buf.name for buf in output_buffers]

        # Create OpNode for the new op
        new_op.name = self.naming_policy.get_op_name(new_op)
        new_node = OpNode(new_op, input_names, output_names, self.src_axis_order)

        # Replace the op in buffers
        input_shapes = []
        for buf in input_buffers:
            buf.consumers.remove(old_node)
            buf.consumers.add(new_node)
            input_shapes.append(buf.shape)

        try:
            output_shapes = new_op.infer_shape(input_shapes, len(output_names), self.src_axis_order)
        except NotImplementedError as e:
            if self.shape_inference_policy:
                try:
                    output_shapes = self.shape_inference_policy.infer_shape(new_op, input_shapes)
                except KeyError as e:
                    log_error("Node %s: %s", new_op.name, e)
                    raise e
            else:
                log_error("Node %s: %s", new_op.name, e)
                raise e

        for i, buf in enumerate(output_buffers):
            buf.producer = new_node
            buf.shape = output_shapes[i]
            new_op.populate_axis_format(buf, self.src_axis_order)

        # Replace the op in op-lists
        idx = self.nodes_in_order.index(old_node)
        self.nodes_by_name[new_op.name] = new_node
        if idx == -1:
            self.nodes_in_order.append(new_node)
        else:
            self.nodes_in_order.insert(idx, new_node)

        del self.nodes_by_name[old_node.op.name]
        self.nodes_in_order.remove(old_node)

    def add_input(self, name, shape, axis_format=None, input_type=None):
        if not input_type:
            input_type = self.get_input_type(name)

        input_encoding = self.get_input_encoding(name)

        if input_encoding == InputEncodings.TIME_SERIES:
            log_assert(len(shape) == 3,
                       code_to_message.get_error_message("ERROR_TIMESERIES_UNEXPECTED_RANK")
                       (name, len(shape)))
        if input_encoding == InputEncodings.OTHER:
            axis_format = AxisTracker.AxisFormat.NONTRIVIAL

        op = op_adapter.InputOp(name, shape,
                                input_encoding_in=input_encoding,
                                input_encoding_out=InputEncodings.BGR,  # always default to BGR
                                input_type=input_type)
        output_names = self.naming_policy.get_output_names(op, [name])

        node = OpNode(op, [], output_names, self.src_axis_order)
        axis_formats = None if axis_format is None else [axis_format]
        self.__insert_node(node, [shape], axis_formats)

        # return the added input node
        return node

    def inject(self, op, input_name, output_name, consumer_names=None, axis_format=None):
        op.name = self.naming_policy.get_op_name(op)
        if input_name not in self.buffers:
            raise KeyError("Cannot inject op %s onto nonexistent buffer %s" % (op.name, input_name))

        input_buffer = self.buffers[input_name]
        if consumer_names is None:
            old_consumers = list(input_buffer.consumers)
            input_buffer.consumers.clear()
        else:
            old_consumers = []
            for name in consumer_names:
                if name not in self.nodes_by_name:
                    raise KeyError("Cannot inject op %s with nonexistent consumer %s" % (op.name, name))
                consumer = self.nodes_by_name[name]
                if consumer not in input_buffer.consumers:
                    raise KeyError("Cannot inject op %s, specified consumer %s does not actually consume input"
                                   " buffer %s" % (op.name, name, input_name))

                old_consumers.append(consumer)
                input_buffer.consumers.remove(consumer)

        output_name = self.naming_policy.get_output_names(op, [output_name])[0]
        producer_idx = self.nodes_in_order.index(input_buffer.producer)
        try:
            output_shapes = op.infer_shape([input_buffer.shape], 1, self.src_axis_order)
        except NotImplementedError as e:
            if self.shape_inference_policy:
                try:
                    output_shapes = self.shape_inference_policy.infer_shape(op, [input_buffer.shape])
                except KeyError as e:
                    log_error("Node %s: %s", op.name, e)
                    raise e
            else:
                log_error("Node %s: %s", op.name, e)
                raise e
        node = OpNode(op, [input_name], [output_name], self.src_axis_order)
        axis_formats = None if axis_format is None else [axis_format]
        self.__insert_node(node, output_shapes, axis_formats, producer_idx+1)

        output_buffer = self.buffers[output_name]
        for consumer in old_consumers:
            output_buffer.consumers.add(consumer)
            for i, name in enumerate(consumer.input_names):
                if name == input_name:
                    consumer.input_names[i] = output_name

    def inject_implicit_permute(self, input_name, target_format, permute_order, consumers=None):
        permute_name = input_name + '.' + target_format.lower()
        input_buf = self.get_buffer(input_name)
        log_assert(input_buf.rank() == len(permute_order),
                   "Error: length of buf to permute({}) does not match length of permute order({})"
                   " for input name: {}",
                   input_buf.rank(), len(permute_order), input_name)
        implicit_permute = op_adapter.PermuteOp(permute_name, permute_order)
        # since the implicit permute won't be visited in this pass, go
        # ahead and set the correct order for its buffer here.
        self.inject(implicit_permute, input_name, permute_name, consumers, axis_format=target_format)

    def prune(self, node, force_remove=False):
        """Remove a node and its output buffers from the graph completely.
        Will raise an exception if force_remove is False and the node has any successors."""

        # Disconnect output nodes
        output_buffers = self.get_output_buffers(node)
        consumers = []
        for buf in output_buffers:
            consumers.extend(buf.consumers)

        if len(consumers) > 0:
            if force_remove:
                for buf in output_buffers:
                    for c in buf.consumers:
                        try:
                            c.input_names.remove(buf.name)
                        except Exception as e:
                            log_error("Buffer {} not found in consumers for node {}".format(buf.name,
                                                                                            c.op.name))
                            raise e
            else:
                consumer_name_list = [c.op.name for c in consumers]
                raise RuntimeError("Cannot prune node %s, which has the following successors: %s"
                                   % (node.op.name, consumer_name_list))

        for buf in output_buffers:
            del self.buffers[buf.name]

        # Disconnect input nodes
        # loop through as set to support scenarios where a node is listed as input more than once
        for buf in set(self.get_input_buffers(node)):
            buf.consumers.remove(node)
        del self.nodes_by_name[node.op.name]
        self.nodes_in_order.remove(node)

    def squash(self, node, input_name):
        # remove the input buffer, causing that buffer's
        # producer to producer the output buffer instead.
        if input_name not in self.buffers:
            raise KeyError("Cannot squash node %s onto non-existent input buffer %s" % (node.op.name, input_name))
        input_buffer = self.buffers[input_name]
        output_buffer = self.buffers[node.output_names[0]]

        if len(input_buffer.consumers) > 1:
            raise ValueError("Cannot squash node %s onto input buffer %s, which has more than one consumer"
                             % (node.op.name, input_name))
        if node not in input_buffer.consumers:
            raise ValueError("Cannot squash node %s onto input buffer %s that it doesn't consume"
                             % (node.op.name, input_name))

        prev = input_buffer.producer
        if prev.op.type != op_adapter.InputOp.TRANSLATION_KEY:
            output_idx = prev.output_names.index(input_name)
            prev.output_names[output_idx] = output_buffer.name
            output_buffer.producer = prev

            other_input_names = [in_name
                                 for in_name
                                 in node.input_names
                                 if in_name != input_name]

            for in_buf_name in other_input_names:
                in_buf = self.get_buffer(in_buf_name)
                in_producer = self.get_producer_node(in_buf_name)
                if len(in_buf.consumers) == 1 and len(in_producer.output_names) == 1:
                    self.prune(in_producer, True)
                else:
                    in_buf.consumers.remove(node)
                    if not len(in_buf.consumers):
                        in_producer.output_names.remove(in_buf_name)
                        del self.buffers[in_buf_name]

            del self.buffers[input_name]
        elif self.get_op_output_nodes(node):
            # If input op is of type InputOp, we cannot change buffer name of input.
            # So instead of squashing into prev, we squash into next.
            next_op_nodes = output_buffer.consumers
            for next_op_node in next_op_nodes:
                input_idx = next_op_node.input_names.index(output_buffer.name)
                next_op_node.input_names[input_idx] = input_name
                input_buffer.consumers.add(next_op_node)
            del self.buffers[output_buffer.name]
            input_buffer.consumers.remove(node)
        else:
            # Squashing not possible
            return False

        del self.nodes_by_name[node.op.name]
        self.nodes_in_order.remove(node)
        return True

    def get_matched_nodes(self, sequence, validator=None, ignore_constants=False):
        """
        Traverses each node in graph to find the requested pattern
        :param sequence: list[tuples] a list of node translation keys with their inputs and outputs. i.e:
                         each tuple contains ("opdapter.<op_name>.TRANSLATION_KEY", ([inputs]), ([outputs]))
                         The tuple for inputs/outputs should state BufferCriteria to verify list length; additionally,
                         each input/output should state specific BufferCriteria to determine how many(if any) of the
                         buffer should be in the matched sequence.
             E.g for format:
             sequence = [
                   # node type A
                   (op_adapter.<op_name>.TRANSLATION_KEY,
                       # inputs
                       (BufferCriteria.<criteria>, [(op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    (op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    ...]),
                       # outputs
                       (BufferCriteria.<criteria>, [(op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    (op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    ...])
                   ),
                   # node type B
                   (op_adapter.<op_name>.TRANSLATION_KEY,
                       # inputs
                       (),
                       # outputs
                       ()
                   ),
                   ...
             ]
             E.g (Channel Shuffle). Note: we can pass strings instead of class.xxx for convenience,
                                          this function handles both.
             sequence = [
                        ("reshape",
                            (),
                            ("MATCH_NUM_BUFS", [("permute", "ALL")])
                        ),
                        ("permute",
                            (),
                            ("MATCH_NUM_BUFS", [("reshape", "ALL")])
                        ),
                        ("reshape",
                            (),
                            ()
                        )
                       ]
             Note 1: both inputs and outputs should also be translation keys
             Note 2: BufferCriteria can either be one of the BufferCriteria Enums or an INT to match a specific index
             Note 3: it is not required to have inputs or outputs, they can be left empty.
        :param validator: function to run if a match is found based on sequence. The matched sequence will be passed as
                          {"node_tuples": (nodes_matched)}
                          If not provided, function will return based on only matching the sequence as criteria.
        :param ignore_constants: if constant nodes need to be filtered during matching, this flag will be set to True.
        :return: list of node tuples that match the sequence provided, where each tuple contains the corresponding nodes
                 for each TRANSLATION_KEY in the sequence.
        """

        matched_nodes = []
        requested_types_seq = [entry[0].lower() for entry in sequence]
        start = 0
        end = len(sequence)
        nodes_list = self.list_nodes()

        if ignore_constants:
            nodes_list = [node for node in nodes_list if node.op.type != 'constant']

        log_debug2("Evaluating to match Sequence {}...", requested_types_seq)

        # we want to allow use of strings for op translation_keys(i.e op_types) to make sequence length minimal
        # so validate user has asked to match op_types that are supported in op_adapter
        log_assert(self.verify_op_types_exist(requested_types_seq) is True,
                   code_to_message.get_error_message("ERROR_UNKNOWN_OP_TYPE(S)_FOUND")(requested_types_seq))

        while end <= len(nodes_list):
            nodes_tuple = tuple(nodes_list[start:end])  # get number of nodes based on length of sequence
            current_types_seq = [node.op.type for node in nodes_tuple]
            if (current_types_seq == requested_types_seq and self._validate_nodes_topology(nodes_tuple, sequence)) and \
                    (validator is None or validator(nodes_tuple)):
                matched_nodes.append(nodes_tuple)
                start = end  # start next node by skipping over the length of the sequence matched
                end += len(sequence)
            else:
                start += 1
                end = start + len(sequence)

        log_debug2("Found {} match(es)", len(matched_nodes))

        return matched_nodes

    def _validate_nodes_topology(self, nodes_tuple, sequence):
        """
        validates the input and output buffers for each matched node sequence in graph

        :param nodes_tuple: a tuple of matched nodes based on pattern
        :param sequence: the original list of sequences provided by user
        :return: True if each node's input and output buffer match the expected ones in sequence, False otherwise
        :raises: AssertionError if length and node types of node_list and sequence do not match
        """

        log_assert(len(nodes_tuple) == len(sequence), "Matched node list length must be same as requested sequence. "
                                                      "Expected {}, Got {}", len(nodes_tuple), len(sequence))

        for i in range(0, len(nodes_tuple)):
            node_type_actual = nodes_tuple[i].op.type
            node_type_expected = sequence[i][0]
            log_assert(node_type_actual == node_type_expected,
                       "Cannot validate topology for nodes of different types. Expected {}, Got{}",
                       node_type_expected, node_type_actual)

            inputs_actual = self.get_input_op_types(nodes_tuple[i])
            outputs_actual = self.get_output_op_types(nodes_tuple[i])
            inputs_expected, outputs_expected = sequence[i][1:]

            # providing inputs_expected and outputs_expected is not required from user
            # since user might just care to match a sequence of node types for any given inputs/outputs
            if (len(inputs_expected) and not self._validate_buffers(inputs_expected, inputs_actual)) or \
               (len(outputs_expected) and not self._validate_buffers(outputs_expected, outputs_actual)):
                    log_debug2("Sequence pattern {} matched, but not input/output buffers for node {} of type {} in "
                               "sequence.", [entry[0] for entry in sequence], nodes_tuple[i].op.name,
                               nodes_tuple[i].op.type)
                    return False

        return True

    def _validate_buffers(self, expected_buffers, actual_buffers):
        """
        validates the actual buffers(inputs or outputs of nodes) against the criteria set in the expected buffers
        :param expected_buffers: a tuple with BufferCriteria for matching the list of buffers, list of tuple pairs
                                 with each tuple containing the type of op and a buffer criteria
                        (BufferCriteria.<criteria>, [(op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    (op_adapter.<op_name>.TRANSLATION_KEY, BufferCriteria.<criteria>)
                                                    ...])
        :param actual_buffers: list of actual buffer types for the current node being evaluated
        :return: true if actual buffers pass criteria set in the expected buffers, False otherwise

        raises Assertion error: if unknown buffer criteria,
               Value error: if ALL criteria given and there exists more expected inputs
        """

        # remove matching criteria from expected buffers and validate
        matching_criteria, expected_buffers = expected_buffers
        matching_criteria = matching_criteria.upper()
        log_assert(matching_criteria in [BufferCriteria.MATCH_NUM_BUFS, BufferCriteria.FLEXIBLE_NUM_BUFS],
                   code_to_message.get_error_message("ERROR_UNKNOWN_MATCHING_CRITERIA")
                   ([BufferCriteria.MATCH_NUM_BUFS, BufferCriteria.FLEXIBLE_NUM_BUFS], matching_criteria))

        if matching_criteria == BufferCriteria.MATCH_NUM_BUFS and len(expected_buffers) != len(actual_buffers):
            return False

        for op_type, buf_criteria in expected_buffers:
            op_type = op_type.lower()
            log_assert(self.verify_op_types_exist(op_type) is True,
                       code_to_message.get_error_message("ERROR_UNKNOWN_OP_TYPE(S)_FOUND")(op_type))

            if type(buf_criteria) == int:
                if matching_criteria == BufferCriteria.MATCH_NUM_BUFS:
                    # User knows the number of input/output buffers to expect, hence it is an error to request
                    # an out-of-range index
                    log_assert(buf_criteria < len(actual_buffers),
                               code_to_message.get_error_message("ERROR_BUFFER_CRITERIA_INDEX")
                               (op_type, buf_criteria, len(actual_buffers)))
                    # In this case, user doesnt know/care for the number of input/output buffers of a node but want to
                    # match ops that fit a certain criteria e.g. when the 2nd input is a particular op type;
                    # in this instance an out-of-range index is not an error.

                    if buf_criteria >= len(actual_buffers) or actual_buffers[buf_criteria] != op_type:
                        return False
                elif matching_criteria == BufferCriteria.FLEXIBLE_NUM_BUFS:
                    # In this case, user knows exactly how many of this type to expect but does not care
                    # about the position in the inputs
                    op_type_count = len([actual_op_type for actual_op_type in actual_buffers
                                         if actual_op_type == op_type])
                    if op_type_count != buf_criteria:
                        return False
            elif buf_criteria.upper() == BufferCriteria.ALL:
                if len(expected_buffers) != 1:
                    raise ValueError(code_to_message.get_error_message("ERROR_BUFFER_CRITERIA_ALL")
                                     (op_type, len(expected_buffers)))
                if not all(buf == op_type for buf in actual_buffers):
                    return False

            elif buf_criteria.upper() == BufferCriteria.ANY:
                if not any(buf == op_type for buf in actual_buffers):
                    return False

            elif buf_criteria.upper() == BufferCriteria.NONE:
                if any(buf == op_type for buf in actual_buffers):
                    return False

            # Unknown buffer criteria, so raise error
            else:
                raise ValueError(code_to_message.get_error_message("ERROR_UNKNOWN_BUFFER_CRITERIA")
                                 (op_type, ["ALL", "ANY", "NONE"], buf_criteria))

        return True

    @staticmethod
    def verify_op_types_exist(op_list):
        if type(op_list) is not list:
            op_list = [op_list]
        # get all supported op_types in op_adapter module
        supported_op_list = [class_[1].TRANSLATION_KEY if hasattr(class_[1], 'TRANSLATION_KEY') else ''
                             for class_ in inspect.getmembers(op_adapter, inspect.isclass)]
        return all(op in supported_op_list for op in op_list)

    def get_input_buffers(self, node):
        return [self.buffers[name] for name in node.input_names]

    def get_output_buffers(self, node):
        return [self.buffers[name] for name in node.output_names]

    def get_op_output_nodes(self, node):
        output_nodes = set()
        for buf in self.get_output_buffers(node):
            output_nodes.update(buf.consumers)
        return list(output_nodes)

    def get_op_input_nodes(self, node):
        return [buf.producer for buf in self.get_input_buffers(node)]

    def get_input_op_types(self, node):
        return [self.buffers[name].producer.op.type for name in node.input_names]

    def get_output_op_types(self, node):
        consumer_nodes = []
        consumer_nodes_types = []
        for name in node.output_names:
            for consumer in self.buffers[name].consumers:
                # consumer already existing in our list can happen if one consumer takes 2 or more outputs of a node.
                # e.g: if node_a has buf_1, buf_2 as outputs and next layer(node_b) has both of these buffers as input,
                # both buf_1 and buf_2 will list node_b as consumers so we don't want to have [node_b, node_b]
                # for outputs
                if consumer not in consumer_nodes:
                    consumer_nodes.append(consumer)
                    consumer_nodes_types.append(consumer.op.type)
        return consumer_nodes_types

    def get_buffer(self, buffer_name):
        return self.buffers[buffer_name]

    def has_buffer(self, buffer_name):
        return buffer_name in self.buffers

    def get_producer_node(self, buffer_name):
        return self.buffers[buffer_name].producer

    def get_producer_op(self, buffer_name):
        return self.buffers[buffer_name].producer.op

    def get_input_nodes_to_graph(self):
        input_nodes = []
        for node in self.list_nodes():
            if node.op.TRANSLATION_KEY == op_adapter.InputOp.TRANSLATION_KEY:
                input_nodes.append(node)
        return input_nodes

    def get_output_nodes_of_graph(self):
        output_nodes = []
        # matches list of given output node names to graph node names
        for node in self.list_nodes():
            for output_node_name in self.output_nodes:
                if ((output_node_name + ":0") in node.output_names) or (output_node_name in node.output_names):
                    output_nodes.append(node)
        return output_nodes

    def list_nodes(self):
        return self.nodes_in_order[:]

    def list_buffers(self):
        return list(self.buffers.values())

    def has_op(self, op):
        nodes = self.list_nodes()
        for node in nodes:
            if node.op == op:
                return True
        return False

    def has_node(self, node):
        nodes = self.list_nodes()
        for node_ in nodes:
            if node_.is_equal(node):
                return True
        return False

    def get_quantizable_tensors(self):
        params = {}
        for node in self.list_nodes():
            if node in self.list_nodes():
                tmp = {'type' : node.op.type}
                for kv in node.attrs.items():
                    if isinstance(kv[1], np.ndarray) and kv[1].dtype == np.float32 and \
                        node.op.hasattr('quantizable') and node.op['quantizable']:
                            tmp[kv[0]] = kv[1]
                if len(tmp) > 1:
                    params[node.name] = tmp
        return params
