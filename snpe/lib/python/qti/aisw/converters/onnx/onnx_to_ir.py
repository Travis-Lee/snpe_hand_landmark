# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import sys
import traceback

from qti.aisw.converters.common.utils import code_to_message

try:
    import onnx
except ImportError as e:
    raise Exception(code_to_message.get_error_message("ERROR_ONNX_NOT_FOUND")(e.msg, str(sys.path)))

from qti.aisw.converters.common.converter_ir import op_graph_optimizations, op_policies
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrders
from qti.aisw.converters.common.utils.converter_base import ConverterBase, CustomOpBackend
from .util import *
from . import onnx_translations


# ------------------------------------------------------------------------------
#   The Converter Class
# ------------------------------------------------------------------------------
class OnnxConverter(ConverterBase):
    class ArgParser(ConverterBase.ArgParser):
        def __init__(self, backend_framework):
            super(OnnxConverter.ArgParser, self).__init__("onnx", backend_framework)
            # add command-line options custom to onnx converter
            self.parser.add_optional_argument("--dry_run", type=str, nargs='?', const='info', default=None,
                                              help='Evaluates the model without actually converting any ops, and '
                                                   'returns unsupported ops/attributes as well as unused inputs and/or '
                                                   'outputs if any. Leave empty or specify "info" to see dry run as a '
                                                   'table, or specify "debug" to show more detailed messages only"')
            self.parser.add_optional_argument('-d', '--input_dim', nargs=2, action='append',
                                              metavar=('INPUT_NAME', 'INPUT_DIM'),
                                              help="The name and dimension of all the input buffers to the network specified in\n"
                                                    "the format [input_name comma-separated-dimensions],\n"
                                                    "for example: 'data' 1,224,224,3. \n"
                                                    "Note that the quotes should always be included in order to handle special\n"
                                                    "characters, spaces, etc.\n"
                                                    "NOTE: This feature works only with Onnx 1.6.0 and above")

    def __init__(self, args):
        super(OnnxConverter, self).__init__(args,
                                            naming_policy=OnnxNamePolicy(),
                                            shape_inference_policy=OnnxShapeInferencePolicy(),
                                            axis_order=AxisOrders.ONNX)
        self.translations = onnx_translations.OnnxTranslations
        self.dry_run = args.dry_run
        self.op_info = onnx_translations.OpVersionInfo()
        if args.input_dim is not None:
            (in_node, in_dim) = list(zip(*args.input_dim))
            self.input_node = in_node
            self.input_dim = in_dim
        else:
            self.input_node = None
            self.input_dim = None

    def evaluate(self, model):
        """
        Performs a dry-run of the Onnx Model without actually converting it, highlighting potential issues with
        attributes, inputs/outputs or opset versions.
        :param model: An Onnx model
        :return:
        """
        from qti.aisw.converters.onnx import model_evaluator
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            log_warning("Potential errors found in {} as per Onnx's in-built checker tool".format(self.input_model_path))
            log_warning("{}: {}", type(e), str(e))
        log_info('Proceeding with model evaluation...................................\n')
        model_evaluator.setup_dry_run(model, self.dry_run)

    def convert(self):
        model = onnx.load(self.input_model_path)
        self.op_info.set_global_op_ver(model)

        if self.dry_run:
            self.evaluate(model)
            sys.exit(0)

        if self.input_dim and self.input_node:
            self._update_input_node(model.graph)

        self.graph.weights = WeightProvider(model)

        # check to give priority to custom output node provided as argument over graph output
        if not self.output_nodes and model.graph.output:
            for value_info in model.graph.output:
                self.output_nodes.append(str(value_info.name))

        # TODO: Remove after Custom Op API is refactored for both backends
        if self.custom_op_backend is not CustomOpBackend.SNPE_UDO:
            self.populate_custom_op_collection(model, 'onnx')
        else:
            self.populate_udo_collection(model, "onnx")

        # extract inputs
        parameter_names = set()
        for tensor in model.graph.initializer:
            parameter_names.add(str(tensor.name))

        for value_info in model.graph.input:
            name = str(value_info.name)
            if name in parameter_names:
                # weights are usually listed as inputs too.
                continue
            self.translations.apply_method_to_op(converter_type("input", "onnx"),
                                                 onnx_translations.OnnxTranslationBase.ADD_INPUT, value_info, self.graph)

        # extract parameters, infer shapes, etc.
        for i, src_op in enumerate(model.graph.node):
            log_debug(code_to_message.get_debugging_message("DEBUG_CONVERTING_NODE")(i, src_op.op_type))
            src_type = converter_type(src_op.op_type, "onnx")

            if self.custom_op_factory and src_op.op_type in self.custom_op_factory.op_collection:
                if self.custom_op_backend is not CustomOpBackend.SNPE_UDO:
                    src_type = converter_type('custom', "onnx")
                else:
                    src_type = converter_type('udo', "onnx")
                try:
                    self.translations.apply_method_to_op(src_type,
                                                         onnx_translations.OnnxTranslationBase.ADD_OP,
                                                         src_op,
                                                         self.graph)
                except Exception as e:
                    if self.debug:
                        traceback.print_exc()
                    log_error("Node %s: %s" % (src_op.name, e))
                    sys.exit(-1)

                continue

            try:
                supported_version = self.translations.apply_method_to_op(src_type,
                                                                         onnx_translations.OnnxTranslationBase.SUPPORTED_VERSION)
                self.op_info.validate_op_ver(src_op, supported_version)
            except Exception as e:
                if self.debug:
                    traceback.print_exc()
                log_error("Node %s: %s" % (src_op.name, e))
                sys.exit(-1)

            try:
                self.translations.apply_method_to_op(src_type,
                                                     onnx_translations.OnnxTranslationBase.ADD_OP,
                                                     src_op,
                                                     self.graph)
            except Exception as e:
                if self.debug:
                    traceback.print_exc()
                log_error("Node %s: %s" % (src_op.name, e))
                sys.exit(-1)

        return self.graph

    def _update_input_node(self, graph):
        if onnx.version.version < '1.6.0':
            raise ValueError("--input_dim command supported with onnx versions >= 1.6.0")
        inputs = [node.name for node in graph.input]
        initializers = [node.name for node in graph.initializer]
        for input_node in graph.input:
            if input_node.name not in initializers:
                graph.input.remove(input_node)
        bufs_to_remove = set()
        for i, src_op in enumerate(graph.node):
            for output_buf_name in src_op.output:
                if output_buf_name in self.input_node:
                    position = self.input_node.index(output_buf_name)
                    dims = tuple(map(int, self.input_dim[position].split(',')))
                    input_new = onnx.helper.make_tensor_value_info(output_buf_name,
                        onnx.TensorProto.FLOAT, dims)
                    graph.input.insert(0, input_new)
                    bufs_to_remove.add(output_buf_name)
        # Cleaning all the nodes before the input node
        nodes_to_remove = []
        while bufs_to_remove:
            node_name = bufs_to_remove.pop()
            node_list = [node for node in graph.node if (node.name == node_name or node_name in node.output)]
            if not node_list:
                continue
            node = node_list[0]
            bufs_to_remove.update(set(node.input))
            if node not in nodes_to_remove:
                nodes_to_remove.append(node)

        remaining_nodes = [node for node in graph.node if node not in nodes_to_remove]

        # Throw error when all buffers in a branch are not specified
        for node in nodes_to_remove:
            for output in node.output:
                for remaining_node in remaining_nodes:
                    if output in remaining_node.input and output not in self.input_node:
                        raise ValueError("Cannot disconnect node with outputs: {} as output buffer: {} is still in use and was not specified".format
                            (str(node.output), str(output)))
            graph.node.remove(node)

    def ir_optimize(self, graph, **kwargs):
        try:
            # apply graph transformations
            op_graph_optimizations.apply_graph_optimizations(graph, self.disable_batchnorm_folding, **kwargs)
            return graph
        except Exception as e:
            if self.debug:
                traceback.print_exc()
            log_error(str(e))
            sys.exit(-1)


# ------------------------------------------------------------------------------
#   Policies
# ------------------------------------------------------------------------------
class OnnxNamePolicy(op_policies.ConversionNamePolicy):
    def __init__(self):
        op_policies.ConversionNamePolicy.__init__(self)

    def get_op_name(self, op):
        count = self.type_count.get(op.type, 0)
        self.type_count[op.type] = count + 1
        if op.name:
            return str(op.name)
        elif op.type == 'custom':
            return "%s_%s_%d" % (str(op.custom_type).lower(), op.type, count)
        else:
            return "%s_%d" % (op.type, count)


class OnnxShapeInferencePolicy(op_policies.ConversionShapeInferencePolicy):

    def infer_shape(self, op, input_shapes):
        return onnx_translations.OnnxTranslations.apply_method_to_op(op.type,
                                                                     onnx_translations.OnnxTranslationBase.INFER_SHAPE,
                                                                     op,
                                                                     input_shapes)
