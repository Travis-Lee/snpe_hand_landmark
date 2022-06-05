# ==============================================================================
#
#  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import importlib
import os
import traceback
from google.protobuf import text_format

from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrders
from qti.aisw.converters.common.converter_ir import op_graph_optimizations, op_adapter, translation
from qti.aisw.converters.common.utils import converter_utils, code_to_message
from qti.aisw.converters.common.utils.converter_base import ConverterBase
from qti.aisw.converters.common.utils.converter_utils import *
from .caffe_base_translation import CaffeTranslations, CaffeTranslationBase
from .caffe_policies import CaffeNamePolicy, CaffeShapeInferencePolicy
from .weight_provider import RandomWeightProvider, BlobWeightProvider


# ------------------------------------------------------------------------------
#   The Converter Class
# ------------------------------------------------------------------------------
class CaffeConverter(ConverterBase):
    class ArgParser(ConverterBase.ArgParser):
        def __init__(self, backend_framework):
            super(CaffeConverter.ArgParser, self).__init__("caffe", backend_framework)
            # add command-line options custom to onnx converter
            self.parser.add_optional_argument('-b', '--caffe_bin', type=str,
                                              help='Input caffe binary file containing the weight data')
            self.parser.add_optional_argument('--udl', type=str, nargs=2, metavar=('UDL_MODULE', 'FACTORY_FUNCTION'),
                                              help='Option to add User Defined Layers. Provide Filename, Function name.'
                                                   '1.Filename: Name of python module to load for registering custom '
                                                   'udl(note: must be in PYTHONPATH). If file part of package list the '
                                                   'package.filename as you would when doing a python import.'
                                                   '2.Function name: Name of the udl factory function that return a '
                                                   'dictionary of key layer type and value function callback.',
                                              default=[])
            self.parser.add_optional_argument('--enable_preprocessing',
                                              action="store_const", const=True, default=False,
                                              help='If specified, converter will enable preprocessing specified by a data'
                                                   'layer transform_param subtract_mean is supported.')


    def __init__(self, args):
        super(CaffeConverter, self).__init__(args,
                                             naming_policy=CaffeNamePolicy(),
                                             shape_inference_policy=CaffeShapeInferencePolicy(),
                                             axis_order=AxisOrders.CAFFE)
        self.translations = CaffeTranslations
        self.caffe_weights_path = args.caffe_bin
        self.udl_args = args.udl
        self.input_dim = list()
        self.network_dim = []
        self.udl_layer_dict = {}
        self.spec = None
        self.enable_preprocessing = args.enable_preprocessing

        # Caffe specific:  This control caffe's output with verbose option
        if not converter_utils.is_log_level_debug():
            # The levels are
            # 0 - debug
            # 1 - info (still a LOT of outputs)
            # 2 - warnings
            # 3 - errors
            os.environ['GLOG_minloglevel'] = '2'

    def convert(self):
        # import of Caffe has to come after the setting of GLOG_minloglevel for it to take effect
        try:
            import caffe
            import caffe.proto.caffe_pb2 as caffe_pb2
        except ImportError as e:
            raise Exception(code_to_message.get_error_message("ERROR_CAFFE_NOT_FOUND")(e.msg, str(sys.path)))

        # these need to be imported so they are evaluated and translations are registered.
        # importing them here since the modules import caffe as well.
        from . import input_translations, data_translations, math_translations, nn_translations, proposal_translation,\
            noop_translations, rnn_translations, udl_translation

        # Add udl module if provided
        if len(self.udl_args):
            log_info("Loading UDLs from module {} using factory function {}", self.udl_args[0], self.udl_args[1])
            udl_factory_func = getattr(importlib.import_module(self.udl_args[0]), self.udl_args[1])
            self.udl_layer_dict = udl_factory_func()
            # standardize the layer types so that type matching is easier below
            self.udl_layer_dict = {converter_type(k, "caffe"): v for k, v in self.udl_layer_dict.items()}

        caffe.set_mode_cpu()
        # get caffe spec
        try:
            self.spec = caffe_pb2.NetParameter()
            with open(self.input_model_path, 'rb') as text_file:
                text_format.Merge(text_file.read(), self.spec)
        except Exception as e:
            print(code_to_message.get_error_message('ERROR_CAFFE_CAFFE_PARSING_ERROR')(self.input_model_path,
                                                                                       str(e)))
            print(code_to_message.get_progress_message('INFO_CAFFE_CAFFE_INSTALLATION_ERROR')(caffe.__file__))
            sys.exit(1)

        # get weight provider
        caffe_net=None
        if self.caffe_weights_path is None:
            self.graph.weights = RandomWeightProvider(self.spec, self.graph)
        else:
            caffe_net = caffe.Net(self.input_model_path, self.caffe_weights_path, caffe.TEST)
            self.graph.weights = BlobWeightProvider(caffe_net.params)

        # Need to add a data layer
        if len(self.spec.input_shape) > 0:
            self.input_dim = list(map(int, self.spec.input_shape[0].dim))
        elif len(self.spec.input_dim) > 0:
            self.input_dim = list(map(int, self.spec.input_dim))
        # TODO: why need to change to 4D? commenting out for now
        # if self.input_dim and len(self.input_dim) != 4:
        #     # input_dim, but not 4-D, add 1 as batch
        #     log_debug("Length of input dim " + str(len(self.input_dim)))
        #     self.input_dim.insert(0, 1)

        # assign input type as data when inputs are not written  as layers in prototxt
        input_type = converter_type("data", "caffe")
        if self.input_dim and len(self.spec.input):
            self.translations.apply_method_to_op(input_type, translation.ConversionTranslationBase.ADD_INPUT_OP, self.spec.input[0],
                                                 self.input_dim, self.graph)

        # If there are additional inputs. create data layers for these. Note that
        # only the input {} input_shape {} syntax is supported here.
        for index in range(1, len(self.spec.input)):
            data_name = str(self.spec.input[index])
            if len(self.spec.input_shape[index].dim) == 4:
                data_dims = list(map(int, self.spec.input_shape[index].dim))
            elif len(self.spec.input_shape[index].dim) == 2:
                # 2-D input_dim. Treat as (batch, depth). TODO: why need to change to 4D? commenting out for now
                # data_dims = list(map(int, [self.spec.input_shape[index].dim[0], self.spec.input_shape[index].dim[1], 1, 1]))
                data_dims = list(map(int, [self.spec.input_shape[index].dim[0], self.spec.input_shape[index].dim[1]]))
            else:
                raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_UNSUPPORTED_INPUT_DIMS')
                                 (str(data_name)))  # TODO: remove restriction if applicable
            self.translations.apply_method_to_op(input_type, translation.ConversionTranslationBase.ADD_INPUT_OP,
                                                 data_name, data_dims, self.graph)

        # extract parameters, infer shapes, etc.
        layers = self.spec.layer if len(self.spec.layer) != 0 else self.spec.layers

        # Populate the custom op using the provided model
        if caffe_net is None and self.config_paths:
            raise RuntimeError('The caffe model binary is required to ingest a custom op. Please provide '
                               'caffe model via the \'-b\' converter option')
        elif self.config_paths:
            if self.custom_op_backend is CustomOpBackend.SNPE_UDO:
                from . import udo_translations
                self.populate_udo_collection(layers, converter_type='caffe', caffe_net=caffe_net)
            else:
                from . import custom_op_translations
                self.populate_custom_op_collection(layers, converter_type='caffe', caffe_net=caffe_net)

        for i, layer in enumerate(layers):
            if self._is_in_train_phase(layer):
                # Skip train layers
                continue
            log_debug(code_to_message.get_debugging_message("DEBUG_CONVERTING_NODE")(i, layer.type))
            src_type = converter_type(layer.type, "caffe")

            # cases where input is a layer
            if src_type == converter_type("data", "caffe"):
                if len(self.input_dim) == 0:
                    raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_DATA_LAYER_ERR_NO_INPUT_DIM')
                                     (str(layer.name)))
                self.translations.apply_method_to_op(src_type, translation.ConversionTranslationBase.ADD_INPUT_OP,
                                                     str(layer.top[0]), self.input_dim, self.graph)
                if self.enable_preprocessing:
                    self.setup_preprocessing(src_type, layer, self.graph)
            elif src_type == converter_type("input", "caffe"):
                if not hasattr(layer, "input_param"):
                    raise ValueError(code_to_message.get_error_message('ERROR_CAFFE_NO_INPUT_PARAM_SPECIFIED')
                                     (str(layer.name)))

                input_param = layer.input_param
                self.input_dim = list(map(int, input_param.shape[0].dim))
                self.translations.apply_method_to_op(src_type, translation.ConversionTranslationBase.ADD_INPUT_OP,
                                                     str(layer.top[0]), self.input_dim, self.graph)
                if self.enable_preprocessing:
                    self.setup_preprocessing(src_type, layer, self.graph)
            else:
                try:
                    # first check if layer is a registered custom op in an op collection. If so, the layer is added
                    # and the loop continues.
                    # TODO: Move this check once snpe has its own backend converter base
                    if self.custom_op_factory and layer.type in self.custom_op_factory.op_collection:
                        if self.custom_op_backend is CustomOpBackend.SNPE_UDO:
                            src_type = converter_type('udo', "caffe")
                        else:
                            src_type = converter_type('custom', "caffe")
                        self.translations.apply_method_to_op(src_type,
                                                             translation.ConversionTranslationBase.ADD_OP,
                                                             layer,
                                                             self.graph)
                        continue

                    # Check if layer is UDL. If so call factory function to extract blob info
                    # and then add to graph
                    if src_type in self.udl_layer_dict:
                        udl_obj = self.udl_layer_dict[src_type]
                        udl_layer = (layer, udl_obj)
                        self.translations.apply_method_to_op(op_adapter.UdlOp.TRANSLATION_KEY,
                                                             translation.ConversionTranslationBase.ADD_OP,
                                                             udl_layer,
                                                             self.graph)
                    else:
                        self.translations.apply_method_to_op(src_type,
                                                             translation.ConversionTranslationBase.ADD_OP,
                                                             layer,
                                                             self.graph)
                except Exception as e:
                    if converter_utils.is_log_level_debug():
                        traceback.print_exc()
                    log_error("Node %s: Type : %s %s" % (layer.name, layer.type, e))
                    sys.exit(-1)

        return self.graph

    def ir_optimize(self, graph, **kwargs):
        try:
            # apply graph transformations
            op_graph_optimizations.apply_graph_optimizations(graph, self.disable_batchnorm_folding, **kwargs)
            return graph
        except Exception as e:
            if converter_utils.is_log_level_debug():
                traceback.print_exc()
            log_error(str(e))
            sys.exit(-1)

    @staticmethod
    def _is_in_train_phase(layer):
        if layer.include:
            import caffe.proto.caffe_pb2 as caffe_pb2
            caffe_phases = {pair[0]: pair[1] for pair in list(caffe_pb2.Phase.items())}
            phases = [state.phase for state in layer.include if state.phase is not None]
            return caffe_phases['TRAIN'] in phases
        return False

    def setup_preprocessing(self, src_type, layer, graph):
        if layer.transform_param.mean_value:
            src_type = op_adapter.SubtractMeanOp.TRANSLATION_KEY
            self.translations.apply_method_to_op(src_type,
                                                 CaffeTranslationBase.ADD_INPUT_PREPROCESSING_OP,
                                                 layer,
                                                 self.graph)

