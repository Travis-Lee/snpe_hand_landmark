# ==============================================================================
#
#  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


import traceback
from abc import abstractmethod, ABCMeta
import qti.aisw.converters.common.converter_ir.op_graph as op_graph
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrder
from qti.aisw.converters.common.converter_ir.op_policies import ConversionNamePolicy
from qti.aisw.converters.common.utils import validation_utils
from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper, CustomHelpFormatter
from qti.aisw.converters.common.utils.converter_utils import *


class ConverterBase(object):
    __metaclass__ = ABCMeta

    class ArgParser(object):
        def __init__(self, src_framework, backend_framework, **kwargs):
            # Initialize Logger with default level so that Deprecated argument WARNINGs can be printed
            setup_logging(-1)
            self.parser = ArgParserWrapper(
                description='Script to convert ' + src_framework + ' model into '
                            + backend_framework,
                formatter_class=CustomHelpFormatter,
                **kwargs)
            # TODO: as deprecation step, setting required to False for now so that scripts don't break.
            #       please adjust for 1.31.0 release
            self.parser.add_required_argument("--input_network", "-i", type=str,
                                              action=validation_utils.ValidateStringArgs,
                                              help="Path to the source framework model.")

            self.parser.add_optional_argument('-o', '--output_path', type=str,
                                              action=validation_utils.ValidateStringArgs,
                                              help='Path where the converted Output model should be saved.If not '
                                                   'specified, the converter model will be written to a file with same '
                                                   'name as the input model')
            self.parser.add_optional_argument('--out_node', type=str, action='append', default=[],
                                              help="Name of the graph\'s output nodes. Multiple output nodes "
                                                   "should be provided separately like: \n"
                                                   "    --out_node out_1 --out_node out_2")
            self.parser.add_optional_argument('--copyright_file', type=str,
                                              action=validation_utils.ValidateStringArgs,
                                              help='Path to copyright file. If provided, the content of the file will '
                                                   'be added to the output model.')
            self.parser.add_optional_argument('--model_version', type=str, default=None,
                                              help='User-defined ASCII string to identify the model, only first '
                                                   '64 bytes will be stored')
            self.parser.add_optional_argument("--disable_batchnorm_folding",
                                              help="If not specified, converter will try to fold batchnorm into "
                                                   "previous convolution layer",
                                              action="store_true")
            self.parser.add_optional_argument('--input_type', "-t", nargs=2, action='append',
                                              help='Type of data expected by each input op/layer. Type for each input '
                                                   'is |default| if not specified. For example: "data" image.Note that '
                                                   'the quotes should always be included in order to handle special '
                                                   'characters, spaces,etc. For multiple inputs specify multiple '
                                                   '--input_type on the command line.\n'
                                                   'Eg: \n'
                                                   '   --input_type "data1" image --input_type "data2" opaque \n'
                                                   'These options get used by DSP runtime and following descriptions '
                                                   'state how input will be handled for each option.\n'
                                                   'Image: \n'
                                                   'Input is float between 0-255 and the input\'s mean is 0.0f '
                                                   'and the input\'s max is 255.0f. We will cast the float to uint8ts '
                                                   'and pass the uint8ts to the DSP. \n'
                                                   'Default: \n'
                                                   'Pass the input as floats to the dsp directly and the DSP '
                                                   'will quantize it.\n'
                                                   'Opaque: \n'
                                                   'Assumes input is float because the consumer layer(i.e next '
                                                   'layer) requires it as float, therefore it won\'t be quantized.\n'
                                                   'Choices supported:\n   ' + '\n   '.join(
                                                  op_graph.InputType.
                                                      get_supported_types()),
                                              metavar=('INPUT_NAME', 'INPUT_TYPE'), default=[])
            self.parser.add_optional_argument('--input_encoding', "-e", nargs=2, action='append',
                                              help='Image encoding of the source images. Default is bgr. \n'
                                                   'Eg usage: \n'
                                                   '   "data" rgba \n'
                                                   'Note the quotes should always be included in order to handle '
                                                   'special characters, spaces, etc. For multiple '
                                                   'inputs specify --input_encoding for each on the command line.\n'
                                                   'Eg:\n'
                                                   '    --input_encoding "data1" rgba --input_encoding "data2" other\n'
                                                   'Use options:\n '
                                                   'color encodings(bgr,rgb, nv21...) if input is image; \n'
                                                   'time_series: for inputs of rnn models; \n'
                                                   'other: if input doesn\'t follow above categories or is unknown. \n'
                                                   'Choices supported:\n   ' + '\n   '.join(
                                                  op_graph.InputEncodings.
                                                      get_supported_encodings()),
                                              metavar=('INPUT_NAME', 'INPUT_ENCODING'), default=[])
            self.parser.add_optional_argument('--validation_target', nargs=2,
                                              action=validation_utils.ValidateTargetArgs,
                                              help="A combination of processor and runtime target against which model "
                                                   "will be validated. \n"
                                                   "Choices for RUNTIME_TARGET: \n   {cpu, gpu, dsp}. \n"
                                                   "Choices for PROCESSOR_TARGET: \n"
                                                   "   {snapdragon_801, snapdragon_820, snapdragon_835}.\n"
                                                   "If not specified, will validate model against "
                                                   "{snapdragon_820, snapdragon_835} across all runtime targets.",
                                              metavar=('RUNTIME_TARGET', 'PROCESSOR_TARGET'),
                                              default=[], )
            self.parser.add_optional_argument('--strict', dest="enable_strict_validation",
                                              action="store_true",
                                              default=False,
                                              help="If specified, will validate in strict mode whereby model will not "
                                                   "be produced if it violates constraints of the specified validation "
                                                   "target. If not specified, will validate model in permissive mode "
                                                   "against the specified validation target.")
            self.parser.add_optional_argument("--debug", type=int, nargs='?', const=0, default=-1,
                                              help="Run the converter in debug mode.")
            # TODO: Move once converter base hierarchy is refactored
            self.parser.add_optional_argument("--udo_config_paths", "-udo", nargs='+',
                                              help="Path to the UDO configs (space separated, if multiple)")

        def parse_args(self, args=None, namespace=None):
            return self.parser.parse_args(args, namespace)

    def __init__(self, args,
                 naming_policy=ConversionNamePolicy(),
                 shape_inference_policy=None,
                 axis_order=AxisOrder(),
                 custom_op_backend: CustomOpBackend = CustomOpBackend.SNPE_UDO):

        setup_logging(args.debug)

        self.output_nodes = args.out_node
        self.graph = op_graph.IROpGraph(naming_policy, shape_inference_policy,
                                        args.input_type, args.input_encoding, axis_order,
                                        output_nodes=self.output_nodes)
        self.input_model_path = args.input_network
        self.output_model_path = args.output_path
        self.copyright_str = get_string_from_txtfile(args.copyright_file)
        self.model_version = args.model_version
        self.disable_batchnorm_folding = args.disable_batchnorm_folding
        self.validation_target = args.validation_target
        self.enable_strict_validation = args.enable_strict_validation
        self.converter_command = sanitize_args(args,
                                               args_to_ignore=['input_network', 'i', 'output_path',
                                                               'o'])
        self.debug = args.debug
        self.config_paths = None
        self.custom_op_factory = None
        self.custom_op_backend = custom_op_backend
        # TODO: Move once converter base hierarchy is refactored
        # Relying on the fact that this code is removed for qnn
        from qti.aisw.converters.common.udo import udo_factory
        self.config_paths = self.udo_config_paths = args.udo_config_paths
        self.udo_factory = udo_factory.UDOFactory()

    @abstractmethod
    def convert(self):
        """
        Convert the input framework model to IROpGraph
        """
        pass

    def populate_udo_collection(self,
                                model,
                                converter_type='onnx',
                                **kwargs):
        self.config_paths = self.udo_config_paths
        self.custom_op_factory = self.udo_factory
        self.custom_op_factory.op_collection = {}
        self.populate_custom_op_collection(model, converter_type, **kwargs)

    # TODO: Move once converter base hierarchy is refactored
    def populate_custom_op_collection(self,
                                      model,
                                      converter_type='onnx',
                                      **kwargs):
        # Create a custom op collection based on configs provided by user
        if self.config_paths is not None:
            for config_path in self.config_paths:
                try:
                    self.custom_op_factory.parse_config(config_path,
                                                        model=model,
                                                        converter_type=converter_type,
                                                        **kwargs)
                except Exception as e:
                    if not is_log_level_debug():
                        traceback.print_exc()
                    log_error("Error populating custom ops from: {}\n {}".format(config_path,
                                                                                 str(e)))
                    sys.exit(-1)

            # TODO: Remove after Custom Op API refactor
            if self.custom_op_backend is CustomOpBackend.SNPE_UDO:
                if not len(self.custom_op_factory.udo_collection):
                    raise LookupError("UDO_OP_NOT_FOUND: "
                                      "None of the Udo Ops present "
                                      "in the config were found in the provided model.")
                self.custom_op_factory.op_collection = self.udo_factory.udo_collection
            else:
                if not len(self.custom_op_factory.op_collection) and \
                        not self.custom_op_factory.default_op_collection:
                    raise LookupError("CUSTOM_OP_NOT_FOUND: "
                                      "None of the custom Ops present in the "
                                      "config were found in the provided model.")
