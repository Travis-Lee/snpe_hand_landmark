# ==============================================================================
#
#  Copyright (c) 2018-2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.converter_ir import translation, op_adapter
from qti.aisw.converters.onnx.op_schema import OpSchemaBase, OpSchemaDict, OP_SCHEMA_REGISTRY
from qti.aisw.converters.common.utils import translation_utils
from .util import *

OnnxTranslations = translation.TranslationBank()


class OnnxTranslationBase(translation.ConversionTranslationBase):
    # onnx specific translation method keys
    ADD_INPUT = "ADD_INPUT"
    SUPPORTED_VERSION = "SUPPORTED_VERSION"

    def __init__(self):
        translation.ConversionTranslationBase.__init__(self)
        self.register_method(self.SUPPORTED_VERSION, self.get_supported_version)
        self._op_schema = OpSchemaDict()  # dictionary-style class that maps {version:op_schema}

    def extract_parameters(self, src_op, graph):
        raise NotImplementedError("extract_parameters for {} not implemented ".format(str(self.__class__.__name__)))

    def extract_input_names(self, src_op, graph):
        return list(map(str, src_op.input))

    def extract_output_names(self, src_op, graph):
        return list(map(str, src_op.output))

    def get_supported_version(self):
        try:
            version = list(map(int, self._op_schema.get_schemas().keys()))
            return version
        except Exception as e:
            raise NotImplementedError("get_supported_version for {} not implemented ".format(str(self.__class__.__name__)))

    def register_op_schema(self, name, versions, unsupported_attrs=None):
        """
           Wraps Onnx's internal schema definition into a condensed op_schema_dict internal object (OpSchemaDict)
           which contains individual op_schema(s)(OpSchemaBase) that tie supported attributes,
           number of inputs and outputs to the appropriate op version

           :param name: The type of op to be registered
           :param versions : list of versions of the op to be registered. Note the versions must be available in
                             the Onnx spec.
           :param unsupported_attrs: A list of lists of unsupported attrs, which are in the Onnx spec
                                    for an op version but are not supported by the translation

           registers the resulting op_schema dictionary with the translation, as well as with a
           global schema registry

        """
        if unsupported_attrs:
            while len(unsupported_attrs) < len(versions):
                unsupported_attrs.append(unsupported_attrs[0])
        else:
            unsupported_attrs = [[] for _ in range(len(versions))]

        for i, version in enumerate(versions):
            try:
                # Note: get_schema uses version as maxInclusiveVersion and returns the schema with the
                #       biggest version, which is not greater than specified version in specified domain
                schema = defs.get_schema(name, version, '')
                op_schema = OpSchemaBase()
                op_schema.populate_op_schema(schema, unsupported_attrs[i])
                self._op_schema.add_schema(op_schema, version)
            except RuntimeError as e:
                # Only warn user here since even though their onnx installation doesnt have all the ops(and the
                # different versions) we support, their model might not contain that Op. If it does, it will be caught
                # at conversion time later.
                # Note: need to use print here instead of log_warning since Ops are registered at module import time
                #       and the logger is not yet set.
                print(code_to_message.get_warning_message("WARNING_OP_NOT_SUPPORTED_BY_ONNX")(name, version, str(e)))

        OP_SCHEMA_REGISTRY[name.lower()] = self._op_schema

    def op_schema(self, version=None):
        if version is not None:
            return self._op_schema.get_schemas(version)
        values = list(self._op_schema.get_schemas().values())
        return values[-1]


class ElementwiseBinaryTranslationBase(OnnxTranslationBase):
    """
    Additional BaseClass for elementWiseBinary Ops(mul, prod, div and sub) since they need add_op to handle constant Op
    addition to graph
    """
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.input_names = []

    def add_op(self, src_op, graph):
        op = self.extract_parameters(src_op, graph)
        input_names = self.extract_input_names(src_op, graph)
        output_names = self.extract_output_names(src_op, graph)
        elementwise_const_input_names = []
        for input_name in self.input_names:
            # Add constant op if one of the inputs is Constant and not added to graph, then add the input name to the
            # list of known constant inputs
            # if the op input is already produced by a Constant op, it is also added to the aforementioned list
            if graph.weights.has(input_name) and not graph.has_buffer(input_name):
                tensor = graph.weights.fetch(input_name, prunable=False)
                graph.add(op_adapter.ConstantOp(input_name, tensor), [], input_name)
                elementwise_const_input_names.append(input_name)
            elif graph.get_buffer(input_name).producer.op.TRANSLATION_KEY == op_adapter.ConstantOp.TRANSLATION_KEY:
                elementwise_const_input_names.append(input_name)

        # if any of the op inputs need to be broadcasted, then this is handled here.
        # The assumption is that the input to be broadcasted must be contained in a constant op, which is either
        # added above or has been previously added .
        if is_broadcast(src_op, graph):
            input_shapes = [graph.get_buffer(input_name).shape for input_name in self.input_names]
            broadcast_shape = translation_utils.get_broadcasted_shape(input_shapes)
            for name in elementwise_const_input_names:
                producer = graph.get_buffer(name).producer
                broadcast_value = broadcast_to(getattr(producer.op, 'tensor'), list(broadcast_shape))
                setattr(producer.op, 'tensor', broadcast_value)
                setattr(graph.buffers[name], 'shape', broadcast_shape)

        if hasattr(op, 'bias'):
            if not graph.has_buffer(src_op.input[1]):
                graph.add(op_adapter.ConstantOp(src_op.input[1], op.bias), [], src_op.input[1])
            input_names.append(src_op.input[1])
        graph.add(op, input_names, output_names)

    def extract_input_names(self, src_op, graph):
        return self.input_names


# -----------------------------------------------------------------
# Converter translations
# Note: ONNX doesn't have input op(s) but we create one for the IR
# -----------------------------------------------------------------
class OnnxInputTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_method(self.ADD_INPUT, self.add_input)

    def add_input(self, input_, graph):
        name = str(input_.name)
        tensor_shape = input_.type.tensor_type.shape
        shape = [int(dim.dim_value) for dim in tensor_shape.dim]
        neg_idx = [idx for idx in range(len(shape)) if shape[idx] < 0]

        if neg_idx:
            raise RuntimeError('Negative/placeholder dimensions is not supported.'
                               'Expected shape: {} > 0'.format(shape))

        graph.add_input(name, shape)


OnnxTranslations.register_translation(OnnxInputTranslation(),
                                      converter_type('input', 'onnx'),
                                      op_adapter.InputOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Dropout and other Noops
# ------------------------------------------------------------------------------
class OnnxNoopTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Dropout', [1])

    def extract_parameters(self, src_op, graph):
        return op_adapter.NoopOp(src_op.name)

    def extract_output_names(self, src_op, graph):
        return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxNoopTranslation(),
                                      converter_type('Dropout', 'onnx'),
                                      op_adapter.NoopOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   StaticOp
# ------------------------------------------------------------------------------
# 'Static' ops are transformations applied to weights, which do not produce
# an actual runtime output.
class OnnxStaticTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_parameters(self, src_op, graph):
        return op_adapter.StaticOp(src_op.name)

    def infer_output_shapes(self, op, input_shapes):
        return []

    def get_supported_version(self):
        return {}


OnnxTranslations.register_translation(OnnxStaticTranslation(), op_adapter.StaticOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Class OpVersionInfo
# ------------------------------------------------------------------------------
# Returns name and version information about an op from a particular model
class OpVersionInfo:
    def __init__(self):
        self.model_opset_version = 0

    @staticmethod
    def update_schema_registry(src_op_type, op_version):
        """ Updates the schema registry so that get_op_schema(src_op_type) will always return the appropriate schema
            for the global model opset version """
        op_schema_dict = OP_SCHEMA_REGISTRY[src_op_type.lower()]
        op_schema_keys = list(op_schema_dict.get_schemas().keys())
        if op_schema_keys[-1] != str(op_version):
           op_schema_dict.reorder_op_schemas(str(op_version))

    def validate_op_ver(self, src_op, supported_version):
        """

        :param src_op: The op from the Onnx framework
        :param supported_version: The version of the op supported by the Onnx Converter
        :return: a warning if the opset_version for the source op does not match any version supported
                 by the converter
                 updates the schema registry if the src_op version is supported, so that any schema calls (self.op_schema()
                 or get_op_schema) will return the src_op_version.
        """

        # This uses the model version to extract the associated opset version for a given op
        # For example:
        # The scenarios are described below:
        # supported_version = [1, 6, 7]
        # Model_opset_version = 3,    Model_opset_version = 7,   Model_opset_version = 7,    Model_opset_version = 9
        # current_op_version = 1,     current_op_version = 7,    current_op_version = 1      current_op_version = 8
        #                                                        returns a warning for       returns a warning for
        #                                                        onnx installation support   converter support
        try:
            current_op_version = int(defs.C.get_schema(src_op.op_type, self.model_opset_version, '').since_version)
            if current_op_version not in supported_version:
                log_warning(code_to_message.get_warning_message("WARNING_OP_VERSION_NOT_SUPPORTED")
                            (src_op.op_type, list(map(int, supported_version)), [current_op_version]))
            else:
                if self.model_opset_version != current_op_version and self.model_opset_version in supported_version:
                    log_warning(code_to_message.get_warning_message("WARNING_OP_VERSION_NOT_SUPPORTED_BY_ONNX")
                                (src_op.op_type, self.model_opset_version, current_op_version))
                self.update_schema_registry(src_op.op_type, current_op_version)
        except RuntimeError as e:
            # Throw an error here since model contains an op or a max_version that is not part of the current onnx
            # installation.
            # Note: re-raising error since the onnx error message is not very informative
            raise RuntimeError(code_to_message.get_error_message("ERROR_OP_NOT_SUPPORTED_BY_ONNX")(src_op.op_type,
                               self.model_opset_version, str(e)))

    def set_global_op_ver(self, model):
        """ Sets the highest global op version supported by the model"""
        # Get the global opset version
        if len(model.opset_import) > 1:
            log_warning(code_to_message.get_warning_message("WARNING_OPSET_VERSION"))

        for opset in model.opset_import:
            if opset.version > self.model_opset_version:
                self.model_opset_version = opset.version
