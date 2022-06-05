# ==============================================================================
#
#  Copyright (c) 2018-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import math
from qti.aisw.converters.common.utils import translation_utils
from qti.aisw.converters.common.utils.translation_utils import IRPaddingStrategies
from qti.aisw.converters.common.utils import converter_utils
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrders


class Op(object):
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.attrs = {}

    def addattr(self, key, source, default):
        attr = source.get(key, default)
        # Use the default's type when value/type is not None
        if attr is None or type(default) is type(None):
            self.attrs[key] = attr
        else:
            self.attrs[key] = type(default)(attr)

    def assertattr(self, key, source):
        if key in source:
            self.attrs[key] = source[key]
        else:
            raise KeyError("Op %s missing required argument %s" % (self.name, key))

    def hasattr(self, key):
        return key in self.attrs

    def __getitem__(self, key):
        return self.attrs[key]

    def __setitem__(self, key, value):
        self.attrs[key] = value

    def __getattr__(self, name):
        try:
            return self.attrs[name]
        except KeyError:
            raise AttributeError("op %s has no attribute %s" % (self.name, name))

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        raise NotImplementedError("infer_shape for {} not implemented ".format(str(self.__class__.__name__)))

    # This bypass is provided so that if some Op needs to override a format,
    # it can override this function and do so
    def populate_axis_format(self, buf, axis_order):
        buf.populate_axis_format(axis_order)

    def list_params(self):
        """ This gets instance variables of this class as key/value"""

        instance_vars = dict(self.__dict__)
        # above will get the attrs as {'attrs': {name1:val1...} instead we want to expand that
        del instance_vars['attrs']
        op_attrs = self.attrs
        instance_vars.update(op_attrs)

        return instance_vars

    def is_equal(self, other_op):
        """
        Compares another op instance to current one based on type and attribute matching
        :param other_op: an op_adapter object
        :return: bool, msg. True if type and attr/params match, False otherwise. Plus message detailing what was
                            different
        """
        # instance equality check
        if not isinstance(other_op, self.__class__):
            return False, "{} is not an instance of current Op {}".format(other_op, self.__class__)

        # attr/param list equality check
        other_op_params = other_op.list_params()
        current_op_params = self.list_params()
        if not other_op_params.keys() == current_op_params.keys():
            return False, "Op adapter for {} not set with same attribute as current Op. Expected keys: {}. Got {}".\
                format(type(other_op.type), current_op_params.keys(), other_op_params.keys())
        # loop through attributes. Since we verified above both objects are same instance and have same attrs/params
        # we can use one to list all
        for attr_ in list(current_op_params.keys()):
            if not translation_utils.compare_values(other_op_params[attr_], current_op_params[attr_]):
                return False, "Attribute match error for Op:{} Attribute: {}. Expected {}, Got {} ".format(
                    str(other_op.type), attr_, str(current_op_params[attr_]), str(other_op_params[attr_]))

        return True, "Op {} is equal to current Op instance".format(other_op)

    def __eq__(self, other_op):
        return self.is_equal(other_op)[0]


class InputOp(Op):
    TRANSLATION_KEY = 'input'

    def __init__(self, name, shape, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.shape = shape
        self.assertattr('input_encoding_in', kargs)
        self.assertattr('input_encoding_out', kargs)
        self.assertattr('input_type', kargs)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [self.shape[:]]


class ArgMaxOp(Op):
    TRANSLATION_KEY = 'argmax'

    def __init__(self, name, axis, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.axis = axis
        self.addattr('keep_dims', kargs, False)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        output_shape = []
        for axis, shape in enumerate(input_shapes[0][:]):
            if axis == self.axis:
                if self.keep_dims:
                    output_shape.append(1)
                continue
            output_shape.append(shape)
        return [output_shape]


class ArgMinOp(Op):
    TRANSLATION_KEY = 'argmin'

    def __init__(self, name, axis, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.axis = axis
        self.addattr('keep_dims', kargs, False)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        output_shape = []
        for axis, shape in enumerate(input_shapes[0][:]):
            if axis == self.axis:
                if self.keep_dims:
                    output_shape.append(1)
                continue
            output_shape.append(shape)
        return [output_shape]


class BatchnormOp(Op):
    TRANSLATION_KEY = 'batchnorm'

    def __init__(self, name, weights, bias, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.addattr('compute_statistics', kargs, False)
        self.addattr('use_mu_sigma', kargs, False)
        self.addattr('across_spatial', kargs, False)
        self.addattr('epsilon', kargs, 1e-9)
        self.addattr('normalize_variance', kargs, True)
        self.addattr('gamma', kargs, [])
        self.addattr('beta', kargs, [])

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class ChannelShuffleOp(Op):
    TRANSLATION_KEY = 'channel_shuffle'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('groups', kargs)
        self.addattr('shuffle_mode', kargs, "CHANNEL_SHUFFLE_GROUPED")

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class ConcatOp(Op):
    TRANSLATION_KEY = 'concatenation'

    def __init__(self, name, axis):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.axis = axis

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        axis = self.axis
        output_shape = input_shapes[0][:]
        output_shape[axis] = sum(shape[axis] for shape in input_shapes)
        return [output_shape]


class ConstantOp(Op):
    TRANSLATION_KEY = 'constant'

    def __init__(self, name, tensor, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.tensor = tensor
        self.addattr('quantizable', kargs, True)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [list(self.tensor.shape)]


class ConvolutionOp(Op):
    TRANSLATION_KEY = 'convolution'

    def __init__(self, name, weights, bias, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.assertattr('padx_before', kargs)
        self.assertattr('padx_after', kargs)
        self.assertattr('pady_before', kargs)
        self.assertattr('pady_after', kargs)
        self.assertattr('stridex', kargs)
        self.assertattr('stridey', kargs)
        self.assertattr('dilationx', kargs)
        self.assertattr('dilationy', kargs)
        self.addattr('groups', kargs, 1)
        self.addattr('padding_mode', kargs, "PADDING_ZERO")
        self.addattr('padding_size_strategy', kargs, IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        batch_size, input_height, input_width, input_depth = axis_order.extract_spatial_dims(input_shapes[0])
        filter_size_height, filter_size_width, _, _ = axis_order.extract_conv_weights_dims(self.weights.shape)

        output_height = translation_utils.calc_conv_output_dim(input_height,
                                                               filter_size_height,
                                                               self.pady_before,
                                                               self.pady_after,
                                                               self.stridey,
                                                               self.dilationy,
                                                               self.padding_size_strategy)
        output_width = translation_utils.calc_conv_output_dim(input_width,
                                                              filter_size_width,
                                                              self.padx_before,
                                                              self.padx_after,
                                                              self.stridex,
                                                              self.dilationx,
                                                              self.padding_size_strategy)
        output_depth = self.bias.shape[0]

        output_shape = axis_order.format_spatial_output_shape(batch_size=batch_size,
                                                              depth=output_depth,
                                                              height=output_height,
                                                              width=output_width)

        return [output_shape]


class CropOp(Op):
    TRANSLATION_KEY = 'crop'

    def __init__(self, name, offsets, counts, output_shape):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.offsets = offsets
        self.counts = counts
        self.output_shape = output_shape

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [self.output_shape[:]]


class CropAndResizeOp(Op):
    TRANSLATION_KEY = 'crop_and_resize'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr("num_boxes", kwargs)
        self.assertattr("crop_height", kwargs)
        self.assertattr("crop_width", kwargs)
        self.assertattr("interpolation_method", kwargs)
        self.assertattr("extrapolation_value", kwargs)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        _, _, _, depth = axis_order.extract_spatial_dims(input_shapes[0])
        output_shape = axis_order.format_spatial_output_shape(batch_size=self.num_boxes,
                                                              depth=depth,
                                                              height=self.crop_height,
                                                              width=self.crop_width)
        return [output_shape]


class CrossCorrelationOp(Op):
    TRANSLATION_KEY = 'cross_correlation'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class CustomOp(Op):
    TRANSLATION_KEY = 'custom'

    def __init__(self,
                 name,
                 package_name,
                 custom_type,
                 inputs,
                 outputs,
                 output_dims,
                 scalar_params,
                 tensor_params,
                 axis_orders):

        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.custom_type = custom_type
        self.output_dims = output_dims
        self.package_name = package_name
        self.axis_orders = axis_orders
        self.inputs = inputs
        self.outputs = outputs
        self.scalar_params = scalar_params
        self.tensor_params = tensor_params

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return self.output_dims

    def populate_axis_format(self, buf, axis_order):
        # if the buffer is rank 4 and the axis order has been defined (note only 4-d axis orders
        # can be defined in a CustomOp) or the axis format is non-trivial, then we keep the
        # format set by the CustomOp object. Otherwise, the axis format will be set according to
        # framework AxisOrder class using the buffer rank when we call populate axis format.

        if (buf.rank() == 4 and self.axis_orders[buf.name] != 'NOT_YET_DEFINED') or \
                self.axis_orders[buf.name] == 'NON-TRIVIAL':
            buf.axis_format = self.axis_orders[buf.name]
        else:
            buf.populate_axis_format(axis_order)


class DeconvolutionOp(Op):
    TRANSLATION_KEY = 'deconvolution'

    def __init__(self, name, weights, bias, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.addattr('stridex', kargs, 1)
        self.addattr('stridey', kargs, 1)
        self.addattr('padx_before', kargs, 0)
        self.addattr('padx_after', kargs, 0)
        self.addattr('pady_before', kargs, 0)
        self.addattr('pady_after', kargs, 0)
        self.addattr('padding_size_strategy', kargs, IRPaddingStrategies.PADDING_SIZE_EXPLICIT_FLOOR)
        self.addattr('output_paddingx', kargs, 0)
        self.addattr('output_paddingy', kargs, 0)
        self.assertattr('output_height', kargs)
        self.assertattr('output_width', kargs)
        self.addattr('groups', kargs, 1)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        batch_size, input_height, input_width, _ = axis_order.extract_spatial_dims(input_shapes[0])
        filter_size_height, filter_size_width, _, _ = axis_order.extract_deconv_weights_dims(self.weights.shape)

        if self.output_height == 0:
            # calculate according to provided formula
            output_height = translation_utils.calc_deconv_output_dim(input_height,
                                                                     filter_size_height,
                                                                     self.pady_before,
                                                                     self.pady_after,
                                                                     self.stridey,
                                                                     self.padding_size_strategy)

            output_width = translation_utils.calc_deconv_output_dim(input_width,
                                                                    filter_size_width,
                                                                    self.padx_before,
                                                                    self.padx_after,
                                                                    self.stridex,
                                                                    self.padding_size_strategy)
        else:
            output_height = self.output_height
            output_width = self.output_width

        output_depth = self.bias.shape[0]

        output_shape = axis_order.format_spatial_output_shape(batch_size=batch_size,
                                                              depth=output_depth,
                                                              height=output_height,
                                                              width=output_width)

        return [output_shape]


class DetectionOutputOp(Op):
    TRANSLATION_KEY = 'detection_output'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('output_dims', kargs)
        self.assertattr('num_classes', kargs)
        self.assertattr('share_location', kargs)
        self.assertattr('background_label_id', kargs)
        self.assertattr('nms_threshold', kargs)
        self.assertattr('confidence_threshold', kargs)
        self.assertattr('nms_top_k', kargs)
        self.assertattr('nms_eta', kargs)
        self.assertattr('code_type', kargs)
        self.assertattr('keep_top_k', kargs)
        self.assertattr('variance_encoded_in_target', kargs)
        self.addattr('priorbox_data', kargs, None)  # gets filled out in optimization
        self.addattr('priorbox_center_size_data', kargs, None)  # gets filled out in optimization
        self.addattr('scale_h', kargs, 0)  # gets filled out in optimization
        self.addattr('scale_w', kargs, 0)  # gets filled out in optimization
        self.addattr('scale_y', kargs, 0)  # gets filled out in optimization
        self.addattr('scale_x', kargs, 0)  # gets filled out in optimization

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return self.output_dims[:]


class DropoutOp(Op):
    TRANSLATION_KEY = 'dropout'

    def __init__(self, name, keep):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.keep = keep

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class ElementwiseBinarySubOp(Op):
    TRANSLATION_KEY = 'elementwise_binary_sub'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        if len(input_shapes) != 2:
            converter_utils.log_assert(ValueError, "Binary Ops need exactly 2 inputs, got {}",
                                       len(input_shapes))
        # Binary Ops support broadcasting so need to find the largest shape of the 2 Ops
        output_shape = translation_utils.get_broadcasted_shape(input_shapes)
        return [output_shape]


class ElementwiseBinaryMinOp(Op):
    TRANSLATION_KEY = 'elementwise_binary_min'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class ElementwiseBinaryMaxOp(Op):
    TRANSLATION_KEY = 'elementwise_binary_max'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class ElementwiseBinaryDivOp(Op):
    TRANSLATION_KEY = 'elementwise_binary_div'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        if len(input_shapes) != 2:
            converter_utils.log_assert(ValueError, "Binary Ops need exactly 2 inputs, got {}",
                                       len(input_shapes))
        # Binary Ops support broadcasting so need to find the largest shape of the 2 Ops
        output_shape = translation_utils.get_broadcasted_shape(input_shapes[:])
        return [output_shape]


class ElementwiseBinaryProductOp(Op):
    TRANSLATION_KEY = 'elementwise_binary_product'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        if len(input_shapes) != 2:
            converter_utils.log_assert(ValueError, "Binary Ops need exactly 2 inputs, got {}",
                                       len(input_shapes))
        # Binary Ops support broadcasting so need to find the largest shape of the 2 Ops
        output_shape = translation_utils.get_broadcasted_shape(input_shapes[:])
        return [output_shape]


class ElementwiseDivOp(Op):
    TRANSLATION_KEY = 'elementwise_div'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class ElementwiseMaxOp(Op):
    TRANSLATION_KEY = 'elementwise_max'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class ElementwiseMinOp(Op):
    TRANSLATION_KEY = 'elementwise_min'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class ElementwiseProductOp(Op):
    TRANSLATION_KEY = 'elementwise_product'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class ElementwiseSubOp(Op):

    TRANSLATION_KEY = 'elementwise_sub'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class ElementwiseSumOp(Op):
    TRANSLATION_KEY = 'elementwise_sum'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.addattr('coeffs', kargs, [])

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        if not isinstance(input_shapes, list):
            raise TypeError("input_shapes is not of type List")
        output_shape = translation_utils.get_broadcasted_shape(input_shapes[:])
        return [output_shape]


class ElementwiseUnaryAbsOp(Op):
    TRANSLATION_KEY = 'elementwise_unary_abs'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class ElementwiseUnaryExpOp(Op):
    TRANSLATION_KEY = 'elementwise_unary_exp'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class ElementwiseUnaryFloorOp(Op):
    TRANSLATION_KEY = 'elementwise_unary_floor'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class ElementwiseUnaryLogOp(Op):
    TRANSLATION_KEY = 'elementwise_unary_log'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class ElementwiseUnaryNegOp(Op):
    TRANSLATION_KEY = 'elementwise_unary_neg'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class ElementwiseUnarySinOp(Op):
    TRANSLATION_KEY = 'elementwise_unary_sin'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class ElementwiseUnarySqrtOp(Op):
    TRANSLATION_KEY = 'elementwise_unary_sqrt'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class EmbeddingOp(Op):
    TRANSLATION_KEY = 'embedding'

    def __init__(self, name, output_dim,  **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.output_dim = output_dim
        self.addattr('embedding_strategy', kwargs, 'EMBEDDING_PARTITION_STRATEGY_MOD')

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [self.output_dim]


class ExtractGlimpseOp(Op):
    TRANSLATION_KEY = 'extract_glimpse'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr("glimpse_width", kwargs)
        self.assertattr("glimpse_height", kwargs)
        self.assertattr("centered", kwargs)
        self.assertattr("normalized", kwargs)
        self.assertattr("uniform_noise", kwargs)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        input_shape = input_shapes[0][:]
        if len(input_shape) != 4:
            raise ValueError("ExtractGlimpse accepts only 4-D tensor input")
        batch_size, _, _, depth = axis_order.extract_spatial_dims(input_shape)
        output_shape = axis_order.format_spatial_output_shape(batch_size=batch_size,
                                                              depth=depth,
                                                              height=self.glimpse_height,
                                                              width=self.glimpse_width)
        return [output_shape]


class FullyConnectedOp(Op):
    TRANSLATION_KEY = 'fully_connected'

    def __init__(self, name, weights, bias, output_shape=None, transpose_a=False, transpose_b=True):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.output_shape=output_shape
        self.transpose_a=transpose_a
        self.transpose_b=transpose_b

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        if self.output_shape == None:
            batch = input_shapes[0][0]
            out_channels, in_channels = axis_order.extract_fc_weights_dims(list(self.weights.shape))
            return [[batch, out_channels]]
        return [self.output_shape]


class GatherOp(Op):
    TRANSLATION_KEY = 'gather'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.addattr('axis', kargs, 0)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        output_shape = input_shapes[0][:self.axis] + list(input_shapes[1]) + input_shapes[0][self.axis + 1:]
        return [output_shape]


class GenerateProposalsOp(Op):
    TRANSLATION_KEY = 'generate_proposals'

    def __init__(self, name, anchors, im_info, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('spatial_scale', kargs)
        self.assertattr('pre_nms_top_n', kargs)
        self.assertattr('post_nms_top_n', kargs)
        self.assertattr('nms_thresh', kargs)
        self.assertattr('min_size', kargs)
        self.addattr('correct_transform_coords', kargs, True)


class GruOp(Op):
    TRANSLATION_KEY = 'gru'

    def __init__(self, name, state_gate, forget_gate, control_gate, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.state_gate = state_gate
        self.forget_gate = forget_gate
        self.control_gate = control_gate
        self.addattr('activation', kargs, "NEURON_LOGISTIC")
        self.addattr('gate_activation', kargs, "NEURON_LOGISTIC")
        self.addattr('rec_gate_activation', kargs, "NEURON_TANH")
        self.addattr('backwards', kargs, False)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        def get_c_h_output_dims(axis_order, batch_size, output_depth):
            if axis_order == AxisOrders.ONNX:
                c_t_dims = [1, batch_size, output_depth]
                h_t_dims = [1, batch_size, output_depth]
            else:
                c_t_dims = [batch_size, output_depth]
                h_t_dims = [batch_size, output_depth]
            return [c_t_dims, h_t_dims]

        input_shape = input_shapes[0][:]
        time_steps = 1
        batch_size = 1
        output_dims = []
        if len(input_shape) == 3:
            batch_size, time_steps, _ = axis_order.extract_time_series_dims(input_shape)
        output_depth = self.control_gate['rec_weights'].shape[1]  # Num of hidden units
        output_dims.append(
            axis_order.format_time_series_output_shape(batch_size=batch_size,
                                                       time_steps=time_steps,
                                                       feature=output_depth)
        )

        if self.c_0_input_name and self.h_0_input_name:
            # Layer has exposed recurrent inputs, therefore we need to add c_T and h_T outputs
            c_dims, h_dims = get_c_h_output_dims(axis_order, batch_size, output_depth)
            output_dims.append(c_dims)
            output_dims.append(h_dims)

        return output_dims


class ImageProjectiveTransformOp(Op):
    TRANSLATION_KEY = 'image_projective_transform'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.addattr("interpolation_mode", kwargs, "BILINEAR")
        self.addattr("output_shape", kwargs, None)
        if self.output_shape is not None and len(self.output_shape) != 2:
            raise ValueError("Output Shape specified in {0} needs to be 2-D in shape".format(name))

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        num_images, input_height, input_width, num_channels = input_shapes[0][:]
        if self.output_shape:
            return [[num_images, self.output_shape[0], self.output_shape[1], num_channels]]
        else:
            return [[num_images, input_height, input_width, num_channels]]


class L2NormOp(Op):
    TRANSLATION_KEY = 'l2_norm'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.addattr('axis', kwargs, -1)
        self.addattr('epsilon', kwargs, 1e-12)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class LstmOp(Op):
    TRANSLATION_KEY = 'lstm'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('input_weights', kwargs)
        self.assertattr('gate_bias', kwargs)
        self.assertattr('hidden_state_weights', kwargs)
        self.addattr('w_xc_static', kwargs, None)
        self.addattr('backward', kwargs, False)
        self.addattr('activations', kwargs, [])
        self.addattr('reset_state_at_time_step_0', kwargs, False)
        self.addattr('h_0_input_name', kwargs, '')
        self.addattr('c_0_input_name', kwargs, '')
        self.addattr('sequence_continuation_name', kwargs, '')
        self.addattr('x_static_name', kwargs, '')
        self.addattr('w_cc', kwargs, None)
        self.addattr('cell_clip', kwargs, 0.0)
        self.addattr('w_p', kwargs, None)
        self.addattr('b_p', kwargs, None)
        self.addattr('projection_clip', kwargs, 0.0)
        self.addattr('w_n', kwargs, 0.0)
        self.addattr('epsilon', kwargs, 0.0)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        def get_c_h_output_dims(axis_order, batch_size, output_depth):
            if axis_order == AxisOrders.ONNX:
                c_t_dims = [1, batch_size, output_depth]
                h_t_dims = [1, batch_size, output_depth]
            else:
                c_t_dims = [batch_size, output_depth]
                h_t_dims = [batch_size, output_depth]
            return [c_t_dims, h_t_dims]

        input_shape = input_shapes[0][:]
        time_steps = 1
        batch_size = 1
        output_dims = []
        if len(input_shape) == 3:
            batch_size, time_steps, _ = axis_order.extract_time_series_dims(input_shape)
        output_depth = self.input_weights.shape[-1]  # Num of hidden units
        output_dims.append(
            axis_order.format_time_series_output_shape(batch_size=batch_size,
                                                       time_steps=time_steps,
                                                       feature=output_depth)
        )

        if self.c_0_input_name and self.h_0_input_name:
            # Layer has exposed recurrent inputs, therefore we need to add c_T and h_T outputs
            c_dims, h_dims = get_c_h_output_dims(axis_order, batch_size, output_depth)
            output_dims.append(c_dims)
            output_dims.append(h_dims)

        return output_dims

class MatMulOp(Op):
    TRANSLATION_KEY = 'matmul'

    def __init__(self, name, bias, transpose_a, transpose_b):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.bias = bias
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        if len(input_shapes[0]) == 1 or len(input_shapes[1]) == 1:
            converter_utils.log_assert(ValueError, "Shape of matrix must be rank 2 but rank is 1")
        if len(input_shapes[0]) != len(input_shapes[1]):
            converter_utils.log_assert(ValueError, "Mismatch in input shapes to MatMul OP")
        input1 = input_shapes[0]
        input2 = input_shapes[1]
        output_shape = []
        # matrix1: axb matrix2: cxd
        a = input1[-2]
        b = input1[-1]
        c = input2[-2]
        d = input2[-1]
        if self.transpose_a:
            a,b = b,a
        if self.transpose_b:
            c,d = d,c
        if len(input1) == 4:
            output_shape = [[input1[0], input1[1], a, d]]
        elif len(input1) == 3:
            output_shape = [[input1[0], a, d]]
        elif len(input1) == 2:
            output_shape = [[a, d]]
        return output_shape

class MaxYOp(Op):
    TRANSLATION_KEY = 'max_y'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        output_shape = input_shapes[0][:]
        if len(output_shape) < 3:
            raise ValueError("MaxY layer expects input shape of at lesat Rank 3")
        idx = len(output_shape) - 3
        output_shape[idx] = 1
        return [output_shape]


class MomentOp(Op):
    TRANSLATION_KEY = 'moment'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axes', kwargs)
        self.addattr('keep_dims', kwargs, False)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        output_shape = []
        input_shape = input_shapes[0]
        for i in range(len(input_shape)):
            dim = input_shape[i]
            if i in self.axes:
                dim = 1
                if self.keep_dims:
                    output_shape.append(dim)
            else:
                output_shape.append(dim)
        return [output_shape, output_shape]


class NegOp(Op):
    TRANSLATION_KEY = 'neg'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return input_shapes[:]


class NeuronOp(Op):
    TRANSLATION_KEY = 'neuron'

    def __init__(self, name, neuron_type, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.neuron_type = neuron_type
        self.addattr('a', kargs, 0.0)
        self.addattr('b', kargs, 0.0)
        self.addattr('min_clamp', kargs, 0.0)
        self.addattr('max_clamp', kargs, 0.0)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return input_shapes[:]


class NonMaxSuppresionOp(Op):
    TRANSLATION_KEY = 'non_max_suppression'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr("max_total_detections", kwargs)
        self.addattr("max_detections_per_class", kwargs, self.max_total_detections)
        self.assertattr("iou_threshold", kwargs)
        self.addattr("score_threshold", kwargs, 0)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        input_shape = input_shapes[0][:]
        out_dim_1 = [input_shape[0], self.max_total_detections, 4]
        out_dim_2 = [input_shape[0], self.max_total_detections]
        out_dim_3 = [input_shape[0], self.max_total_detections]

        output_shape = [out_dim_1, out_dim_2, out_dim_3]
        if num_outputs == 4 and len(input_shapes) == 2:
            # add dim for num_det. (i.e SSD case, (nms + gather) will have more than 2 inputs
            # to have more than 3 outputs.
            output_shape.append([input_shape[0]])

        # TODO: following outputs added for VIVO support of nms + gather in 1.23.0 to support features as inputs
        # required until nms layer supported independently
        for i in range(0, len(input_shapes)):
            if i >= 2:
                shape = input_shapes[i][:]
                shape[0] = self.max_total_detections
                output_shape.append(shape)

        return output_shape


class NoopOp(Op):
    TRANSLATION_KEY = 'noop'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return input_shapes[:num_outputs]


class PackOp(Op):
    TRANSLATION_KEY = 'pack'

    def __init__(self, name, output_dim, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.output_dim = output_dim
        self.addattr('axis', kwargs, 0)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [self.output_dim for _ in range(num_outputs)]


class PadOp(Op):
    TRANSLATION_KEY = 'pad'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pads', kargs)
        self.addattr('mode', kargs, "PADDING_CONSTANT")
        self.addattr('constant_value', kargs, 0.0)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        input_shape = input_shapes[0]
        output_shape = []

        for i in range(0, len(input_shape)):
            output_shape.append(input_shape[i] + self.pads[i][0] + self.pads[i][1])

        return [output_shape]


class PermuteOp(Op):
    TRANSLATION_KEY = 'permute'

    def __init__(self, name, order):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.order = order

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        output_shape = []
        input_shape = input_shapes[0][:]

        for axis in self.order:
            output_shape.append(input_shape[axis])

        return [output_shape]


class PixelShuffleOp(Op):
    TRANSLATION_KEY = 'pixel_shuffle'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr("upscale_factor", kwargs)
        self.addattr("data_format", kwargs, "NHWC")

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        input_shape = input_shapes[0][:]

        batch = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        depth = input_shape[3]

        output_shape = [batch,
                        height*self.upscale_factor,
                        width*self.upscale_factor,
                        depth/(self.upscale_factor**2)]
        return [output_shape]


class PoolOp(Op):
    TRANSLATION_KEY = 'pool'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pool_type', kargs)
        self.assertattr('size_x', kargs)
        self.assertattr('size_y', kargs)
        self.addattr('stride_x', kargs, 1)
        self.addattr('stride_y', kargs, 1)
        self.addattr('padx_before', kargs, 0)
        self.addattr('padx_after', kargs, 0)
        self.addattr('pady_before', kargs, 0)
        self.addattr('pady_after', kargs, 0)
        self.addattr('padding_size_strategy', kargs, IRPaddingStrategies.PADDING_SIZE_EXPLICIT)
        self.addattr('pool_region_include_padding', kargs, True)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        batch_size, input_height, input_width, depth = axis_order.extract_spatial_dims(input_shapes[0])
        output_height = translation_utils.calc_pool_output_dim(input_height,
                                                               self.size_y,
                                                               self.pady_before,
                                                               self.pady_after,
                                                               self.stride_y,
                                                               self.padding_size_strategy)
        output_width = translation_utils.calc_pool_output_dim(input_width,
                                                              self.size_x,
                                                              self.padx_before,
                                                              self.padx_after,
                                                              self.stride_x,
                                                              self.padding_size_strategy)

        output_shape = axis_order.format_spatial_output_shape(batch_size=batch_size,
                                                              depth=depth,
                                                              height=output_height,
                                                              width=output_width)
        converter_utils.log_debug(code_to_message.get_debugging_message("DEBUG_INFERRED_SHAPE")(self.name,
                                                                                                output_shape))
        return [output_shape]


class PowerOp(Op):
    TRANSLATION_KEY = 'power'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('power', kargs)
        self.addattr('scale', kargs, 1.0)
        self.addattr('shift', kargs, 0.0)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0][:]]


class PreluOp(Op):
    TRANSLATION_KEY = 'prelu'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('coeff', kargs)
        self.addattr('channel_shared', kargs, False)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        # Frontend frameworks support broadcasting of input with coeff. It's up to backend to check shape if
        # multi-directional broadcasting is not supported.
        # so need to find the largest shape of the 2 Ops
        prelu_shapes = [input_shapes[0], self.coeff.shape]
        output_shape = translation_utils.get_broadcasted_shape(prelu_shapes)
        return [output_shape]


class ProposalOp(Op):
    TRANSLATION_KEY = 'proposal'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('feat_stride', kargs)
        self.assertattr('scales', kargs)
        self.assertattr('ratios', kargs)
        self.assertattr('anchor_base_size', kargs)
        self.assertattr('min_bbox_size', kargs)
        self.assertattr('max_num_proposals', kargs)
        self.assertattr('max_num_rois', kargs)
        self.assertattr('iou_threshold_nms', kargs)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        output_shape = [1, 1, self.max_num_rois, 5]
        return [output_shape]


class ReduceMaxOp(Op):
    TRANSLATION_KEY = 'reduce_max'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axes', kargs)
        self.addattr('keep_dims', kargs, True)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        output_shape = []
        input_shape = input_shapes[0]
        for i in range(len(input_shape)):
            dim = input_shape[i]
            if i in self.axes:
                dim = 1
                if self.keep_dims:
                    output_shape.append(dim)
            else:
                output_shape.append(dim)
        return [output_shape]


class ReduceMeanOp(Op):
    TRANSLATION_KEY = 'reduce_mean'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axes', kargs)
        self.addattr('keep_dims', kargs, True)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        output_shape = []
        input_shape = input_shapes[0]
        for i in range(len(input_shape)):
            dim = input_shape[i]
            if i in self.axes:
                dim = 1
                if self.keep_dims:
                    output_shape.append(dim)
            else:
                output_shape.append(dim)
        return [output_shape]


class ReduceMinOp(Op):
    TRANSLATION_KEY = 'reduce_min'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axes', kargs)
        self.addattr('keep_dims', kargs, True)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        output_shape = []
        input_shape = input_shapes[0]
        for i in range(len(input_shape)):
            dim = input_shape[i]
            if i in self.axes:
                dim = 1
                if self.keep_dims:
                    output_shape.append(dim)
            else:
                output_shape.append(dim)
        return [output_shape]


class ReduceProdOp(Op):
    TRANSLATION_KEY = 'reduce_prod'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axes', kargs)
        self.addattr('keep_dims', kargs, True)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        output_shape = []
        input_shape = input_shapes[0]
        for i in range(len(input_shape)):
            dim = input_shape[i]
            if i in self.axes:
                dim = 1
                if self.keep_dims:
                    output_shape.append(dim)
            else:
                output_shape.append(dim)
        return [output_shape]


class ReduceSumOp(Op):
    TRANSLATION_KEY = 'reduce_sum'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axes', kargs)
        self.addattr('keep_dims', kargs, True)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        output_shape = []
        input_shape = input_shapes[0]
        for i in range(len(input_shape)):
            dim = input_shape[i]
            if i in self.axes:
                dim = 1
                if self.keep_dims:
                    output_shape.append(dim)
            else:
                output_shape.append(dim)
        return [output_shape]


class ReshapeOp(Op):
    TRANSLATION_KEY = 'reshape'

    def __init__(self, name, output_shape):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        if not isinstance(output_shape, list):
            output_shape = list(output_shape)
        self.output_shape = output_shape

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [self.output_shape[:]]


class RNormOp(Op):
    TRANSLATION_KEY = 'rnorm'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('size', kwargs)
        self.assertattr('alpha', kwargs)
        self.assertattr('beta', kwargs)
        self.assertattr('k', kwargs)
        self.addattr('across_channels', kwargs, True)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        if not self.across_channels:
            input_rank = len(input_shapes[0])
            if input_rank != 4:
                raise ValueError("Input rank (%d) must equal 4 for WITHIN LRN on node %s" (input_rank, self.name))

        return input_shapes[:]


class RoiAlignOp(Op):
    TRANSLATION_KEY = 'roi_align'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('spatial_scale', kargs)
        self.assertattr('pooled_size_h', kargs)
        self.assertattr('pooled_size_w', kargs)
        self.assertattr('sampling_ratio', kargs)
        # implode batch parameters
        self.addattr('tiled_batch_h', kargs, -1)
        self.addattr('tiled_batch_w', kargs, -1)
        self.addattr('batch_pad_h', kargs, -1)
        self.addattr('batch_pad_w', kargs, -1)
        self.addattr('pad_value', kargs, 0.0)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        def calc_tiled_height(in_height):
            return self.tiled_batch_h * in_height + (self.tiled_batch_h - 1) * self.batch_pad_h

        def calc_tiled_width(in_width):
            return self.tiled_batch_w * in_width + (self.tiled_batch_w - 1) * self.batch_pad_w

        input_shape = input_shapes[0][:]
        _, _, _, depth = axis_order.extract_spatial_dims(input_shape)

        if self.tiled_batch_h > 0:
            output_shape = axis_order.format_spatial_output_shape(batch_size=1,
                                                                  height=calc_tiled_height(self.pooled_size_h),
                                                                  width=calc_tiled_width(self.pooled_size_w),
                                                                  depth=depth)
        else:
            output_shape = axis_order.format_spatial_output_shape(batch_size=1,
                                                                  height=self.pooled_size_h,
                                                                  width=self.pooled_size_w,
                                                                  depth=depth)
        return [output_shape]


class RoiPoolingOp(Op):
    TRANSLATION_KEY = 'roi_pooling'

    def __init__(self, name, output_shape, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pooled_size_h', kargs)
        self.assertattr('pooled_size_w', kargs)
        self.assertattr('spatial_scale', kargs)
        self.output_shape = output_shape

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [self.output_shape[:]]


class ResizeOp(Op):
    TRANSLATION_KEY = 'resize'

    def __init__(self, name, output_shape, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.output_shape = output_shape
        self.addattr('pad_value', kargs, 0.0)
        self.addattr('maintain_aspect_ratio', kargs, False)
        self.addattr('resize_mode', kargs, "RESIZE_BILINEAR")
        self.addattr('scale_height', kargs, 0.0)
        self.addattr('scale_width', kargs, 0.0)
        self.addattr('align_corners', kargs, False)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [self.output_shape[:]]


class RnnTransformationOp(Op):
    TRANSLATION_KEY = 'rnn_transformation'

    def __init__(self, name, weights, bias, activation):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        def get_c_h_output_dims(axis_order, batch_size, output_depth):
            if axis_order == AxisOrders.ONNX:
                c_t_dims = [1, batch_size, output_depth]
                h_t_dims = [1, batch_size, output_depth]
            else:
                c_t_dims = [batch_size, output_depth]
                h_t_dims = [batch_size, output_depth]
            return [c_t_dims, h_t_dims]

        input_shape = input_shapes[0][:]
        time_steps = 1
        batch_size = 1
        output_dims = []
        if len(input_shape) == 3:
            batch_size, time_steps, _ = axis_order.extract_time_series_dims(input_shape)
        output_depth = self.weights.shape[-2]  # Num of hidden units
        output_dims.append(
            axis_order.format_time_series_output_shape(batch_size=batch_size,
                                                       time_steps=time_steps,
                                                       feature=output_depth)
        )

        if self.c_0_input_name and self.h_0_input_name:
            # Layer has exposed recurrent inputs, therefore we need to add c_T and h_T outputs
            c_dims, h_dims = get_c_h_output_dims(axis_order, batch_size, output_depth)
            output_dims.append(c_dims)
            output_dims.append(h_dims)

        return output_dims


class ScaleOp(Op):
    TRANSLATION_KEY = 'scale'

    def __init__(self, name, weights, bias, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.weights = weights
        self.bias = bias
        self.assertattr('axis', kargs)
        self.assertattr('num_axes', kargs)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class SliceOp(Op):
    TRANSLATION_KEY = 'slice'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('axis', kargs)
        self.assertattr('slice_points', kargs)
        self.assertattr('output_shape', kargs)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return self.output_shape[:]


class StridedSliceOp(Op):
    TRANSLATION_KEY = 'strided_slice'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('begin', kargs)
        self.assertattr('end', kargs)
        self.assertattr('strides', kargs)
        self.addattr('shrink_axis_mask', kargs, 0)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        # deep copy so that original is not modified
        output_shape = input_shapes[0][:]

        for i, shape in enumerate(output_shape):
            output_shape[i] = int(math.ceil(float(self.end[i] - self.begin[i]) / self.strides[i]))
            if self.shrink_axis_mask & (1 << i):
                output_shape[i] = 1

        return [output_shape]


class StaticOp(Op):
    TRANSLATION_KEY = 'static'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return []


class SoftmaxOp(Op):
    TRANSLATION_KEY = 'softmax'

    def __init__(self, name):
        Op.__init__(self, name, self.TRANSLATION_KEY)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return input_shapes[:]


class SpaceToDepthOp(Op):
    TRANSLATION_KEY = 'space_to_depth'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr("downscale_factor", kwargs)
        self.addattr("data_format", kwargs, "NHWC")

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        input_shape = input_shapes[0][:]

        batch, height, width, depth = axis_order.extract_spatial_dims(input_shape)

        output_shape = axis_order.format_spatial_output_shape(batch_size=batch,
                                                              depth=depth * (self.downscale_factor**2),
                                                              height=height/self.downscale_factor,
                                                              width=width/self.downscale_factor)

        return [output_shape]


class SsdOp(Op):
    TRANSLATION_KEY = 'ssd'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr("scale_y", kwargs)
        self.assertattr("scale_x", kwargs)
        self.assertattr("scale_h", kwargs)
        self.assertattr("scale_w", kwargs)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class SubtractMeanOp(Op):
    TRANSLATION_KEY = 'subtract_mean'

    def __init__(self, name, mean_values):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.mean_values = mean_values

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class TileOp(Op):
    TRANSLATION_KEY = 'tile'

    def __init__(self, name, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr("multiples", kwargs)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        input_rank = len(input_shapes[0])
        multiples_len = len(self.multiples)

        if input_rank != multiples_len:
            raise ValueError("Multiples length (%d) doesn't equal input rank (%d) on node %s" % (multiples_len,
                                                                                                 input_rank,
                                                                                                 self.name))
        return [[input_shapes[0][i] * self.multiples[i] for i in range(input_rank)]]


class UdoOp(Op):
    TRANSLATION_KEY = 'udo'

    def __init__(self,
                 name,
                 package_name,
                 udo_type,
                 inputs,
                 outputs,
                 params,
                 output_dims,
                 infer_shape_method,
                 axis_orders):

        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.udo_type = udo_type
        self.output_dims = output_dims
        self.package_name = package_name
        self.infer_shape_method = infer_shape_method
        self.axis_orders = axis_orders
        self.inputs = inputs
        self.outputs = outputs
        for attr_name, attr in params.items():
            self.attrs[attr_name] = attr

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        if self.infer_shape_method is None:
            return self.output_dims
        elif callable(self.infer_shape_method):
            # method has already been checked, but is verified again here.
            # this means the user has provided their own infer shapes method,
            # which we will now call here.
            return self.infer_shape_method(input_shapes, self.attrs)

    def populate_axis_format(self, buf, axis_order):
        # if the buffer is rank 4 and the axis order has been defined
        # (note only 4-d axis orders can be defined in a UdoOp) or the axis format is non-trivial,
        # then we keep the format set by the UdoOp object. Otherwise, the axis format will be set
        # according to framework AxisOrder class using the buffer rank when we call
        # populate axis format.
        if (buf.rank() == 4 and self.axis_orders[buf.name] != 'NOT_YET_DEFINED') or \
                self.axis_orders[buf.name] == 'NON-TRIVIAL':
            buf.axis_format = self.axis_orders[buf.name]
        else:
            buf.populate_axis_format(axis_order)


class UdlOp(Op):
    TRANSLATION_KEY = 'udl'

    def __init__(self, name, layer_type, blob, output_dims, expected_input_axis_orders, expected_output_axis_orders):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.layer_type = layer_type
        self.blob = blob
        self.output_dims = output_dims
        self.expected_input_axis_orders = expected_input_axis_orders
        self.expected_output_axis_orders = expected_output_axis_orders


class UnpackOp(Op):
    TRANSLATION_KEY = 'unpack'

    def __init__(self, name, output_dim, **kwargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.output_dim = output_dim
        self.addattr('axis', kwargs, 0)
        self.addattr('num', kwargs, 2)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [self.output_dim for _ in range(self.num)]


class Upsample(ResizeOp):
    TRANSLATION_KEY = "upsample"


class UpsampleIndexBasedOp(Op):
    TRANSLATION_KEY = 'upsample_index_based'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pool_size', kargs)
        self.addattr('pool_stride', kargs, 1)
        self.addattr('pad', kargs, 0)
        self.addattr('output_height', kargs, -1)
        self.addattr('output_width', kargs, -1)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]


class UpsampleSparseOp(Op):
    TRANSLATION_KEY = 'upsample_sparse'

    def __init__(self, name, **kargs):
        Op.__init__(self, name, self.TRANSLATION_KEY)
        self.assertattr('pool_size', kargs)
        self.addattr('pool_stride', kargs, 1)
        self.addattr('pad', kargs, 0)
        self.addattr('output_height', kargs, -1)
        self.addattr('output_width', kargs, -1)

    def infer_shape(self, input_shapes, num_outputs, axis_order):
        return [input_shapes[0]]
