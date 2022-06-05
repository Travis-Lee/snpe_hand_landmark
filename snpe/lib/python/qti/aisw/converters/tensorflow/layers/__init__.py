# =============================================================================
#
#  Copyright (c) 2015-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import numpy as np


from qti.aisw.converters.tensorflow.layers.fullyconnected import (
    FullyConnectedLayerResolver,
    FullyConnectedLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.convolution import (
    ConvolutionLayerResolver,
    ConvolutionLayerBuilder,
    GroupedConvolutionLayerResolver,
    DilatedConvolutionLayerResolver,
    DepthwiseConvolutionLayerResolver,
    DilatedDepthwiseConvolutionLayerResolver
)
from qti.aisw.converters.tensorflow.layers.concat import (
    ConcatLayerResolver,
    ConcatLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.relu import (
    ReluLayerResolver,
    ReluLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.relu_min_max import (
    ReluMinMaxLayerResolver,
    ReluMinMaxLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.relu6 import (
    Relu6LayerResolver
)
from qti.aisw.converters.tensorflow.layers.sigmoid import (
    SigmoidLayerResolver,
    SigmoidLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.tanh import (
    TanhLayerResolver,
    TanhLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.softmax import (
    SoftmaxLayerResolver,
    SoftmaxLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.lrn import (
    LrnLayerResolver,
    LrnLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.embedding import (
    EmbeddingLayerResolver,
    EmbeddingLayerBuilder,
    GatherLayerResolver,
    GatherLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.deconvolution import (
    DeConvolutionOptimizedLayerResolver,
    DeConvolutionLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.batchnorm import (
    BatchNormLayerResolver,
    BatchNormWithEltwiseLayerResolver,
    BatchNormWithGlobalNormLayerResolver,
    BatchNormLayerBuilder,
    FusedBatchNormNormLayerResolver,
    GenericBatchNormLayerResolver
)

from qti.aisw.converters.tensorflow.layers.instance_norm import (
    InstanceNormLayerBuilder,
    InstanceNormLayerResolver
)

from qti.aisw.converters.tensorflow.layers.pooling import (
    AvgPoolingLayerResolver,
    MaxPoolingLayerResolver,
    PoolingLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.eltwise import (
    EltWiseSumLayerResolver,
    EltWiseSumLayerBuilder,
    EltWiseSubLayerResolver,
    EltWiseSubLayerBuilder,
    EltWiseMulLayerResolver,
    EltWiseMulLayerBuilder,
    EltWiseMaxLayerResolver,
    EltWiseMaxLayerBuilder,
    EltWiseMinLayerResolver,
    EltWiseMinLayerBuilder,
    EltWiseDivLayerResolver,
    EltWiseDivLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.add_n import (
    AddNLayerResolver,
    AddNLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.slice import (
    SliceLayerResolver,
    SliceLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.prelu import (
    PReLuLayerResolver,
    LeakyReLuLayerResolver,
    PReLuLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.reshape import (
    ReshapeLayerResolver,
    ReshapeLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.resize import (
    ResizeNearestNeighborLayerResolver,
    ResizeBilinearLayerResolver,
    ResizeLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.l2_normalize import (
    L2NormLayerResolver,
    L2NormLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.lstm import (
    LstmLayerResolver,
    LstmLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.ignored_patterns import (
    IgnoredLayersResolver,
    IgnoredLayersBuilder
)

from qti.aisw.converters.tensorflow.layers.fill import (
    FillLayerResolver,
    FillLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.ssd import (
    SSDDecoderResolver,
    SSDDecoderLayersBuilder,
    SSDNmsResolver,
    SSDNmsLayersBuilder,
    SSDAnchorGeneratorResolver,
)

from qti.aisw.converters.tensorflow.layers.crop import (
    CropLayerResolver,
    CropLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.constant import (
    ConstantLayerResolver,
    ConstantLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.pad import (
    PadLayerResolver,
    PadLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.strided_slice import (
    StridedSliceLayerResolver,
    StridedSliceLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.permute import (
    PermuteLayerResolver,
    PermuteLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.argmax import (
    ArgMaxLayerResolver,
    ArgMaxLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.argmin import (
    ArgMinLayerResolver,
    ArgMinLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.channel_shuffle import (
    ChannelShuffleLayerResolver,
    ChannelShuffleLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.elu import (
    EluLayerResolver,
    EluLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.reduction import (
    ReductionMeanLayerResolver,
    ReductionMeanLayerBuilder,
    ReductionProdLayerResolver,
    ReductionProdLayerBuilder,
    ReductionSumLayerResolver,
    ReductionSumLayerBuilder,
    ReductionMinLayerResolver,
    ReductionMinLayerBuilder,
    ReductionMaxLayerResolver,
    ReductionMaxLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.eltwise_unary import (
    EltWiseUnarySqrtLayerResolver,
    EltWiseUnarySqrtLayerBuilder,
    EltWiseUnaryAbsLayerResolver,
    EltWiseUnaryAbsLayerBuilder,
    EltWiseUnaryFloorLayerResolver,
    EltWiseUnaryFloorLayerBuilder,
    EltWiseUnaryExpLayerResolver,
    EltWiseUnaryExpLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.pow import (
    PowLayerResolver,
    PowLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.tile import (
    TileLayerResolver,
    TileLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.extract_glimpse import (
    ExtractGlimpseLayerResolver,
    ExtractGlimpseLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.image_projective_transform import (
    ImageProjectiveTransformLayerResolver,
    ImageProjectiveTransformLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.fake_quant import (
    FakeQuantLayerResolver,
    FakeQuantLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.matmul import(
    MatMulLayerResolver,
    MatMulLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.pixel_shuffle import (
    PixelShuffleLayerResolver,
    PixelShuffleLayerBuilder
)


from qti.aisw.converters.tensorflow.layers.crop_and_resize import (
    CropAndResizeLayerResolver,
    CropAndResizeLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.non_max_suppression import (
    NonMaxSuppressionLayerResolver,
    NonMaxSuppressionLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.moments import (
    MomentsLayerResolver,
    MomentsLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.space_to_depth import (
    SpaceToDepthLayerResolver,
    SpaceToDepthLayerBuilder
)

from qti.aisw.converters.tensorflow.layers.caffe_ssd import (
    CaffeSsdLayerResolver,
    CaffeSsdLayerBuilder
)
from qti.aisw.converters.tensorflow.layers.pack import (
    PackLayerResolver,
    PackLayerBuilder,
    UnPackLayerResolver,
    UnpackLayerBuilder
)

# TODO: Remove after Custom Op API Restructure
try:
    from qti.aisw.converters.tensorflow.layers.custom import (
        CustomLayerResolver,
        CustomLayerBuilder,
    )
except ImportError:
    # temporary fix since this file is not enabled on SNPE
    pass

from qti.aisw.converters.tensorflow.layers.udo import (
    UdoLayerResolver,
    UdoLayerBuilder,
)

from qti.aisw.converters.tensorflow.common import (
    LayerDescriptor,
    LayerResolver,
    LayerBuilder
)

layer_resolvers = [
    IgnoredLayersResolver,
    FakeQuantLayerResolver,
    CaffeSsdLayerResolver,
    SSDAnchorGeneratorResolver,
    SSDNmsResolver,
    ConvolutionLayerResolver,
    ReshapeLayerResolver,
    ConcatLayerResolver,
    FullyConnectedLayerResolver,
    ReluLayerResolver,
    Relu6LayerResolver,
    ReluMinMaxLayerResolver,
    SigmoidLayerResolver,
    TanhLayerResolver,
    AvgPoolingLayerResolver,
    MaxPoolingLayerResolver,
    NonMaxSuppressionLayerResolver,
    SoftmaxLayerResolver,
    L2NormLayerResolver,
    LrnLayerResolver,
    DeConvolutionOptimizedLayerResolver,
    InstanceNormLayerResolver,
    EltWiseSumLayerResolver,
    EltWiseSubLayerResolver,
    EltWiseMulLayerResolver,
    EltWiseMaxLayerResolver,
    EltWiseMinLayerResolver,
    EltWiseDivLayerResolver,
    BatchNormWithEltwiseLayerResolver,
    BatchNormWithGlobalNormLayerResolver,
    GenericBatchNormLayerResolver,
    GroupedConvolutionLayerResolver,
    SliceLayerResolver,
    PackLayerResolver,
    UnPackLayerResolver,
    PReLuLayerResolver,
    LeakyReLuLayerResolver,
    DilatedConvolutionLayerResolver,
    ResizeBilinearLayerResolver,
    ResizeNearestNeighborLayerResolver,
    DilatedDepthwiseConvolutionLayerResolver,
    DepthwiseConvolutionLayerResolver,
    AddNLayerResolver,
    LstmLayerResolver,
    FillLayerResolver,
    SSDDecoderResolver,
    CropLayerResolver,
    FusedBatchNormNormLayerResolver,
    EmbeddingLayerResolver,
    PadLayerResolver,
    PowLayerResolver,
    PixelShuffleLayerResolver,
    StridedSliceLayerResolver,
    PermuteLayerResolver,
    ArgMaxLayerResolver,
    ArgMinLayerResolver,
    ChannelShuffleLayerResolver,
    EluLayerResolver,
    TileLayerResolver,
    GatherLayerResolver,
    ReductionMeanLayerResolver,
    ReductionProdLayerResolver,
    ReductionSumLayerResolver,
    ReductionMinLayerResolver,
    ReductionMaxLayerResolver,
    EltWiseUnarySqrtLayerResolver,
    EltWiseUnaryAbsLayerResolver,
    EltWiseUnaryFloorLayerResolver,
    EltWiseUnaryExpLayerResolver,
    ExtractGlimpseLayerResolver,
    ImageProjectiveTransformLayerResolver,
    CropAndResizeLayerResolver,
    MomentsLayerResolver,
    MatMulLayerResolver,
    SpaceToDepthLayerResolver,
]
"""
type: list[type(LayerResolver)]
"""

layer_builders = {
    BatchNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    BatchNormWithGlobalNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    GenericBatchNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    CaffeSsdLayerResolver.Descriptor: CaffeSsdLayerBuilder,
    GatherLayerResolver.Descriptor: GatherLayerBuilder,
    ConcatLayerResolver.Descriptor: ConcatLayerBuilder,
    ConvolutionLayerResolver.Descriptor: ConvolutionLayerBuilder,
    DeConvolutionOptimizedLayerResolver.Descriptor: DeConvolutionLayerBuilder,
    EltWiseMaxLayerResolver.Descriptor: EltWiseMaxLayerBuilder,
    EltWiseMinLayerResolver.Descriptor: EltWiseMinLayerBuilder,
    EltWiseMulLayerResolver.Descriptor: EltWiseMulLayerBuilder,
    EltWiseSumLayerResolver.Descriptor: EltWiseSumLayerBuilder,
    EltWiseSubLayerResolver.Descriptor: EltWiseSubLayerBuilder,
    EltWiseDivLayerResolver.Descriptor: EltWiseDivLayerBuilder,
    InstanceNormLayerResolver.Descriptor: InstanceNormLayerBuilder,
    AddNLayerResolver.Descriptor: AddNLayerBuilder,
    TileLayerResolver.Descriptor: TileLayerBuilder,
    FullyConnectedLayerResolver.Descriptor: FullyConnectedLayerBuilder,
    FakeQuantLayerResolver.Descriptor: FakeQuantLayerBuilder,
    L2NormLayerResolver.Descriptor: L2NormLayerBuilder,
    LrnLayerResolver.Descriptor: LrnLayerBuilder,
    ReluLayerResolver.Descriptor: ReluLayerBuilder,
    Relu6LayerResolver.Descriptor: ReluMinMaxLayerBuilder,
    ReluMinMaxLayerResolver.Descriptor: ReluMinMaxLayerBuilder,
    SigmoidLayerResolver.Descriptor: SigmoidLayerBuilder,
    SoftmaxLayerResolver.Descriptor: SoftmaxLayerBuilder,
    TanhLayerResolver.Descriptor: TanhLayerBuilder,
    AvgPoolingLayerResolver.Descriptor: PoolingLayerBuilder,
    MaxPoolingLayerResolver.Descriptor: PoolingLayerBuilder,
    NonMaxSuppressionLayerResolver.Descriptor: NonMaxSuppressionLayerBuilder,
    GroupedConvolutionLayerResolver.Descriptor: ConvolutionLayerBuilder,
    SliceLayerResolver.Descriptor: SliceLayerBuilder,
    PixelShuffleLayerResolver.Descriptor: PixelShuffleLayerBuilder,
    PReLuLayerResolver.Descriptor: PReLuLayerBuilder,
    LeakyReLuLayerResolver.Descriptor: PReLuLayerBuilder,
    DilatedConvolutionLayerResolver.Descriptor: ConvolutionLayerBuilder,
    ReshapeLayerResolver.Descriptor: ReshapeLayerBuilder,
    ResizeBilinearLayerResolver.Descriptor: ResizeLayerBuilder,
    ResizeNearestNeighborLayerResolver.Descriptor: ResizeLayerBuilder,
    LstmLayerResolver.UnrolledTimeStepDescriptor: LstmLayerBuilder,
    LstmLayerResolver.StateDescriptor: LstmLayerBuilder,
    IgnoredLayersResolver.Descriptor: IgnoredLayersBuilder,
    FillLayerResolver.Descriptor: FillLayerBuilder,
    SSDDecoderResolver.Descriptor: SSDDecoderLayersBuilder,
    CropLayerResolver.Descriptor: CropLayerBuilder,
    SSDNmsResolver.Descriptor: SSDNmsLayersBuilder,
    ConstantLayerResolver.Descriptor: ConstantLayerBuilder,
    FusedBatchNormNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    EmbeddingLayerResolver.Descriptor: EmbeddingLayerBuilder,
    PackLayerResolver.Descriptor: PackLayerBuilder,
    PadLayerResolver.Descriptor: PadLayerBuilder,
    UnPackLayerResolver.Descriptor: UnpackLayerBuilder,
    StridedSliceLayerResolver.Descriptor: StridedSliceLayerBuilder,
    PermuteLayerResolver.Descriptor: PermuteLayerBuilder,
    ArgMaxLayerResolver.Descriptor: ArgMaxLayerBuilder,
    ArgMinLayerResolver.Descriptor: ArgMinLayerBuilder,
    ChannelShuffleLayerResolver.Descriptor: ChannelShuffleLayerBuilder,
    EluLayerResolver.Descriptor: EluLayerBuilder,
    PowLayerResolver.Descriptor: PowLayerBuilder,
    ReductionMeanLayerResolver.Descriptor: ReductionMeanLayerBuilder,
    ReductionProdLayerResolver.Descriptor: ReductionProdLayerBuilder,
    ReductionSumLayerResolver.Descriptor: ReductionSumLayerBuilder,
    ReductionMinLayerResolver.Descriptor: ReductionMinLayerBuilder,
    ReductionMaxLayerResolver.Descriptor: ReductionMaxLayerBuilder,
    EltWiseUnarySqrtLayerResolver.Descriptor: EltWiseUnarySqrtLayerBuilder,
    EltWiseUnaryAbsLayerResolver.Descriptor: EltWiseUnaryAbsLayerBuilder,
    EltWiseUnaryFloorLayerResolver.Descriptor: EltWiseUnaryFloorLayerBuilder,
    EltWiseUnaryExpLayerResolver.Descriptor: EltWiseUnaryExpLayerBuilder,
    ExtractGlimpseLayerResolver.Descriptor: ExtractGlimpseLayerBuilder,
    ImageProjectiveTransformLayerResolver.Descriptor: ImageProjectiveTransformLayerBuilder,
    CropAndResizeLayerResolver.Descriptor: CropAndResizeLayerBuilder,
    MomentsLayerResolver.Descriptor: MomentsLayerBuilder,
    SpaceToDepthLayerResolver.Descriptor: SpaceToDepthLayerBuilder,
    MatMulLayerResolver.Descriptor: MatMulLayerBuilder,
    UdoLayerResolver.Descriptor: UdoLayerBuilder
}

# TODO: Remove after Custom Op API Restructure
try:
    layer_builders.update({CustomLayerResolver.Descriptor: CustomLayerBuilder})
except NameError:
    pass

"""
type: dict[type(LayerDescriptor), type(LayerBuilder)]
"""
