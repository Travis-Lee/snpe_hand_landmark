#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import struct

from qti.aisw.converters.common.utils.converter_utils import NpUtils
from qti.aisw.converters.backend import snpe_udl_utils

npUtils = NpUtils()


class LayerType:
    MY_CUSTOM_SCALE_LAYER = 1
    MY_CUSTOM_NORMALIZE_LAYER = 2
    MY_ANOTHER_LAYER = 3


class UdlBlobMyCustomNormalize(snpe_udl_utils.UdlBlob):
    """
    Wrapper class for MyCustomNormalize layer blob
    """
    def __init__(self, layer, weight_provider):
        snpe_udl_utils.UdlBlob.__init__(self)

        # MyCustomNormalize layer reuses  the Caffe Normalize layer params
        caffe_params = layer.norm_param

        # Initialize the SNPE params
        snpe_params = UdlBlobMyCustomNormalize.MyCustomNormalizeLayerParam()

        # fill the params
        snpe_params.across_spatial = caffe_params.across_spatial
        snpe_params.channel_shared = caffe_params.channel_shared

        # fill the weights
        caffe_weights = npUtils.blob2arr(weight_provider.weights_map[layer.name][0])
        snpe_params.weights_dim = list(caffe_weights.shape)
        snpe_params.weights_data = list(caffe_weights.astype(float).flat)

        self._blob = snpe_params.serialize()
        self._size = len(self._blob)

    def get_blob(self):
        return self._blob

    def get_size(self):
        return self._size

    class MyCustomNormalizeLayerParam:
        """
        Helper class for packing blob data
        """
        def __init__(self):
            self.type = LayerType.MY_CUSTOM_NORMALIZE_LAYER
            self.across_spatial = None
            self.channel_shared = None
            self.weights_dim = []
            self.weights_data = []

        def serialize(self):
            packed = struct.pack('i', self.type)
            packed += struct.pack('?', self.across_spatial)
            packed += struct.pack('?', self.channel_shared)
            packed += struct.pack('I%sI' % len(self.weights_dim),
                                  len(self.weights_dim), *self.weights_dim)
            packed += struct.pack('I%sf' % len(self.weights_data),
                                  len(self.weights_data), *self.weights_data)
            return packed


def udl_mycustomnormalize_func(layer, weight_provider, input_dims):
    """
    Conversion callback function for MyCustomScale layer
    """
    # Initialize blob for our custom layer with the wrapper class
    blob = UdlBlobMyCustomNormalize(layer, weight_provider)

    # Input and output dims are the same for MyCustomScale layer
    return snpe_udl_utils.UdlBlobOutput(blob=blob, out_dims=input_dims)
