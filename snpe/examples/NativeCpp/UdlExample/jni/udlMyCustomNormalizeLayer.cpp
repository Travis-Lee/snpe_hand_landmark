//==============================================================================
//
//  Copyright (c) 2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <algorithm>
#include <cstring>
#include <iostream>
#include <iterator>
#include <math.h>
#include <numeric>

#include "udlMyCustomNormalizeLayer.hpp"

namespace {

   std::size_t getSizeByDim(const std::vector<size_t>& dim) {
      return std::accumulate(std::begin(dim), std::end(dim), 1,
                             std::multiplies<size_t>());
   }

} // ns

namespace myudl {

   bool UdlMyCustomNormalize::setup(void *cookie,
                                    size_t insz, const size_t **indim, const size_t *indimsz,
                                    size_t outsz, const size_t **outdim, const size_t *outdimsz) {

      std::cout << "UdlMyCustomNormalize::setup() of name " << m_Context.getName()
                << std::endl;

      if (cookie != (void*) 0xdeadbeaf) {
         std::cerr << "UdlMyCustomNormalize::setup() cookie should be 0xdeadbeaf"
                   << std::endl;
         return false;
      }
      if (insz != 1 or outsz != 1) {
         std::cerr << "UdlMyCustomNormalize::setup() insz=" << insz << " outsz="
                   << outsz << std::endl;
         std::cerr
                 << "UdlMyCustomNormalize::setup() multi-input or multi-output not supported"
                 << std::endl;
         return false;
      }
      if (indimsz[0] != outdimsz[0]) {
         std::cerr << "UdlMyCustomNormalize::setup() not the same number of dim, in:"
                   << indimsz[0] << " != : " << outdimsz[0] << std::endl;
         return false;
      }
      // compute dims and compare. keep the output dim
      size_t inszdim = getSizeByDim(
              std::vector<size_t>(indim[0], indim[0] + indimsz[0]));
      m_OutSzDim = getSizeByDim(
              std::vector<size_t>(outdim[0], outdim[0] + outdimsz[0]));
      std::cout << "UdlMyCustomNormalize::setup() input size dim: " << inszdim
                << ", output: " << m_OutSzDim << std::endl;
      if (inszdim != m_OutSzDim) {
         std::cerr << "UdlMyCustomNormalize::setup() not the same overall dim, in:"
                   << inszdim << " != out: " << m_OutSzDim << std::endl;
         return false;
      }
      // parse the params
      const void* blob = m_Context.getBlob();
      std::cout << "UdlMyCustomNormalize::setup() got blob size "
                << m_Context.getSize() << std::endl;
      if (!blob) {
         std::cerr << "UdlMyCustomNormalize::setup() got null blob " << std::endl;
         return false;
      }

      if (!ParseMyCustomLayerParams(blob, m_Context.getSize(), m_Params)) {
         std::cerr << "UdlMyCustomNormalize::setup() failed to parse layer params "
                   << std::endl;
         return false;
      }

      // Check the params
      if (m_Params.across_spatial) {
         std::cerr << "UdlMyCustomNormalize::setup() across_spatial not supported! "
                   << std::endl;
         return false;
      }
      if (m_Params.channel_shared) {
         std::cerr << "UdlMyCustomNormalize::setup() channel_shared not supported! "
                   << std::endl;
         return false;
      }

      std::cout << "UdlMyCustomNormalize::setup() across_spatial=" << m_Params.across_spatial << std::endl;
      std::cout << "UdlMyCustomNormalize::setup() channel_shared=" << m_Params.channel_shared << std::endl;
      std::cout << "UdlMyCustomNormalize::setup() weight dimensions: (";
      for(size_t i=0; i<m_Params.weights_dim.size(); i++) {
         std::cout << m_Params.weights_dim[i] << ",";
      }
      std::cout << ")" << std::endl;
      std::cout << "UdlMyCustomNormalize::setup() # weights=" << m_Params.weights_data.size() << std::endl;

      return true;
   }

   void UdlMyCustomNormalize::close(void *cookie) noexcept {
      if (cookie != (void*) 0xdeadbeaf) {
         std::cerr << "UdlMyCustomNormalize::close() cookie should be 0xdeadbeaf"
                   << std::endl;
      }
      std::cout << "UdlMyCustomNormalize::close()" << std::endl;
      delete this;
   }

   bool UdlMyCustomNormalize::execute(void *cookie, const float **input,
                                      float **output) {
      if (cookie != (void*) 0xdeadbeaf) {
         std::cerr << "UdlMyCustomNormalize::execute() cookie should be 0xdeadbeaf"
                   << std::endl;
         return false;
      }
      std::cout << "UdlMyCustomNormalize::execute()" << std::endl;
      float norm_factor[128] = {0.0};
      for (int i = 0; i < 128; i++) {
         for (int j = 0; j < 1444; j++) {
            norm_factor[i] += pow(input[0][j * 128 + i], 2.0);
         }
         norm_factor[i] = sqrt(norm_factor[i]);
      }
      for (int i = 0; i < 128; i++) {
         for (int j = 0; j < 1444; j++) {
            output[0][j * 128 + i] = (input[0][j * 128 + i] / norm_factor[i]) * m_Params.weights_data[i];
         }
      }
      return true;
   }

   bool UdlMyCustomNormalize::ParseMyCustomLayerParams(const void* buffer, size_t size,
                                                       MyCustomNormalizeParams& params) {
      if(!ParseCommonLayerParams(buffer, size, m_Params.common_params)) return false;

      size_t r_size = size - sizeof(CommonLayerParams);
      uint8_t* r_buffer = (uint8_t*) buffer;
      r_buffer += sizeof(CommonLayerParams);

      // across_spatial
      if(r_size < sizeof(bool)) return false;
      params.across_spatial = *reinterpret_cast<bool*>(r_buffer);
      r_size -= sizeof(bool);
      r_buffer += sizeof(bool);

      // channel_shared
      if(r_size < sizeof(bool)) return false;
      params.channel_shared = *reinterpret_cast<bool*>(r_buffer);
      r_size -= sizeof(bool);
      r_buffer += sizeof(bool);

      // weights_dim
      // packing order:
      //   uint32_t containing # elements
      //   uint32_t[] containing values
      if(r_size < sizeof(uint32_t)) return false;
      uint32_t num_dims = *reinterpret_cast<uint32_t*>(r_buffer);
      r_size -= sizeof(uint32_t);
      r_buffer += sizeof(uint32_t);

      if(r_size < num_dims*sizeof(uint32_t)) return false;
      uint32_t* dims = reinterpret_cast<uint32_t*>(r_buffer);
      params.weights_dim = std::vector<uint32_t>(dims, dims+num_dims);
      r_size -= num_dims*sizeof(uint32_t);
      r_buffer += num_dims*sizeof(uint32_t);

      // weights_data
      // packing order:
      //   uint32_t containing # elements
      //   float[] containins values
      if(r_size < sizeof(uint32_t)) return false;
      uint32_t num_weights = *reinterpret_cast<uint32_t*>(r_buffer);
      r_size -= sizeof(uint32_t);
      r_buffer += sizeof(uint32_t);

      if(r_size < num_weights*sizeof(float)) return false;
      float* weights = reinterpret_cast<float*>(r_buffer);
      params.weights_data = std::vector<float>(weights, weights+num_weights);
      r_size -= num_weights*sizeof(float);
      r_buffer += num_weights*sizeof(float);

      return r_size == 0;
   }

} // ns batchrun