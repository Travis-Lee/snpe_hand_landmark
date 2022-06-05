//=============================================================================
//
//  Copyright (c) 2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <string>
#include "ConvolutionImplLibCpu.hpp"
#include <chrono>
#include <cmath>

std::unique_ptr<UdoUtil::UdoOperation>
ConvolutionOpDef::createOp(void *perOpInfrastructure,
                           uint32_t numOfInputs,
                           SnpeUdo_TensorParam_t *inputs,
                           uint32_t numOfOutputs,
                           SnpeUdo_TensorParam_t *outputs,
                           uint32_t numOfStaticParams,
                           SnpeUdo_Param_t* params)
{
    return std::unique_ptr<UdoUtil::UdoCpuOperation>
            (new ConvolutionOp(inputs, numOfInputs, outputs, numOfOutputs,
                               static_cast<SnpeUdo_CpuInfrastructure_t*>(perOpInfrastructure),
                               numOfStaticParams, params));
}

SnpeUdo_ErrorType_t
ConvolutionOp::snpeUdoExecute(bool blocking, const uint32_t ID, SnpeUdo_ExternalNotify_t notifyFunc)
{

    auto startTime = std::chrono::high_resolution_clock::now();
    if(!blocking) { return SNPE_UDO_UNSUPPORTED_FEATURE; }
    if(m_Inputs.empty() || m_Outputs.empty() || !m_PerOpFactoryInfrastructure) { return SNPE_UDO_INVALID_ARGUMENT; }

    //Initialise params
    int32_t numFilters = 0;
    int32_t kernelSize = 0;
    int32_t biasTerm = 1;
    int32_t pad = 0;
    int32_t stride = 1;
    int32_t padH = 0;
    int32_t padW = 0;
    int32_t strideH = 1;
    int32_t strideW = 1;
    int32_t kernelH = 0;
    int32_t kernelW = 0;
    int32_t groups = 1;
    size_t filterHeight = 0;
    size_t filterWidth = 0;
    size_t filterDepth = 0;
    float* filter = nullptr;
    float* bias = nullptr;

    //Params for quantization
    uint8_t* quantized_filter  = nullptr;
    uint8_t* quantized_bias = nullptr;
    bool isQuantizable = false;
    float weight_delta = 0;
    float weight_offset = 0;
    float bias_delta = 0;
    float bias_offset = 0;

    std::map<std::string,int32_t*> dataMapScalar;
    dataMapScalar["num_output"] = &numFilters;
    dataMapScalar["bias_term"]  = &biasTerm;
    dataMapScalar["group"]      = &groups;
    dataMapScalar["kernel_h"]   = &kernelH;
    dataMapScalar["kernel_w"]   = &kernelW;
    dataMapScalar["pad_h"]      = &padH;
    dataMapScalar["pad_w"]      = &padW;
    dataMapScalar["stride_h"]   = &strideH;
    dataMapScalar["stride_w"]   = &strideW;

    std::map<std::string,int32_t*> dataMapTensor;
    dataMapTensor["kernel_size"] = &kernelSize;
    dataMapTensor["stride"]      = &stride;
    dataMapTensor["pad"]         = &pad;

    for (auto it = m_Params.begin(); it != m_Params.end(); it++)
    {
        std::string paramName = it->first;
        auto& param = it->second;
        if (paramName.compare("weight_filler") == 0)
        {
            if(param->tensorParam.dataType == SNPE_UDO_DATATYPE_FLOAT_32)
            {
                filter = reinterpret_cast<float*>(param->tensorParam.tensorData);
            }
            else
            {
                // Running for quantized dlc
                isQuantizable = true;
                quantized_filter = reinterpret_cast<uint8_t*>(param->tensorParam.tensorData);
                float minVal = static_cast<float>(param->tensorParam.quantizeParams.TFParams.minValue);
                float maxVal = static_cast<float>(param->tensorParam.quantizeParams.TFParams.maxValue);
                float range  = maxVal - minVal;
                weight_delta = range / 255.0f;
                weight_offset = round(minVal / weight_delta);
            }
            //For filters, tensor layout is kept same as we are getting through model. Here for caffe it is NCHW.
            //Filter height, width and depth
            filterHeight = param->tensorParam.currDimensions[2];
            filterWidth = param->tensorParam.currDimensions[3];
            filterDepth = param->tensorParam.currDimensions[1];
        }
        else if ( paramName.compare("bias_filler") == 0 )
        {
            if(param->tensorParam.dataType == SNPE_UDO_DATATYPE_FLOAT_32)
            {
                bias = reinterpret_cast<float*>(param->tensorParam.tensorData);
            }
            else
            {
                isQuantizable = true;
                quantized_bias = reinterpret_cast<uint8_t*>(param->tensorParam.tensorData);
                float minVal = static_cast<float>(param->tensorParam.quantizeParams.TFParams.minValue);
                float maxVal = static_cast<float>(param->tensorParam.quantizeParams.TFParams.maxValue);
                float range  = maxVal - minVal;
                bias_delta = range / 255.0f;
                bias_offset = round(minVal / bias_delta);
            }
        }
        else if ( param->paramType == SNPE_UDO_PARAMTYPE_SCALAR )
        {
            *(dataMapScalar[paramName]) = static_cast<int32_t>(param->scalarParam.dataValue.int32Value);
        }
        else if ( param->paramType == SNPE_UDO_PARAMTYPE_TENSOR )
        {
            *(dataMapTensor[paramName]) = *(static_cast<int32_t*>(param->tensorParam.tensorData));
        }
    }

    if (kernelSize != 0)
    {
        kernelH = kernelSize;
        kernelW = kernelSize;
    }
    if (pad != 0)
    {
        padH = pad;
        padW = pad;
    }
    if (stride != 0)
    {
        strideH = stride;
        strideW = stride;
    }

    //The tensor layout for input should be NHWC.
    //Input height, width and depth.
    const size_t inputHeight = m_Inputs[0]->currDimensions[1];
    const size_t inputWidth = m_Inputs[0]->currDimensions[2];
    const size_t inputDepth = m_Inputs[0]->currDimensions[3];

    //The tensor layout for output should be NHWC.
    //Output height, width and depth
    const size_t outputHeight = m_Outputs[0]->currDimensions[1];
    const size_t outputWidth = m_Outputs[0]->currDimensions[2];
    const size_t outputDepth = m_Outputs[0]->currDimensions[3];

    // set the depth for each group of filters
    uint32_t outputGroupDepth = numFilters / groups;

    float outputActivationMin = std::numeric_limits<float>::lowest();
    float outputActivationMax = std::numeric_limits<float>::max();

    const float* in = (float*)m_PerOpFactoryInfrastructure->getData(m_Inputs[0]->tensorData);
    float* out = (float*)m_PerOpFactoryInfrastructure->getData(m_Outputs[0]->tensorData);

    const float* filterbase = nullptr;
    const uint8_t* quantized_filterbase = nullptr;

    for (int32_t oh = 0; oh < outputHeight; oh++)
    {
        for (int32_t ow = 0; ow < outputWidth; ow++)
        {
            if(!isQuantizable)
            {
                filterbase = filter;
            }
            else
            {
                quantized_filterbase = quantized_filter;
            }

            int32_t inputOriginH = oh * strideH - padH;
            int32_t inputOriginW = ow * strideW - padW;
            for (int32_t g = 0; g < groups; g++)
            {
                for (int32_t d = 0; d < outputGroupDepth; d++)
                {
                    float sum = 0.0f;
                    for (int32_t fd = 0; fd < filterDepth; fd++)
                    {
                        for (int32_t fh = 0; fh < filterHeight; fh++)
                        {
                            int32_t inputH = inputOriginH + fh;
                            if (inputH < 0 || inputH >= static_cast<int32_t>(inputHeight))
                                continue;
                            for (int32_t fw = 0; fw < filterWidth; fw++)
                            {
                                int32_t inputW = inputOriginW + fw;
                                if (inputW < 0 || inputW >= static_cast<int32_t>(inputWidth))
                                    continue;
                                uint32_t inputD = filterDepth * g + fd;
                                uint32_t filterIndex = fd * filterHeight * filterWidth + fh * filterWidth + fw;
                                uint32_t inputIndex = inputDepth * (inputH * inputWidth + inputW) + inputD;

                                if (!isQuantizable) {
                                    sum += filterbase[filterIndex] * in[inputIndex];
                                }
                                else
                                {
                                    float x = weight_delta * (quantized_filterbase[filterIndex] + weight_offset);
                                    sum += x * in[inputIndex];
                                }
                            }
                        }
                    }
                    if (biasTerm)
                    {
                        if (!isQuantizable)
                        {
                            sum += bias[g * outputGroupDepth + d];
                        }
                        else
                        {
                            float y = bias_delta * (quantized_bias[g * outputGroupDepth + d] + bias_offset);
                            sum += y;
                        }
                    }
                    sum = std::max(std::min(sum, outputActivationMax), outputActivationMin);

                    out[d] = sum;
                    if (!isQuantizable)
                    {
                        filterbase += (filterHeight * filterWidth * filterDepth);
                    }
                    else
                    {
                        quantized_filterbase += (filterHeight * filterWidth * filterDepth);
                    }
                }
                out += outputGroupDepth;
            }
        }
    }
    auto elapsedTime = std::chrono::high_resolution_clock::now() - startTime;
    uint32_t elapsedTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(elapsedTime).count();
    m_ExecutionTime = elapsedTimeUs;
    return SNPE_UDO_NO_ERROR;
}
