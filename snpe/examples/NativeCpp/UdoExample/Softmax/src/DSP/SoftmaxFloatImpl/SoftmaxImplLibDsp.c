//=============================================================================
//
//  Copyright (c) 2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <string.h>
#include <math.h>
#include "SoftmaxImplLibDsp.h"
#include "SnpeUdo/UdoImplDsp.h"

// operations info
char SoftmaxOpType [] = "Softmax";
uint32_t SoftmaxStaticParamsNum = 0;
uint32_t SoftmaxInputsNum = 1;
uint32_t SoftmaxOutputsNum = 1;
SnpeUdo_QuantizationType_t SoftmaxInputQuantizationTypes [] = {SNPE_UDO_QUANTIZATION_NONE};
SnpeUdo_QuantizationType_t SoftmaxOutputQuantizationTypes [] =  {SNPE_UDO_QUANTIZATION_NONE};
SnpeUdo_HexNNTensorLayout_t* SoftmaxLayout = NULL;

UdoDspShared* new_Softmax(SnpeUdo_HexNNv2GlobalInfra_t* infra)
{
    UdoDspShared *pOpObj = (*(infra->udoMalloc))(sizeof(UdoDspShared));
    if (pOpObj == NULL)
    {
        return NULL;
    }

    pOpObj->QueryOp = Softmax_QueryOperation;
    pOpObj->ValidateOp = Softmax_ValidateOperation;
    pOpObj->CreateOp = Softmax_CreateOpFactory;
    pOpObj->ExecuteOp = Softmax_ExecuteOp;

    return pOpObj;
}

SnpeUdo_ErrorType_t
Softmax_QueryOperation (SnpeUdo_String_t operationType, uint32_t numOfStaticParams,
                        const SnpeUdo_Param_t* staticParams, uint32_t* numOfInputs,
                        SnpeUdo_QuantizationType_t** inputsQuantTypes,
                        SnpeUdo_HexNNTensorLayout_t** inputsLayouts, uint32_t* numOfOutputs,
                        SnpeUdo_QuantizationType_t** outputsQuantTypes,
                        SnpeUdo_HexNNTensorLayout_t** outputsLayouts)
{
    if(strcmp(operationType, SoftmaxOpType) == 0)
    {
        *numOfInputs = SoftmaxInputsNum;
        *inputsQuantTypes = SoftmaxInputQuantizationTypes;
        *inputsLayouts = SoftmaxLayout;
        *numOfOutputs = SoftmaxOutputsNum;
        *outputsQuantTypes = SoftmaxOutputQuantizationTypes;
        *outputsLayouts = SoftmaxLayout;
    } else
    {
        return SNPE_UDO_WRONG_OPERATION;
    }
    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
Softmax_ValidateOperation (SnpeUdo_String_t operationType, uint32_t numOfStaticParams,
                           const SnpeUdo_Param_t* staticParams)
{
    if(strcmp(operationType, SoftmaxOpType) == 0)
    {
        if (numOfStaticParams != SoftmaxStaticParamsNum)
        {
            return SNPE_UDO_UNSUPPORTED_FEATURE;
        }
    } else
    {
        return SNPE_UDO_WRONG_OPERATION;
    }
    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
Softmax_CreateOpFactory (SnpeUdo_HexNNv2GlobalInfra_t* infra, SnpeUdo_CoreType_t udoCoreType,
                         void* perFactoryInfrastructure,
                         SnpeUdo_String_t operationType, uint32_t numOfStaticParams,
                         SnpeUdo_Param_t* staticParams, SnpeUdo_OpFactory_t* opFactory)
{
    if(operationType == NULL || operationType == 0)
    {
        return SNPE_UDO_INVALID_ARGUMENT;
    }
    // no static parameters, two inputs and one output
    if(strcmp(operationType, SoftmaxOpType) == 0)
    {
        SoftmaxOpFactory* this_factory = (*(infra->udoMalloc))(sizeof(SoftmaxOpFactory));
        int size = strlen(operationType) + 1; // +1 to hold the '\0' character
        this_factory->opType = (*(infra->udoMalloc))(size);
        strlcpy((this_factory->opType), operationType, size);
        *opFactory = (SnpeUdo_OpFactory_t) this_factory;
    }
    else
    {
        return SNPE_UDO_WRONG_OPERATION;
    }
    return SNPE_UDO_NO_ERROR;
}

/*
 * function to be passed into multi threading infrastructure function
 * each thread calls this same function
 */
void worker_thread_Softmax (void* perOpInfrastructure, void* userData)
{
    SoftmaxOpInfo * data = (SoftmaxOpInfo *) userData;
    float* input = data->input;
    uint32_t inputsLen = data->inputsLen;
    uint32_t depth = data->depth;
    float* output = data->output;

    float sum = 0.0f;
    float max;
    float * in;
    float * out;
    for(size_t i = 0; i < inputsLen / depth; i++) {
        in = input + i * depth;
        out = output + i * depth;
        // find the maximum
        max = (float) in[0];
        for(size_t j = 0; j < depth; j++) {
            max = (max < in[j]) ? in[j] : max;
        }
        // calculate output as e^(input - max)
        // calculate sum
        sum = 0.0f;
        for(size_t j = 0; j < depth; j++) {
            float exp = expf(in[j] - max);
            out[j] = exp;
            sum += exp;
        }
        // normalize outputs
        for(size_t j = 0; j < depth; j++) {
            out[j] = out[j] / sum;
        }
    }
}

SnpeUdo_ErrorType_t
Softmax_ExecuteOp (SnpeUdo_HexNNv2GlobalInfra_t* infra, SnpeUdo_Operation_t operation,
                    bool blocking, const uint32_t ID, SnpeUdo_ExternalNotify_t notifyFunc)
{
    if(operation == NULL)
    {
        return SNPE_UDO_INVALID_ARGUMENT;
    }

    OpParams* m_Operation = (OpParams*) operation;
    char* m_OpType = ((SoftmaxOpFactory*) (m_Operation->opFactory))->opType;
    if(m_OpType == NULL)
    {
        return SNPE_UDO_INVALID_ARGUMENT;
    }
    if(strcmp(m_OpType, SoftmaxOpType) == 0)
    {
        SnpeUdo_TensorParam_t* input = &(m_Operation->InputParams[0]);
        SnpeUdo_TensorParam_t* out = m_Operation->outputParams;

        if (input->layout == SNPE_UDO_LAYOUT_NULL || input->layout == SNPE_UDO_LAYOUT_NULL
            || out->layout == SNPE_UDO_LAYOUT_NULL) {
            return SNPE_UDO_UNSUPPORTED_FEATURE;
        }

        uint32_t inputLen = sizeof(uint32_t);
        for(int k = 0; k < input->tensorRank; k++) {
            inputLen *= input->currDimensions[k];
            out->currDimensions[k] = input->currDimensions[k];
        }

        float* inputTensorData = (float*)(input->tensorData);
        float* outputTensorData = (float*)(out->tensorData);

        out->dataType = SNPE_UDO_DATATYPE_FLOAT_32;
        // required to set output tensor sizes
        if( (*(infra->udoSetOutputTensorSize)) (m_Operation->opInfra, 0, inputLen) != 0 ) {
            return SNPE_UDO_UNSUPPORTED_FEATURE;
        }

        SoftmaxOpInfo workerThreadIn = {inputTensorData, inputLen / sizeof(uint32_t),
                                        input->currDimensions[3], outputTensorData};
        (*(infra->udoRunWorkerThreads))(m_Operation->opInfra, 1, worker_thread_Softmax, &workerThreadIn);

        return SNPE_UDO_NO_ERROR;
    }
    else
    {
        return SNPE_UDO_WRONG_OPERATION;
    }
}

