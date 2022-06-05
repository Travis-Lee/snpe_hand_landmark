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
SnpeUdo_QuantizationType_t SoftmaxInputQuantizationTypes [] = {SNPE_UDO_QUANTIZATION_TF};
SnpeUdo_QuantizationType_t SoftmaxOutputQuantizationTypes [] =  {SNPE_UDO_QUANTIZATION_TF};
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

static float expf_approx(float x)
{
    float val = x * (float)(16384.0/0.69314718056);
    int xfrac = (int)(val + copysignf(0.5f,val));
    float xf  = (xfrac & 0x3FFF) * (float)(1./16384.);
    float exp0 = 1.0f + xf*(0.69051585f + xf*(0.23793659f + xf*0.07154756f));
    float exp = powf(2.0f, xfrac>>14);

    return (exp * exp0);
}

void worker_thread_SoftmaxQuant (void* perOpInfrastructure, void* userData)
{
    SoftmaxOpInfo * data = (SoftmaxOpInfo *) userData;
    uint8_t* input = data->input;
    uint32_t inputsLen = data->inputsLen;
    uint8_t* output = data->output;
    float inputMin = data->inputMin;
    float inputMax = data->inputMax;
    uint32_t depth = data->depth;

    float stepsize = (inputMax - inputMin) * (1.0f / 255.0f);

    float sum = 0.0f;
    uint8_t maxval;
    float temp_slice[depth];
    uint8_t * in;
    uint8_t * out;
    float recip;

    if( stepsize < 0.63529f)
    {
        for (size_t i = 0; i < inputsLen / depth; i++)
        {
            in = input + i * depth;
            out = output + i * depth;
            sum = 0.0f;

            for (size_t j = 0; j < depth; j++)
            {
                float exp = expf_approx(stepsize * in[j] - 83.0f);
                temp_slice[j] = exp;
                sum += exp;
            }
            recip = 255.0f/sum;
            for (size_t j = 0; j < depth; j++)
            {
                int val = roundf(recip * temp_slice[j]);
                out[j] = (val < 0) ? 0 : ((val > 255) ? 255 : val);
            }
        }
    }
    else
    {
        for(size_t i = 0; i < inputsLen / depth; i++)
        {
            in = input + i * depth;
            out = output + i * depth;
            sum = 0.0f;
            maxval = (uint8_t) in[0];

            for(size_t j = 0; j < depth; j++)
            {
                maxval = (maxval < in[j]) ? in[j] : maxval;
            }

            for(size_t j = 0; j < depth; j++)
            {
                float exp = expf_approx(stepsize * (in[j] - maxval));
                temp_slice[j] = exp;
                sum += exp;
            }
            recip = 255.0f/sum;
            for(size_t j = 0; j < depth; j++)
            {
                int val = roundf(recip * temp_slice[j]);
                out[j] = (val < 0) ? 0 : ((val > 255) ? 255 : val);
            }
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

        if (input->layout == SNPE_UDO_LAYOUT_NULL || out->layout == SNPE_UDO_LAYOUT_NULL) {
            return SNPE_UDO_UNSUPPORTED_FEATURE;
        }

        uint32_t inputLen = sizeof(uint8_t);
        for(int k = 0; k < input->tensorRank; k++) {
            inputLen *= input->currDimensions[k];
            out->currDimensions[k] = input->currDimensions[k];
        }

        float inputMin = input->quantizeParams.TFParams.minValue;
        float inputMax = input->quantizeParams.TFParams.maxValue;

        float outputMin = out->quantizeParams.TFParams.minValue;
        float outputMax = out->quantizeParams.TFParams.maxValue;

        uint8_t* inputTensorData = (uint8_t*)(input->tensorData);
        uint8_t* outputTensorData = (uint8_t*)(out->tensorData);

        out->dataType = SNPE_UDO_DATATYPE_FIXED_8;
        // required to set output tensor sizes
        if( (*(infra->udoSetOutputTensorSize)) (m_Operation->opInfra, 0, inputLen) != 0 ) {
            return SNPE_UDO_UNSUPPORTED_FEATURE;
        }

        SoftmaxOpInfo workerThreadIn = {inputTensorData, inputLen,
                                                   inputMin, inputMax,
                                                   outputMin, outputMax,
                                                   input->currDimensions[3],
                                                   outputTensorData};
        (*(infra->udoRunWorkerThreads))(m_Operation->opInfra, 1, worker_thread_SoftmaxQuant, &workerThreadIn);

        return SNPE_UDO_NO_ERROR;
    }
    else
    {
        return SNPE_UDO_WRONG_OPERATION;
    }
}
