//=============================================================================
//
//  Copyright (c) 2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "SnpeUdo/UdoBase.h"
#include "SoftmaxUdoPackageCpuImplValidationFunctions.hpp"
#include <string.h>
using namespace UdoUtil;

SnpeUdo_ErrorType_t
SoftmaxCpuValidationFunction::validateOperation(SnpeUdo_OpDefinition_t* def) {

    SnpeUdo_ErrorType_t status = SNPE_UDO_NO_ERROR;

    if (def == nullptr)
    {
        return SNPE_UDO_INVALID_ARGUMENT;
    }

    if (strcmp(def->operationType, "Softmax"))
        return SNPE_UDO_WRONG_OPERATION;

    if (def->numOfStaticParams != 0)
        return SNPE_UDO_WRONG_OPERATION;


    if (def->numOfInputs != 1 || def->numOfOutputs != 1)
        return SNPE_UDO_WRONG_OPERATION;

    return SNPE_UDO_NO_ERROR;
}

