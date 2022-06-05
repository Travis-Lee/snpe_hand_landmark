//==============================================================================
//
// Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <vector>
#include "utils/UdoOperation.hpp"
#include "utils/UdoMacros.hpp"
#include "SnpeUdo/UdoImplGpu.h"
namespace UdoUtil {

class UdoAdrenoOperation : public UdoOperation
{
public:
    UdoAdrenoOperation(SnpeUdo_GpuOpFactoryInfrastructure_t* infrastructure,
                       cl_kernel kernel,
                       cl_program program,
                       std::vector<size_t>& globalDim,
                       std::vector<size_t>& localDim)
    : m_Kernel(kernel),
      m_Program(program),
      m_Infrastructure(infrastructure),
      m_GlobalDim(globalDim),
      m_LocalDim(localDim)
      {}

    SnpeUdo_ErrorType_t snpeUdoExecute(bool blocking, uint32_t id, SnpeUdo_ExternalNotify_t notify) override;

    SnpeUdo_ErrorType_t snpeUdoSetIo(SnpeUdo_TensorParam_t* inputs, SnpeUdo_TensorParam_t* outputs) override;

    SnpeUdo_ErrorType_t snpeUdoProfile(uint32_t* executionTime) override;

    ~UdoAdrenoOperation();

protected:
    cl_kernel m_Kernel;
    cl_program m_Program;
    cl_event m_Event;
    SnpeUdo_GpuOpFactoryInfrastructure_t* m_Infrastructure;
    std::vector<size_t> m_GlobalDim;
    std::vector<size_t> m_LocalDim;
    uint32_t m_ExecTime;
};

}
