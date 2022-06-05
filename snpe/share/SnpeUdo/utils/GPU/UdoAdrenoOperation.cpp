//==============================================================================
//
// Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "utils/GPU/UdoAdrenoOperation.hpp"
#include <cstdlib>

using namespace UdoUtil;

#define CL_ASSERT(rc, msg) \
    UDO_VALIDATE_MSG((rc) != CL_SUCCESS, SNPE_UDO_UNKNOWN_ERROR, \
    msg<<";CL_ERROR_CODE: " <<(int)rc)

SnpeUdo_ErrorType_t
UdoAdrenoOperation::snpeUdoExecute(bool blocking,
                                   const uint32_t id,
                                   SnpeUdo_ExternalNotify_t notify) {

    UDO_VALIDATE_MSG(!blocking, SNPE_UDO_UNSUPPORTED_FEATURE,
                     "Snpe Udo does not support non-blocking (async) execution mode on GPU")


    CL_ASSERT(clEnqueueNDRangeKernel(m_Infrastructure->commandQueue,
                                     m_Kernel,
                                     m_GlobalDim.size(),
                                     NULL,
                                     m_GlobalDim.data(),
                                     m_LocalDim.data(),
                                     0,
                                     NULL,
                                     &m_Event), "Failure occurred while enqueueing cl kernel")

    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
UdoAdrenoOperation::snpeUdoSetIo(SnpeUdo_TensorParam_t*, SnpeUdo_TensorParam_t*) {
    // SetIo method is not applicable for AdrenoOperation
    UDO_VALIDATE_MSG(false, SNPE_UDO_UNSUPPORTED_FEATURE,
                     "Direct use of snpeUdoSetIo is not supported on GPU");
}

SnpeUdo_ErrorType_t
UdoAdrenoOperation::snpeUdoProfile(uint32_t* executionTime) {
    cl_ulong startNs, stopNs;

    CL_ASSERT(clGetEventProfilingInfo(m_Event,
                                      CL_PROFILING_COMMAND_START,
                                      sizeof(startNs),
                                      &startNs,
                                      nullptr), "Failure occurred during profile start")

    CL_ASSERT(clGetEventProfilingInfo(m_Event,
                                      CL_PROFILING_COMMAND_END,
                                      sizeof(stopNs),
                                      &stopNs,
                                      nullptr), "Failure occurred during profile end")

// microseconds = 1000 ns
    * executionTime = (stopNs - startNs) / 1000;
    return SNPE_UDO_NO_ERROR;
}

UdoAdrenoOperation::~UdoAdrenoOperation() {
    clReleaseKernel(m_Kernel);
    clReleaseProgram(m_Program);
}