//=============================================================================
//
//  Copyright (c) 2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "SoftmaxImplLibGpu.hpp"

const char* udo_kernel_program_softmax = R"SrcRaw(
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  __kernel void softmax_kernel(
     __global half* input,
           int numChannels,
     __global half* output)
  {
     int x = get_global_id(0);
     half maxPixelValue = -MAXFLOAT;
     for(int ch = 0 ; ch< numChannels; ch++)
     {
        maxPixelValue = max(maxPixelValue, input[ch+x*numChannels]);
     }
     half expSum = 0.0;
     for(int ch =0 ; ch< numChannels; ch++)
     {
        half tempIn = input[ch+x*numChannels] - maxPixelValue;
        half expIn = native_exp(tempIn);
        output[ch+x*numChannels] = expIn;
        expSum+=expIn;
     }
     for(int ch =0 ; ch< numChannels; ch++)
     {
        output[ch+x*numChannels] = output[ch+x*numChannels]/expSum;
     }
  }
  )SrcRaw";

std::unique_ptr<UdoUtil::UdoOperation>
SoftmaxAdrenoOpDef::createOp(void *perOpInfrastructure,
                             uint32_t numOfInputs,
                             SnpeUdo_TensorParam_t *inputs,
                             uint32_t numOfOutputs,
                             SnpeUdo_TensorParam_t *outputs,
                             uint32_t numOfStaticParams,
                             SnpeUdo_Param_t* params)
{
   cl_int err;
   SnpeUdo_ErrorType_t snpeUdoErr;

   m_GpuPerOpInfra = static_cast<SnpeUdo_GpuOpFactoryInfrastructure_t*>(perOpInfrastructure);

   const char* programName = "udo_kernel_program_softmax";
   if(!m_GpuPerOpInfra->programCache){
      return nullptr;
   }

   if(!m_GlobalInfra){
      return nullptr;
   }

   cl_program program;
   //retrieve opencl program if it exists in the cache
   snpeUdoErr = m_GlobalInfra->SnpeUdo_getProgram(m_GpuPerOpInfra->programCache, programName, &program );

   if(snpeUdoErr!=SNPE_UDO_NO_ERROR )
   {
      program = clCreateProgramWithSource(m_GpuPerOpInfra->context,
                                            1,
                                            (const char**)&udo_kernel_program_softmax,
                                            NULL,
                                            &err);
      if(err!=CL_SUCCESS){
         return nullptr;
      }

      err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

      if(err!=CL_SUCCESS){
         clReleaseProgram(program);
         return nullptr;
      }

      //store opencl program such that it can be cached and is referenced
      //for the next instance of UDO
      snpeUdoErr = m_GlobalInfra->SnpeUdo_storeProgram(m_GpuPerOpInfra->programCache,programName, program );
      if(snpeUdoErr!=SNPE_UDO_NO_ERROR) {
         clReleaseProgram(program);
      }
   }

   cl_kernel kernel = clCreateKernel(program, "softmax_kernel", &err );

   if(err!=CL_SUCCESS){
      clReleaseProgram(program);
      return nullptr;
   }

   //finding tensorlength without including depth
   uint32_t tensorRank = outputs[0].tensorRank;
   uint32_t tensorLength = 1;
   for(uint32_t k = 0; k < tensorRank-1; ++k)
   {
      tensorLength *= outputs[0].currDimensions[k];
   }

   int numChannels = outputs[0].currDimensions[tensorRank-1];

   //Set kernel dimensions
   //setting global dims equal to N*H*W
   m_GlobalKernelDim = {tensorLength,1,1};
   int xDim = std::min(64,static_cast<int>(tensorLength));
   m_LocalKernelDim = {static_cast<unsigned long>(xDim),1,1};

   SnpeUdo_GpuTensorData_t* tensorData0 = static_cast<SnpeUdo_GpuTensorData_t*>(inputs[0].tensorData);
   SnpeUdo_GpuTensorData_t* tensorData1 = static_cast<SnpeUdo_GpuTensorData_t *>(outputs[0].tensorData);

   err = 0;
   //Set Kernel Arguments
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)tensorData0->mem);
   err |= clSetKernelArg(kernel, 1, sizeof(int), (void *) &numChannels);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)tensorData1->mem);

   if(err!=CL_SUCCESS){
      clReleaseKernel(kernel);
      clReleaseProgram(program);
      return nullptr;
   }

   return std::unique_ptr<SoftmaxAdrenoOp>(new SoftmaxAdrenoOp(m_GpuPerOpInfra, kernel, program,
                                                               m_GlobalKernelDim, m_LocalKernelDim ));

}

SnpeUdo_ErrorType_t
SoftmaxAdrenoOp::snpeUdoExecute(bool blocking,
                                const uint32_t ID,
                                SnpeUdo_ExternalNotify_t notifyFunc)
{
    // uses base class implementation by default, user can override here.
    return UdoUtil::UdoAdrenoOperation::snpeUdoExecute(blocking, ID, notifyFunc);
}
