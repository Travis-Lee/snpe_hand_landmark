<%doc> define all relevant variables</%doc>
<%doc>
# ==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
<%
 package_name = package.name
 runtimes = package.supported_runtimes
 coretypes = package.core_types
 op_catalog = package.op_catalog_info
 operators = package.package_info.operators
 calculation_types = package.calculation_types
%>
<%namespace file="/helpers.mako" import="*" />
//==============================================================================
// Auto Generated Code for ${package_name}
//==============================================================================
#include <iostream>
#include "utils/UdoUtil.hpp"
%for runtime in runtimes:
#include "${package_name}${runtime.title()}ImplValidationFunctions.hpp"
%endfor
extern "C"
{

std::unique_ptr<UdoUtil::UdoVersion> regLibraryVersion;
std::unique_ptr<UdoUtil::UdoRegLibrary> regLibraryInfo;

SnpeUdo_ErrorType_t
SnpeUdo_initRegLibrary(void)
{
    regLibraryInfo.reset(new UdoUtil::UdoRegLibrary("${package_name}",
                                                   ${_to_bitmask(coretypes)}));

    regLibraryVersion.reset(new UdoUtil::UdoVersion);

    regLibraryVersion->setUdoVersion(1, 0, 0);

<%doc> Add library names : one for each supported coretype </%doc>
    /*
    ** User should fill in implementation library path here as needed.
    ** Note: The Implementation library path set here is relative, meaning each library to be used
    ** must be discoverable by the linker.
    */
%for idx, runtime in enumerate(runtimes):
    regLibraryInfo->addImplLib("libUdo${package_name}Impl${str(runtime).title()}.so", ${_to_bitmask(coretypes[idx])}); //adding implementation libraries
%endfor

    %for operator in operators:
    //==============================================================================
    // Auto Generated Code for ${operator.name}
    //==============================================================================
    auto ${operator.name}Info = regLibraryInfo->addOperation("${operator.name}", ${_to_bitmask(operator.core_types)}, ${len(operator.input)}, ${len(operator.input)});

    %for i, coretype in enumerate(operator.core_types):
    ${operator.name}Info->addCoreInfo(${coretype}, ${calculation_types[i]}); //adding core info
    %endfor

    %for scalar_param in operator.scalar_param:
    ${operator.name}Info->addScalarParam("${scalar_param.name}", ${scalar_param.data_type}); // adding scalar param
    %endfor

    %for tensor_param in operator.tensor_param:
    ${operator.name}Info->addTensorParam("${tensor_param.name}", ${tensor_param.data_type}, ${tensor_param.tensor_layout}); // adding tensor param
    %endfor

    //inputs and outputs need to be added as tensor params
    %for i, input in enumerate(operator.input):
    ${operator.name}Info->addTensorParam("${input.name}", ${input.data_type}, ${input.tensor_layout}); //adding tensor param
    %endfor

    //adding outputs
    %for output in operator.output:
    ${operator.name}Info->addTensorParam("${output.name}", ${output.data_type}, ${output.tensor_layout}); //adding tensor param
    %endfor

    // adding validation functions
    %for coretype in operator.core_types:
    UDO_VALIDATE_RETURN_STATUS(regLibraryInfo->registerValidationFunction("${operator.name}",
                                                ${str(coretype)},
                                                std::unique_ptr<${operator.name}${str(coretype).split('_')[-1].title()}ValidationFunction>
                                                    (new ${operator.name}${str(coretype).split('_')[-1].title()}ValidationFunction())))

    %endfor
    %endfor
    UDO_VALIDATE_RETURN_STATUS(regLibraryInfo->createRegInfoStruct())

    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
SnpeUdo_getVersion(SnpeUdo_LibVersion_t** version) {

    UDO_VALIDATE_RETURN_STATUS(regLibraryVersion->getLibraryVersion(version))

    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
SnpeUdo_getRegInfo(SnpeUdo_RegInfo_t** registrationInfo) {

    UDO_VALIDATE_RETURN_STATUS(regLibraryInfo->getLibraryRegInfo(registrationInfo))

    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
SnpeUdo_terminateRegLibrary(void) {
    regLibraryInfo.release();
    regLibraryVersion.release();

    return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t
SnpeUdo_validateOperation(SnpeUdo_OpDefinition_t* opDefinition) {
    UDO_VALIDATE_RETURN_STATUS(regLibraryInfo->snpeUdoValidateOperation(opDefinition))

    return SNPE_UDO_NO_ERROR;
}
}; //extern C
