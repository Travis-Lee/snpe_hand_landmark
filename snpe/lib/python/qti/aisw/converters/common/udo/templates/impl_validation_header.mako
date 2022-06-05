<%doc>
# ==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
//==============================================================================
// Auto Generated Code for ${package_name}
//==============================================================================

#pragma once

#include "utils/UdoUtil.hpp"

%for operator in operators:
class ${operator.name}${runtime}ValidationFunction : public UdoUtil::ImplValidationFunction {
public:

    ${operator.name}${runtime}ValidationFunction()
            : ImplValidationFunction() {}

    SnpeUdo_ErrorType_t
    validateOperation(SnpeUdo_OpDefinition_t* def) override;
};
%endfor
