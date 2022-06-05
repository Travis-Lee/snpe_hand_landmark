<%doc>
# ==============================================================================
#
#  Copyright (c) 2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================</%doc>
#================================================================================
# Auto Generated Code for ${package.name}
#================================================================================

# define relevant directories
SRC_DIR := ./

%if str(runtime).lower() != 'dsp':
# define library name and corresponding directory
%if str(runtime).lower() != 'cpu':
export RUNTIME := ${str(runtime).upper()}
export LIB_DIR := ../../../libs/$(TARGET)/$(RUNTIME)
%else:
export LIB_DIR := ../../../libs/$(TARGET)
%endif

library := $(LIB_DIR)/libUdo${package.name}Impl${runtime}.so

%if str(runtime).lower() == 'gpu':
# Note: add CL include path here to compile Gpu Library or set as env variable
# export CL_INCLUDE_PATH = <my_cl_include_path>
%endif

# define target architecture if not previously defined, default is x86
ifndef TARGET_AARCH_VARS
TARGET_AARCH_VARS:= -march=x86-64
endif

# specify package paths, should be able to override via command line?
UDO_PACKAGE_ROOT =${package.root}

include ../../../common.mk

%else:
# NOTE:
# - this Makefile is going to be used only to create DSP skels, so no need for android.min

ifndef HEXAGON_SDK_ROOT
$(error "HEXAGON_SDK_ROOT needs to be defined to compile a dsp library. Please set HEXAGON_SDK_ROOT to hexagon sdk installation.(Supported Version : 3.5.1)")
endif

ifndef HEXAGON_TOOLS_ROOT
$(error "HEXAGON_TOOLS_ROOT needs to be defined to compile a dsp library. Please set HEXAGON_TOOLS_ROOT to HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/8.3.07")
endif

ifndef SDK_SETUP_ENV
$(error "SDK_SETUP_ENV needs to be defined to compile a dsp library. Please set SDK_SETUP_ENV=Done")
endif

# define variant as V=hexagon_Release_dynamic_toolv83_v60 - it can be hardcoded too
ifndef V
V = hexagon_Release_dynamic_toolv83_v60
endif

V_TARGET = $(word 1,$(subst _, ,$(V)))
ifneq ($(V_TARGET),hexagon)
$(error Unsupported target '$(V_TARGET)' in variant '$(V)')
endif

# define package include paths and check API header path
# set package include paths, note package root will take precedence
# if includes are already present in package
UDO_PACKAGE_ROOT =${package.root}

# must list all variants supported by this project
SUPPORTED_VS = $(default_VS)

# must list all the dependencies of this project
DEPENDENCIES = ATOMIC RPCMEM TEST_MAIN TEST_UTIL

# each dependency needs a directory definition
#  the form is <DEPENDENCY NAME>_DIR
#  for example:
#    DEPENDENCIES = FOO
#    FOO_DIR = $(HEXAGON_SDK_ROOT)/examples/common/foo
#

ATOMIC_DIR = $(HEXAGON_SDK_ROOT)/libs/common/atomic
RPCMEM_DIR = $(HEXAGON_SDK_ROOT)/libs/common/rpcmem
TEST_MAIN_DIR = $(HEXAGON_SDK_ROOT)/test/common/test_main
TEST_UTIL_DIR = $(HEXAGON_SDK_ROOT)/test/common/test_util

include $(HEXAGON_SDK_ROOT)/build/make.d/$(V_TARGET)_vs.min
include $(HEXAGON_SDK_ROOT)/build/defines.min

# set include paths as compiler flags
CC_FLAGS += -I $(UDO_PACKAGE_ROOT)/include

# if SNPE_ROOT is set and points to the SDK path, it will be used. Otherwise ZDL_ROOT will be assumed to point
# to a build directory
ifdef SNPE_ROOT
CC_FLAGS += -I $(SNPE_ROOT)/include/zdl
else ifdef ZDL_ROOT
CC_FLAGS += -I $(ZDL_ROOT)/x86_64-linux-clang/include/zdl
else
$(error SNPE_ROOT: Please set SNPE_ROOT or ZDL_ROOT to obtain Udo headers necessary to compile the package)
endif

# only build the shared object if dynamic option specified in the variant
ifeq (1,$(V_dynamic))
BUILD_DLLS = libUdo${package.name}Impl${runtime}
endif

# sources for the DSP implementation library in src directory
SRC_DIR = ./
libUdo${package.name}Impl${runtime}.C_SRCS := $(wildcard $(SRC_DIR)/*.c)

# copy final build products to the ship directory
BUILD_COPIES = $(DLLS) $(EXES) $(LIBS) $(UDO_PACKAGE_ROOT)/libs/dsp/

# always last
include $(RULES_MIN)

# define destination library directory, and copy files into lib/dsp
# this code will create it
SHIP_LIBS_DIR   := $(UDO_PACKAGE_ROOT)/jni/src/DSP/$(V)
LIB_DIR         := $(UDO_PACKAGE_ROOT)/libs/dsp
OBJ_DIR         := $(UDO_PACKAGE_ROOT)/obj/local/dsp

.PHONY: dsp

dsp: tree
	mkdir -p ${"${OBJ_DIR}"};  ${"\\"}
	cp -Rf ${"${SHIP_LIBS_DIR}"}/. ${"${OBJ_DIR}"} ;${"\\"}
	rm -rf ${"${SHIP_LIBS_DIR}"};

%endif
