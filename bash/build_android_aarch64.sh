#!/bin/bash
var=`pwd`
echo "=======>$var"

if [ ! -d build_android_aarch64 ]; then
    mkdir -p build_android_aarch64
fi
cd build_android_aarch64
rm -rf *

BUILD_DIR="aarch64-release"
if [ ! -d $BUILD_DIR ]; then
    mkdir -p $BUILD_DIR
fi
cd $BUILD_DIR

export ANDROID_NDK=$var/3rdparty/android-ndk-r17c

if [ "$ANDROID_NDK" = "" ]; then
    echo "ERROR: Please set ANDROID_NDK_HOME environment"
    exit
fi

cmake ../.. \
 -DTARGET_OS=android_64 \
 -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
 -DANDROID_ABI="arm64-v8a" \
 -DANDROID_STL=c++_shared \
 -DANDROID_TOOLCHAIN=clang \
 -DANDROID_PLATFORM=android-24 \

make -j4 VERBOSE=1
make install/strip

cd ../../
DEMO_DIR=build_android_aarch64/$BUILD_DIR/test_demo
if [ ! -d $DEMO_DIR ]; then
    mkdir -p $DEMO_DIR
fi

echo "build_dir:$BUILD_DIR"
cp  build_android_aarch64/$BUILD_DIR/example/exe/* $DEMO_DIR
echo DEMO_DIR ${DEMO_DIR}
cp  build_android_aarch64/$BUILD_DIR/src/*.so $DEMO_DIR
cp  ./model/hlandmark.dlc $DEMO_DIR
cp  ./snpe/lib/aarch64-android-clang6.0/*.so $DEMO_DIR
cp  -rf ./snpe/lib/dsp $DEMO_DIR
cp  ./data/test.jpg $DEMO_DIR
