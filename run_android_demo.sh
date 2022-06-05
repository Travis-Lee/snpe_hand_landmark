#!/usr/bin/env bash

SNPE_LIB=build_android_aarch64/aarch64-release/test_demo

adb push $SNPE_LIB /data/local/tmp/
bin_path="/data/local/tmp/test_demo"
adb shell "chmod +x ${bin_path}/hand_landmark"
adb shell "cd ${bin_path} \
          && export ADSP_LIBRARY_PATH=${bin_path}/dsp; \
          export LD_LIBRARY_PATH=${bin_path}:${LD_LIBRARY_PATH} \
          && ./hand_landmark ./hlandmark.dlc 2 ./test.jpg ./res.jpg"

adb pull /data/local/tmp/test_demo/res.jpg ~/Desktop
