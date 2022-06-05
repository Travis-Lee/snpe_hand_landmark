#
# Copyright (c) 2018, 2019 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

#############################################################
# Inception V3 setup
#############################################################

mkdir -p inception_v3
mkdir -p inception_v3/images

cd inception_v3

cp -R ../../../../models/inception_v3/data/cropped/*.jpg images
FLOAT_DLC="../../../../models/inception_v3/dlc/inception_v3.dlc"
QUANTIZED_DLC="../../../../models/inception_v3/dlc/inception_v3_quantized.dlc"
UDO_DLC="../../../../models/inception_v3/dlc/inception_v3_udo.dlc"
UDO_QUANTIZED_DLC="../../../../models/inception_v3/dlc/inception_v3_udo_quantized.dlc"
if [ -f ${UDO_QUANTIZED_DLC} ]; then
    cp -R ${UDO_QUANTIZED_DLC} model.dlc
else
    if [ -f ${UDO_DLC} ]; then
        cp -R ${UDO_DLC} model.dlc
    else
        if [ -f ${QUANTIZED_DLC} ]; then
            cp -R ${QUANTIZED_DLC} model.dlc
        else
            cp -R ${FLOAT_DLC} model.dlc
        fi
    fi
fi
cp -R ../../../../models/inception_v3/data/imagenet_slim_labels.txt labels.txt

zip -r inception_v3.zip ./*
mkdir -p ../app/src/main/res/raw/
cp inception_v3.zip ../app/src/main/res/raw/

cd ..
rm -rf ./inception_v3
