/*
 * Copyright (c) 2019 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import android.util.Log;

import com.qualcomm.qti.psnpedemo.utils.Util;
import com.qualcomm.qti.psnpe.PSNPEManager;

import java.io.File;

public class VGGPreProcessor extends PreProcessor {
    private static String TAG = VGGPreProcessor.class.getSimpleName();

    @Override
    public float[] preProcessData(File data) {
        String dataName = data.getName().toLowerCase();
        if(!(dataName.contains(".jpg") || dataName.contains(".jpeg"))) {
            Log.d(TAG, "data format invalid, dataName: " + dataName);
            return null;
        }

        int [] tensorShapes = PSNPEManager.getInputDimensions(); // nhwc
        int length = tensorShapes.length;
        if(length != 4 || tensorShapes[length-1] != 3) {
            Log.d(TAG, "data format should be BGR" + length + " " + tensorShapes[length-1]);
            return null;
        }

        double [] meanRGB = {103.939d, 116.779d, 123.68d};
        return Util.imagePreprocess(data, tensorShapes[1], meanRGB, 1, true, 256);
    }
}
