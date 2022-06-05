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

public class InceptionV3PreProcessor extends PreProcessor {
    private static String TAG = InceptionV3PreProcessor.class.getSimpleName();
    public InceptionV3PreProcessor(){}

    @Override
    public float[] preProcessData(File data) {
        String dataName = data.getName().toLowerCase();
        if(!(dataName.contains(".jpg") || dataName.contains(".jpeg"))) {
            Log.d(TAG, "data format invalid, dataName: " + dataName);
            return null;
        }

        int [] tensorShapes = PSNPEManager.getInputDimensions(); // nhwc
        int length = tensorShapes.length;
        if(tensorShapes.length != 4 || tensorShapes[length-1] != 3) {
            Log.d(TAG, "data format should be BGR" + length + " " + tensorShapes[length-1]);
            return null;
        }

        double [] meanRGB = {128.0d, 128.0d, 128.0d};
        return Util.imagePreprocess(data, tensorShapes[1], meanRGB, 128.0, false, 310);
    }
}
