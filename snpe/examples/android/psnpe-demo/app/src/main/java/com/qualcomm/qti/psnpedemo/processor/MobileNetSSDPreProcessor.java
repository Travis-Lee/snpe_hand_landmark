/*
 * Copyright (c) 2019 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import android.util.Log;

import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;
import com.qualcomm.qti.psnpedemo.utils.Util;
import com.qualcomm.qti.psnpe.PSNPEManager;

import java.io.File;

public class MobileNetSSDPreProcessor extends PreProcessor {
    private static String TAG = MobileNetPreProcessor.class.getSimpleName();
    @Override
    public float[] preProcessData(File data) {
        String dataName = data.getName().toLowerCase();
        if(!(dataName.contains(".jpg") || dataName.contains(".jpeg") || dataName.contains("png"))) {
            Log.d(TAG, "data format invalid, dataName: " + dataName);
            return null;
        }

        int [] tensorShapes = PSNPEManager.getInputDimensions(); // nhwc
        int length = tensorShapes.length;
        if(tensorShapes.length != 4 || tensorShapes[length-1] != 3) {
            Log.d(TAG, "data format should be BGR");
            return null;
        }

        double [] meanRGB = {127.5, 127.5, 127.5};
        float [] result = Util.imagePreprocess(data, tensorShapes[1], meanRGB, 127.5, false, 300);

        String inputPath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("input_list").getAbsolutePath();
        Util.write2file(inputPath + "/mobilenetssd_input_list.txt", data.getName());

        //String filePath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("").getAbsolutePath();
        //float [] result = Util.readArrayFromTxt(filePath + "/000000090956.txt");
        //Util.writeData("mobilenetssd", result);
        return result;
    }
}
