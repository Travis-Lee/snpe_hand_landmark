/*
 * Copyright (c) 2019 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import android.util.Log;

import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.networkEvaluation.EvaluationCallBacks;
import com.qualcomm.qti.psnpedemo.networkEvaluation.Result;

import java.io.File;
import java.util.ArrayList;
import java.util.Map;

public class DetectionPostprocessor extends PostProcessor {
    private static String TAG = "DetectionPostprocessor";
    public DetectionPostprocessor(EvaluationCallBacks evaluationCallBacks, int imageNumber) {
        super(imageNumber);
        this.evaluationCallBacks = evaluationCallBacks;
    }
    @Override
    public boolean postProcessResult(ArrayList<File> bulkImage) {
        Log.d(TAG, "start into detection post process!");
        int imageNum = bulkImage.size();
        for(int i=0; i<imageNum; i++) {
            /* output:
             * <image1><image2>...<imageBulkSize>
             * split output and handle one by one.
             */
            Map<String, float []> outputMap = PSNPEManager.getOutputSync(i);
            String[] outputNames = PSNPEManager.getOutputTensorNames();
        }
        return true;
    }

    @Override
    public void setResult(Result result) {

    }

    @Override
    public void getOutputCallback(String fileName, Map<String, float[]> outputs) {

    }
}
