/*
 * Copyright (c) 2019 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import com.qualcomm.qti.psnpedemo.networkEvaluation.Result;

import java.io.File;
import java.util.ArrayList;
import java.util.Map;

public class SegmentationPostProcessor extends PostProcessor {
    public SegmentationPostProcessor(int imageNumber) {
        super(imageNumber);
    }

    @Override
    public boolean postProcessResult(ArrayList<File> bulkImage) {
        return false;
    }

    @Override
    public void setResult(Result result) {

    }

    @Override
    public void getOutputCallback(String fileName, Map<String, float[]> outputs) {

    }
}
