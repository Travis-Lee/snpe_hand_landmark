/*
 * Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import android.util.Log;

import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;
import com.qualcomm.qti.psnpedemo.networkEvaluation.Result;
import com.qualcomm.qti.psnpedemo.networkEvaluation.ModelInfo;
import com.qualcomm.qti.psnpedemo.utils.Util;

import java.io.File;
import java.util.ArrayList;
import java.util.Map;

import static java.sql.Types.NULL;

public class SuperresolutionPostprocessor extends PostProcessor {
    private static String TAG = SuperresolutionPostprocessor.class.getSimpleName();
    private String groundTruthPath;

    private double totalPSNR;
    private double totalSSIM;
    private double averagePSNR;
    private double averageSSIM;
    private int imgHeight;
    private int imgWidth;

    public SuperresolutionPostprocessor(ModelInfo modelInfo, int inputSize) {
        super(inputSize);
        totalPSNR = 0.0;
        totalSSIM = 0.0;
        averagePSNR = 0.0;
        averageSSIM = 0.0;
        imgHeight = 256;
        imgWidth = 256;
        String truthRelPath = "datasets/"+modelInfo.getDataSetName()+"/GroundTruth";
        File truthDir = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir(truthRelPath);
        this.groundTruthPath = truthDir.getAbsolutePath();
    }

    @Override
    public boolean postProcessResult(ArrayList<File> bulkImage) {
        int imageNum = bulkImage.size();

        float [] output = null;
        float [] groundTruth = null;

        for(int i=0; i<imageNum; i++) {
            Map<String, float []> outputMap =  PSNPEManager.getOutputSync(i);
            if(outputMap.size() == 0){
                Log.e(TAG, "postProcessResult error: outputMap is null");
                return false;
            }
            output = outputMap.get(PSNPEManager.getOutputTensorNames()[0]);
            if(null == output){
                Log.e(TAG, "postProcessResult error: output is null");
                return false;
            }
            String truthFileName = bulkImage.get(i).getName().replace("jpg", "raw");
            groundTruth = Util.readFloatArrayFromFile(this.groundTruthPath + "/" + truthFileName);
            if(null == groundTruth){
                Log.e(TAG, "postProcessResult error: groundTruth is null");
                return false;
            }
            this.count.incrementAndGet();
            this.totalPSNR += computePSNR(groundTruth, output);
            this.totalSSIM += computeSSIM(groundTruth, output, 7);
        }
        this.averagePSNR = this.totalPSNR / this.count.doubleValue();
        this.averageSSIM = this.totalSSIM / this.count.doubleValue();
        return true;
    }

    @Override
    public void setResult(Result result) {
        result.setPnsr(averagePSNR);
        result.setSsim(averageSSIM);
    }

    @Override
    public void getOutputCallback(String fileName, Map<String, float[]> outputs) {
        float [] output = null;
        float [] groundTruth = null;
        if(outputs.size() == 0){
            Log.e(TAG, "getOutputCallback error: outputMap is null");
            return;
        }
        output = outputs.get(PSNPEManager.getOutputTensorNames()[0]);
        if(null == output){
            Log.e(TAG, "getOutputCallback error: output is null");
            return;
        }

        String truthFileName = fileName.replace("jpg", "raw");
        groundTruth = Util.readFloatArrayFromFile(this.groundTruthPath + "/" + truthFileName);
        if(null == groundTruth){
            Log.e(TAG, "postProcessResult error: groundTruth is null");
            return;
        }

        this.totalPSNR = computePSNR(groundTruth, output);
        this.totalSSIM = computeSSIM(groundTruth, output, 7);
        this.averagePSNR = this.totalPSNR / this.count.doubleValue();
        this.averageSSIM = this.totalSSIM / this.count.doubleValue();
    }

    /*
     * Calculate MSE(Mean Square Error) between img1 and img2
     * */
    public double computeMSE(float[] img1, float[] img2) {
        if(img1.length != img2.length) {
            Log.e(TAG, "mse computing error with mismatch length of img1 and img2");
            return NULL;
        }
        float[] square = new float[img1.length];
        for (int i = 0; i < img1.length; i++) {
            square[i] = (img1[i] - img2[i]) * (img1[i] - img2[i]);
        }
        return mean(square);
    }

    /*
     * Calculate SSIM(Structural SIMilarity) between img1 and img2.
     * */
    public double computeSSIM(float[] img1, float[] img2, int windowSize) {
        int length = imgWidth * imgHeight;

        // means of img1
        float[] ux = uniformFilter1d(img1, windowSize);
        // means of img2
        float[] uy = uniformFilter1d(img2, windowSize);

        int ndim = 1;
        double NP = Math.pow(windowSize, ndim);
        double cov_norm = NP / (NP - 1);
        float[] uxx = uniformFilter1d(multiply(img1, img1), windowSize);
        float[] uyy = uniformFilter1d(multiply(img2, img2), windowSize);
        float[] uxy = uniformFilter1d(multiply(img1, img2), windowSize);
        float[] vx = new float[length];
        float[] vy = new float[length];
        float[] vxy = new float[length];
        for (int i = 0; i < length; i++) {
            // variances of img1
            vx[i] = (float) (cov_norm * (uxx[i] - ux[i] * ux[i]));
            // variances of img2
            vy[i] = (float) (cov_norm * (uyy[i] - uy[i] * uy[i]));
            // covariances of img1 and img2
            vxy[i] = (float) (cov_norm * (uxy[i] - ux[i] * uy[i]));
        }

        int data_range = 2;
        double K1 = 0.01;
        double K2 = 0.03;
        double C1 = Math.pow(K1 * data_range, 2);
        double C2 = Math.pow(K2 * data_range, 2);

        /* calculate all SSIM for img1 and img2 */
        float[] allSSIM = new float[length];
        for (int i = 0; i < length; i++) {
            double luminance = (2 * ux[i] * uy[i] + C1) / (ux[i] * ux[i] + uy[i] * uy[i] + C1);
            double contrast =  (2 * vxy[i] + C2) / (vx[i] + vy[i] + C2);
            allSSIM[i] = (float) (luminance * contrast);
        }

        int pad = (windowSize - 1) /2;
        float[] croppedSSIM = new float[allSSIM.length - pad * 2];
        for (int i = 0; i < croppedSSIM.length; i++) {
            croppedSSIM[i] = allSSIM[i + pad];
        }

        return mean(croppedSSIM);
    }

    /*
     * Calculate PSNR(Peak Signal to Noise Ratio) of img1 and img2.
     * */
    public double computePSNR(float[] im1, float[] im2) {
        double mse = computeMSE(im1, im2);
        return 10 * (Math.log10(1.0/mse));
    }

    /*
     * Calculate a 1-D minimum uniform filter
     * */
    private float[] uniformFilter1d(float[] input, int windowSize) {
        float[] output = new float[input.length];
        float[] paddingInput = new float[input.length + 2*(windowSize/2)];
        int start = 0;
        int end = 0;
        for (int i = 0; i < input.length + windowSize - 1; i++) {
            if(i < windowSize/2) {
                paddingInput[i] = input[windowSize/2 -1 - i];
            }
            else if(i >= input.length + windowSize/2) {
                paddingInput[i] = input[input.length -1 - end];
                end++;
            }
            else {
                paddingInput[i] = input[start++];
            }
        }

        start = windowSize /2;

        for (int i = 0; i < input.length; i++) {
            double average = 0.0;
            for (int j = 0; j <= windowSize/2; j++) {
                average += paddingInput[start - j];
            }
            for (int k = 1; k <= windowSize/2; k++) {
                average += paddingInput[start + k];
            }
            output[i] = (float)average / windowSize;
            start++;
        }
        return output;
    }

    private double mean(float[] data) {
        if(data.length == 0) {
            return Double.NaN;
        }
        double mean = 0.0;
        for(int i = 0; i < data.length; i++) {
            mean += data[i];
        }
        mean = mean / (double)data.length;
        return mean;
    }

    private float[] multiply(float[] input1, float[] input2){
        float[] output = new float[input1.length];
        for (int i = 0; i < output.length; ++i){
            output[i] = input1[i] * input2[i];
        }
        return output;
    }
}
