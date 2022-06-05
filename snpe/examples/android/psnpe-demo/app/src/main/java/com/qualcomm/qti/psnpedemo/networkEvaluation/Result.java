/*
 * Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.networkEvaluation;

public class Result {
    private double fps;
    private double buildTime;
    private double inferenceTime;
    private float top1;
    private float top5;
    private float tops;
    private double psnr;
    private double ssim;

    public void updateFromProfiler(TimeProfiler timeProfiler) {
        buildTime = timeProfiler.getTime(TimeProfiler.TIME_TYPE.BUILD_TIME) / 1000.0;
        inferenceTime = timeProfiler.getTime(TimeProfiler.TIME_TYPE.EXECUTE_TIME) / 1000.0;
    }

    public void setBuildTime(double buildTime) {
        this.buildTime = buildTime;
    }

    public void setInferenceTime(double inferenceTime) {
        this.inferenceTime = inferenceTime;
    }

    public void setTop1(float top1) {
        this.top1 = top1;
    }

    public void setTop5(float top5) {
        this.top5 = top5;
    }

    public void setTops(float tops) {
        this.tops = tops;
    }

    public void setFPS(double fps){
        this.fps = fps;
    }

    public float getTops() {
        return tops;
    }

    public float getTop1() {
        return top1;
    }

    public float getTop5() {
        return top5;
    }

    public double getFPS(){
        return fps;
    }

    public double getPnsr() {
        return psnr;
    }

    public void setPnsr(double pnsr) {
        this.psnr = pnsr;
    }

    public double getSsim() {
        return ssim;
    }

    public void setSsim(double ssim) {
        this.ssim = ssim;
    }

    public double getBuildTime() {
        return buildTime;
    }

    public double getInferenceTime() {
        return inferenceTime;
    }

    public void clear() {
        fps = 0;
        buildTime = 0;
        inferenceTime = 0;
        top1 = 0;
        top5 = 0;
        tops = 0;
        psnr = 0;
        ssim = 0;
    }

    public Result() {
        clear();
    }

    @Override
    public String toString() {
        String result = "";
        result = result + "FPS: " + getFPS()
                + "\nInference Time: "+ getInferenceTime() + "s\nTop1: " + getTop1()*100
                + "%\nTop5: " + getTop5()*100 + "%\nTops: " + getTops()
                + "\nPSNR:" + getPnsr() +"\nSSIM:" + getSsim();
        return result;
    }
}
