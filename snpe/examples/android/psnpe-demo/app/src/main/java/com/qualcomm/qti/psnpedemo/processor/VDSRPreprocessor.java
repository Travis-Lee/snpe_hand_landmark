/*
 * Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.util.Log;

import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;
import com.qualcomm.qti.psnpedemo.networkEvaluation.ModelInfo;
import com.qualcomm.qti.psnpedemo.utils.Util;

import java.io.File;


public class VDSRPreprocessor extends PreProcessor {
    private static String TAG = VDSRPreprocessor.class.getSimpleName();
    private int radioSize;
    private String groundTruthPath;

    public VDSRPreprocessor(ModelInfo modelInfo) {
        String truthRelPath = "datasets/"+modelInfo.getDataSetName()+"/GroundTruth";
        File truthDir = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir(truthRelPath);
        this.groundTruthPath = truthDir.getAbsolutePath();
        this.radioSize = 4;
    }

    public void setRadioSize(int radioSize) {
        this.radioSize = radioSize;
    }

    @Override
    public float[] preProcessData(File data) {
        String dataName = data.getName().toLowerCase();
        if(!(dataName.contains(".jpg"))) {
            Log.e(TAG, "data format invalid, dataName: " + dataName);
            return null;
        }

        float result[] = getYLowDataRaw(data, radioSize);
        String inputPath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("input_list").getAbsolutePath();
        Util.write2file(inputPath + "/vdsr_input_list.txt", data.getName());
        return result;
    }

    private float[] getYLowDataRaw(File imageName, int lowRatio){
        float pixelsYFloat[] = null;
        try{
            Bitmap imgRGB = BitmapFactory.decodeFile(imageName.getAbsolutePath());

            int originImgWidth = imgRGB.getWidth();
            int originImgHeight = imgRGB.getHeight();
            int startX = Math.max((int)(originImgWidth - 256) / 2, 0);
            int startY = Math.max((int)(originImgHeight - 256) / 2, 0);
            int width = Math.min(originImgWidth, 256);
            int height = Math.min(originImgHeight, 256);

            Bitmap newImage = Bitmap.createBitmap(imgRGB, startX, startY, width, height, null, false);

            float R, G, B, Y;
            float[][] MatrixY = new float[height][width];
            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    int pixel = newImage.getPixel(col, row);
                    R = Color.red(pixel);
                    G = Color.green(pixel);
                    B = Color.blue(pixel);
                    /*convert RGB to YCbCr
                     * convert formula:
                     * Y = (0.256789 * R + 0.504129 * G + 0.097906 * B + 16.0)/255.0
                     * Cb = (-0.148223 * R - 0.290992 * G + 0.439215 * B + 128.0)/255.0
                     * Cr = (0.439215 * R  - 0.367789 * G - 0.071426 * B + 128.0)/255.0
                     * We only use Y channel here.
                     * */
                    Y = (float)((0.256789 * R + 0.504129 * G + 0.097906 * B + 16.0)/255.0);
                    MatrixY[row][col] = Y;
                }
            }

            if(width < 256 || height < 256){
                /*if input img size is smaller than 256*256, adjust it to 256*256*/
                MatrixY = resizeInsertLinear(MatrixY, 256, 256);
                width = 256;
                height = 256;
            }

            pixelsYFloat = new float[width * height * 1];
            int i = 0;
            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    pixelsYFloat[i++] = MatrixY[row][col];
                }
            }
            String truthFileName = imageName.getName().replace("jpg", "raw");
            Util.writeArrayTofile(this.groundTruthPath + '/' + truthFileName, pixelsYFloat);

            int resize_width = width / lowRatio;
            int resize_height = height / lowRatio;

            float[][] resizeTmp = resizeInsertLinear(MatrixY, resize_width, resize_height);
            float[][] finalImg = resizeInsertLinear(resizeTmp, width, height);

            i = 0;
            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    pixelsYFloat[i++] = finalImg[row][col];
                }
            }
        }catch (Exception e){
            Log.e(TAG, "Exception in image pre-processing: "+ e);
        }
        return pixelsYFloat;
    }

    private static float[][] resizeInsertLinear(float [][] src, int dstWidth, int dstHeight){
        if(null == src){
            throw new IllegalArgumentException("src buffer is null");
        }
        if(0 == src.length || 0 == src[0].length){
            throw new IllegalArgumentException(String.format("Wrong resize src buffer size", dstWidth, dstHeight));
        }
        if(0 == dstWidth || 0 == dstHeight){
            throw new IllegalArgumentException(String.format("Wrong resize dstSize(%d, %d)", dstWidth, dstHeight));
        }

        int srcHeight = src.length;
        int srcWidth = src[0].length;
        float[][] dst = new float[dstHeight][dstWidth];

        double scaleX = (double)srcWidth / (double)dstWidth;
        double scaleY = (double)srcHeight / (double)dstHeight;

        for(int dstY = 0; dstY < dstHeight; ++dstY){
            double fy = ((double)dstY + 0.5) * scaleY - 0.5;
            int sy = (int)fy;
            fy -= sy;
            if(sy < 0){
                fy = 0.0; sy = 0;
            }
            if(sy >= srcHeight - 1){
                fy = 0.0; sy = srcHeight - 2;
            }

            for(int dstX = 0; dstX < dstWidth; ++dstX){
                double fx = ((double)dstX + 0.5) * scaleX - 0.5;
                int sx = (int)fx;
                fx -= sx;
                if(sx < 0){
                    fx = 0.0; sx = 0;
                }
                if(sx >= srcWidth - 1){
                    fx = 0.0; sx = srcWidth - 2;
                }

                dst[dstY][dstX] = (float) ((1.0-fx) * (1.0-fy) * src[sy][sx]
                                    + fx * (1.0-fy) * src[sy][sx+1]
                                    + (1.0-fx) * fy * src[sy+1][sx]
                                    + fx * fy * src[sy+1][sx+1]);
            }
        }

        return dst;
    }
}
