/*
 * Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.utils;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.util.Log;
import android.view.Gravity;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;


public class Util {
    private static String TAG = Util.class.getSimpleName();

    public static int R(int c) {
        return (c >> 16) & 0xff;
    }

    public static int G(int c) {
        return (c >> 8) & 0xff;
    }

    public static int B(int c) {
        return c & 0xff;
    }

    public static double r(int c) {
        return R(c)/255.0;
    }

    public static double g(int c) {
        return G(c)/255.0;
    }

    public static double b(int c) {
        return B(c)/255.0;
    }

    static public void writeData(String imageName, float []data) {
        File file = new File("/storage/emulated/0/Android/data/com.demo.qcbenchmark/files/" + imageName + ".txt");
        if(!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            FileWriter fileWriter = new FileWriter(file, false);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            StringBuilder sb = new StringBuilder();
            for(int i=0; i<data.length; i++) {
                sb.append(data[i]).append("\n");
            }
            sb.deleteCharAt(sb.length()-1);
            //Log.d("Utils", sb.toString());
            bufferedWriter.write(sb.toString());
            bufferedWriter.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static float[] imagePreprocess(File imageName, int inputsize, double[] meanRGB, double div, boolean useBGR, int resize_target) {
        float[] pixelFloats = new float[inputsize * inputsize * 3];
        Bitmap img;
        Bitmap imgcrop;
        Bitmap imgresize;
        try{
            img = BitmapFactory.decodeStream(new FileInputStream(imageName));
            //resize image
            int short_dim = Math.min(img.getHeight(), img.getWidth());
            if(resize_target == 300) {
                imgresize = img;
                imgcrop = Bitmap.createScaledBitmap(imgresize, inputsize, inputsize, true);//resize
            }else {
                int px = (img.getWidth() - short_dim) / 2;
                int py = (img.getHeight() - short_dim) / 2;
                imgresize = Bitmap.createBitmap(img, px, py, short_dim, short_dim, null, false);//crop
                imgcrop = Bitmap.createScaledBitmap(imgresize, inputsize, inputsize, true);//resize
            }
            final int[] pixels = new int[imgcrop.getWidth() * imgcrop.getHeight()];
            imgcrop.getPixels(pixels, 0, imgcrop.getWidth(), 0, 0,
                    imgcrop.getWidth(), imgcrop.getHeight());
            int z = 0;
            for (int y = 0; y < imgcrop.getHeight(); y++) {
                for (int x = 0; x < imgcrop.getWidth(); x++) {
                    final int rgb = pixels[y * imgcrop.getWidth() + x];
                    float b = (((rgb) & 0xFF) - (float) meanRGB[2]) / (float)div;
                    float g = (((rgb >> 8) & 0xFF) - (float) meanRGB[1]) / (float)div;
                    float r = (((rgb >> 16) & 0xFF) - (float) meanRGB[0]) / (float)div;
                    if (useBGR) {
                        pixelFloats[z++] = b;
                        pixelFloats[z++] = g;
                        pixelFloats[z++] = r;
                    }
                    else {
                        pixelFloats[z++] = r;
                        pixelFloats[z++] = g;
                        pixelFloats[z++] = b;
                    }
                }
            }
            img.recycle();
            imgcrop.recycle();
            imgresize.recycle();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return pixelFloats;
    }

    public static void write2file(String inputName, String content) {
        File file = new File(inputName);
        if(!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            FileWriter fileWriter = new FileWriter(file, true);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            bufferedWriter.write(content+"\n");
            bufferedWriter.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void writeData(String imageName, double []data) {
        File file = new File("/storage/emulated/0/Android/data/com.demo.qcbenchmark/files/" + imageName + ".txt");
        if(!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            FileWriter fileWriter = new FileWriter(file, false);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            StringBuilder sb = new StringBuilder();
            for(int i=0; i<data.length; i++) {
                sb.append(data[i]).append("\n");
            }
            sb.deleteCharAt(sb.length()-1);
            //Log.d("Utils", sb.toString());
            bufferedWriter.write(sb.toString());
            bufferedWriter.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void writeTopK(String imageName, HashMap<String, Boolean> data) {
        File file = new File("/storage/emulated/0/Android/data/com.demo.qcbenchmark/files/" + imageName + ".txt");
        if(!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            FileWriter fileWriter = new FileWriter(file, false);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            StringBuilder sb = new StringBuilder();
            for(Map.Entry<String, Boolean> entry : data.entrySet()) {
                sb.append(entry.getKey()).append("   " + entry.getValue() + "\n");
            }
            sb.deleteCharAt(sb.length()-1);
            //Log.d("Utils", sb.toString());
            bufferedWriter.write(sb.toString());
            bufferedWriter.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void displayAtToast(Context context, String str, int showTime) {
        Toast toast = Toast.makeText(context, str, showTime);
        toast.setGravity(Gravity.CENTER , 0, 0);  //display position
        LinearLayout layout = (LinearLayout) toast.getView();
        layout.setBackgroundColor(Color.GRAY);
        TextView v = toast.getView().findViewById(android.R.id.message);
        v.setTextColor(Color.RED);     //font color
        toast.show();
    }

    public static boolean deleteFolderFile(File path) {
        if (path.isDirectory()) {
            if (path.listFiles().length != 0) {
                for (File file : path.listFiles()) {
                    boolean res = deleteFolderFile(file);
                    if (!res) {
                        return false;
                    }
                }
            }
            return true;
        } else {
            return path.delete();
        }
    }

    public static float[] readArrayFromTxt(String filePath) {
        float [] result = null;
        try{
            File file = new File(filePath);
            if(file.exists()) {
                result = new float[(int) file.length()];
                Log.d(TAG, "file length " + file.length());
                BufferedReader br = new BufferedReader(new FileReader(filePath));//构造一个BufferedReader类来读取文件
                String s;
                int i = 0;
                while ((s = br.readLine()) != null) {//使用readLine方法，一次读一行
                    result[i++] = Float.parseFloat(s);
                }
                br.close();
            } else {
                Log.d(TAG, "file not exist： " +  filePath);
            }

        }catch(Exception e){
            e.printStackTrace();
        }
        return result;
    }

    public static int checkImageDirValidation(String imagePath) {
        // check if imagePath contains images
        File images = new File(imagePath);
        if(!images.exists() || images.listFiles().length == 0) {
            Log.d(TAG, "Image path not exists:" + imagePath + " or No data exists, image length: " + images.listFiles().length);
            return 0;
        }
        return images.listFiles().length;
    }

    public static void clearInputList(String modelName) {
        String inputPath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("input_list").getAbsolutePath();
        if(modelName.contains("deeplab"))
            inputPath = inputPath + "/deeplabv3_input_list.txt";
        else if(modelName.contains("ssd"))
            inputPath = inputPath + "/mobilenetssd_input_list.txt";
        else if(modelName.contains("vdsr"))
            inputPath = inputPath + "/vdsr_input_list.txt";

        File file = new File(inputPath);
        if(file.exists())
            file.delete();
    }

    public static void writeArrayTofile(String filePath, float[] data){
        try {
            FileOutputStream os = new FileOutputStream(filePath);
            ByteBuffer byteBuf = ByteBuffer.allocate(4 * data.length);
            FloatBuffer floatBuf = byteBuf.asFloatBuffer();
            floatBuf.put(data);
            os.write(byteBuf.array());
            os.close();
        }
        catch (Exception e){
            Log.e(TAG, e.toString());
            e.printStackTrace();
        }
    }

    public static float[] readFloatArrayFromFile(String filePath){
        try {
            FileInputStream is = new FileInputStream(filePath);
            byte[] byteArray = new byte[is.available()];
            is.read(byteArray);
            ByteBuffer byteBuf = ByteBuffer.wrap(byteArray);
            FloatBuffer floatBuf = byteBuf.asFloatBuffer();
            float[] floatArray = new float[floatBuf.limit()];
            floatBuf.get(floatArray);
            return floatArray;
        }
        catch (Exception e){
            Log.e(TAG, e.toString());
            e.printStackTrace();
        }
        return null;
    }
}
