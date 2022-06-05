/*
 * Copyright (c) 2019 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.components;

import android.app.Application;
import android.content.Context;
import android.util.Log;

public class BenchmarkApplication extends Application {

    private static Context mContext;
    private static String mNativeLibraryPath;


    @Override
    public void onCreate() {
        super.onCreate();
        mContext = getApplicationContext();
        mNativeLibraryPath = getApplicationInfo().nativeLibraryDir;
    }

    public static Context getCustomApplicationContext() {
        return mContext;
    }

    public static String getNativeLibraryPath() {
        return mNativeLibraryPath;
    }

}
