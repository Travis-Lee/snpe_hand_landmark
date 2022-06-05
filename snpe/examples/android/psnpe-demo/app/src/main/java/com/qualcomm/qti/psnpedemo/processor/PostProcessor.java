/*
 * Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.processor;


import android.util.Log;

import com.qualcomm.qti.psnpedemo.networkEvaluation.EvaluationCallBacks;
import com.qualcomm.qti.psnpedemo.networkEvaluation.Result;

import java.io.File;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;

public abstract class PostProcessor {
    protected String TAG;
    protected int imageNumber;
    protected AtomicInteger count;
    protected EvaluationCallBacks evaluationCallBacks;
    protected ExecutorService producer;
    protected ExecutorService consumer;
    protected Future consumerResult;
    protected static class OutputData{
        public String fileName;
        public Map<String, float[]> data;
        public OutputData(String fileName, Map<String, float[]> data) {
            this.fileName = fileName;
            this.data = data;
        }
    }
    protected BlockingQueue<OutputData> dataQueue;

    public PostProcessor(final int imageNumber){
        TAG = this.getClass().getSimpleName();
        count = new AtomicInteger(0);
        this.imageNumber = imageNumber;
        dataQueue = new LinkedBlockingQueue<>();
        producer = Executors.newCachedThreadPool();
        consumer = Executors.newFixedThreadPool(1);
    }

    public void setEvaluationCallBacks(EvaluationCallBacks evaluationCallBacks) {
        this.evaluationCallBacks = evaluationCallBacks;
    }

    public EvaluationCallBacks getEvaluationCallBacks() {
        return evaluationCallBacks;
    }

    public void start() {
        consumerResult = consumer.submit(new Callable() {
            @Override
            public Integer call() {
                while(count.get() != imageNumber) {
                    try {
                        OutputData output = dataQueue.take();
                        getOutputCallback(output.fileName, output.data);
                        count.incrementAndGet();
                        //Log.i(TAG, "Consume data: " + output.fileName + " count: " + count + " total: " + imageNumber);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                Log.i(TAG, "Consumer finished!" + " " + count.get() + " image number: " + imageNumber);
                return count.get();
            }
        });
    }

    public void waitForResult(Result result) {
        try {
            Log.i(TAG, "wait for result generation");
            consumerResult.get();
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
            return;
        }
        Log.i(TAG, "Post process done. Update result.");
        setResult(result);
        count.set(0);
        dataQueue.clear();
    }

    public void addToProcessList(final String fileName, final Map<String, float[]> output) {
        producer.submit(new Runnable() {
            @Override
            public void run() {
                try {
                    dataQueue.put(new OutputData(fileName, output));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }
    public abstract void getOutputCallback(String fileName, Map<String, float[]> outputs);

    // Sync API
    public abstract boolean postProcessResult(ArrayList<File> bulkImage);
    public abstract void setResult(Result result);

}
