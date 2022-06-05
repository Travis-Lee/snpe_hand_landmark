/*
 * Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.networkEvaluation;
import android.util.Log;

import com.qualcomm.qti.psnpedemo.components.BenchmarkApplication;
import com.qualcomm.qti.psnpedemo.processor.ClassificationPostprocessor;
import com.qualcomm.qti.psnpedemo.processor.DeeplabV3PreProcessor;
import com.qualcomm.qti.psnpedemo.processor.DetectionPostprocessor;
import com.qualcomm.qti.psnpedemo.processor.FCN8SPreProcessor;
import com.qualcomm.qti.psnpedemo.processor.InceptionV3PreProcessor;
import com.qualcomm.qti.psnpedemo.processor.MobileNetPreProcessor;
import com.qualcomm.qti.psnpedemo.processor.MobileNetSSDPreProcessor;
import com.qualcomm.qti.psnpedemo.processor.PostProcessor;
import com.qualcomm.qti.psnpedemo.processor.PreProcessor;
import com.qualcomm.qti.psnpedemo.processor.ResnetPreProcessor;
import com.qualcomm.qti.psnpedemo.processor.SegmentationPostProcessor;
import com.qualcomm.qti.psnpedemo.processor.SuperresolutionPostprocessor;
import com.qualcomm.qti.psnpedemo.processor.VGGPreProcessor;
import com.qualcomm.qti.psnpedemo.processor.VDSRPreprocessor;
import com.qualcomm.qti.psnpedemo.utils.Util;
import com.qualcomm.qti.psnpe.PSNPEConfig;
import com.qualcomm.qti.psnpe.PSNPEManager;
import com.qualcomm.qti.psnpe.PSNPEManagerListener;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import static com.qualcomm.qti.psnpedemo.networkEvaluation.TimeProfiler.TIME_TYPE.BUILD_TIME;
import static com.qualcomm.qti.psnpedemo.networkEvaluation.TimeProfiler.TIME_TYPE.EXECUTE_TIME;


public class NetworkEvaluator {
    private static String TAG = NetworkEvaluator.class.getSimpleName();
    private int inputSize;
    private ModelInfo modelInfo;
    private PSNPEConfig psnpeConfig;
    private Result result;
    private PreProcessor imagePreprocessor;
    private PostProcessor resultPostProcessor;
    private EvaluationCallBacks evaluationCallBacks;
    private PSNPEManagerListener listener;
    private HashMap<Integer, String> imageMap;
    private Lock asyncLock;
    private Condition asyncCondition;
    private TimeProfiler timeProfiler;
    private AtomicBoolean stressRunningStatus;

    public enum FILE_TYPE {
        MODEL,
        SCENARIO,
        IMAGE,
        GROUND_TRUTH,
        MODEL_LABEL,
        RESULT,
        MAX_FILE_TYPE,
    }

    public NetworkEvaluator(ModelInfo modelInfo) {
        this.modelInfo = modelInfo;
        String imagePath = getFilePath(FILE_TYPE.IMAGE);
        inputSize = Util.checkImageDirValidation(imagePath);
        initPreProcessor();
        initPostProcessor(inputSize);
        result = new Result();
        psnpeConfig = null;
        imageMap = new HashMap<>();
        asyncLock = new ReentrantLock();
        asyncCondition = asyncLock.newCondition();
        timeProfiler = new TimeProfiler(true);
        listener = new PSNPEManagerListener() {
            @Override
            public void getOutputCallback(int index, Map<String, float[]> data, int errorCode) {
                Log.d(TAG, "On output call back." + index + " " + NetworkEvaluator.this.psnpeConfig.bulkSize);
                resultPostProcessor.addToProcessList(imageMap.get(index), data);
            }

            @Override
            public void onInferenceDone() {
                timeProfiler.endProfile(EXECUTE_TIME);
                Log.i(TAG, "OnInferenceDone()");
                if(psnpeConfig.transmissionMode.equals("inputOutputAsync")) {
                    asyncLock.lock();
                    asyncCondition.signal();
                    asyncLock.unlock();
                }
            }

            @Override
            public void onOutputProcessDone() {
                Log.i(TAG, "OnOutputProcessDone()");
                if(psnpeConfig.transmissionMode.equals("outputAsync")) {
                    asyncLock.lock();
                    asyncCondition.signal();
                    asyncLock.unlock();
                }
            }
            @Override
            public float[] IOAsyncInputCallback(String s) {
                File file = new File(s);
                if(!file.exists())
                {
                    Log.e(TAG, "File does not exist:"+s);
                    return new float[0];
                }
                float[] data = imagePreprocessor.preProcessData(file);
                if (data == null) {
                    Log.e(TAG, "Preprocess data failed, Image path: " + file.getAbsolutePath());
                    evaluationCallBacks.setExecuteStatus("Preprocess data failed, Image path: " + file.getAbsolutePath());
                    return new float[0];
                }
                return data;
            }
        };
        stressRunningStatus = new AtomicBoolean(false);
        PSNPEManager.registerPSNPEManagerListener(listener);
    }

    public void initPreProcessor() {
        String modelName = modelInfo.getModelName();
        if(modelName.contains("inception")) {
            imagePreprocessor = new InceptionV3PreProcessor();
        } else if(modelName.contains("resnet")) {
            imagePreprocessor = new ResnetPreProcessor();
        } else if(modelName.contains("vgg")) {
            imagePreprocessor = new VGGPreProcessor();
        } else if(modelName.contains("ssd")) {
            imagePreprocessor = new MobileNetSSDPreProcessor();
        } else if(modelName.contains("mobilenet")) {
            imagePreprocessor = new MobileNetPreProcessor();
        } else if(modelName.contains("deeplabv3")) {
            imagePreprocessor = new DeeplabV3PreProcessor();
        } else if(modelName.contains("fcn8s")) {
            imagePreprocessor = new FCN8SPreProcessor();
        }
        else if(modelName.contains("vdsr")){
            imagePreprocessor = new VDSRPreprocessor(this.modelInfo);
        }
    }

    public void initPostProcessor(int inputSize) {
        if(inputSize <= 0) {
            Log.e(TAG, "Inputsize<= 0 error when init post processor");
            return;
        }
        String scenarioName = modelInfo.getScenarioName();
        if(scenarioName.contains("classification")) {
            resultPostProcessor = new ClassificationPostprocessor(evaluationCallBacks, modelInfo, inputSize);
        }else if(scenarioName.contains("detection")) {
            resultPostProcessor = new DetectionPostprocessor(evaluationCallBacks, inputSize);
        }else if(scenarioName.contains("segmentation")) {
            resultPostProcessor = new SegmentationPostProcessor(inputSize);
        }else if(scenarioName.contains("superresolution")) {
            resultPostProcessor = new SuperresolutionPostprocessor(modelInfo, inputSize);
        }
    }

    public void setPsnpeConfig(PSNPEConfig config) {
        this.psnpeConfig = config;
    }

    public void setEvaluationCallBacks(EvaluationCallBacks evaluationCallBacks) {
        this.evaluationCallBacks = evaluationCallBacks;
        resultPostProcessor.setEvaluationCallBacks(evaluationCallBacks);
    }

    public PSNPEConfig getPsnpeConfig() {
        return psnpeConfig;
    }

    public Result getResult() {
        return result;
    }

    public ModelInfo getModelInfo() {
        return modelInfo;
    }

    public boolean run() {
        // clear result from last time.
        result.clear();
        Util.clearInputList(modelInfo.getModelName());
        timeProfiler.setAccumulate(true);
        // get user config information
        String imagePath = getFilePath(FILE_TYPE.IMAGE);
        int bulkSize = psnpeConfig.bulkSize;
        int imageNums = Util.checkImageDirValidation(imagePath);

        int executeTimes = 0;
        if(bulkSize == 0) {
            Log.e(TAG, "BulkSize=0 error");
	    return false;
        }
        else {
            executeTimes = (imageNums + bulkSize - 1)/bulkSize;
        }
        if(imageNums == 0) return false;

        evaluationCallBacks.setExecuteStatus("Building...");
        timeProfiler.startProfile(BUILD_TIME);
        if (!PSNPEManager.buildFromFile(modelInfo.getModelName())) {
            Log.e(TAG, "Build failed, images number: " + imageNums + "\n model name: " + modelInfo.getModelName());
            PSNPEManager.release();
            evaluationCallBacks.setExecuteStatus("Build failed");
            evaluationCallBacks.showErrorText("Build failed, imagesNums" + imageNums);
            return false;
        }
        timeProfiler.endProfile(BUILD_TIME);
        // PSNPE will handle bulkSize of image at a time.
        int index = 0;
        File[] imagesList = new File(imagePath).listFiles();
        ArrayList<File> bulkImage = new ArrayList<>();

        for(int time = 0; time < executeTimes; time++) {
            int handleSize = time == executeTimes - 1? imageNums - time*bulkSize : bulkSize;
            Log.i(TAG, "Iterator " + time + " handleSize: " + handleSize);

            for(int i=0; i<handleSize; i++) {
                File image = imagesList[index++];
                float[] data = imagePreprocessor.preProcessData(image);
                evaluationCallBacks.setExecuteStatus("Loading Images (" + (i+1) + "/" + handleSize + ") ...");

                if (data == null) {
                    PSNPEManager.release();
                    Log.e(TAG, "Preprocess data failed, Image path: " + image.getAbsolutePath());
                    evaluationCallBacks.setExecuteStatus("Preprocess data failed, Image path: " + image.getAbsolutePath());
                    return false;
                }

                if (!PSNPEManager.loadData(data, i)) {
                    PSNPEManager.release();
                    Log.e(TAG, "Load Data Failed, index； " + i + " path: " + image.getAbsolutePath()) ;
                    evaluationCallBacks.setExecuteStatus("Load Data Failed, index； " + i + " path: " + image.getAbsolutePath());
                    return false;
                }
                bulkImage.add(image);
            }
            evaluationCallBacks.setExecuteStatus("Executing...");

            timeProfiler.startProfile(EXECUTE_TIME);
            if (!PSNPEManager.executeSync()) {
                PSNPEManager.release();
                Log.e(TAG, "Execute failed");
                evaluationCallBacks.setExecuteStatus("Execute failed");
                return false;
            }
            timeProfiler.endProfile(EXECUTE_TIME);

            Log.i(TAG, "Execute time: " + time);
            resultPostProcessor.postProcessResult(bulkImage);
            bulkImage.clear();
        }

        result.updateFromProfiler(timeProfiler);
        result.setFPS((double)imageNums/result.getInferenceTime());
        if(modelInfo.getModelName().contains("vgg13") && psnpeConfig.runtimeConfigs[0].userBufferMode.equals("TF8")) {
            // tops are only support on vgg13. vgg13 is not used for classification
            result.setTops((float) (2 * 15.35 * result.getFPS()) / 1000);
            result.setTop1(0);
            result.setTop5(0);
        }
        resultPostProcessor.setResult(result);
        PSNPEManager.release();
        timeProfiler.reset();
        Log.d(TAG, "Execute Finished");
        evaluationCallBacks.setExecuteStatus("Success");
        return true;
    }

    public boolean runOutputAsync() {
        imageMap.clear();
        result.clear();
        timeProfiler.setAccumulate(true);
        // get user config information
        String imagePath = getFilePath(FILE_TYPE.IMAGE);
        int bulkSize = psnpeConfig.bulkSize;
        int imageNums = Util.checkImageDirValidation(imagePath);
        if(imageNums == 0) return false;

        evaluationCallBacks.setExecuteStatus("Building...");
        timeProfiler.startProfile(BUILD_TIME);
        if (!PSNPEManager.buildFromFile(modelInfo.getModelName())) {
            Log.e(TAG, "Build failed, images number: " + imageNums + "\n model name: " + modelInfo.getModelName());
            PSNPEManager.release();
            evaluationCallBacks.setExecuteStatus("Build failed");
            evaluationCallBacks.showErrorText("Build failed, imagesNums" + imageNums);
            return false;
        }
        timeProfiler.endProfile(BUILD_TIME);

        // BulkSnpe will handle bulkSize of image at a time.
        File[] imagesList = new File(imagePath).listFiles();
        int executeTimes = (imageNums + bulkSize -1)/bulkSize;

        resultPostProcessor.start();
        int index = 0;
        for(int time = 0; time < executeTimes; time++) {
            int handleSize = time == executeTimes - 1 ? imageNums - time * bulkSize : bulkSize;
            Log.i(TAG, "Iterator: " + time + " handleSize: " + handleSize);

            for (int i = 0; i < handleSize; i++, index++) {
                File image = imagesList[index];
                float[] data = imagePreprocessor.preProcessData(image);
                evaluationCallBacks.setExecuteStatus("Loading Images (" + (i + 1) + "/" + handleSize + ") ...");

                if (data == null) {
                    PSNPEManager.release();
                    Log.e(TAG, "Preprocess data failed, Image path: " + image.getAbsolutePath());
                    evaluationCallBacks.setExecuteStatus("Preprocess data failed, Image path: " + image.getAbsolutePath());
                    return false;
                }

                if (!PSNPEManager.loadData(data, i)) {
                    PSNPEManager.release();
                    Log.e(TAG, "Load Data Failed, index； " + i + " path: " + image.getAbsolutePath());
                    evaluationCallBacks.setExecuteStatus("Load Data Failed, index； " + i + " path: " + image.getAbsolutePath());
                    return false;
                }
                imageMap.put(i, image.getName());
            }
            evaluationCallBacks.setExecuteStatus("Executing...");
            timeProfiler.startProfile(EXECUTE_TIME);
            PSNPEManager.executeOutputAsync();
            asyncLock.lock();
            try {
                asyncCondition.await();
            } catch (InterruptedException e) {
                Log.e(TAG, "Interrupted Exception");
            }
            asyncLock.unlock();
        }
        resultPostProcessor.waitForResult(result);
        result.updateFromProfiler(timeProfiler);
        result.setFPS((double)imageNums/(result.getInferenceTime()));
        if(modelInfo.getModelName().contains("vgg13") && psnpeConfig.runtimeConfigs[0].userBufferMode.equals("TF8")) {
            // tops are only support on vgg13. vgg13 is not used for classification
            result.setTops((float) (2 * 15.35 * result.getFPS()) / 1000);
            result.setTop1(0);
            result.setTop5(0);
        }
        PSNPEManager.release();
        timeProfiler.reset();
        Log.i(TAG, "Execute Finished");
        evaluationCallBacks.setExecuteStatus("Success");
        return true;
    }

    public boolean runInputOutputAsync() {
        imageMap.clear();
        result.clear();
        timeProfiler.setAccumulate(true);
        // get user config information
        String imagePath = getFilePath(FILE_TYPE.IMAGE);
        int imageNums = Util.checkImageDirValidation(imagePath);
        if(imageNums == 0) return false;

        evaluationCallBacks.setExecuteStatus("Building...");
        timeProfiler.startProfile(BUILD_TIME);
        if (!PSNPEManager.buildFromFile(modelInfo.getModelName())) {
            Log.e(TAG, "Build failed, images number: " + imageNums + "\n model name: " + modelInfo.getModelName());
            PSNPEManager.release();
            evaluationCallBacks.setExecuteStatus("Build failed");
            evaluationCallBacks.showErrorText("Build failed, imagesNums" + imageNums);
            return false;
        }
        timeProfiler.endProfile(BUILD_TIME);

        resultPostProcessor.start();
        // BulkSnpe will handle bulkSize of image at a time.
        File[] imagesList = new File(imagePath).listFiles();
        Log.i(TAG, "Image num: " + imagesList.length);
        List<String> files = new ArrayList<String>();

        for (int i = 0; i < imagesList.length; i++) {
            if (imagesList[i].isFile()) {
                files.add(imagesList[i].toString());
            }
        }

        for(int i = 0; i < imagesList.length; i++) {
            evaluationCallBacks.setExecuteStatus("Loading Images (" + (i+1) + "/" + imagesList.length + ") ...");
            if(i == 0) timeProfiler.startProfile(EXECUTE_TIME);
            List<String> file = new ArrayList<String>();
            file.add(files.get(i));
            PSNPEManager.executeInputOutputAsync(file, i, imagesList.length);
        }
        asyncLock.lock();
        try {
            asyncCondition.await();
        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted Exception in runInputOutputAsync()");
        }
        asyncLock.unlock();
        evaluationCallBacks.setExecuteStatus("Executing...");
        resultPostProcessor.waitForResult(result);
        result.updateFromProfiler(timeProfiler);
        result.setFPS((double)imageNums/result.getInferenceTime());

        if(modelInfo.getModelName().contains("vgg13") && psnpeConfig.runtimeConfigs[0].userBufferMode.equals("TF8")) {
            // tops are only support on vgg13. vgg13 is not used for classification
            result.setTops((float) (2 * 15.35 * result.getFPS()) / 1000);
            result.setTop1(0);
            result.setTop5(0);
        }

        timeProfiler.reset();
        PSNPEManager.release();
        Log.i(TAG, "Execute Finished");
        evaluationCallBacks.setExecuteStatus("Success");
        return true;
    }

    public boolean startStressTest() {
        stressRunningStatus.set(true);
        timeProfiler.setAccumulate(true);
        // get user config information
        String imagePath = getFilePath(FILE_TYPE.IMAGE);
        int imageNumber = Util.checkImageDirValidation(imagePath);
        if(imageNumber == 0) {
            evaluationCallBacks.setExecuteStatus("No data exist: " + imagePath);
            return false;
        }

        evaluationCallBacks.setExecuteStatus("Building...");
        timeProfiler.startProfile(BUILD_TIME);
        if (!PSNPEManager.buildFromFile(modelInfo.getModelName())) {
            Log.e(TAG, "Build failed, images number: " + imageNumber + "\n model name: " + modelInfo.getModelName());
            PSNPEManager.release();
            timeProfiler.reset();
            evaluationCallBacks.setExecuteStatus("Build failed, imageNumber" + imageNumber);
            return false;
        }
        timeProfiler.endProfile(BUILD_TIME);

        // Load Data at a time.
        File[] imagesList = new File(imagePath).listFiles();

        int testCycle = 0;
        for(int index = 0; index < imageNumber; index++) {
            File image = imagesList[index];
            float[] data = imagePreprocessor.preProcessData(image);
            evaluationCallBacks.setExecuteStatus("Loading Images (" + (index+1) + "/" + imageNumber + ") ...");

            if (data == null) {
                PSNPEManager.release();
                timeProfiler.reset();
                Log.e(TAG, "Preprocess data failed, Image path: " + image.getAbsolutePath());
                evaluationCallBacks.setExecuteStatus("Preprocess data failed, Image path: " + image.getAbsolutePath());
                return false;
            }

            if (!PSNPEManager.loadData(data, index)) {
                PSNPEManager.release();
                timeProfiler.reset();
                Log.e(TAG, "Load Data Failed, index； " + index + " path: " + image.getAbsolutePath()) ;
                evaluationCallBacks.setExecuteStatus("Load Data Failed, index； " + index + " path: " + image.getAbsolutePath());
                return false;
            }
        }

        String currentFPS = "FPS:";
        String executeStatus;

        while(stressRunningStatus.get()) {
            executeStatus = "Test-" + testCycle + " Executing...";
            evaluationCallBacks.setExecuteStatus(executeStatus + "\n" + currentFPS);
            timeProfiler.startProfile(EXECUTE_TIME);
            if (!PSNPEManager.executeSync()) {
                PSNPEManager.release();
                timeProfiler.reset();
                Log.e(TAG, "Test-" + testCycle + " Execute failed");
                evaluationCallBacks.setExecuteStatus("Test-" + testCycle + " Execute failed");
                stressRunningStatus.set(false);
                return false;
            }
            timeProfiler.endProfile(EXECUTE_TIME);
            currentFPS = "FPS: " + ((double)imageNumber * 1000.0 / timeProfiler.getTime(EXECUTE_TIME));
            Log.d(TAG, "Execute time: " + timeProfiler.getTime(EXECUTE_TIME) + "ms");
            evaluationCallBacks.setExecuteStatus(executeStatus + "\n" + currentFPS);
            timeProfiler.reset();
            testCycle++;
        }

        PSNPEManager.release();
        timeProfiler.reset();

        asyncLock.lock();
        //Notify the STOP button to quit
        asyncCondition.signal();
        asyncLock.unlock();

        Log.d(TAG, "Stress Test Finished");
        evaluationCallBacks.setExecuteStatus("Stress Test Finished");
        return true;
    }

    public void stopStressTest() {
        stressRunningStatus.set(false);
        evaluationCallBacks.setExecuteStatus("Wait For Last Execution Finished");
        asyncLock.lock();
        try {
            asyncCondition.await();
        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted Exception");
        } finally {
            asyncLock.unlock();
        }
    }
    public String getFilePath(FILE_TYPE fileType) {
        String packagePath = BenchmarkApplication.getCustomApplicationContext().getExternalFilesDir("").getAbsolutePath();
        String filePath = "";

        switch (fileType) {
            case MODEL:
                filePath = packagePath + "/models/" + modelInfo.getScenarioName() + "/" + modelInfo.getModelName() + ".dlc";
                break;
            case IMAGE:
                filePath = packagePath + "/datasets/" + modelInfo.getDataSetName() + "/images/";
                break;
            case RESULT:
                filePath = packagePath + "/results/" + modelInfo.getScenarioName() + "/"
                        + modelInfo.getModelName() + "/radio_3";
                break;
            case SCENARIO:
                filePath = packagePath + "/models/";
                break;
            case GROUND_TRUTH:
                filePath = packagePath + "/datasets/" + modelInfo.getDataSetName() + "/labels.txt";
                break;
            case MODEL_LABEL:
                filePath = packagePath + "/models/" + modelInfo.getScenarioName() + "/"
                        + modelInfo.getModelName() + "_labels.txt";
                break;
            default:
                Log.d(TAG, "Invalid file type: " + fileType);
                break;
        }
        Log.d(TAG, "Filepath: " + filePath);
        return filePath;
    }
}
