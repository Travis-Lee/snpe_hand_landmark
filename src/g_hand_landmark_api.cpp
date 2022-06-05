#include "g_hand_config.h"
#include "hand_landmark.h"
#include "g_hand_landmark_api.h"
#include "math.h"
#include <iostream>
#include <vector>
#include <string>
#include <string.h>
#include <pthread.h>
#include "g_hand_error.h"
#include "cJSON.h"

#define G_HAND_POSE_GET_RES 20

class DetectParam{
public:
   HandConfig cfg;
   HandLandmark handNet;
   int num;
   std::vector<KeyPoint> results;
};

int GHandLandmarkInit(const char* config_file, Hhandle* handle){
     if(NULL == handle) {return G_IMAGE_PARAM_INVALID;}
     DetectParam* param = new DetectParam();
     if (HandLandmarkConfig(config_file,&param->cfg)){
          delete param;
          return G_IMAGE_CONFIG_ERR;
     }

     param->results.resize(param->cfg.object_limit);
     
     std::string mp(param->cfg.model_path);
     int ret=param->handNet.init(param->cfg.comp_type, mp);
     
     if(ret !=0){
          delete param;
          return G_IMAGE_INIT_FAILED;
     }
   
     *handle = (Hhandle)param;

     return 0;
}

cv::Mat normalize(cv::Mat& mat_src) {
    cv::Mat mat_src_float;
    mat_src.convertTo(mat_src_float, CV_32FC3);

    cv::Mat mat_mean;
    cv::Mat mat_stddev;
    cv::meanStdDev(mat_src_float, mat_mean, mat_stddev);
    cv::Mat mat_dst;

    if (mat_src.channels() == 1) {
        auto m = *((double*)mat_mean.data);
        auto s = *((double*)mat_stddev.data);
        mat_dst = (mat_src_float - m) / (s + 1e-6);
    } else {
        std::vector<cv::Mat> mats;
        cv::split(mat_src_float, mats);
        int c = 0;
        for (auto& mat : mats) {
            auto m = ((double *)mat_mean.data)[c];
            auto s = ((double *)mat_stddev.data)[c];
            mat = (mat - m) / (s + 1e-6);
            c++;
        }
        cv::merge(mats, mat_dst);
    }
    return mat_dst;
}


int GHandLandmarkProcess(Hhandle handle, cv::Mat& img, int w,int h, int channel, int* point_num){

  if(NULL==handle){
    printf("Handle is Empty\n");
    return G_IMAGE_PARAM_INVALID; 
  }
  
  DetectParam* param =(DetectParam*)handle;
  std::vector<KeyPoint> keypoints;
  
  int ret= param->handNet.detect(img,w,h,channel,keypoints);
  if(ret!=0){
   printf("No get result for inference\n");
   return G_IMAGE_NO_RESULT;        
  }
  
  printf("points num:%d\n",keypoints.size());
  
  int object_count=0;
  for(int i=0;i<keypoints.size();++i){
	 if(object_count >= param->cfg.object_limit) break;
	 param->results[object_count].p.x = keypoints[i].p.x;
	 param->results[object_count].p.y = keypoints[i].p.y;
	 param->results[object_count].prob = keypoints[i].prob;
	 object_count++;
   }
    *point_num = object_count;
	param->num = object_count;
	std::cout<<"count=======>"<<param->num <<std::endl;
	keypoints.clear();

    return 0;
}

int GHandLandmarkGetResult(Hhandle handle, int type, void* value){
   
    DetectParam* param =(DetectParam*)handle;
    
    if(NULL==value||NULL==handle){
      return  G_IMAGE_PARAM_INVALID;      
    }

    KeyPoint* res=(KeyPoint*)value;
   
   /*
    *According Type Value For Get Diff Result 
    */
   switch(type){
        case G_HAND_POSE_GET_RES:
          memcpy(res,&(param->results[0]), param->num*sizeof(KeyPoint));
          break;
        default:
     	   printf("Invalid Get Type Value:%d\n",type);
    	   return  G_IMAGE_PARAM_INVALID;      
    }
    return 0;
}


void GHandLandmarkRelease(Hhandle handle){
  DetectParam* param=(DetectParam*)handle;
  if(NULL == handle){
    delete param;
    return;
  }
  delete param;
}

