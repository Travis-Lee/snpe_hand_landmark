#include "g_hand_config.h"
#include <iostream>
#include "g_hand_landmark_api.h"
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

typedef struct _Hand_Landmark_KeyPoint {
  cv::Point2f p;
  float prob;
} KeyPoint;

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
            mat = (mat) / 255;
            c++;
        }
        cv::merge(mats, mat_dst);
    }
    return mat_dst;
}

int print_usage(int argc,char* argv[]){
  std::cout<<argv[0]<<"modelPath comp_type image_path dst_image"<<std::endl;
  return -1;
}

int main(int argc,char* argv[]){
   if(argc!=5){
     print_usage(argc,argv);
     return -1;
   }
   char* config_str=NULL;
   const char* model_path=argv[1];
   const char* infer_type=argv[2];
   
   GCreateHandJson(model_path,infer_type,1,22,&config_str);
   std::cout<<"config str:"<<config_str<<std::endl;
   
   void* handle=NULL;
   int ret=0;
   ret= GHandLandmarkInit(config_str,&handle);
   if(NULL==handle || ret!=0){
    std::cout<<"error:initialize engine failed.\n"<<std::endl;
    free(config_str);
   }
  
    const char* img_path=argv[3];
    cv::Mat org_img=cv::imread(img_path);
    

    cv::Mat tmp_img;
    cv::cvtColor(org_img,tmp_img,CV_BGR2GRAY);
    cv::Mat img;
    cv::cvtColor(tmp_img,img,CV_GRAY2BGR);

    int src_w = img.cols;
    int src_h = img.rows;
    int channel = img.channels();

    printf("org width:%d,org height:%d\n",src_w,src_h);
    
    int input_width =368; 
    int input_height = 368;
    printf("========>width:%d,height:%d\n",input_width,input_height);
    cv::Mat mat_resize=cv::Mat(input_height,input_width,CV_8UC3,cv::Scalar::all(0));
    cv::resize(img, mat_resize, cv::Size(input_width, input_height), cv::INTER_LINEAR);
    printf("debug111");

    cv::Mat input_mat = normalize(mat_resize);
   int object_num=0;
 
   for (int i=0;i<1;i++){
      ret= GHandLandmarkProcess(handle, input_mat, src_w,
                                src_h,channel, &object_num);

      if(NULL==handle || ret!=0){
       std::cout<<"Object Process is  failed.\n"<<std::endl;
       return -2;
     }
    }
   printf("Object Number:%d\n",object_num);
   std::vector<KeyPoint> det_results;
   det_results.resize(22);
   ret=GHandLandmarkGetResult(handle,20,&det_results[0]);

   static const int    joint_pairs[22][2]=    // POSE_PAIRS[20][2] =
   {
     {0,1}, {1,2}, {2,3}, {3,4},         // thumb
     {0,5}, {5,6}, {6,7}, {7,8},         // index
     {0,9}, {9,10}, {10,11}, {11,12},    // middle
     {0,13}, {13,14}, {14,15}, {15,16},  // ring
     {0,17}, {17,18}, {18,19}, {19,20}   // small
   };

    for (int i = 0; i < 22; i++)
    {
        const KeyPoint& p1 = det_results[joint_pairs[i][0]];
        const KeyPoint& p2 = det_results[joint_pairs[i][1]];

        cv::line(img, p1.p, p2.p, cv::Scalar(255, 0, 0), 2);
    }

    // draw joint
    for (size_t i = 0; i < det_results.size(); i++)
    {
        const KeyPoint& keypoint = det_results[i];
        cv::circle(img, keypoint.p, 3, cv::Scalar(0, 255, 0), -1);
    }


    const char* dst_img=argv[4];

    cv::imwrite(dst_img,img);

    GHandLandmarkRelease(handle);

    free(config_str);

    return 0;
}
