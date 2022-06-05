#ifndef VISION_HAND_LANDMARK_H
#define VISION_HAND_LANDMARK_H


#include <iostream>
#include <memory>
#include <string.h>
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "SNPE/SNPE.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef struct _Hand_Landmark_KeyPoint{
   cv::Point2f p;
   float prob;
}KeyPoint;

class HandLandmark {
public:
    HandLandmark();
    virtual ~HandLandmark()=default;
    int init(const char* infer_type,std::string model_path);
    int inference(cv::Mat& input_mat, std::pair<int, float*>& pair);
    int detect(cv::Mat& input_mat, int w, int h, int channel, std::vector<KeyPoint>& keypoint);

private:
    int setRuntime(std::string infer_type);
    void build_tensor(cv::Mat& mat);

    // snpe model
    std::unique_ptr<zdl::SNPE::SNPE> _engine;
    std::unique_ptr<zdl::DlContainer::IDlContainer> _container;

    // snpe input & output
    zdl::DlSystem::StringList _output_tensor_names;
    // if use ITensor
    std::unique_ptr<zdl::DlSystem::ITensor> _input_tensor;
    zdl::DlSystem::TensorMap _output_tensor_map;

    // snpe builder config
    // _runtime_list : runtime order list
    zdl::DlSystem::RuntimeList _runtime_list;
};

#endif //
