#include "hand_landmark.h"
#include <iostream>
#include <fstream>
using namespace std;

HandLandmark::HandLandmark() {}

template <typename T>
void chw_to_hwc(T* in, T* out, int w, int h, int c) {
    int count = 0;
    int step = h * w;
    for (int i = 0; i < step; ++i) {
        for (int j = 0; j < c; ++j) {
            out[count] = in[j * step + i];
            count += 1;
        }
    }
}
template <typename T>
void hwc_to_chw(T* in, T* out, int w, int h, int c) {
    int count = 0;
    int step = h * w;
    for (int i = 0; i < c; ++i) {
        for (int j = 0; j < step; ++j) {
            out[count] = in[j * c + i];
            count += 1;
        }
    }
}

int HandLandmark::init(const char* infer_type, std::string  model_path) {

    // 1.set runtime
    std::string it(infer_type);
    int ret = setRuntime(it);
    if(ret != 0) {
        cout << "setRuntime error : " << zdl::DlSystem::getLastErrorString() << endl;
        return -1;
    }

    // 2. load model
    /*
    FILE* fp=fopen(model_path.c_str(),"rb");
    if (NULL==fp){
       printf("load detect model error:%s",model_path.c_str());
       return -1;
    }
    uint8_t* buffer=NULL;
    size_t size=20;
    _container = zdl::DlContainer::IDlContainer::open(buffer,size);
    */
    _container = zdl::DlContainer::IDlContainer::open(model_path);
    if (_container == nullptr) {
        cout << "load model error : " << zdl::DlSystem::getLastErrorString() << endl;
        return -1;
    }
    // 3. build engine
    zdl::SNPE::SNPEBuilder snpe_builder(_container.get());
    _engine = snpe_builder
            .setOutputLayers({})
            .setRuntimeProcessorOrder(_runtime_list)
            .build();
    if (_engine == nullptr) {
        cout << "build engine error : " << zdl::DlSystem::getLastErrorString() << endl;
        return -1;
    }

    const auto &strList_opt = _engine->getInputTensorNames();
    if (!strList_opt) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &strList = *strList_opt;
    const auto &inputDims_opt = _engine->getInputDimensions(strList.at(0));
    const auto &inputShape = *inputDims_opt;
    printf("net input width:%d,net input height:%d\n",inputShape[2],inputShape[1]);
    cout << "init success..." << endl;
    return 0;
}

int HandLandmark::setRuntime(std::string infer_type){
    zdl::DlSystem::Runtime_t runtime_t;

    if (infer_type == "1") {
        runtime_t = zdl::DlSystem::Runtime_t::GPU_FLOAT32_16_HYBRID;
    } else if (infer_type == "2") {
        runtime_t = zdl::DlSystem::Runtime_t::DSP;
    } else {
      runtime_t = zdl::DlSystem::Runtime_t::CPU;
    }

    const char* runtime_string = zdl::DlSystem::RuntimeList::runtimeToString(runtime_t);

    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime_t) ||
        (infer_type == "1" && !zdl::SNPE::SNPEFactory::isGLCLInteropSupported())) {
        cout << "SNPE runtime " <<  runtime_string << " not support" << endl;
        return -1;
    }

    cout << "SNPE model init, using runtime " <<  runtime_string << endl;

    _runtime_list.add(runtime_t);
    return 0;
}

void HandLandmark::build_tensor(cv::Mat& mat) {

    zdl::DlSystem::Dimension dims[4];
    dims[0] = 1;
    dims[1] = mat.rows;
    dims[2] = mat.cols;
    dims[3] = mat.channels();
    size_t size = 4; // fp32
    zdl::DlSystem::TensorShape tensorShape(dims, size);

    int mem_size = mat.rows * mat.cols * mat.channels();
    float* src = (float*) mat.data;
    _input_tensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(tensorShape);
    std::copy(src, src + mem_size, _input_tensor->begin());


}
int HandLandmark::detect(cv::Mat& input_mat, int w, int h, int channel,std::vector<KeyPoint>& keypoints) {
    
    printf("w:%d\n",w);
    printf("h:%d\n",h);
    printf("c:%d\n",channel);

    keypoints.clear();
    build_tensor(input_mat);

    bool ret = _engine->execute(_input_tensor.get(), _output_tensor_map);
    if (!ret) {
        cout << "engine inference error : " << zdl::DlSystem::getLastErrorString() << endl;
        return -1;
    } else {
        cout << "engine inference success..." << endl;
    }

    const zdl::DlSystem::Optional<zdl::DlSystem::StringList> &outputTensorNames = _engine->getOutputTensorNames();
    auto itensor = _output_tensor_map.getTensor((*outputTensorNames).at(0));
    if (itensor == nullptr) {
        cout << "output tensot is null : " << zdl::DlSystem::getLastErrorString() << endl;
        return -1;
    }
    auto itensor_shape = itensor->getShape();
    auto* dims = itensor_shape.getDimensions();
    
    std::cout<<"n====>"<<dims[0]<<std::endl;
    std::cout<<"h====>"<<dims[1]<<std::endl;
    std::cout<<"w====>"<<dims[2]<<std::endl;
    std::cout<<"c====>"<<dims[3]<<std::endl;
    size_t dim_count = itensor_shape.rank();
    int output_len = 1;
    for (int i = 0; i< dim_count; i++) {
        output_len *=dims[i];
    }
    //float* output_data_hwc = (float*)malloc(sizeof(float) * output_len);
    std::vector<float> output_data_hwc;
    output_data_hwc.resize(output_len);
    int i = 0;
    for(auto it = itensor->begin(); it!=itensor->end();it++)
    {
        output_data_hwc[i++] = *it;
    }

   for (int p = 0; p < dims[3]; ++p) {
        float max_prob = 0.f;
        int max_x = 0;
        int max_y = 0;
        for (int y = 0; y < dims[1]; ++y) {
            for (int x = 0; x < dims[2]; ++x) {
                int offset = (y * dims[2] + x) * dims[3] + p;
//                printf("%.f ", *(output_data_hwc + offset));
                //float prob = *(output_data_hwc + offset);
                float prob = output_data_hwc[offset];
                //float prob = *(output_data_hwc + offset);
                if (prob > max_prob) {
                    max_prob = prob;
                    max_x = x;
                    max_y = y;
                }
            }
        }
        KeyPoint keypoint;
        keypoint.p = cv::Point2f(max_x * w / (float)dims[2], max_y * h / (float)dims[1]);
        keypoint.prob = max_prob;
        keypoints.push_back(keypoint);
    }

  return 0;
}


