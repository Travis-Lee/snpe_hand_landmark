#ifndef G_HAND_POSE_ENGINE_INCLUDED_H
#define G_HAND_POSE_ENGINE_INCLUDED_H

#if defined MAKING_LIB
#define DLL_PUBLIC
#define DLL_LOCAL
#else
#if defined _WIN32 || defined __CYGWIN__
#ifdef MAKING_DLL
#ifdef __GNUC__
#define DLL_PUBLIC __attribute__((dllexport))
#else
#define DLL_PUBLIC \
  __declspec(      \
      dllexport)  // Note: actually gcc seems to also supports this syntax.
#endif
#else
#ifdef __GNUC__
#define DLL_PUBLIC __attribute__((dllimport))
#else
#define DLL_PUBLIC \
  __declspec(      \
      dllimport)  // Note: actually gcc seems to also supports this syntax.
#endif
#endif
#define DLL_LOCAL
#else
#if __GNUC__ >= 4
#define DLL_PUBLIC __attribute__((visibility("default")))
#define DLL_LOCAL __attribute__((visibility("hidden")))
#else
#define DLL_PUBLIC
#define DLL_LOCAL
#endif
#endif
#endif


#include "opencv2/opencv.hpp"
typedef void* Hhandle;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief initialize hand landmark 
 *
 * @param[in] config_file - config file
 * @param[out] handle - pointer to handle of engine
 * @return 0: success, <0: failed
 */
DLL_PUBLIC int GHandLandmarkInit(const char* config_file, Hhandle* handle);

/**
 * @brief Get Hand Pose landmark 
 *
 * @param[in] handle - handle of API
 * @param[in] threadNum - thread numbers
 * @param[in] img - image data
 * @param[in] w - image width
 * @param[in] h - image height
 * @param[in] channel - image channel, such as 3(RGB) or 4(RGBA)
 * @param[out] points_num - hand landmark numbers
 * @return  >=0: success <0: failed
 */
DLL_PUBLIC int GHandLandmarkProcess(Hhandle handle, cv::Mat& img, int w,int h, int channel, int* point_num);

/**
 * @brief get landmark result 
 *
 * @param[in] handle - handle to API
 * @param[in] type - get type
 * @param[out] value - result for getting
 * @return 0: success <0: failed
 */
DLL_PUBLIC int GHandLandmarkGetResult(Hhandle handle, int type, void* value);

/**
 * @brief release resource
 *
 * @param[in] handle - handle of engine
 */
DLL_PUBLIC void GHandLandmarkRelease(Hhandle handle);

#ifdef __cplusplus
}
#endif

#endif
