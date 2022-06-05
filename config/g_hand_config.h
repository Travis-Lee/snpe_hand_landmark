#ifndef HAND_CONFIG_INCLUDED_H
#define HAND_CONFIG_INCLUDED_H

typedef struct _Object_Hand_Config
{
    int thread_num;
    int object_limit;
    char model_path[512];     
    char comp_type[16];     
} HandConfig;

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief load hand config, an extension interface
 *
 * @param[in] config_str - config text, key=value
 * @param[out] cfg - concrete config
 * @return 0: success <0:failed
 */
int HandLandmarkConfig(const char* config_str, HandConfig* cfg);

/**
 * @brief generate detect json config
 *
 * @param[in] model_path - model path
 * @param[in] comp_type - CPU/GPU(1)/DSP(2)
 * @param[in] thread_num - thread number
 * @param[in] object_limit - object limit
 * @param[out] json_str - json string to generate
 * @return 0: success <0:failed
 */
int GCreateHandJson(const char* model_path, const char* comp_type, int thread_num, int object_limit, char** json_str);

#ifdef __cplusplus
}
#endif

#endif

