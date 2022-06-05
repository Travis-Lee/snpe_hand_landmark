#include "cJSON.h"
#include "g_hand_config.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

#define DETECT_MODEL_PATH "ModelPath"
#define DETECT_INFER_TYPE "compType"
#define DETECT_THREAD_NUM "ThreadNum"
#define DETECT_LANDMARK_LIMIT "ObjectLimit"

/**
 * @brief load hand config, an extension interface
 *
 * @param[in] config_str - config text, key=value
 * @param[out] cfg - concrete config
 * @return 0: success <0:failed
 */

int HandLandmarkConfig(const char* config_str, HandConfig* cfg){
  cJSON* json = NULL;
  cJSON* p_sub = NULL;
  
  if(NULL == config_str || NULL == cfg){
    fprintf(stderr,"input param is invaild.\n");
    return -1;
  }
   
  json = cJSON_Parse((char*)config_str);
  if(NULL == json){
    fprintf(stderr,"Parse json string failed.\n");
    return -1;
  }
  
  p_sub = cJSON_GetObjectItem(json,"modelPath");
  if(NULL == p_sub){
    fprintf(stderr,"json format is incorrect.\n");
    cJSON_Delete(json);
    return -1;
  }
  strcpy(cfg->model_path, p_sub->valuestring);
  
  p_sub = cJSON_GetObjectItem(json,"compType");
  if(NULL == p_sub){
    fprintf(stderr,"json format is incorrect.\n");
    cJSON_Delete(json);
    return -1;
  }
  strcpy(cfg->comp_type, p_sub->valuestring);

  p_sub = cJSON_GetObjectItem(json,"threadNum");
  if(NULL == p_sub){
    fprintf(stderr,"json format is incorrect.\n");
    cJSON_Delete(json);
    return -1;
  }
  cfg->thread_num = p_sub->valueint;
  cfg->thread_num = (cfg->thread_num > 1) ? cfg->thread_num : 1;


  p_sub = cJSON_GetObjectItem(json,"ObjectLimit");
  if(NULL == p_sub){
    fprintf(stderr,"json format is incorrect.\n");
    cJSON_Delete(json);
    return -1;
  }
  cfg->object_limit = p_sub->valueint;
  cfg->object_limit = (cfg->object_limit > 1) ? cfg->object_limit : 1;

  cJSON_Delete(json);
  return 0;
}

/**
 * @brief generate hand json config
 *
 * @param[in] model_path - model path
 * @param[in] comp_type - CPU/GPU(1)/DSP(2)
 * @param[in] thread_num - thread number
 * @param[in] object_limit - object limit
 * @param[out] json_str - json string to generate
 * @return 0: success <0:failed
*/
 int GCreateHandJson(const char* model_path, const char* comp_type, int thread_num, int object_limit, char** json_str){
  char* p = NULL;
  cJSON* p_root = NULL;
  
  p_root = cJSON_CreateObject();

  if(NULL==p_root){
    cJSON_Delete(p_root);
    fprintf(stderr,"create json object failed.\n");
    return -1;
  }
  
  cJSON_AddStringToObject(p_root,"modelPath",model_path);
  cJSON_AddStringToObject(p_root,"compType",comp_type);
  cJSON_AddNumberToObject(p_root,"threadNum",thread_num);
  cJSON_AddNumberToObject(p_root,"ObjectLimit",object_limit);

  p = cJSON_Print(p_root);
  if(NULL == p){
    cJSON_Delete(p_root);
    fprintf(stderr,"convert json to string failed.\n");
    return -1;
  }
  *json_str = p;
  cJSON_Delete(p_root);
  return 0;
}






