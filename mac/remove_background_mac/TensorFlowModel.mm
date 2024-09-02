//
//  TensorFlowModel.m
//  remove_background_mac
//
//  Created by normal on 2024/8/27.
//
#import <Foundation/Foundation.h>
#import <tensorflow/c/c_api.h>


@interface TensorFlowModel : NSObject


- (instancetype)initWithModelPath:(NSString *)modelPath;
- (int)runInference:(float *)data length:(int)data_length;


@end


@implementation TensorFlowModel {
    TF_Graph* _graph;
    TF_Session* _session;
    TF_Status* _status;
}


void NoOpDeallocator(void *data, size_t a, void *b) {}


void FreeBuffer(void* data, size_t length) {
    free(data);
}


- (instancetype)initWithModelPath:(NSString *)modelPath {
    self = [super init];
    
    const char *TAGS = "serve";
    
    if (self) {
        const char* model_path = [modelPath UTF8String];
        _status = TF_NewStatus();
        _graph = TF_NewGraph();
        TF_SessionOptions* options = TF_NewSessionOptions();
        TF_Buffer *runOpts = nullptr;
        
        if (TF_GetCode(_status) != TF_OK) {
            NSLog(@"Failed to init: %s", TF_Message(_status));
            return nil;
        }
        
        _session = TF_LoadSessionFromSavedModel(options, runOpts, model_path, &TAGS, 1, _graph, nullptr, _status);
        
        if (TF_GetCode(_status) != TF_OK) {
            NSLog(@"Failed to TF_NewSession: %s", TF_Message(_status));
            return nil;
        }
        
    }
    return self;
}


- (float *)runInference:(float *)data length:(int)data_length {
    
    const int NumInputs = 1;
    const int NumOutputs = 1;
    const char *INPUT_OPER_NAME = "serving_default_input.1";
    const char *OUTPUT_OPER_NAME = "StatefulPartitionedCall";
    const int INPUT_SIZE = 320;
    const int PIXEL_SIZE = 3;
    const int OUTPUT_SIZE = 320;
    
    float *result = (float *)malloc(OUTPUT_SIZE*OUTPUT_SIZE*sizeof(float));
    memset(result, 0, OUTPUT_SIZE*OUTPUT_SIZE*sizeof(float));
    
    TF_Output t_input = {TF_GraphOperationByName(_graph, INPUT_OPER_NAME), 0};
    TF_Output t_output = {TF_GraphOperationByName(_graph, OUTPUT_OPER_NAME), 0};
    
    if (t_input.oper == nullptr || t_output.oper == nullptr) {
        NSLog(@"Failed to get op");
        return result;
    }
    
    TF_Output Input[NumInputs] = {t_input};
    TF_Output Output[NumOutputs] = {t_output};
    TF_Tensor *InputValues[NumInputs];
    TF_Tensor *OutputValues[NumOutputs];
    
    int64_t dims[] = {1, PIXEL_SIZE, INPUT_SIZE, INPUT_SIZE};
    
    size_t ndata = sizeof(float) * data_length;
    TF_Tensor *in_tensor = TF_NewTensor(TF_FLOAT, dims, 4, data, ndata, &NoOpDeallocator, nullptr);
    
    InputValues[0] = in_tensor;
    TF_SessionRun(_session, nullptr, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, nullptr, 0, nullptr, _status);
    if (TF_GetCode(_status) == TF_OK) {
        void *buff = TF_TensorData(OutputValues[0]);
        memcpy(result, buff, OUTPUT_SIZE*OUTPUT_SIZE*sizeof(float));
    } else {
        NSLog(@"Failed to TF_SessionRun: %s", TF_Message(_status));
    }
    
    TF_DeleteTensor(in_tensor);
    TF_DeleteTensor(OutputValues[0]);
    
    return result;
}


- (void)dealloc {
    TF_DeleteSession(_session, _status);
    TF_DeleteGraph(_graph);
    TF_DeleteStatus(_status);
}


TF_Buffer* ReadFile(const char* file) {
    FILE *f = fopen(file, "rb");
    if (f == NULL) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    void* data = malloc(fsize);
    fread(data, fsize, 1, f);
    fclose(f);
    
    TF_Buffer* buffer = TF_NewBuffer();
    buffer->data = data;
    buffer->length = fsize;
    buffer->data_deallocator = FreeBuffer;
    return buffer;
}


@end
