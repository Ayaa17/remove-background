//
//  TensorFlowModel.m
//  remove_background_mac
//
//  Created by normal on 2024/8/27.
//
#import <Foundation/Foundation.h>
#import "TensorFlowLiteC/TensorFlowLiteC.h"


@interface TensorFlowModel : NSObject


- (instancetype)initWithModelPath:(NSString *)modelPath;
- (int)runInference:(float *)data length:(int)data_length;


@end


@implementation TensorFlowModel {
    TfLiteInterpreter* _interpreter;
}


void NoOpDeallocator(void *data, size_t a, void *b) {}


void FreeBuffer(void* data, size_t length) {
    free(data);
}


- (instancetype)initWithModelPath:(NSString *)modelPath {
    self = [super init];
    
    const char *TAGS = "serve";
    
    if (self) {
        TfLiteModel *model = TfLiteModelCreateFromFile([modelPath UTF8String]);
        TfLiteInterpreterOptions *interpreterOptions = TfLiteInterpreterOptionsCreate();

        TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, interpreterOptions);
        TfLiteInterpreterAllocateTensors(interpreter);

        TfLiteInterpreterOptionsDelete(interpreterOptions);
        TfLiteModelDelete(model);

        _interpreter = interpreter;
        
    }
    return self;
}


- (float *)runInference:(float *)data length:(int)data_length {
    
    const int OUTPUT_SIZE = 320;
    
    float *result = (float *)malloc(OUTPUT_SIZE*OUTPUT_SIZE*sizeof(float));
    memset(result, 0, OUTPUT_SIZE*OUTPUT_SIZE*sizeof(float));
    
    TfLiteTensor *inputTensor = TfLiteInterpreterGetInputTensor(_interpreter, 0);
    const TfLiteTensor *outputTensor = TfLiteInterpreterGetOutputTensor(_interpreter, 0);
    
    if(!inputTensor || !outputTensor) {
        NSLog(@"Failed to get tensor");
        return NULL;
    }

    memcpy(TfLiteTensorData(inputTensor), data, data_length);
    
    if (TfLiteInterpreterInvoke(_interpreter) != kTfLiteOk) {
        NSLog(@"Failed to invoke interpreter");
        return NULL;
    }
    
    float *output = (float *)TfLiteTensorData(outputTensor);
    
    int output_length = TfLiteTensorByteSize(outputTensor);
    
    memcpy(result, output, OUTPUT_SIZE*OUTPUT_SIZE*sizeof(float));
    NSLog(@"prediction success");
    
    return result;
}


- (void)dealloc {
    // todo:
}

@end
