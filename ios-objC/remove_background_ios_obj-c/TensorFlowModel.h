//
//  TensorFlowModel.h
//  remove_background_mac
//
//  Created by normal on 2024/8/27.
//

#import <Foundation/Foundation.h>
#import "TensorFlowLiteC/TensorFlowLiteC.h"

NS_ASSUME_NONNULL_BEGIN

@interface TensorFlowModel : NSObject


- (instancetype)initWithModelPath:(NSString *)modelPath;
- (float *)runInference:(float *)data length:(int)data_length;


@end

NS_ASSUME_NONNULL_END
