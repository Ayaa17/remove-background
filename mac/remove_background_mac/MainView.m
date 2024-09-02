#import "MainView.h"
#import "tensorflow/c/c_api.h"
#import "TensorFlowModel.h"
#import <Cocoa/Cocoa.h>

@implementation MainView

// 初始化方法，适用于通过代码创建视图
- (instancetype)initWithFrame:(NSRect)frameRect {
    self = [super initWithFrame:frameRect];
    if (self) {
        // 你可以在这里做一些初始化工作
    }
    return self;
}

// 当视图从XIB加载时调用这个方法
- (void)awakeFromNib {
    NSLog(@"BawakeFromNib");
    [super awakeFromNib];
    // 你可以在这里做一些初始化工作，例如设置默认的图片
    self.imageView.image = [NSImage imageNamed:NSImageNameUser];
}

// 处理第一个按钮点击事件
- (IBAction)button1Clicked:(id)sender {
    NSLog(@"Button 1 Clicked");
    // 在这里处理按钮1的点击事件，例如更换图片
    
    // 创建NSOpenPanel对象
    NSOpenPanel *openPanel = [NSOpenPanel openPanel];
    [openPanel setCanChooseFiles:YES];
    [openPanel setCanChooseDirectories:NO];
    [openPanel setAllowedFileTypes:@[@"png", @"jpg", @"jpeg", @"gif", @"tiff"]]; // 允许的文件类型
    
    // 显示面板并处理用户选择
    [openPanel beginWithCompletionHandler:^(NSModalResponse result) {
        if (result == NSModalResponseOK) {
            NSURL *selectedFileURL = [openPanel URL]; // 获取用户选择的文件URL
            
            // 创建NSImage对象并设置到imageView
            NSImage *image = [[NSImage alloc] initWithContentsOfURL:selectedFileURL];
            self.imageView.image = image;
        }
    }];
}

// 处理第二个按钮点击事件
- (IBAction)button2Clicked:(id)sender {
    NSLog(@"Button 2 Clicked");
    // 在这里处理按钮2的点击事件，例如更换图片
    [self runTensorFlowModel];
}

- (void)runTensorFlowModel {
    NSLog(@"load model...");
    
    NSString *modelDirectory = [[NSBundle mainBundle] pathForResource:@"tf_u2netp" ofType:nil];
    NSLog(@"modelPath: %@",modelDirectory);
    
    TensorFlowModel *tensorFlowModel = [[TensorFlowModel alloc] initWithModelPath:modelDirectory];
    
    NSLog(@"load model success");
    
    if (tensorFlowModel == nil) {
        NSLog(@"load model fail");
        return;
    }
    
    int dataLength = 320*320*3;
    
    float *imageBuffer = [self processImage:self.imageView.image];
    
    float *resultIndex = [tensorFlowModel runInference:imageBuffer length:dataLength];
    
    NSImageRep *rep = [[self.imageView.image representations] objectAtIndex:0];
    int ori_Width = (int)[rep pixelsWide];
    int ori_Height = (int)[rep pixelsHigh];
    
    NSLog(@"image: w= %d, h=%d",ori_Width,ori_Height);
    
    
    float *resizedMask = [self resizeGrayscaleImageWithInput:resultIndex
                                                originalWidth:320
                                               originalHeight:320
                                                     newWidth:ori_Width
                                                    newHeight:ori_Height];
    
    
    free(imageBuffer);
    free(resultIndex);
    
    NSImage *resultImage = [self modifyAlphaAtPoint:resizedMask oriImage:self.imageView.image];
    
    [self.imageView setImage:resultImage];
    free(resizedMask);
    
    NSLog(@"predict success");
    
}

- (NSImage *)resizeImage:(NSImage *)image toWidth:(int)width height:(int)height {
    NSImage *resizedImage = [[NSImage alloc] initWithSize:NSMakeSize(width, height)];
    
    [resizedImage lockFocus];
    [image setSize:NSMakeSize(width, height)];
    [image drawInRect:NSMakeRect(0, 0, width, height)
             fromRect:NSZeroRect
            operation:NSCompositingOperationCopy
             fraction:1.0];
    [resizedImage unlockFocus];
    
    return resizedImage;
}

- (NSImage *)imageResize:(NSImage *)anImage width:(NSInteger)width height:(NSInteger)height {
    NSImage *sourceImage = anImage;
    [sourceImage setScalesWhenResized:YES];
    
    // Report an error if the source isn't a valid image
    if (![sourceImage isValid]){
        NSLog(@"Invalid Image");
        return nil;
    }
    
    NSSize newSize = NSMakeSize(width, height);
    NSImage *resizedImage = [[NSImage alloc] initWithSize:newSize];
    [resizedImage lockFocus];
    
    [sourceImage setSize:newSize];
    [[NSGraphicsContext currentContext] setImageInterpolation:NSImageInterpolationHigh];
    [sourceImage drawInRect:NSMakeRect(0, 0, width, height) fromRect:NSZeroRect operation:NSCompositingOperationCopy fraction:1.0];
    
    [resizedImage unlockFocus];
    return resizedImage;
}

- (float *)resizeImage2:(NSImage *)image newWidth:(NSInteger)newWidth newHeight:(NSInteger)newHeight {
    NSBitmapImageRep *bitmapRep = [[NSBitmapImageRep alloc] initWithData:[image TIFFRepresentation]];
    NSInteger width = bitmapRep.pixelsWide;
    NSInteger height = bitmapRep.pixelsHigh;
    
    if (width == 0 || height == 0) {
        return NULL; // 防止除以零
    }
    
    float xScale = (float)width / newWidth;
    float yScale = (float)height / newHeight;
    
    float *resizedData = (float *)malloc(newWidth * newHeight * 3 * sizeof(float));
    if (!resizedData) {
        return NULL; // 如果内存分配失败
    }
    
    unsigned char *data = [bitmapRep bitmapData];
    
    for (NSInteger y = 0; y < newHeight; y++) {
        for (NSInteger x = 0; x < newWidth; x++) {
            NSInteger srcX = (NSInteger)(x * xScale);
            NSInteger srcY = (NSInteger)(y * yScale);
            NSInteger srcIndex = (srcY * width + srcX) * 4; // 原图的索引
            
            // 提取 RGB 通道值，并进行归一化
            float redValue = data[srcIndex] / 255.0f;
            float greenValue = data[srcIndex + 1] / 255.0f;
            float blueValue = data[srcIndex + 2] / 255.0f;
            
            // 存储 RGB 数据，按 [R, G, B] 顺序
            NSInteger dstIndex = (y * newWidth + x) * 3;
            resizedData[dstIndex] = redValue;
            resizedData[dstIndex + 1] = greenValue;
            resizedData[dstIndex + 2] = blueValue;
        }
    }
    
    return resizedData;
}

- (float *)normalizeImage:(NSImage *)image width:(int *)width height:(int *)height {
    NSBitmapImageRep *bitmapRep = [[NSBitmapImageRep alloc] initWithData:[image TIFFRepresentation]];
    if (!bitmapRep) {
        NSLog(@"Failed to get bitmap representation from image.");
        return NULL;
    }
    
    NSInteger imageWidth = [bitmapRep pixelsWide];
    NSInteger imageHeight = [bitmapRep pixelsHigh];
    NSInteger bytesPerRow = [bitmapRep bytesPerRow];
    NSInteger bitsPerPixel = [bitmapRep bitsPerPixel];
    NSInteger bytesPerPixel = bitsPerPixel / 8;
    
    
    NSLog(@"normalizeImage: %d,%d,%d,%d",imageWidth,imageHeight,bytesPerRow,bitsPerPixel);
    
    // 取得圖像的像素數據
    unsigned char *pixels = [bitmapRep bitmapData];
    
    // 計算正規化後的 float 陣列大小
    NSInteger pixelCount = imageWidth * imageHeight * bytesPerPixel;
    float *normalizedData = (float *)malloc(pixelCount * sizeof(float));
    
    for (NSInteger y = 0; y < imageHeight; y++) {
        for (NSInteger x = 0; x < imageWidth; x++) {
            NSInteger index = y * bytesPerRow + x * bytesPerPixel;
            float r = pixels[index] / 255.0f;     // Red
            float g = pixels[index + 1] / 255.0f; // Green
            float b = pixels[index + 2] / 255.0f; // Blue
            // 如果有 alpha 通道，可以選擇處理它
            // float a = pixels[index + 3] / 255.0f; // Alpha
            
            // 計算在 float 陣列中的位置
            NSInteger floatIndex = (y * imageWidth + x) * 3;
            normalizedData[floatIndex] = r;
            normalizedData[floatIndex + 1] = g;
            normalizedData[floatIndex + 2] = b;
        }
    }
    
    // 設置寬度和高度參數
    if (width) *width = (int)imageWidth;
    if (height) *height = (int)imageHeight;
    
    return normalizedData;
}

- (float *)convertHWCtoCHW:(float *)hwcData height:(int)height width:(int)width channels:(int)channels {
    // 計算新的陣列大小
    int chwSize = channels * height * width;
    float *chwData = (float *)malloc(chwSize * sizeof(float));
    
    // 將數據從 [height, width, channels] 轉換為 [channels, height, width]
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int hwcIndex = (h * width + w) * channels + c;
                int chwIndex = (c * height + h) * width + w;
                chwData[chwIndex] = hwcData[hwcIndex];
            }
        }
    }
    
    return chwData;
}

- (float *)processImage:(NSImage *)image {
    
    int width = 320;
    int height = 320;
    
    // Resize the image to 320x320
    //    NSImage *resizedImage = [self resizeImage:image toWidth:320 height:320];
    //    NSImage *resizedImage = [self imageResize:image width:320 height:640];
    float *resizedImage = [self resizeImage2:image newWidth:320 newHeight:320];
    
    // Normalize the image
    //    float *normalizedData = [self normalizeImage:resizedImage width:&width height:&height];
    float *resultData = [self convertHWCtoCHW:resizedImage height:width width:height channels:3];
    //    free(normalizedData);
    return resultData;
}

- (float *)resizeGrayscaleImageWithInput:(float *)input
                           originalWidth:(int)originalWidth
                          originalHeight:(int)originalHeight
                                newWidth:(int)newWidth
                               newHeight:(int)newHeight {
    // 分配输出数组
    float *output = (float *)malloc(newWidth * newHeight * sizeof(float));
    if (output == NULL) {
        NSLog(@"Failed to allocate memory for output array.");
        return NULL;
    }
    
    double xScale = (double)originalWidth / (double)newWidth;
    double yScale = (double)originalHeight / (double)newHeight;
    
    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            int srcX = (int)(x * xScale);
            int srcY = (int)(y * yScale);
            int srcIndex = srcY * originalWidth + srcX;
            output[y * newWidth + x] = input[srcIndex];
        }
    }
    
    return output;
}

// Function to modify alpha channel of NSImage
- (NSImage *)modifyAlphaAtPoint:(float *)resizeMask oriImage:(NSImage *)oriImage {
    
    NSBitmapImageRep *bitmapRep = [[NSBitmapImageRep alloc] initWithData:[oriImage TIFFRepresentation]];
    NSInteger width = bitmapRep.pixelsWide;
    NSInteger height = bitmapRep.pixelsHigh;
    NSInteger bytesPerRow = bitmapRep.bytesPerRow;
    NSInteger bitsPerPixel = bitmapRep.bitsPerPixel;
    
    unsigned char *data = [bitmapRep bitmapData];
    // rawData 現在包含了圖片的像素數據
    NSLog(@"%d,%d,%d,%d",width,height,bytesPerRow,bitsPerPixel);
    
    if (resizeMask == NULL) {
        return nil;
    }
    
    unsigned char *data_2 = (unsigned char *)malloc(height*bytesPerRow);
    memcpy(data_2, data, height*bytesPerRow);
    
    // Modify the alpha channel
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t index = (y * width + x);
            float alphaValue = resizeMask[index]; // Get output data as alpha value
            
            data_2[index * 4 + 3] = (int)(alphaValue * 255);
        }
    }
    
    NSBitmapImageRep *bitmapRep_2 = [[NSBitmapImageRep alloc] initWithBitmapDataPlanes:&data_2
                                                                            pixelsWide:width
                                                                            pixelsHigh:height
                                                                         bitsPerSample:8
                                                                       samplesPerPixel:4
                                                                              hasAlpha:YES
                                                                              isPlanar:NO
                                                                        colorSpaceName:NSCalibratedRGBColorSpace
                                                                           bytesPerRow:bytesPerRow
                                                                          bitsPerPixel:bitsPerPixel];
    
    NSImage *newImage = [[NSImage alloc] initWithSize:NSMakeSize(width, height)];
    [newImage addRepresentation:bitmapRep_2];
    
    
    return newImage;
}


@end
