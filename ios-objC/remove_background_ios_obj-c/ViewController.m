//
//  ViewController.m
//  remove_background_ios_obj-c
//
//  Created by normal on 2024/9/2.
//

#import "ViewController.h"
#import "TensorFlowModel.h"
#import "CoreImage/CoreImage.h"

@interface ViewController ()

@property (nonatomic, strong) UIImageView *imageView;
@property (nonatomic, strong) UIButton *button1;
@property (nonatomic, strong) UIButton *button2;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.view.backgroundColor = [UIColor whiteColor];
    
    UIImage *placeholderImage = [UIImage systemImageNamed:@"photo"];
    self.imageView = [[UIImageView alloc] initWithImage:placeholderImage];
    self.imageView.contentMode = UIViewContentModeScaleAspectFit;
    self.imageView.translatesAutoresizingMaskIntoConstraints = NO; // 關閉自動約束
    [self.view addSubview:self.imageView];
    
    // 創建第一個 UIButton
    self.button1 = [UIButton buttonWithType:UIButtonTypeSystem];
    [self.button1 setTitle:@"Button 1" forState:UIControlStateNormal];
    self.button1.translatesAutoresizingMaskIntoConstraints = NO;
    [self.button1 addTarget:self action:@selector(button1Tapped:) forControlEvents:UIControlEventTouchUpInside];
    [self.view addSubview:self.button1];
    
    // 創建第二個 UIButton
    self.button2 = [UIButton buttonWithType:UIButtonTypeSystem];
    [self.button2 setTitle:@"Button 2" forState:UIControlStateNormal];
    self.button2.translatesAutoresizingMaskIntoConstraints = NO;
    [self.button2 addTarget:self action:@selector(button2Tapped:) forControlEvents:UIControlEventTouchUpInside];
    [self.view addSubview:self.button2];
    
    // 使用 Auto Layout 來置中對齊 UI 元素
    [NSLayoutConstraint activateConstraints:@[
        // UIImageView 填滿整個螢幕，並預留底部空間
        [self.imageView.topAnchor constraintEqualToAnchor:self.view.topAnchor],
        [self.imageView.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor constant:-80], // 預留底部空間
        [self.imageView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.imageView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        
        // Button 1 置中，位於圖片下方
        [self.button1.centerXAnchor constraintEqualToAnchor:self.view.centerXAnchor],
        [self.button1.bottomAnchor constraintEqualToAnchor:self.button2.topAnchor constant:-10],
        [self.button1.widthAnchor constraintEqualToConstant:100],
        [self.button1.heightAnchor constraintEqualToConstant:50],
        
        // Button 2 置中，位於圖片底部
        [self.button2.centerXAnchor constraintEqualToAnchor:self.view.centerXAnchor],
        [self.button2.bottomAnchor constraintEqualToAnchor:self.view.safeAreaLayoutGuide.bottomAnchor constant:-20],
        [self.button2.widthAnchor constraintEqualToConstant:100],
        [self.button2.heightAnchor constraintEqualToConstant:50]
    ]];
    
}

// 第一個 UIButton 的事件處理方法
- (void)button1Tapped:(UIButton *)sender {
    NSLog(@"Button 1 Tapped");
    
    UIImagePickerController *imagePicker = [[UIImagePickerController alloc] init];
    imagePicker.delegate = self;
    imagePicker.sourceType = UIImagePickerControllerSourceTypePhotoLibrary; // 選擇圖庫
    
    [self presentViewController:imagePicker animated:YES completion:nil];
    
}

// 第二個 UIButton 的事件處理方法
- (void)button2Tapped:(UIButton *)sender {
    NSLog(@"Button 2 Tapped");
    [self runTensorFlowModel];
}

// 選擇圖片後的回調方法
- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary<UIImagePickerControllerInfoKey,id> *)info {
    UIImage *selectedImage = info[UIImagePickerControllerOriginalImage];
    self.imageView.image = selectedImage;
    
    [picker dismissViewControllerAnimated:YES completion:nil];
}

// 取消選擇圖片的回調方法
- (void)imagePickerControllerDidCancel:(UIImagePickerController *)picker {
    [picker dismissViewControllerAnimated:YES completion:nil];
}

- (void)runTensorFlowModel {
    NSLog(@"load model...");
    
    NSString *modelDirectory = [[NSBundle mainBundle] pathForResource:@"tf_u2netp_model" ofType:@"tflite"];
    NSLog(@"modelPath: %@",modelDirectory);

    TensorFlowModel *tensorFlowModel = [[TensorFlowModel alloc] initWithModelPath:modelDirectory];

    if(!tensorFlowModel) {
        NSLog(@"model init fail");
    }

    float *imageBuffer = [self processImage:self.imageView.image];
    
    float *resultIndex = [tensorFlowModel runInference:imageBuffer length:1228800];
    
    CGSize originalSize = self.imageView.image.size;
    int ori_Width = (int)originalSize.width;
    int ori_Height = (int)originalSize.height;

    float *resizedMask = [self resizeGrayscaleImageWithInput:resultIndex
                                                originalWidth:320
                                               originalHeight:320
                                                     newWidth:ori_Width
                                                    newHeight:ori_Height];
    

    UIImage *resultImage = [self modifyAlphaAtPoint:resizedMask oriImage:self.imageView.image];

    [self.imageView setImage:resultImage];
    
    NSLog(@"all success");
    
}

- (float *)processImage:(UIImage *)image {
    
    int width = 320;
    int height = 320;
    
    UIImage *resizeImage =  [self resizeUIImage:self.imageView.image width:320 height:320];
    float *input = [self transformDimensionsWithImage:resizeImage n:1 w:320 h:320 c:3];

    return input;
}

- (UIImage *)resizeUIImage:(UIImage *)image width:(NSInteger)width height:(NSInteger)height {
    CGSize targetSize = CGSizeMake(width, height);
    
    UIGraphicsBeginImageContextWithOptions(targetSize, NO, 1.0);
    [image drawInRect:CGRectMake(0, 0, targetSize.width, targetSize.height)];
    
    UIImage *resizedImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    return resizedImage;
}


- (float *)transformDimensionsWithImage:(UIImage *)image n:(NSInteger)n w:(NSInteger)w h:(NSInteger)h c:(NSInteger)c {
    CGImageRef cgImage = image.CGImage;
    if (!cgImage) {
        return nil;
    }
    
    NSInteger width = CGImageGetWidth(cgImage);
    NSInteger height = CGImageGetHeight(cgImage);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    NSInteger bytesPerPixel = 4;
    NSInteger bytesPerRow = bytesPerPixel * width;
    NSInteger bitsPerComponent = 8;
    
    UInt8 *imageData = (UInt8 *)malloc(width * height * bytesPerPixel);
    if (!imageData) {
        CGColorSpaceRelease(colorSpace);
        return nil;
    }
    
    CGContextRef context = CGBitmapContextCreate(imageData, width, height, bitsPerComponent, bytesPerRow, colorSpace, kCGImageAlphaPremultipliedLast);
    if (!context) {
        free(imageData);
        CGColorSpaceRelease(colorSpace);
        return nil;
    }
    
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), cgImage);
    
    float *floatArray = (float *)malloc(width * height * c * sizeof(float));
    if (!floatArray) {
        CGContextRelease(context);
        free(imageData);
        CGColorSpaceRelease(colorSpace);
        return nil;
    }
    
    for (NSInteger i = 0; i < width * height; i++) {
        for (NSInteger j = 0; j < c; j++) {
            floatArray[i * c + j] = (float)imageData[i * bytesPerPixel + j] / 255.0f;
        }
    }
    
    NSInteger newSize = n * c * w * h;
    float *transformed = (float *)malloc(newSize * sizeof(float));
    if (!transformed) {
        free(floatArray);
        CGContextRelease(context);
        free(imageData);
        CGColorSpaceRelease(colorSpace);
        NSLog(@"transformDimensionsWithImage fail");
        return nil;
    }
    
    for (NSInteger i = 0; i < n; i++) {
        for (NSInteger j = 0; j < w; j++) {
            for (NSInteger k = 0; k < h; k++) {
                for (NSInteger l = 0; l < c; l++) {
                    NSInteger oldIndex = i * (w * h * c) + j * (h * c) + k * c + l;
                    NSInteger newIndex = i * (c * w * h) + l * (w * h) + j * h + k;
                    transformed[newIndex] = floatArray[oldIndex];
                }
            }
        }
    }
    
    
    free(floatArray);
    CGContextRelease(context);
    free(imageData);
    CGColorSpaceRelease(colorSpace);
    
    return transformed;
}

- (float *)resizeGrayscaleImageWithInput:(float *)input
                           originalWidth:(int)originalWidth
                          originalHeight:(int)originalHeight
                                newWidth:(int)newWidth
                               newHeight:(int)newHeight {

    float *output = (float *)malloc(newWidth * newHeight * sizeof(float));
    if (!output) {
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

- (UIImage *)modifyAlphaAtPoint:(float *)resizeMask oriImage:(UIImage *)oriImage {
    CGImageRef cgImage = [oriImage CGImage];
    if (!cgImage) { return nil; }
    
    NSInteger width = CGImageGetWidth(cgImage);
    NSInteger height = CGImageGetHeight(cgImage);

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGBitmapInfo bitmapInfo = kCGImageAlphaPremultipliedLast;
    int bytesPerPixel = 4;
    int bytesPerRow = bytesPerPixel * (int)width;
    
    void *bitmapData = malloc(height * bytesPerRow);
    
    CGContextRef context = CGBitmapContextCreate(
        bitmapData,
        width,
        height,
        8,
        bytesPerRow,
        colorSpace,
        bitmapInfo);
    
    if (!context) {
        free(bitmapData);
        CGColorSpaceRelease(colorSpace);
        return nil;
    }
    
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), cgImage);
    
    unsigned char *data = (unsigned char *)CGBitmapContextGetData(context);
    
    for (NSInteger y = 0; y < height; y++) {
        for (NSInteger x = 0; x < width; x++) {
            NSInteger index = (y * width + x);
            float alphaValue = resizeMask[index];
            data[index * 4 + 3] = (unsigned char)(alphaValue * 255);
        }
    }
    
    CGImageRef newCGImage = CGBitmapContextCreateImage(context);
    UIImage *newImage = [UIImage imageWithCGImage:newCGImage];
    
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    free(bitmapData);
    free(resizeMask);
    CGImageRelease(newCGImage);
    
    return newImage;
}

@end
