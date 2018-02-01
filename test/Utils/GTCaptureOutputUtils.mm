//
//  GTCaptureOutputUtils.m
//  gmv_opt
//
//  Created by Andrea Chen on 12/14/17.
//  Copyright © 2017 Andrea Chen. All rights reserved.
//

#import "GTCaptureOutputUtils.h"


@implementation GTCaptureOutputUtils

+ (UIImage *)imageFromCvMat:(const cv::Mat *)mat {
    // code from Patrick O'Keefe (http://www.patokeefe.com/archives/721)
    NSData *data = [NSData dataWithBytes:mat->data length:mat->elemSize() * mat->total()];
    
    CGColorSpaceRef colorSpace;
    
    if (mat->elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)data);
    NSLog(@"ROW %lu",mat->step.p[0]);
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(mat->cols,                                 //width
                                        mat->rows,                                 //height
                                        8,                                          //bits per component
                                        8 * mat->elemSize(),                       //bits per pixel
                                        mat->step.p[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

+ (cv::Mat *)cvMatFromImage:(const UIImage *)img gray:(BOOL)gray {
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(img.CGImage);
    
    const int w = [img size].width;
    const int h = [img size].height;
    
    // create cv::Mat
    cv::Mat *mat = new cv::Mat(h, w, CV_8UC4);
    
    // create context
    CGContextRef contextRef = CGBitmapContextCreate(mat->ptr(),
                                                    w, h,
                                                    8,
                                                    mat->step[0],
                                                    colorSpace,
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault);
    
    if (!contextRef) {
        delete mat;
        
        return NULL;
    }
    
    // draw the image in the context
    CGContextDrawImage(contextRef, CGRectMake(0, 0, w, h), img.CGImage);
    
    CGContextRelease(contextRef);
    //    CGColorSpaceRelease(colorSpace);  // "colorSpace" is not owned, only referenced
    
    // convert to grayscale data if necessary
    if (gray) {
        cv::Mat *grayMat = new cv::Mat(h, w, CV_8UC1);
        cv::cvtColor(*mat, *grayMat, CV_RGBA2GRAY);
        delete mat;
        
        return grayMat;
    }
    
    return mat;
}

+ (CGImageRef)CGImageFromCvMat:(const cv::Mat &)mat {
    NSData *data = [NSData dataWithBytes:mat.data length:mat.elemSize() * mat.total()];
    
    CGColorSpaceRef colorSpace;
    
    if (mat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(mat.cols,                                   //width
                                        mat.rows,                                   //height
                                        8,                                          //bits per component
                                        8 * mat.elemSize(),                         //bits per pixel
                                        mat.step.p[0],                              //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,//bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return imageRef;
}

+ (void)convertYUVSampleBuffer:(CMSampleBufferRef)buf toGrayscaleMat:(cv::Mat &)mat {
    CVImageBufferRef imgBuf = CMSampleBufferGetImageBuffer(buf);
    
    // lock the buffer
    CVPixelBufferLockBaseAddress(imgBuf, 0);
    
    // get the address to the image data
    //    void *imgBufAddr = CVPixelBufferGetBaseAddress(imgBuf);
    // this is wrong! see http://stackoverflow.com/a/4109153
    //void *imgBufAddr = CVPixelBufferGetBaseAddressOfPlane(imgBuf, 0);
    void *imgBufAddr = CVPixelBufferGetBaseAddress(imgBuf);
    
    // get image properties
    int w = (int)CVPixelBufferGetWidth(imgBuf);
    int h = (int)CVPixelBufferGetHeight(imgBuf);
    size_t bytesPerRow;
    bytesPerRow = CVPixelBufferGetBytesPerRow(imgBuf);
    
    //for (int i = 0; i < (w * h); i++) {
    // Calculate the combined grayscale weight of the RGB channels
    //     int weight = (imgBufAddr[0] * 0.11) + (imgBufAddr[1] * 0.59) + (imgBufAddr[2] * 0.3);
    //}
    
    // create the cv mat
    //cv::Mat mat_c= cv::Mat(h, w, CV_8UC4,imgBuf);
    mat.create(h, w, CV_8UC4);              // 8 bit unsigned chars for grayscale data
    //cv::Mat gray = cv::Mat(h,w,CV_8UC1);
    memcpy(mat.data, imgBufAddr, w * h);    // the first plane contains the grayscale data
    //cv::cvtColor(mat_c,mat, CV_BGRA2GRAY);
    //memcpy(mat.data, &gray, w * h);
    //mat.data = gray;
    // therefore we use <imgBufAddr> as source
    
    // unlock again
    CVPixelBufferUnlockBaseAddress(imgBuf, 0);
}

+ (cv::Mat) matFromImageBuffer: (CMSampleBufferRef) buffer {
    CVImageBufferRef imgBuf = CMSampleBufferGetImageBuffer(buffer);
    
    cv::Mat mat ;
    
    CVPixelBufferLockBaseAddress(imgBuf, 0);
    
    void *address = CVPixelBufferGetBaseAddress(imgBuf);
    int width = (int) CVPixelBufferGetWidth(imgBuf);
    int height = (int) CVPixelBufferGetHeight(imgBuf);
    
    mat   = cv::Mat(height, width, CV_8UC4, address, 0);
    //cv::cvtColor(mat, _mat, CV_BGRA2BGR);
    
    CVPixelBufferUnlockBaseAddress(imgBuf, 0);
    
    return mat;
}

// Create a UIImage from sample buffer data
+ (UIImage *)imageFromSampleBuffer:(CMSampleBufferRef)sampleBuffer
{
    // Get a CMSampleBuffer's Core Video image buffer for the media data
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    // Lock the base address of the pixel buffer
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    
    // Get the number of bytes per row for the pixel buffer
    void *baseAddress = CVPixelBufferGetBaseAddress(imageBuffer);
    
    // Get the number of bytes per row for the pixel buffer
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    // Get the pixel buffer width and height
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    
    // Create a device-dependent RGB color space
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    // Create a bitmap graphics context with the sample buffer data
    CGContextRef context = CGBitmapContextCreate(baseAddress, width, height, 8,
                                                 bytesPerRow, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
    // Create a Quartz image from the pixel data in the bitmap graphics context
    CGImageRef quartzImage = CGBitmapContextCreateImage(context);
    // Unlock the pixel buffer
    CVPixelBufferUnlockBaseAddress(imageBuffer,0);
    
    // Free up the context and color space
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    
    // Create an image object from the Quartz image
    UIImage *image = [UIImage imageWithCGImage:quartzImage];
    
    // Release the Quartz image
    CGImageRelease(quartzImage);
    
    return (image);
}

@end
