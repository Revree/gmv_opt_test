//
//  GTCaptureOutputUtils.h
//  gmv_opt
//
//  Created by Andrea Chen on 12/14/17.
//  Copyright © 2017 Andrea Chen. All rights reserved.
//

#ifndef GTCaptureOutputUtils_h
#define GTCaptureOutputUtils_h


#endif /* GTCaptureOutputUtils_h */
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <AVFoundation/AVFoundation.h>
#import <opencv2/opencv.hpp>

@interface GTCaptureOutputUtils : NSObject

// Convert cv::mat image data to UIImage
// code from Patrick O'Keefe (http://www.patokeefe.com/archives/721)
+ (UIImage *)imageFromCvMat:(const cv::Mat *)mat;

// get a cvMat image from an UIImage
+ (cv::Mat *)cvMatFromImage:(const UIImage *)img gray:(BOOL)gray;

// create a CGImage from a cv::Mat
// you will need to destroy the returned object later!
+ (CGImageRef)CGImageFromCvMat:(const cv::Mat &)mat;

/**
 * Convert a sample buffer <buf> from the camera (YUV 4:2:0 [NV12] pixel format) to an
 * OpenCV <mat> that will contain only the luminance (grayscale) data
 * See http://www.fourcc.org/yuv.php#NV12 and https://wiki.videolan.org/YUV/#NV12.2FNV21
 * for details about the pixel format
 */
+ (void)convertYUVSampleBuffer:(CMSampleBufferRef)buf toGrayscaleMat:(cv::Mat &)mat;

// Create a UIImage from sample buffer data
+ (UIImage *)imageFromSampleBuffer:(CMSampleBufferRef)sampleBuffer;

@end

