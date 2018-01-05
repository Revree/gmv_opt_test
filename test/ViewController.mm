//
//  ViewController.m
//  gmv_opt
//
//  Created by Andrea Chen on 12/3/17.
//  Copyright © 2017 Andrea Chen. All rights reserved.
//
#import <GoogleMobileVision/GoogleMobileVision.h>

#import "ViewController.h"
#import <AVFoundation/AVFoundation.h>
#import <opencv2/opencv.hpp>
#import "GTCaptureOutputUtils.h"
#import "GTPreviewView.h"
#import <GoogleMobileVision/GMVUtility.h>
#import <GoogleMobileVision/GMVDetector.h>
#import <GoogleMobileVision/GMVFeature.h>

using namespace cv;


string window_name = "optical flow tracking";
const int MAX_POINTS_COUNT = 10;
int maxCount = 100;
double qLevel = 0.1;
double minDist = 10;
const int32_t MAX_FPS = 30;
const CGSize resolutionSize = CGSizeMake(640,480);

CGFloat xScale = 1;
CGFloat yScale = 1;
int frameid = 0;

@interface ViewController ()<AVCaptureVideoDataOutputSampleBufferDelegate,GTPreviewViewDelegate>
{
    UIImageView *liveView_;
    AVCaptureSession *_session;
    AVCaptureDevice *_device;
    AVCaptureVideoPreviewLayer *previewLayer;
    AVCaptureVideoDataOutput *videoDataOutput;
    UIDeviceOrientation lastKnownDeviceOrientation;
    
    
    IBOutlet GTPreviewView          *_previewView;
    IBOutlet UIView                 *overlayView;
    IBOutlet UIView                 *placeHolder;
    IBOutlet UITapGestureRecognizer *_tapGestureRecognizer;
    IBOutlet UITapGestureRecognizer *_doubleTapGestureRecognizer;
    GMVDetector *faceDetector;
    
    
    Point2f                 _touchPoint;
    Point2f                 _touchPoint1;
    Point2f                 _touchPoint2;
    Point2f                 _touchPoint3;
    bool                    _addRemovePt;
    
    
    
    cv::Size                    _winSize;
    cv::TermCriteria            _termcrit;
    
    Mat gray;
    Mat gray_prev;
    Mat image;
    vector<uchar> status;
    vector<float> err;
    //cv::Mat err;
    vector<Point2f> points[2];
    vector<Point2f> initial;
    vector<Point2f> features;
    
    
}


@end



@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    // Set up default camera settings.
    [self setupCaptureSession];
    
    // Setup camera preview.
    [self setupCameraPreview];
    
    [self configureDevice:_device frameRate:MAX_FPS];
    
    _winSize = cv::Size(31,31);
    _termcrit = cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
    
    
    [_tapGestureRecognizer requireGestureRecognizerToFail:_doubleTapGestureRecognizer];
    
    [self autoFocusAtPoint:self.view.center];
    
    // Initialize the face detector.
    NSDictionary *options = @{
                              GMVDetectorFaceMinSize : @(0.3),
                              GMVDetectorFaceTrackingEnabled : @(YES),
                              GMVDetectorFaceLandmarkType : @(GMVDetectorFaceLandmarkAll)
                              };
    self->faceDetector = [GMVDetector detectorOfType:GMVDetectorTypeFace options:options];
    
}


- (void)viewDidLayoutSubviews {
    [super viewDidLayoutSubviews];
    
    self->previewLayer.frame = self.view.layer.bounds;
    self->previewLayer.position = CGPointMake(CGRectGetMidX(self->previewLayer.frame),
                                              CGRectGetMidY(self->previewLayer.frame));
}
- (void)willAnimateRotationToInterfaceOrientation:(UIInterfaceOrientation)toInterfaceOrientation
                                         duration:(NSTimeInterval)duration {
    // Camera rotation needs to be manually set when rotation changes.
    if (self->previewLayer) {
        if (toInterfaceOrientation == UIInterfaceOrientationPortrait) {
            self->previewLayer.connection.videoOrientation = AVCaptureVideoOrientationPortrait;
        } else if (toInterfaceOrientation == UIInterfaceOrientationPortraitUpsideDown) {
            self->previewLayer.connection.videoOrientation = AVCaptureVideoOrientationPortraitUpsideDown;
        } else if (toInterfaceOrientation == UIInterfaceOrientationLandscapeLeft) {
            self->previewLayer.connection.videoOrientation = AVCaptureVideoOrientationLandscapeLeft;
        } else if (toInterfaceOrientation == UIInterfaceOrientationLandscapeRight) {
            self->previewLayer.connection.videoOrientation = AVCaptureVideoOrientationLandscapeRight;
        }
    }
}

- (void)viewWillAppear:(BOOL)animated
{
    [super viewWillAppear:animated];
    
    [_previewView becomeFirstResponder];
}

- (void)viewWillDisappear:(BOOL)animated
{
    [super viewWillDisappear:animated];
    
    [_previewView resignFirstResponder];
}

- (BOOL)shouldAutorotate
{
    return NO;
}

- (BOOL)prefersStatusBarHidden
{
    return YES;
}

- (void)dealloc
{
    [_session stopRunning];
}

#pragma mark - AVCaptureVideoPreviewLayer Helper method

- (CGRect)scaledRect:(CGRect)rect
              xScale:(CGFloat)xscale
              yScale:(CGFloat)yscale
              offset:(CGPoint)offset {
    CGRect resultRect = CGRectMake(rect.origin.x * xscale,
                                   rect.origin.y * yscale,
                                   rect.size.width * xscale,
                                   rect.size.height * yscale);
    resultRect = CGRectOffset(resultRect, offset.x, offset.y);
    return resultRect;
}

- (CGPoint)scaledPoint:(CGPoint)point
                xScale:(CGFloat)xscale
                yScale:(CGFloat)yscale
                offset:(CGPoint)offset {
    CGPoint resultPoint = CGPointMake(point.x * xscale + offset.x, point.y * yscale + offset.y);
    return resultPoint;
}

/*
 - (void)setLastKnownDeviceOrientation:(UIDeviceOrientation)orientation {
 if (orientation != UIDeviceOrientationUnknown &&
 orientation != UIDeviceOrientationFaceUp &&
 orientation != UIDeviceOrientationFaceDown) {
 self->lastKnownDeviceOrientation = orientation;
 }
 }
 */
- (void)setupCameraPreview {
    self->previewLayer = [[AVCaptureVideoPreviewLayer alloc] initWithSession:self->_session];
    [self->previewLayer setBackgroundColor:[[UIColor whiteColor] CGColor]];
    [self->previewLayer setVideoGravity:AVLayerVideoGravityResizeAspect];
    CALayer *rootLayer = [self->placeHolder layer];
    [rootLayer setMasksToBounds:YES];
    [self->previewLayer setFrame:[rootLayer bounds]];
    [rootLayer addSublayer:self->previewLayer];
}


// Create and configure a capture session and start it running
- (void)setupCaptureSession
{
    //NSError *error = nil;
    // Create the session
    _session = [[AVCaptureSession alloc] init];
    
    // Configure the session to produce lower resolution video frames, if your
    // processing algorithm can cope. We'll specify medium quality for the
    // chosen device.
    _session.sessionPreset = AVCaptureSessionPreset640x480;
    
    // Find a suitable AVCaptureDevice
    _device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    
    
    // Create a device input with the device and add it to the session.
    AVCaptureDeviceInput *input = [self cameraForPosition:AVCaptureDevicePositionFront];
    //AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:_device error:&error];
    
    if (!input) {
        // Handling the error appropriately.
    }
    [_session addInput:input];
    
    videoDataOutput = [[AVCaptureVideoDataOutput alloc] init];
    //AVCaptureVideoDataOutput *videoDataOutput = [AVCaptureVideoDataOutput new];
    NSDictionary *newSettings =
    @{ (NSString *)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange) };
    videoDataOutput.videoSettings = newSettings;
    
    // create a serial dispatch queue used for the sample buffer delegate as well as when a still image is captured
    // a serial dispatch queue must be used to guarantee that video frames will be delivered in order
    dispatch_queue_t videoDataOutputQueue = dispatch_queue_create("black.grandson.videodataoutputqueue", DISPATCH_QUEUE_SERIAL);
    [videoDataOutput setAlwaysDiscardsLateVideoFrames:YES];
    [videoDataOutput setSampleBufferDelegate:self queue:videoDataOutputQueue];
    
    if ([_session canAddOutput:videoDataOutput])
    {
        [_session addOutput:videoDataOutput];
    }
    
    [self updateCaptureOrientation];
    
    // Start the session running to start the flow of data
    [_session startRunning];
}

- (AVCaptureDeviceInput *)cameraForPosition:(AVCaptureDevicePosition)desiredPosition {
    for (AVCaptureDevice *device in [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo]) {
        if ([device position] == desiredPosition) {
            NSError *error = nil;
            AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:device
                                                                                error:&error];
            if ([self->_session canAddInput:input]) {
                return input;
            }
        }
    }
    return nil;
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection
{
    [GTCaptureOutputUtils convertYUVSampleBuffer:sampleBuffer toGrayscaleMat:gray];
    //connection.videoOrientation = (AVCaptureVideoOrientation)UIInterfaceOrientationPortrait;
    
    gray.copyTo(image);
    
    [self tracking];
    
    UIImage *imageToDisplay = [GTCaptureOutputUtils imageFromCvMat:&image];
    UIImage *image = [GMVUtility sampleBufferTo32RGBA:sampleBuffer];
    AVCaptureDevicePosition devicePosition =  AVCaptureDevicePositionFront;
    
    
    //Define orientation
    UIDeviceOrientation deviceOrientation = [[UIDevice currentDevice] orientation];
    //NSLog(@"%c %c",(char)deviceOrientation,(char)lastKnownDeviceOrientation);
    GMVImageOrientation orientation = [GMVUtility
                                       imageOrientationFromOrientation:deviceOrientation
                                       withCaptureDevicePosition:devicePosition
                                       defaultDeviceOrientation:lastKnownDeviceOrientation];
    NSDictionary *options = @{
                              GMVDetectorImageOrientation : @(orientation)
                              };
    
    
    
    // Detect features using GMVDetector.
    NSArray<GMVFaceFeature *> *faces = [self->faceDetector featuresInImage:image options:options];
    NSLog(@"Detected %lu face(s).", (unsigned long)[faces count]);
    
    // Assume AVLayerVideoGravityResizeAspect
    // The video frames captured by the camera are a different size than the video preview.
    // Calculates the scale factors and offset to properly display the features.
    CMFormatDescriptionRef fdesc = CMSampleBufferGetFormatDescription(sampleBuffer);
    CGRect clap = CMVideoFormatDescriptionGetCleanAperture(fdesc, false);
    CGSize parentFrameSize = self->previewLayer.frame.size;
    
    CGFloat cameraRatio = clap.size.height / clap.size.width;
    CGFloat viewRatio = parentFrameSize.width / parentFrameSize.height;
    CGFloat xScale = 1;
    CGFloat yScale = 1;
    CGRect videoBox = CGRectZero;
    
    if (viewRatio > cameraRatio) {
        videoBox.size.width = parentFrameSize.height * clap.size.width / clap.size.height;
        videoBox.size.height = parentFrameSize.height;
        videoBox.origin.x = (parentFrameSize.width - videoBox.size.width) / 2;
        videoBox.origin.y = (videoBox.size.height - parentFrameSize.height) / 2;
        
        xScale = videoBox.size.width / clap.size.width;
        yScale = videoBox.size.height / clap.size.height;
    } else {
        videoBox.size.width = parentFrameSize.width;
        videoBox.size.height = clap.size.width * (parentFrameSize.width / clap.size.height);
        videoBox.origin.x = (videoBox.size.width - parentFrameSize.width) / 2;
        videoBox.origin.y = (parentFrameSize.height - videoBox.size.height) / 2;
        
        xScale = videoBox.size.width / clap.size.height;
        yScale = videoBox.size.height / clap.size.width;
    }
    
    NSLog(@"box %f %f %f %f",videoBox.size.width,videoBox.size.height,videoBox.origin.x,videoBox.origin.y);
    
    dispatch_sync(dispatch_get_main_queue(), ^{
        _previewView.image = imageToDisplay;
        frameid++;
        
        // Remove previously added feature views.
        //CGFloat scale = 0;
        //scale = resolutionSize.height / _previewView.bounds.size.width;
        
        for(GMVFaceFeature *face in faces){
            CGPoint point_nose,point_botMouth,point_lefteye,point_righteye;
            if (face.hasNoseBasePosition) {
                point_nose = [self scaledPoint:face.noseBasePosition
                                        xScale:xScale
                                        yScale:yScale
                                        offset:videoBox.origin];
                
            }
            if (face.hasBottomMouthPosition) {
                point_botMouth = [self scaledPoint:face.bottomMouthPosition
                                            xScale:xScale
                                            yScale:yScale
                                            offset:videoBox.origin];
                _touchPoint = cv::Point2f(point_botMouth.x,point_botMouth.y);
                _addRemovePt = true;
            }
            if(face.hasLeftEyePosition){
                point_lefteye = [self scaledPoint:face.bottomMouthPosition
                                           xScale:xScale
                                           yScale:yScale
                                           offset:videoBox.origin];
            }
            if(face.hasRightEarPosition){
                point_righteye = [self scaledPoint:face.bottomMouthPosition
                                            xScale:xScale
                                            yScale:yScale
                                            offset:videoBox.origin];
            }
            /* if(frameid%10==0){
             NSLog(@"time %d.",frameid);
             _touchPoint = cv::Point2f(point_nose.x * scale,point_nose.y * scale);
             _touchPoint1 = cv::Point2f(point_botMouth.x * scale,point_botMouth.y * scale);
             _touchPoint2 = cv::Point2f(point_lefteye.x * scale,point_lefteye.y * scale);
             _touchPoint3 = cv::Point2f(point_righteye.x * scale,point_righteye.y * scale);
             NSLog(@"%f,%f,%f,%f",_touchPoint.x,_touchPoint1.x,_touchPoint2.x,_touchPoint3.x);
             _addRemovePt = true;
             }
             */
            
        }
    });
    
    
    //NSLog(@" %lu %lu", (unsigned long)_previewView.bounds.size.width,(unsigned long)_previewView.bounds.size.height);
    
}

- (void)tracking
{
    if (!points[0].empty())
    {
        cv::vector<uchar> status;
        cv::vector<float> err;
        
        if(gray_prev.empty())
        {
            gray.copyTo(gray_prev);
        }
        
        calcOpticalFlowPyrLK(gray_prev, gray, points[0],points[1], status, err, _winSize, 3, _termcrit, 0, 0.001);
        
        size_t i, k;
        for(i = k = 0; i < points[1].size(); i++)
        {
            if(_addRemovePt)
            {
                if(norm(_touchPoint - points[1][i]) <= 5)
                {
                    _addRemovePt = false;
                    continue;
                }
            }
            
            if(!status[i])
                continue;
            
            points[1][k++] = points[1][i];
            circle(image, points[1][i], 10, cv::Scalar(0,255,0), -1, 8);
            
        }
        points[1].resize(k);
    }
    
    if(_addRemovePt && points[1].size() < (size_t)MAX_POINTS_COUNT)
    {
        cv::vector<cv::Point2f> tmp;
        tmp.push_back(_touchPoint);
        //tmp.push_back(_touchPoint1);
        //tmp.push_back(_touchPoint2);
        //tmp.push_back(_touchPoint3);
        cornerSubPix( gray, tmp, _winSize, cv::Size(-1,-1), _termcrit);
        points[1].push_back(tmp[0]);
        _addRemovePt = false;
    }
    
    // reference
    std::swap(points[1], points[0]);
    cv::swap(gray_prev, gray);
    
    //imshow(window_name, output);
}

- (BOOL)addNewPoints
{
    return points[0].size() <= 20;
}

- (BOOL)acceptTrackedPoint:(int)i{
    return status[i] && ((abs(points[0][i].x - points[1][i].x)+abs(points[0][i].y-points[1][i].y)) > 2);
}

- (void)configureDevice:(AVCaptureDevice *)device frameRate:(int32_t)frameRate
{
    if ([device lockForConfiguration:NULL] == YES)
    {
        device.activeVideoMinFrameDuration = CMTimeMake(1, frameRate);
        device.activeVideoMaxFrameDuration = CMTimeMake(1, frameRate);
        [device unlockForConfiguration];
    }
}


- (void)updateCaptureOrientation
{
    AVCaptureConnection *captureConnection = [[[[_session outputs] firstObject] connections] firstObject];
    
    if ([captureConnection isVideoOrientationSupported])
    {
        UIInterfaceOrientation orientation = [[UIApplication sharedApplication] statusBarOrientation];
        [captureConnection setVideoOrientation:(AVCaptureVideoOrientation)orientation];
    }
}

- (void)autoFocusAtPoint:(CGPoint)point
{
    double focus_x = point.x/_previewView.bounds.size.width;
    double focus_y = point.y/_previewView.bounds.size.height;
    
    if([_device isFocusPointOfInterestSupported] && [_device isFocusModeSupported:AVCaptureFocusModeAutoFocus])
    {
        if([_device lockForConfiguration:NULL] == YES)
        {
            [_device setFocusPointOfInterest:CGPointMake(focus_x, focus_y)];
            [_device setFocusMode:AVCaptureFocusModeAutoFocus];
            [_device unlockForConfiguration];
        }
    }
}

/*- (void)processImage:(Mat &)image
 {
 // Do some OpenCV stuff with the image
 
 //[self tracking:downsampled_grey output:result];
 
 cvtColor(image, image_copy, CV_RGB2GRAY);
 GaussianBlur(image_copy, image_copy, cv::Size(5,5), 1.2,1.2);
 Mat edges;
 Canny(image_copy,edges,0,50);
 image.setTo(Scalar::all(255));
 image.setTo(Scalar(0,128,255,255),edges);
 
 //cv::Mat gray_down(image.rows,image.cols,CV_8UC1);
 cvtColor(image, gray, COLOR_BGR2GRAY);
 
 if([self addNewPoints]){
 goodFeaturesToTrack(gray, features, maxCount, qLevel, minDist);
 points[0].insert(initial.end(), features.begin(),features.end());
 initial.insert(initial.end(), features.begin(),features.end());
 }
 
 if (gray_prev.empty())
 {
 gray.copyTo(gray_prev);
 }
 //goodFeaturesToTrack(gray, features, maxCount, qLevel, minDist);
 calcOpticalFlowPyrLK(gray_prev, gray, points[0], points[1], status, err);
 // remove bad features
 int k = 0;
 for (size_t i=0; i<points[1].size(); i++)
 {
 if ([self acceptTrackedPoint:i])
 {
 initial[k] = initial[i];
 points[1][k++] = points[1][i];
 }
 }
 points[1].resize(k);
 initial.resize(k);
 
 // show
 for (size_t i=0; i<points[1].size(); i++)
 {
 //line(image, initial[i], points[1][i], Scalar(0, 0, 255));
 circle(image, points[1][i], 3, Scalar(0, 255, 0), -1);
 }
 //drawArrows(image, features, points[1], status);
 
 
 // reference
 std::swap(points[1], points[0]);
 cv::swap(gray_prev, gray);
 
 //gray.copyTo(gray_prev);
 
 }
 
 #endif
 */

- (void)cleanCirclesOfInterest
{
    _addRemovePt = false;
    
    points[0].clear();
    points[1].clear();
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (IBAction)didClickCreamButton:(id)sender {
    //[videoCamera_ start];
}

#pragma mark - Gesture recongnizer
/*
 - (IBAction)tap:(UITapGestureRecognizer *)sender
 {
 CGPoint locationInView = [sender locationInView:_previewView];
 
 UIInterfaceOrientation orientation = [[UIApplication sharedApplication] statusBarOrientation];
 
 CGFloat scale = 0;
 
 if (UIInterfaceOrientationIsLandscape(orientation)) {
 scale = resolutionSize.width / _previewView.bounds.size.width;
 }
 else
 {
 scale = resolutionSize.height / _previewView.bounds.size.width;
 }
 NSLog(@"scale %f, location %f %f",scale,locationInView.x,locationInView.y);
 
 _touchPoint = cv::Point2f(locationInView.x * scale,locationInView.y * scale);
 _addRemovePt = true;
 }
 */

- (IBAction)doubleTap:(UITapGestureRecognizer *)sender
{
    CGPoint locationInView = [sender locationInView:_previewView];
    
    [self autoFocusAtPoint:locationInView];
}

#pragma mark - GTPreviewViewDelegate

-(void)previewViewMotionShakeDetected
{
    [self cleanCirclesOfInterest];
}

@end

