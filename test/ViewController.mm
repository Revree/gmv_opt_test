//
//  ViewController.m
//  gmv_opt
//
//  Created by Andrea Chen on 12/3/17.
//  Copyright © 2017 Andrea Chen. All rights reserved.
//

#import "ViewController.h"
#import <opencv2/opencv.hpp>
#import "GTCaptureOutputUtils.h"
#import "GTPreviewView.h"


using namespace cv;



string window_name = "optical flow tracking";

int maxCount = 100;
double qLevel = 0.1;
double minDist = 10;


@interface ViewController ()<AVCaptureVideoDataOutputSampleBufferDelegate,GTPreviewViewDelegate>
{
    UIImageView *liveView_;
    CvVideoCamera *videoCamera_;
    
    Mat gray;
    Mat gray_prev;
    Mat output;
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
    // Do any additional setup after loading the view, typically from a nib.
    
    // 1. Setup the your OpenCV view, so it takes up the entire App screen......
    int view_width = self.view.frame.size.width;
    int view_height = (414*view_width)/272; // Work out the view-height assuming 640x480 input
    int view_offset = (self.view.frame.size.height - view_height)/2;
    liveView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, view_offset, view_width, view_height)];
    [self.view addSubview:liveView_]; // Important: add liveView_ as a subview
    
    
    
    // 2. Initialize the camera parameters and start the camera (inside the App)
    videoCamera_ = [[CvVideoCamera alloc] initWithParentView:liveView_];
    videoCamera_.delegate = self;
    
    // This chooses whether we use the front or rear facing camera
    videoCamera_.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    
    // This is used to determine the device orientation
    videoCamera_.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    videoCamera_.defaultFPS = 120;
    
    //orientation shows sideways even if it is portait, these two lines are added to fix this problem
    // CGAffineTransform xform = CGAffineTransformMakeRotation(-M_PI / 2);
    // liveView_.transform = xform;
    self-> videoCamera_.rotateVideo = YES;
    
    // This starts the camera capture
    [videoCamera_ start];
    
}

static void drawArrows(cv::Mat frame, const cv::vector<cv::Point2f>& prevPts, const cv::vector<cv::Point2f>& nextPts, const cv::vector<uchar>& status,
                       cv::Scalar line_color = cv::Scalar(0, 0, 255))

{
    //cv:: Scalar RED = cv::Scalar(255,0,0);
    for (size_t i = 0; i < prevPts.size(); ++i)
    {
        if (status[i])
        {
            int line_thickness = 1;
            
            cv::Point p = prevPts[i];
            cv::Point q = nextPts[i];
            
            double angle = atan2((double) p.y - q.y, (double) p.x - q.x);
            
            double hypotenuse = sqrt( (double)(p.y - q.y)*(p.y - q.y) + (double)(p.x - q.x)*(p.x - q.x) );
            
            if (hypotenuse < 1.0)
                continue;
            
            // Here we lengthen the arrow by a factor of three.
            q.x = (int) (p.x - 3 * hypotenuse * cos(angle));
            q.y = (int) (p.y - 3 * hypotenuse * sin(angle));
            
            // Now we draw the main line of the arrow.
            cv:: Scalar RED = cv::Scalar(255,0,0);
            cv:: Scalar BLUE = cv::Scalar(0,0,255);
            line(frame, p, q, line_color, line_thickness);
            circle(frame, q, 5,RED);
            circle(frame, p, 5,BLUE);
            
            // Now draw the tips of the arrow. I do some scaling so that the
            // tips look proportional to the main line of the arrow.
            
            p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);
            
            p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
            p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
            line(frame, p, q, line_color, line_thickness);
        }
    }
}

void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step,double, const cv::Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
            line(cflowmap, cv::Point(x,y), cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, cv::Point(x,y), 1, CV_RGB(200, 0, 0), -1);
        }
    
}

void drawpoints(Mat& output, vector<Point2f> points[2]){
    for (size_t i=0; i<points[1].size(); i++)
    {
        //line(image, initial[i], points[1][i], Scalar(0, 0, 255));
        circle(output, points[1][i], 3, Scalar(0, 255, 0), -1);
    }
    
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection
{
    [GTCaptureOutputUtils convertYUVSampleBuffer:sampleBuffer toGrayscaleMat:gray];
    
    gray.copyTo(gray_prev);
    
    [self tracking:gray_prev output:gray];
    
    UIImage *imageToDisplay = [GTCaptureOutputUtils imageFromCvMat:&_image];
    
    dispatch_async(dispatch_get_main_queue(), ^{
        _previewView.image = imageToDisplay;
    });
}
/*- (void)processImage:(Mat &)image
 {
 // Do some OpenCV stuff with the image
 
 //[self tracking:downsampled_grey output:result];
 
 /*  cvtColor(image, image_copy, CV_RGB2GRAY);
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
 
 #endif*/

- (void)tracking:(cv::Mat &)frame output:(cv::Mat &)output
{
    
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    frame.copyTo(output);
    if([self addNewPoints]){
        goodFeaturesToTrack(gray, features, maxCount, qLevel, minDist);
        points[0].insert(initial.end(), features.begin(),features.end());
        initial.insert(initial.end(), features.begin(),features.end());
    }
    
    if (gray_prev.empty())
    {
        gray.copyTo(gray_prev);
    }
    
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
        line(output, initial[i], points[1][i], Scalar(0, 0, 255));
        circle(output, points[1][i], 3, Scalar(0, 255, 0), -1);
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


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (IBAction)didClickCreamButton:(id)sender {
    
    [videoCamera_ start];
    
    
}

@end

