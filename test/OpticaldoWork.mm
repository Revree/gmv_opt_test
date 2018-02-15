//
//  OpticalDowork.m
//  gmv_opt
//
//  Created by Andrea Chen on 2/9/18.
//  Copyright © 2018 Andrea Chen. All rights reserved.
//
#import "GTCaptureOutputUtils.h"
#import "OpticalDowork.h"
#import <AVFoundation/AVFoundation.h>

using namespace cv;
@implementation OpticalDowork


cv::Size                    _winSize;
cv::TermCriteria            _termcrit;


Mat image;
vector<uchar>               status;
vector<float>               err;


//These four component should be global
bool _addRemovePt = true;
cv::vector<cv::Point2f> points[2];
Mat gray;
Mat gray_prev;

// Input: GMVinputPoints: correct detection results from GMV
//        CurrentFrame: coordinate frame with the correct results
//        Update: whether there is a new result come in
+(NSArray*)OpticalFlowdowork:(NSArray<NSValue *>*)GMVinputPoints :(CGImageRef)CurrentFrame
{
    NSArray* newPoints;
    _winSize = cv::Size(15,15);
    _termcrit = cv::TermCriteria(cv::TermCriteria::EPS|cv::TermCriteria::COUNT,20,0.01);
    
    //creat UIimage from input CGimage
    UIImage *grayui = [UIImage imageWithCGImage:CurrentFrame];
    
    //convert UIimage to opencv gray mat
    gray =* [GTCaptureOutputUtils cvMatFromImage:grayui gray:true];
    
    //vector for all input GMV results
    vector<Point2f> _touchPointall;
    
    //put GMV results into vector, for further compute
    for(NSValue * value in GMVinputPoints){
        CGPoint point = [value CGPointValue];
        _touchPointall.push_back(cv::Point2f(point.x,point.y));
    }
    
    //These mats should be global.
    //Mat gray;
    //Mat gray_prev;
    
    
    if (!points[0].empty())
    {
        cv::vector<uchar> status;
        cv::vector<float> err;
        
        if(gray_prev.empty())
        {
            gray.copyTo(gray_prev);
        }
        
        calcOpticalFlowPyrLK(gray_prev, gray, points[0],points[1], status, err, _winSize, 3, _termcrit, 0, 0.001);
        
        size_t i;
        //if new result comes in, then check with the tracking results, else just do tracking
        
        for(i = 0; i<_touchPointall.size();i++)
        {
            if(_addRemovePt)
            {
                if( _touchPointall[i]==cv::Point2f(288,-80)){
                    continue;
                }
                if(norm(_touchPointall[i]-points[1][i])<=3)
                {
                    NSLog(@"accpeted");
                }
                
                if(norm(_touchPointall[i] - points[1][i])>= 10 && norm(_touchPointall[i] - points[1][i])<=20)
                {
                    points[1][i]=Point2f((points[1][i].x + _touchPointall[i].x)/2.0,(points[1][i].y+ _touchPointall[i].y)/2.0);
                    points[1][i] = _touchPointall[i];
                    NSLog(@"wrong points, adjusting");
                }
                if(norm(_touchPointall[i] - points[1][i])>20)
                {
                    points[1][i] = _touchPointall[i];
                    NSLog(@"FATALE ERROR");
                }
            }
            if(!status[i])
                continue;
            circle(image, points[1][i], 5, cv::Scalar(0,255,0), -1, 8);
        }
        
    }
    
    //for first init or give new results, put GMV results in points[1]
    if(_addRemovePt)
    {
        //test code -- upload points
        for(size_t i=0;i < _touchPointall.size();i++){
            cv::vector<cv::Point2f> temp;
            temp.push_back(_touchPointall[i]);
            cornerSubPix(gray,temp, _winSize,cv::Size(-1,-1),_termcrit);
            points[1].push_back(temp[0]);
        }
        _addRemovePt = false;
    }
    
    newPoints = [NSArray arrayWithObjects:
                 [NSValue valueWithCGPoint:CGPointMake(points[1][0].x,points[1][0].y)],
                 [NSValue valueWithCGPoint:CGPointMake(points[1][1].x,points[1][1].y)],
                 [NSValue valueWithCGPoint:CGPointMake(points[1][2].x,points[1][2].y)],
                 [NSValue valueWithCGPoint:CGPointMake(points[1][3].x,points[1][3].y)],
                 nil];
    
    //swap currentpoint and previous points for next round
    std::swap(points[1], points[0]);
    cv::swap(gray_prev, gray);
    
    //temp: assume GMV gives me four points' results.
    
    //for(int j = 0;j<points[1].size();j++){
    //    CGPoint point_temp;
    //    point_temp.x = points[1][j].x;
    //    point_temp.y = points[1][j].y;
    //[newPoints addObject:point_temp];
    //}
    //newPoints = [NSArray arrayWithObjects:&(points[1][1].x,points[1][1].y) count:points[1].size()];
    return newPoints;
}

@end

