//
//  OpticalDowork.h
//  gmv_opt
//
//  Created by Andrea Chen on 2/9/18.
//  Copyright © 2018 Andrea Chen. All rights reserved.
//

#ifndef OpticalDowork_h
#define OpticalDowork_h


#endif /* OpticalDowork_h */
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <opencv2/highgui/cap_ios.h>
#import <opencv2/objdetect/objdetect.hpp>
#import <opencv2/imgproc/imgproc_c.h>

@interface OpticalDowork : NSObject
+(NSMutableArray*)OpticalFlowdowork:(NSArray<NSValue *>*)GMVinputPoints :(CGImageRef)CurrentFrame;
@end

