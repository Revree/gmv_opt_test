//
//  GTPreviewView.m
//  gmv_opt
//
//  Created by Andrea Chen on 12/14/17.
//  Copyright © 2017 Andrea Chen. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "GTPreviewView.h"
#import <AVFoundation/AVFoundation.h>


@implementation GTPreviewView

- (void)motionEnded:(UIEventSubtype)motion withEvent:(UIEvent *)event
{
    if (event.subtype == UIEventSubtypeMotionShake)
    {
        if ([self.delegate respondsToSelector:@selector(previewViewMotionShakeDetected)])
        {
            [self.delegate previewViewMotionShakeDetected];
        }
    }
    
    if ([super respondsToSelector:@selector(motionEnded:withEvent:)])
        [super motionEnded:motion withEvent:event];
}

- (BOOL)canBecomeFirstResponder
{
    return YES;
}

@end

