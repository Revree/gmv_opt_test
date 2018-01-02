//
//  GTPreviewView.h
//  gmv_opt
//
//  Created by Andrea Chen on 12/14/17.
//  Copyright © 2017 Andrea Chen. All rights reserved.
//

#ifndef GTPreviewView_h
#define GTPreviewView_h


#endif /* GTPreviewView_h */
#import <UIKit/UIKit.h>

@class GTPreviewView;

@protocol GTPreviewViewDelegate <NSObject>
@optional
- (void)previewViewMotionShakeDetected;

@end

@interface GTPreviewView : UIImageView

@property (nonatomic,weak) IBOutlet id<GTPreviewViewDelegate>delegate;

@end
