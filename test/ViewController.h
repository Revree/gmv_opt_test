//
//  ViewController.h
//  gmv_opt
//
//  Created by Andrea Chen on 12/3/17.
//  Copyright © 2017 Andrea Chen. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <opencv2/highgui/cap_ios.h>
#import <opencv2/objdetect/objdetect.hpp>
#import <opencv2/imgproc/imgproc_c.h>

@interface ViewController : UIViewController<CvVideoCameraDelegate>
{
    IBOutlet UIImageView* imageView;
    IBOutlet UIButton* button;
    
    
}

- (IBAction)didClickCreamButton:(id)sender;

@end

