//
//  MainView.h
//  remove_background_mac
//
//  Created by normal on 2024/8/26.
//

#import <Cocoa/Cocoa.h>

NS_ASSUME_NONNULL_BEGIN

@interface MainView : NSView

@property (weak) IBOutlet NSImageView *imageView;
@property (weak) IBOutlet NSButton *button1;
@property (weak) IBOutlet NSButton *button2;

- (IBAction)button1Clicked:(id)sender;

@end

NS_ASSUME_NONNULL_END
