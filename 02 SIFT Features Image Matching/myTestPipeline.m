clear;clc;
%read both images
img1 = imread('./data/shrek_reference.png');
img2 = imread('./data/shrek_test.png');
%detect corners and save obtained images
[features1, pyr1, imp1,scale1] = detect_corners(img1);
saveas(gcf,'shrek1.png')
[features2, pyr2, imp2,scale2] = detect_corners(img2);
saveas(gcf,'shrek2.png')
%extract descriptor (note that they are not as robust as the descriptors in
%the SIFT paper
descriptor1 = SIFTDescriptor(imp1, features1, scale1);
descriptor2 = SIFTDescriptor(imp2, features2, scale2);
%calculate matches
matches = SIFTSimpleMatcher(descriptor1,descriptor2,0.7);
%plot matches to see visual correspondence.
PlotMatch(im2double(img1),im2double(img2),features1',features2',matches');
saveas(gcf,'shrekMatch.png')