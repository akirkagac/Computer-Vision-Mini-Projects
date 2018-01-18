function [features,pyr,imp,scale] = detect_corners(img)
%read image
height = size(img,1);
width = size(img,2);

sigma = 0.5;
%define sobel filters
SobelVertical = fspecial('sobel');
SobelHorizontal = transpose(SobelVertical);

%compute vertical and horizontal gradients
xGradient = (1/255)*double(imfilter(img,SobelHorizontal));
xGradient = imgaussfilt(xGradient,1.5);
yGradient = (1/255)*double(imfilter(img,SobelVertical));
yGradient = imgaussfilt(yGradient,1.5);

%sketch gradients
figure(1),imshow(uint8(xGradient*255))
figure(2),imshow(uint8(yGradient*255))

%define empty harris score matrix
Harris = zeros(height,width);

%for every pixel, extract a 5x5 vicinity from gradients in both directions
%and use them to first calculate correlation matrix,and then Harris score
%using det(C) - alfa*trace(C)^2
for i = 3:height-2
    for j = 3:width-2
        xVicinity = xGradient(i-2:i+2,j-2:j+2,:);
        xVicinity = reshape(xVicinity,[75,1]);
        yVicinity = yGradient(i-2:i+2,j-2:j+2,:);
        yVicinity = reshape(yVicinity,[75,1]);
        
        D = double([xVicinity,yVicinity]);
        C = (D')*D;
        Harris(i,j) = det(C)-0.04*(trace(C))^2;
        
    end
end

%keep values above threshold, discard others.
binaryHarris = Harris>5;
Harris = Harris.*binaryHarris;

%fill some windows with the max values in that area
maxHarris = colfilt(Harris, [10 10], 'sliding', @max);

%obtain logic difference so that we can locate max locations
%above function filled the whole window with the max value, so if we run an
%AND operator, we will only get the maximum value and its specific
%location. Following two opeations will do this.
a = (Harris == maxHarris & Harris > 0);
%extract the corners where maximum Harris values occur.
[posr, posc] = find(a > 0);
%show image,plot the corners on top of the image.
figure(5),imshow(img);
hold on;
plot(posc,posr,'r.');
hold off;

%output features in the desired format to further use it in SIFT Pipeline.
features = [posc posr];
if size(img,3) > 1
    img = rgb2gray(img);
end
img2 = img;
img = filter_gaussian(img,7,.5);%slightly filter bottom level
imp{1}=img;
A = filter_gaussian(img2,7,sigma);%calculate difference of gaussians
B = filter_gaussian(A,7,sigma);
pyr{1} = A-B;  %store result in cell array
scale = ones(size(features,1),1); %our scale output with a fixed value (1).

end