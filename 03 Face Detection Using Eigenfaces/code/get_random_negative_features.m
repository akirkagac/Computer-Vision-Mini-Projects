% Starter code prepared by James Hays
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

%loading the eigenfaces and mean face
eigstruct = load('EigenFacesMeanFace.mat');
eigfacevec= eigstruct.eigfacevec;
meanFace = eigstruct.meanFace;

image_files = dir(fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);

%BEGIN_RECODE
temp_dim = (feature_params.template_size);
features_neg_raw = zeros(num_samples, temp_dim*temp_dim);
r = 0:(feature_params.template_size - 1);

for i = 1:num_samples
    % Multiple samples from each image, so we determine the image here.
    I = single(imread(fullfile(non_face_scn_path, ...
        image_files(ceil(i / num_images)).name))) / 255;
    h = size(I, 1);
    w = size(I, 2);
    random_patch = I(randi(h - feature_params.template_size) + r, ...
        randi(w - feature_params.template_size) + r);
    random_patch = reshape(random_patch,1,temp_dim*temp_dim);
    features_neg_raw(i, :) = random_patch;
end
%normalizing negative training data
features_neg_raw = features_neg_raw - repmat(meanFace,num_samples,1);
features_neg = zeros(num_samples,size(eigfacevec,2));
%calculating negative features (coefficients) using imported eigenfaces
for k = 1:num_samples
    for j = 1:size(eigfacevec,2)
        features_neg(k,j) = features_neg_raw(k,:) * eigfacevec(:,j);
    end
end
%END_RECODE

