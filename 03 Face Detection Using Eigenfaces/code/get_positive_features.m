% Starter code prepared by James Hays
% This function should return all positive training examples (faces) from
% 36x36 images in 'train_path_pos'. Each face should be converted into a
% HoG template according to 'feature_params'. For improved performance, try
% mirroring or warping the positive training examples.

function features_pos = get_positive_features(train_path_pos, feature_params)
% 'train_path_pos' is a string. This directory contains 36x36 images of
%   faces
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.


% 'features_pos' is N by D matrix where N is the number of faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( train_path_pos, '*.jpg') ); %Caltech Faces stored as .jpg
num_images = length(image_files);

%BEGIN_RECODE
temp_dim = (feature_params.template_size);
features_raw = zeros(num_images, temp_dim*temp_dim);

%storing images in a matrix as vectors
for i = 1:num_images
    I = single(imread(fullfile(train_path_pos, image_files(i).name))) / 255;
    width = size(I,2);
    features_raw(i, :) = reshape(I,1,width*width);
end

%calculating mean face
meanFace = mean(features_raw,1);
%normalizing each face
features_raw = features_raw - repmat(meanFace,num_images,1);
%calculation of eigenfaces
cov_matrix = (1/num_images)*(features_raw'*features_raw);
valcolumn = sort(eig(cov_matrix),'descend');
sum_N = sum(valcolumn);
sum_M = 0;
M=0;
%keeping eigenfaces with high eigenvalues until a threshold
while(sum_M<0.93*sum_N)
    M = M+1;
    sum_M = sum_M + valcolumn(M);    
end
[eigfacevec,~] = eigs(cov_matrix,M);
features_pos = zeros(num_images,M);
%calculating positive coefficients (features)
for i = 1:num_images
    for j = 1:M
        features_pos(i,j) = features_raw(i,:) * eigfacevec(:,j);
    end
end
save('EigenFacesMeanFace.mat','meanFace','eigfacevec');

%END_RECODE
