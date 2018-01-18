% Starter code prepared by James Hays
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = ....
    run_detector(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i.
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')


% Your code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

%BEGIN_RECODE
eigstruct = load('EigenFacesMeanFace.mat');
eigfacevec= eigstruct.eigfacevec;
meanFace = eigstruct.meanFace;
window_dim = feature_params.template_size;  % Range of cells to look at
scale  = [1.1 1.02 0.97, 0.8, 0.7 0.6 0.5];% Scales
%step size
step_size = 7;

for i = 1:1%length(test_scenes)
    
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
    cur_bboxes = [];
    cur_confidences = [];
    cur_image_ids = [];
    
    %running the code for scales predefined in matrix 'scale'
    for s = scale
        I_sc = imresize(img, s);
        
        %calculating the area we can work on
        ih = size(I_sc, 1) - window_dim;
        iw = size(I_sc, 2) - window_dim;
        %number of iterations to be done over dimensions
        horSize=floor(iw/step_size);
        verSize=floor(ih/step_size);
        
        pred = zeros(ih,iw);  % Predictions for each group
        for ii = 1:step_size:(verSize)*step_size+1
            for jj = 1:step_size:(horSize)*step_size+1
                extractedPatch = I_sc(ii:ii+window_dim-1,jj:jj+window_dim-1);
                extractedPatch = extractedPatch(:)' - meanFace;
                eigfacecoeff = zeros(1,size(eigfacevec,2));
                %calculating coefficients
                for j = 1:size(eigfacevec,2)
                    eigfacecoeff(j) = extractedPatch * eigfacevec(:,j);
                end
                %expand the data using kernel trick since w is expanded
                eigfacecoeff = vl_homkermap(eigfacecoeff',2,'kChi2');
                %use classifier to get a score
                score = dot(eigfacecoeff,w) + b;
                pred(ii,jj) = score;
            end
        end
        
        % Row and column indices of cells with possible faces detected
        [det_r, det_c] = find(pred > 0.7);  
               
        sc_bboxes = (1/s) * ...
            [det_c, det_r, det_c+window_dim-1, det_r+window_dim-1];
        sc_confidences = pred(sub2ind(size(pred), det_r, det_c));
        sc_image_ids = repmat({test_scenes(i).name}, size(det_r, 1), 1);
            cur_bboxes = vertcat(cur_bboxes, sc_bboxes);
            cur_confidences = vertcat(cur_confidences, sc_confidences);
            cur_image_ids = vertcat(cur_image_ids, sc_image_ids);        
    end
    
    %END_RECODE
    
    
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));
    
    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
    
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
end




