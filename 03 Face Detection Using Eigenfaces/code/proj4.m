% Sliding window face detection with linear SVM. 
% All code by James Hays, except for pieces of evaluation code from Pascal
% VOC toolkit. Images from CMU+MIT face database, CalTech Web Face
% Database, and SUN scene database.

% Code structure:
% proj4.m <--- You recode parts of this
%  + get_positive_features.m  <--- You recode this
%  + get_random_negative_features.m  <--- You recode this
%   [classifier training]   <--- You recode this
%  + report_accuracy.m
%  + run_detector.m  <--- You recode this
%    + non_max_supr_bbox.m
%  + evaluate_all_detections.m
%    + VOCap.m
%  + visualize_detections_by_ima    ge.m
%  + visualize_detections_by_image_no_gt.m
%  + visualize_detections_by_confidence.m

% Other functions. You don't need to use any of these unless you're trying
% to modify or build a test set:

% Training and Testing data related functions:
% test_scenes/visualize_cmumit_database_landmarks.m
% test_scenes/visualize_cmumit_database_bboxes.m
% test_scenes/cmumit_database_points_to_bboxes.m %This function converts
% from the original MIT+CMU test set landmark points to Pascal VOC
% annotation format (bounding boxes).

% caltech_faces/caltech_database_points_to_crops.m %This function extracts
% training crops from the Caltech Web Face Database. The crops are
% intentionally large to contain most of the head, not just the face. The
% test_scene annotations are likewise scaled to contain most of the head.

    % set up paths to VLFeat functions. 
    % See http://www.vlfeat.org/matlab/matlab.html for VLFeat Matlab documentation
% This should work on 32 and 64 bit versions of Windows, MacOS, and Linux
close all
clear
run('vlfeat/toolbox/vl_setup')
%run('MirrorPositiveImages.m')  if extra positive images wanted
[~,~,~] = mkdir('visualizations');

data_path = '../data/'; %change if you want to work with a network copy
train_path_pos = fullfile(data_path, 'caltech_faces/Caltech_CropFaces'); %Positive training examples. 36x36 head crops
non_face_scn_path = fullfile(data_path, 'train_non_face_scenes'); %We can mine random or hard negatives from here
test_scn_path = fullfile(data_path,'test_scenes/test_jpg'); %CMU+MIT test scenes
%test_scn_path = fullfile(data_path,'extra_test_scenes'); %Bonus scenes
label_path = fullfile(data_path,'test_scenes/ground_truth_bboxes.txt'); %the ground truth face locations in the test set

%The faces are 36x36 pixels, which works fine as a template size. You could
%add other fields to this struct if you want to modify HoG default
%parameters such as the number of orientations, but that does not help
%performance in our limited test.

%BEGIN_RECODE
feature_params = struct('template_size', 36);
%END_RECODE


%% Step 1. Load positive training crops and random negative examples
%YOU CODE 'get_positive_features' and 'get_random_negative_features'

features_pos = get_positive_features( train_path_pos, feature_params );

%normally I worked with 75.000 negative training data, however for less
%time complexity, one can reset it to 10.000. I left it as 75.000

num_negative_examples = 75000; %Higher will work strictly better, but you should start with 10000 for debugging
features_neg = get_random_negative_features( non_face_scn_path, feature_params, num_negative_examples);

    
%% step 2. Train Classifier
% Use vl_svmtrain on your training features to get a linear classifier
% specified by 'w' and 'b'
% [w b] = vl_svmtrain(X, Y, lambda) 
% http://www.vlfeat.org/sandbox/matlab/vl_svmtrain.html
% 'lambda' is an important parameter, try many values. Small values seem to
% work best e.g. 0.0001, but you can try other values

%BEGIN_RECODE  Make sure the outputs are 'w' and 'b'.
%Try using kernel trick here
lambda = 0.0001;
X = vertcat(features_pos, features_neg)';
npos = size(features_pos, 1);
nneg = size(features_neg, 1);
%Y is 1 for positive features and -1 for negative features
Y = horzcat(ones(1, npos), ones(1, nneg) * -1);
% create a structure with kernel map parameters
X = vl_homkermap(X,2,'kChi2');
% train the classifier using expanded input
[w,b] = vl_svmtrain(X, Y, lambda,'MaxNumIterations',1000000000000);
%END_RECODE

%% step 3. Examine learned classifier
% You don't need to modify anything in this section. The section first
% evaluates _training_ error, which isn't ultimately what we care about,
% but it is a good sanity check. Your training error should be very low.

fprintf('Initial classifier performance on train data:\n')
confidences2 = X'*w + b; %this is also slightly modified to match expanded w with the input
label_vector = [ones(size(features_pos,1),1); -1*ones(size(features_neg,1),1)];
[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy( confidences2, label_vector );

% Visualize how well separated the positive and negative examples are at
% training time. Sometimes this can idenfi  ty odd biases in your training
% data, especially if you're trying hard negative mining. This
% visualization won't be very meaningful with the placeholder starter code.
non_face_confs = confidences2( label_vector < 0);
face_confs     = confidences2( label_vector > 0);
figure(2); 
plot(sort(face_confs), 'g'); hold on
plot(sort(non_face_confs),'r'); 
plot([0 size(non_face_confs,1)], [0 0], 'b');
hold off;

%save features for further use
save('features.mat','features_neg','features_pos','confidences2','Y','w','b');
% Visualize the learned detector. This would be a good thing to include in
% your writeup! For PCA you can visualize the reconstructed images based on eigenfaces.
%BEGIN_RECODE

%END_RECODE
    
 
%% step 4. (optional) Mine hard negatives
% Mining hard negatives is extra credit. You can get very good performance 
% by using random negatives, so hard negative mining is somewhat
% unnecessary for face detection. If you implement hard negative mining,
% you probably want to modify 'run_detector', run the detector on the
% images in 'non_face_scn_path', and keep all of the features above some
% confidence level.

%% Step 5. Run detector on test set.
% YOU CODE 'run_detector'. Make sure the outputs are properly structured!
% They will be interpreted in Step 6 to evaluate and visualize your
% results. See run_detector.m for more details.
[bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params);

% run_detector will have (at least) two parameters which can heavily
% influence performance -- how much to rescale each step of your multiscale
% detector, and the threshold for a detection. If your recall rate is low
% and your detector still has high precision at its highest recall point,
% you can improve your average precision by reducing the threshold for a
% positive detection.


%% Step 6. Evaluate and Visualize detections
% These functions require ground truth annotations, and thus can only be
% run on the CMU+MIT face test set. Use visualize_detectoins_by_image_no_gt
% for testing on extra images (it is commented out below).

% Don't modify anything in 'evaluate_detections'!
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections(bboxes, confidences, image_ids, label_path);

%visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp, test_scn_path, label_path)
visualize_detections_by_image_no_gt(bboxes, confidences, image_ids, test_scn_path)

%visualize_detections_by_confidence(bboxes, confidences, image_ids, test_scn_path, label_path);

% performance to aim for
% random (stater code) 0.001 AP
% single scale ~ 0.2 to 0.4 AP
% multiscale, 6 pixel step ~ 0.83 AP
% multiscale, 4 pixel step ~ 0.89 AP
% multiscale, 3 pixel step ~ 0.92 AP
