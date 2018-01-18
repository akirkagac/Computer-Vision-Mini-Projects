data_path = '../data/';
train_path_pos = fullfile(data_path, 'caltech_faces/Caltech_CropFaces');

image_files = dir( fullfile( train_path_pos, '*.jpg') );
num_images = length(image_files);

for i = 1:num_images
    I = single(imread(fullfile(train_path_pos, image_files(i).name))) / 255;
    height = size(I,1);
    width = size(I,2);
    I2 = zeros(height,width);
    for j = 1:floor(width)/2
        
        I2(:,j)=I(:,(width-j+1));
        I2(:,(width-j+1))=I(:,j);
    end
    imwrite(I2, sprintf('../data/caltech_faces/Caltech_CropFaces/swapped%f.jpg',i))
end
