% Short demo for generating image

[parent_dir, ~, ~] = fileparts(pwd);
data_list = dir([parent_dir, '/data/rgb/']);
data_list = data_list(~[data_list.isdir]);

i = 64; % image index, 0-based 

img = imread([parent_dir, '/data/rgb/', data_list(i).name]);
fig = figure('Position',[20 100 size(img,2) size(img,1)]); ...
    axes('Position',[0 0 1 1]);
imshow(img); hold on;

% plot points
M = csvread([parent_dir, '/data/depth/', data_list(i).name(1:end-4), '.dat']);
scatter(M(:,1), M(:,2), 70*ones(size(M,1),1), M(:,3), 'filled');
