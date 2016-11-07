data_dir='.';
data_list=dir([data_dir, '\rgb']);
data_list=data_list(~[data_list.isdir]);

i=15-1 % image index, 0-based 

img = imread([data_dir, '\rgb\', data_list(i).name]);
fig = figure('Position',[20 100 size(img,2) size(img,1)]); axes('Position',[0 0 1 1]);
imshow(img); hold on;
% plot points
M = csvread([data_dir, '\depth\', data_list(i).name(1:end-4), '.dat']);
% M=[u,v,depth] 
scatter(M(:,1), M(:,2), 10*ones(size(M,1),1), M(:,3), 'filled');