function []=visualizeImgDepth2D(para, frame, u, v, depth)
% load and display image
img = imread(sprintf('%s/image_%02d/data/%010d.png',para.base_dir,para.cam,frame));
fig = figure('Position',[20 100 size(img,2) size(img,1)]); axes('Position',[0 0 1 1]);
imshow(img); hold on;
% plot points
scatter(u, v, 10*ones(size(u)), depth, 'filled');