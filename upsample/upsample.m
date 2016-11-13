
%   This is the demo code of the following paper.
%
%   Y. Konno, Y, Monno, D. Kiku, M. Tanaka and M. Okutomi,
%   "Intensity Guided Depth Upsampling by Residual Interpolation",
%   In JSME/RMD International Conference on Advanced Mechatronics, 2015
%  ---------------------------------------------------------------------
%   date    : 2016/04/19
%   Author  : Yosuke Konno <ykonno@ok.ctrl.titech.ac.jp>
%   version : 1.1
%
%   Copyright 2015 Okutomi & Tanaka lab, Tokyo Institute of Technology
%   All rights reserved.
%  ---------------------------------------------------------------------

[parent_dir, ~, ~] = fileparts(pwd);
rgb_data_list = dir([parent_dir, '/data/rgb/']);
depth_data_list = dir([parent_dir, '/data/depth/']);
%------------------------------------------------------------------
i = 64;
scale = 1;
guide_s = double(rgb2gray(imread([parent_dir, '/data/rgb/', rgb_data_list(i).name])));
guide_s = guide_s/max(guide_s(:));

% depth_gt = double(imread('Inputs\art.png'));

% make the input LR depth map.
% h = fspecial('gaussian', 2^scale, 2^scale);
% tmp = imfilter(depth_gt, h, 'replicate');
% depth = tmp(1:2^scale:end, 1:2^scale:end);
depth = csvread([parent_dir, '/data/depth/', depth_data_list(i).name]);
k = depth;
d_max = max(depth(:));
depth = depth/d_max;

if ~exist('Results', 'dir')
    mkdir('Results');
end
savepath = sprintf('Results\\%dx%d_to_%dx%d.png', ...
                    size(depth), size(depth_gt));

% perform x2 upsampling operation iteratively.
for i = 1:scale
    h = fspecial('gaussian', 2^(scale-i), 2^(scale-i));
    guide_tmp = imfilter(guide_s, h, 'replicate');
    guide = guide_tmp(1:2^(scale-i):end, 1:2^(scale-i):end);
    size(guide)
    imshow(guide)
    
    size(depth)
    depth = resint(depth, guide);
end

imshow(uint8(depth*d_max));
imwrite(uint8(depth*d_max), savepath);
