function [u, v, depth]=velo2uvdepth(base_dir, frame, calib)
% load velodyne points
fid = fopen(sprintf('%s/velodyne_points/data/%010d.bin',base_dir,frame),'rb');
velo = fread(fid,[4 inf],'single')';
fclose(fid);

% remove all points behind image plane (approximation
idx = velo(:,1)<5;
velo(idx,:) = [];

% project to image plane (exclude luminance)
velo_img = project(velo(:,1:3),calib.P_velo_to_img);

% extract depth info for each point projected on image plane (w/ distortion)
velo_cam = calib.Tr_velo_to_cam * [velo(:,1:3) ones(length(velo), 1)]';
depth = velo_cam(3, :)';

u=velo_img(:,1);
v=velo_img(:,2);