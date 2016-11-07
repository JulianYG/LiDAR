function [u, v, depth]=run_demoVelodyne (para,frame)
% KITTI RAW DATA DEVELOPMENT KIT
% 
% Demonstrates projection of the velodyne points into the image plane
%
% Input arguments:
% base_dir .... absolute path to sequence base directory (ends with _sync)
% calib_dir ... absolute path to directory that contains calibration files

% load calibration
calib = loadCalibrationCamToCam(fullfile(para.calib_dir,'calib_cam_to_cam.txt'));
Tr_velo_to_cam = loadCalibrationRigid(fullfile(para.calib_dir,'calib_velo_to_cam.txt'));

% compute projection matrix velodyne->image plane
R_cam_to_rect = eye(4);
R_cam_to_rect(1:3,1:3) = calib.R_rect{1};
P_velo_to_img = calib.P_rect{para.cam+1}*R_cam_to_rect*Tr_velo_to_cam;

% load velodyne points
fid = fopen(sprintf('%s/velodyne_points/data/%010d.bin',para.base_dir,frame),'rb');
velo = fread(fid,[4 inf],'single')';
fclose(fid);

% remove all points behind image plane (approximation
idx = velo(:,1)<5;
velo(idx,:) = [];

% project to image plane (exclude luminance)
velo_img = project(velo(:,1:3),P_velo_to_img);

% extract depth info for each point projected on image plane (w/ distortion)
velo_cam = Tr_velo_to_cam * [velo(:,1:3) ones(length(velo), 1)]';
depth = velo_cam(3, :)';

u=velo_img(:,1);
v=velo_img(:,2);

