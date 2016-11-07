function [output_frame_ind, total_frame_i]=generateDepth(base_dir, calib_dir, para, total_frame_i)

% get calibration parameters to para.calib
calib = loadCalibrationCamToCam(fullfile(calib_dir,'calib_cam_to_cam.txt'));
calib.Tr_velo_to_cam = loadCalibrationRigid(fullfile(calib_dir,'calib_velo_to_cam.txt'));

R_cam_to_rect = eye(4);
R_cam_to_rect(1:3,1:3) = calib.R_rect{1};
calib.P_velo_to_img = calib.P_rect{para.cam+1}*R_cam_to_rect*calib.Tr_velo_to_cam;


data_list=dir([base_dir,'\velodyne_points\data']);
data_num=sum(~[data_list.isdir]);
frames=0:(para.frame_skip+1):(data_num-1); % 0-based index

output_frame_ind = [frames'];
for frame_ind = frames
    [u, v, depth] = velo2uvdepth(base_dir, frame_ind, calib);
    csvwrite([para.depth_output, sprintf('%010d.dat',total_frame_i)],[u,v,depth]);
    copyfile(sprintf('%s/image_%02d/data/%010d.png',base_dir,para.cam,frame_ind), [para.image_output, sprintf('%010d.png',total_frame_i)]);
    total_frame_i=total_frame_i+1;
end



