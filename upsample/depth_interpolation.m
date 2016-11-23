data_dir='.';
data_list=dir([data_dir, '/rgb']);
data_list=data_list(~[data_list.isdir]);

i=180; % image index, 0-based

img = imread([data_dir, '/rgb/', data_list(i).name]);
fig = figure('Position',[20 100 size(img,2) size(img,1)]); axes('Position',[0 0 1 1]);
imshow(img); hold on;
% plot points

for num = 1:194,
    M = csvread([data_dir, '/depth/', data_list(num).name(1:end-4), '.dat']);
    % M=[u,v,depth]
    %scatter(M(:,1), M(:,2), 10*ones(size(M,1),1), M(:,3), 'filled');
    
    
    u = M(:, 1);
    u = ceil(u);
    v = M(:, 2);
    v = ceil(v);
    row = 375;
    col = 1242;
    depth_map = zeros(row, col);
    
    for i=1:size(u,1),
        if (u(i) > 0 && u(i) < col+1 && v(i) > 0 && v(i) < row+1)
            depth_map(v(i), u(i)) = M(i, 3);
        end
    end
    if num == 180
        scatter(u, v, 10*ones(size(M,1),1), M(:,3), 'filled');
    end
    
    
    
    for i = 1:row
        for j = 1:col
            if (depth_map(i,j) == 0)
                value = [];
                dist = [];
                for di = -5:5
                    for dj = -5:5
                        newi = i+di;
                        newj = j+dj;
                        if (newi > 0 && newi < row+1 && newj > 0 && newj < col+1)
                            if (depth_map(newi, newj) ~= 0)
                                d = di^2 + dj^2;
                                value = [value; depth_map(newi, newj)];
                                dist = [dist; d];
                            end
                        end
                    end
                end
                weight_sum = sum(dist);
                [dist_descent, I] = sort(dist, 'descend');
                dist_ascent = sort(dist);
                for l = 1:size(dist, 1)
                    depth_map(i,j) = depth_map(i,j) + value(I(l))*dist_ascent(l)/weight_sum;
                end
            end
        end
    end
    filename = [ data_list(num).name(1:end-4), '.dat'];
    save(filename, 'depth_map', '-ascii');
    %save ('test.dat','depth_map','-ASCII');
    if num == 180
        figure; imagesc(depth_map);
    end
end