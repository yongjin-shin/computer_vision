function [] = particle_filter_tracking( seq_path, result_file_path )
%PARTICLE_FILTER_TRACKING Track the given target with color-based particle filter algorithm

dbstop error; % Stop the code when error occurs
% If you stuck in debug-mode due to error, you can take an action in below.
    % type *dbquit* to exit debug mode.
    % type *dbup* and *dbdown* move former/later called function
    % type *dbcont* to execute remaining code

global display_figure
display_figure = 1;
%display_figure = 0; % If you don't want figure window

%% parameters for tracker
% Tune parameter for better tracking result
p.M = 100; % the number of particles
p.Nh = 10;
p.Ns = 10;
p.Nv = 10;
p.lambda = 20;
p.Nbin = p.Nh * p.Ns + p.Nv;

p.sigma_x = 0.1;
p.sigma_y = 0.1;
p.sigma_s = 0.05;
p.threshold_h = 0.1;
p.threshold_s = 0.2;
p.delta = 0.001;

%% load sequence file information
% load paths of image files
img_files = dir([seq_path '/img/*.jpg']);

% the number of frames
Nf = length(img_files);

%load ground truth file
gt_file = [seq_path '/groundtruth_rect.txt'];
Ybox = load_ground_truth(gt_file);

% variable for saving tracking result
% X: center point representations for each frames
X = cell(Nf,1); % use cell data-type for readability

% compute state vector at first frame
% [x1 y1 width height]
% -->  [x_center y_center scale(width)], aspect ratio (height/width)
[X1, p.A] = bbox2xys(Ybox{1});
X{1} = [X1 1];

% ( Ybox: Ground Truth --> make window in the frame )

%% intialize entire pipeline for tracker
initialize_display();

I1 = imread([img_files(1).folder '/' img_files(1).name]);
p.W = size(I1, 2); % image width
p.H = size(I1, 1); % image height

% Need to modify this. 
% has to get histogram info
% with cropping only inside the window size
F = initialize_observation_model(I1, X{1}, p);

% Saving Particle distribution
% for the first time 't', all the frame is same with ground truth; so
% possibilities are same;
P = cell(Nf,1);
P{1} = initialize_particles(X{1}, p.M);
overlapRatio = zeros(Nf,1);

%% main body for tracker
for t=2:min(Nf, size(Ybox,1))
    
    img_file_path = [img_files(t).folder '/' img_files(t).name];
    It = imread(img_file_path);
    
    P{t} = iterate_particle_filter(t, X{1}, X{t-1}, P{t-1}, It, F, p, @compute_likelihood);
    
    
    X{t} = estimate_current_state(P{t});
    
    overlapRatio(t) = display_result(It, X{t}, Ybox{t}, P{t}, p);
end

%% save result
Xmat = cell2mat(X);
Xbox = xys2bbox(Xmat, p.A);
save_tracking_result(Xbox, result_file_path);
mean_overlap = mean(overlapRatio)
histogram(overlapRatio)
end

function [X3d, A] = bbox2xys(Xbox)
%% No need to
    X3d = zeros(size(Xbox,1),3);
    
    % x_center
    X3d(:, 1) = (Xbox(:, 1)*2+Xbox(:,3)-1)/2;
    
    % y_center
    X3d(:, 2) = (Xbox(:, 2)*2+Xbox(:,4)-1)/2;
    
    % scale = width
    X3d(:, 3) = Xbox(:, 3);
    
    % Aspect ratio = height/width
    A = Xbox(:, 4) / Xbox(:, 3);
end

function [] = initialize_display()
%% No need to
global display_figure
if(display_figure)
    figure(1)
end
end

function [F] = initialize_observation_model(I1, Y1, p)
%% No Need to
    F = color_histogram(Y1, I1, p);
end

function [Ht] = color_histogram( Rt, It, p )
%% Modify this
%Rt: Frame
%It: Image
%p : parameter

    Ih = rgb2hsv(It);
    % cellfun for alternative to a for-loop
    Rt_cell = mat2cell(Rt, ones(1, size(Rt,1)), [size(Rt,2)]);
    function [Hist_HSV] = hist_HS_V(Rti)
        % transforming coordinate for cropping image.
        % make the frame with using ceter point.
        
        x1 = Rti(1) - Rti(3)/2;
        y1 = Rti(2) - Rti(3)*p.A/2;
        x2 = x1 + Rti(3) - 1;
        y2 = y1 + Rti(3)*p.A - 1;
        

        x1 = min(x1, x2);
        x2 = max(x1, x2);

        y1 = min(y1, y2);
        y2 = max(y1, y2);
        
        
        % handling out-of-image coordinate
        x1 = max(1, ceil(x1));
        y1 = max(1, ceil(y1));
        x2 = min(p.W, ceil(x2));
        y2 = min(p.H, ceil(y2));
        

                
        % (hint : use *histcounts* and *histcounts2*)
        % divded by number of bins.
        % which represents the probabilities.
        %Hist_HSV = ones(1, p.Nbin)/p.M; % DUMMY IMPLEMENTATION
        Crop_I = Ih(y1:y2, x1:x2, :);
        H = Crop_I(:,:,1);S = Crop_I(:,:,2);V = Crop_I(:,:,3);
        H = H.*(H>p.threshold_h);S=S.*(S>p.threshold_s);
        
        hsbins = p.Nh * p.Ns;
        Nhs = histcounts2(H,S,[p.Nh,p.Ns]);
        Nv = histcounts(V, p.Nv);
        Hist_HSV = cat(1, reshape(Nhs, [hsbins,1]), Nv');
        Hist_sum = sum(Hist_HSV);
        Hist_HSV = Hist_HSV/Hist_sum;
    end
    Ht_cell = cellfun(@hist_HS_V, Rt_cell, 'UniformOutput', false);
    Ht = cell2mat(Ht_cell);

    % if you don't believe, check the time of for-loop.
end

function [P1] = initialize_particles(Y1, M)
%% No need to
    P0 = [Y1 1]; % only ground truth
    P1 = repmat(P0, M, 1);
    P1(:,4) = P1(:,4)/M;
end

% compute likelihood using distance
function [Lt] = compute_likelihood(Rt, It, F, p)
%% Modify this
    % F : observation model at initial frame
    Lt = zeros(p.M, 1);
    for i=1:p.M
        F_t = color_histogram(Rt(i,:),It,p);
        q_q_t = sqrt(F.*F_t);
        Lt(i)=exp(-p.lambda*(1-sum(q_q_t)));
    end
    
end

function [Xt] = estimate_current_state(Pt)
%% No Need to
[~, max_idx] = max(Pt(:, 4));
Xt = Pt(max_idx, :);
end

function [overlapRatio] = display_result(It, Xt, Ybox, Pt, p)
%% No need to
global display_figure
if(display_figure)
    
    Xbox = xys2bbox(Xt, p.A);
    Pbox = xys2bbox(Pt, p.A);
    
    imshow(It);
    hold on
     for i = 1:min(size(Pbox, 1), 20)
         rectangle('Position', Pbox(i, :), 'EdgeColor', 'y', 'LineStyle', '--');
     end
    rectangle('Position', Ybox, 'EdgeColor', 'g');
    rectangle('Position', Xbox, 'EdgeColor', 'r');
    overlapRatio = bboxOverlapRatio(Ybox,Xbox);

    hold off
    drawnow;
end
end

function [Xbox] = xys2bbox(X3d, A)
%% No need to
   W = X3d(:, 3);
   H = X3d(:, 3) .* A;
   X1 = X3d(:, 1) - W/2;
   Y1 = X3d(:, 2) - H/2;
   Xbox = ceil([X1 Y1 W H]);
end

function [Ycell] = load_ground_truth(gt_file)
%% No need to
    Ybox = dlmread(gt_file);
    Ycell = mat2cell(Ybox, ones(1, size(Ybox,1)), [4]);
end

function [] = save_tracking_result(X, result_file_path)
%% No need to
    dlmwrite(result_file_path, X, ',');
end