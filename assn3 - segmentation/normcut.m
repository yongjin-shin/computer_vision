%%
function [seg_] = normcutt(img)
    tic;
    %% Step 0 Matrix Aggumentation----------------------------------
    [row, col, c] = size(img);
    N = row*col;
    img_lab = rgb2lab(img);
    
    tmp = zeros(row, col, 2); %save x-coordiante and y-coordinate
    for i=1:row
        for j=1:col
            tmp(i,j,1)=i;
            tmp(i,j,2)=j;
        end
    end
    img_lab = cat(3, img_lab, tmp);
    
    %% Step 1 Extract centered data N to decrease # of Nodes----------
    % Fast Approximate Spectral Clustering according to Donghui Yan et al.(2009)    
    k_ = 10;
    N = floor(N/100);
    rimg_ = reshape(img_lab, [size(img_lab,1)*size(img_lab,2), 5]);
    [idx__, center] = kmeans(rimg_, N); 
    
    %% Step 2 Constructing Affinity Matrix-------------------
    W = sparse(N, N);
    for i=1:N
        W(:,i) = distance(center(i,:), center(:,:));
        W(i,i) = 1;
    end
    D = sparse(N, N);
    D = diag(sum(W,2));
    
    %% Step 3 Solve the Eigen System-----------------------------
    L = D - W;
    [Y, lambda] = eigs(L, D, N, 'sm'); 
    
    %% Step 4 Partition with the eigen vectors-------------------
    idx = 1:N;
    lambda_ = diag(lambda);
    [lambda_, idx_] = sort(lambda_);
    val=lambda_(2:(N/10));
    idx_ = idx_(2:(N/10));
    target = Y(:, idx_);    
    idx_ = target>median(target);
    [idx_] = kmeans(idx_, k_);

    %% Step 5  Recover the cluster membership -------------------
    result_ = zeros(row*col, 1);
    for i=1:k_
        %original: center,idx__
        A = (idx_==i);
        A = find(idx_&A);
        a = length(A);
        for j=1:a
           B = (idx__ == A(j));
           result_(B==1)=i;
        end
    end
    
    idx_ = result_;
    seg_ = reshape(idx_, [size(img_lab,1) size(img_lab,2)]);    
    toc;
end

%% ----------------Affinity Calculator-------------------
function dist = distance(p1, p2)
    % Normalized spectral clustering according to Shi and Malik (2000)
    r = 20;
    sigma_X = 4;
    sigma_I= 5;

    % Simple calculation of two nodes
    p = p2 - p1;
    p = p .* p;
    
    % calculate dist: sptial location of two points
    d(:,1) = p(:,4)+p(:,5);
    d(sqrt(d)>r)=0;
    d=exp(-d/sigma_X);
    d(d==1)=0;
 
    % calculate dist: feature location of two points
    d_(:,1) = (1*p(:,1))+(3*p(:,2))+(3*p(:,3));
    d_=exp(-d_/sigma_I);
    
    dist = d_ .* d;
end