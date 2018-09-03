function [ Pt ] = iterate_particle_filter( t, X_1, Xt_1, Pt_1, It, Ft_1, p, likelihood_func )
%ITERATE_PARTICLE_FILTER Summary of this function goes here
%   Detailed explanation goes here

Rt_1 = resampling(Pt_1, p);
Rt = state_transion(t, X_1, Rt_1, Xt_1, p);
Lt = likelihood_func(Rt, It, Ft_1, p);
Pt = [Rt Lt];

end

function [Rt_1] = resampling(Pt_1, p)
%% Resample a new particle set from particles
%  which represents posterior distribution at previous time step p(x_(t-1) | z_(1:t-1))

% for example, if we resample M-samples from the prior
% [x1 i1 s1 c1;      [x1 y1 s1 1/M;
%  x2 y2 s2 c2;       x1 y1 s1 1/M;
%  x3 y3 s3 c3;  -->  x3 y3 s3 1/M;
%      ...                ...
%  xn yn sn cn]       xM yM sM 1/M];
%
%  where sum of all ci = 1

% (hint : use function *mnrnd* to get samples from multinormial distribution)
Pt_1_prob = Pt_1(:,4);
Pt_1_prob = Pt_1_prob.^2;
Pt_1_prob = Pt_1_prob/sum(Pt_1_prob);
Pt_smp = mnrnd(p.M,Pt_1_prob);
Rt_1 = zeros(p.M, 3);
rt_count = 1;
for i=1:p.M
    iter = Pt_smp(i);
    while(iter>0)
        Rt_1(rt_count, 1:3) = Pt_1(i, 1:3);
        rt_count = rt_count + 1;
        iter = iter - 1 ;
    end
end

end

function [Rt] = state_transion(t, X_1, Rt_1, Xt_1, p)
%% State transition by random walk 
% p(x_(t-1) | z_(1:t-1)) --> p(x_t | z_(1:t-1))
% random walk for each particle
% move state of each particle using random Gaussian samples
% (hint : use function *mvnrnd* to get samples from Gaussian distribution)

% x_1 = X_1(1); y_1=X_1(2);
% x_t_1 = Xt_1(1); y_t_1 = Xt_1(2);
% grad = zeros(1,3);
% grad(1) = (x_t_1-x_1)/t;
% grad(2) = (y_t_1-y_1)/t;
% Gt_1 = repmat(grad, p.M, 1);

s_t_1 = Xt_1(3);
sigma_tmp = [p.sigma_x*s_t_1,p.sigma_y*s_t_1*p.A,p.sigma_s*s_t_1];
sigma = diag(sigma_tmp);
sigma = sigma.*sigma;
mu = [0,0,0];
e = mvnrnd(mu, sigma, p.M); 

Rt = Rt_1 + e; %Gt_1 
end