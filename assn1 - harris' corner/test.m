%%Problem (a)
Imds = imageDatastore({'img1.jpg','img2.jpg'});
Image = readimage(Imds, 2);
g_Image = rgb2gray(Image);

h = size(g_Image, 1);
w = size(g_Image, 2);
a=0.05;

g = fspecial('gaussian', [9 9], 4);
Sx = fspecial('sobel');
Sy = Sx';

%step1: Find derivatives of the image w/sobel filter
Ix = conv2(g_Image, Sx, 'same');
Iy = conv2(g_Image, Sy, 'same');

%step2: Calculate Ixx, Iyy and Ixy
Ix2 = Ix .^2;
Iy2 = Iy .^ 2;
Ixy = Ix .* Iy;

%step3: Set Gaussian filter into Ixx, Iyy and Ixy,
% which also not making det to be zero
Ix2 = conv2(Ix2, g, 'same');
Iy2 = conv2(Iy2, g, 'same');
Ixy = conv2(Ixy, g, 'same');

%step4: R=det(A)- a*(tr(A))^2
tr = (Ix2+Iy2).^2;
tr = tr.*a;
d_t = Ix2.*Iy2-(Ixy.^2);
R=d_t- tr;

% Step 5: Find local maxima (non maximum suppression)
mx = ordfilt2(R, 9.^ 2, ones(9));
mx_copy = mx; %for a problem (b)

% Step 6: Thresholding
R = (R == mx) & (R >2e8);

% Step 7: Disaplying
[posY, posX] = find(R>0);
imshow(Image);
hold on;
plot(posX, posY, 'g.');


%%problem (b)
%----------------------------------
%Step 1: Find top 10 largest cornerness without redundant values
sortedR = sort(mx_copy(:), 'descend');
Uq = unique(sortedR);
Uq = sort(Uq(:), 'descend');
R10 = Uq(1:10);

% Step 2: Get the indices of the top ten values
[~,ia,~] = intersect(mx(:),R10(:)); 
D = d_t(ia);
T = tr(ia);

% Step 3: Find eigen values
eigA = zeros(10,1);
eigB = zeros(10,1);

for i=1:10
    syms a b;
    eq1 = a+b == D(i);
    eq2 = a*b == T(i);
    [sola, solb] =solve(eq1, eq2);
    eigA(i,1) = sola(1);
    eigB(i,1) = solb(1);    
end

% Step 4: Convert the indices to rows and columns
% and Display on the image
[r,c]=ind2sub(size(mx),ia); 
imshow(Image);
hold on;
plot(c, r, 'g.');
%-----------------------------------


