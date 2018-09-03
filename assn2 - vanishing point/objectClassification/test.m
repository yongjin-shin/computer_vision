final_dir = 'dataset/';

%road dataset from folder 
%and split them into each labels w/foldernames
finalSet = imageDatastore(final_dir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
finalSet.ReadFcn = @(trainingSet)imresize(imread(trainingSet),[256 256]);

%front 30 images will be in training Set
%after splitting, shuffle each set!
[trainingSet,testSet] = splitEachLabel(finalSet,0.6);
trainingSet = shuffle(trainingSet);
testSet = shuffle(testSet);

%find the hog feature size
%to set the size of training feature matrix
cellSize = [5 5];
img = readimage(trainingSet, 87);
[hog_4x4] = hog_feature_vector(img);
hogFeatureSize = length(hog_4x4);

%set labels and
%training feature matrix
numLabels = 5;
numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'double');
trainingLabels = grp2idx(trainingSet.Labels);

%extract hog freature of each images/train
for i = 1:numImages
    img = readimage(trainingSet, i);
    trainingFeatures(i, :) = hog_feature_vector(img);
end

% train one vs all models
model = cell(numLabels, 1);
for k=1:numLabels
    model{k} = svmtrain(double(trainingLabels == k), trainingFeatures, '-b 1');
end

% get prob. estimates of test instance using each model
numTest = numel(testSet.Files);
testLabels = grp2idx(testSet.Labels);
prob = zeros(numTest, numLabels);
p = zeros(numTest, 2);
testFeatures = zeros(numTest, hogFeatureSize, 'double');

%extract hog freature of each images/test
for i = 1:numTest
    img = readimage(testSet, i);
    testFeatures(i, :) = hog_feature_vector(img);
end

for k=1:numLabels
    [~,~,p] = svmpredict(double(testLabels == k), testFeatures, model{k}, '-b 1');
    prob(:,k) = p(:,model{k}.Label == 1);
end

% predict the class with the highest probability
[~, pred] = max(prob, [], 2);
acc = sum(pred == testLabels) ./ numel(testLabels)
C = confusionmat(testLabels, pred)



% to Extract HoG features, input is image,
function [feature] = hog_feature_vector(im)
    
    %block size is [5 5]
    %and make image has gray scale
    B_WIDTH = 5;
    if size(im,3)==3
        im=rgb2gray(im);
    end
    
    im=double(im);
    rows=size(im,1);
    cols=size(im,2);
    Ix=im; Iy=im;

    % Gradients in X and Y direction. Iy is the gradient in X direction
    for i=1:rows-2
        Iy(i,:)=(im(i,:)-im(i+2,:));
    end
    for i=1:cols-2
        Ix(:,i)=(im(:,i)-im(:,i+2));
    end

    %Initialized a gaussian filter with sigma=0.5 * block width.    
    gauss=fspecial('gaussian',B_WIDTH); 

    % Matrix containing the angles of each edge gradient
    %Angles in range (0,180)
    angle=atand(Ix./Iy); 
    angle=imadd(angle,90); 
    magnitude=sqrt(Ix.^2 + Iy.^2);

    % Remove redundant pixels in an image. 
    angle(isnan(angle))=0;
    magnitude(isnan(magnitude))=0;

    %initialized the feature vector
    feature=[]; 

    % Iterations for Blocks
    for i = 0: rows/B_WIDTH - 2
        for j= 0: cols/B_WIDTH -2
            
            %slide image with stride 50%
            mag_patch = magnitude(B_WIDTH*i+1 : B_WIDTH*i+10 , B_WIDTH*j+1 : B_WIDTH*j+10);
            ang_patch = angle(B_WIDTH*i+1 : B_WIDTH*i+10 , B_WIDTH*j+1 : B_WIDTH*j+10);

            %initialized the block vector
            block_feature=[];

            %Iterations for each cells in a block
            for x= 0:1
                for y= 0:1
                    angleA =ang_patch(B_WIDTH*x+1:B_WIDTH*x+B_WIDTH, B_WIDTH*y+1:B_WIDTH*y+B_WIDTH);
                    magA   =mag_patch(B_WIDTH*x+1:B_WIDTH*x+B_WIDTH, B_WIDTH*y+1:B_WIDTH*y+B_WIDTH); 
                    histr  =zeros(1,9);

                    %Iterations for each pixels in one cell
                    for p=1:B_WIDTH
                        for q=1:B_WIDTH

                            %get angle from each pixels
                            alpha= angleA(p,q);

                            % Binning Process (Bi-Linear Interpolation)
                            if alpha>10 && alpha<=30
                                histr(1)=histr(1)+ magA(p,q)*(30-alpha)/20;
                                histr(2)=histr(2)+ magA(p,q)*(alpha-10)/20;
                            elseif alpha>30 && alpha<=50
                                histr(2)=histr(2)+ magA(p,q)*(50-alpha)/20;                 
                                histr(3)=histr(3)+ magA(p,q)*(alpha-30)/20;
                            elseif alpha>50 && alpha<=70
                                histr(3)=histr(3)+ magA(p,q)*(70-alpha)/20;
                                histr(4)=histr(4)+ magA(p,q)*(alpha-50)/20;
                            elseif alpha>70 && alpha<=90
                                histr(4)=histr(4)+ magA(p,q)*(90-alpha)/20;
                                histr(5)=histr(5)+ magA(p,q)*(alpha-70)/20;
                            elseif alpha>90 && alpha<=110
                                histr(5)=histr(5)+ magA(p,q)*(110-alpha)/20;
                                histr(6)=histr(6)+ magA(p,q)*(alpha-90)/20;
                            elseif alpha>110 && alpha<=130
                                histr(6)=histr(6)+ magA(p,q)*(130-alpha)/20;
                                histr(7)=histr(7)+ magA(p,q)*(alpha-110)/20;
                            elseif alpha>130 && alpha<=150
                                histr(7)=histr(7)+ magA(p,q)*(150-alpha)/20;
                                histr(8)=histr(8)+ magA(p,q)*(alpha-130)/20;
                            elseif alpha>150 && alpha<=170
                                histr(8)=histr(8)+ magA(p,q)*(170-alpha)/20;
                                histr(9)=histr(9)+ magA(p,q)*(alpha-150)/20;
                            elseif alpha>=0 && alpha<=10
                                histr(1)=histr(1)+ magA(p,q)*(alpha+10)/20;
                                histr(9)=histr(9)+ magA(p,q)*(10-alpha)/20;
                            elseif alpha>170 && alpha<=180
                                histr(9)=histr(9)+ magA(p,q)*(190-alpha)/20;
                                histr(1)=histr(1)+ magA(p,q)*(alpha-170)/20;
                            end


                        end
                    end
                    % Concatenation of Four histograms to form one block feature
                    block_feature=[block_feature histr]; 

                end
            end
            
            % Normalize the values in the block using L2-Norm
            block_feature=block_feature/sqrt(norm(block_feature)^2+.001);
            for z=1:length(block_feature)
                if block_feature(z)>0.2
                     block_feature(z)=0.2;
                end
            end
            block_feature=block_feature/sqrt(norm(block_feature)^2+.001);
            
            %Features concatenation
            feature=[feature block_feature]; 
        end
    end
end