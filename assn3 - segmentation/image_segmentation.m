function [ seg ] = image_segmentation( img_file_path, result_file_path)
    %image_segmentation: perform generic normalized cut
    %   input : input image path, output segmentation path
    %   output : indexed segmentation image

    %% load image (do not modify this part in submission)
    img = imread(img_file_path); % range 0~255 and dtype with uint8
    img = double(img) / 255; % range 0~1 and dtype double

    %% implement below algorithm
    seg = normcut(img);

    %% save result as png (do not modify this part in submission)
    % the number of clusters should be less than 255 for visualization.
    imwrite(seg, labelcolormap(255), result_file_path);
end