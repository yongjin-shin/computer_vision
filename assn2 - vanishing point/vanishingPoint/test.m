first();
second();

function first(~)
    %load image
    img = imread('image1.jpg');
    img = rgb2gray(img);

    %performing canny dection and hough transform
    BW = edge(img, 'canny');
    [H,theta,rho] = hough(BW,'RhoResolution', 0.5,'Theta', -89:89);

    %find out hough peak and lines
    P = houghpeaks(H,4,'threshold',ceil(0.1*max(H(:))));
    lines = houghlines(BW,theta,rho,P,'FillGap',5);

    %draw lines
    figure, imshow(img), hold on
    li = zeros(length(lines), 2);
    x = get(gca,'XLim');
    for k = 1:length(lines) 
        %line
        x1  = [lines(k).point1(1) lines(k).point2(1)];
        y1  = [lines(k).point1(2) lines(k).point2(2)];

        %fit linear polynomial
        p1 = polyfit(x1,y1,1);
        li(k,:) = p1;
        plot(x, p1(1)*x+p1(2), 'LineWidth', 3, 'Color', 'green');
    end

    %calculate intersection
    intersect = [];
    for k = 1:length(li)
        for l=1:length(li)
            if l ~= k
                x_intersect = fzero(@(x) polyval(li(k,:)-li(l,:),x),3);
                y_intersect = polyval(li(k,:), x_intersect);
                intersect = [intersect;[x_intersect y_intersect]];
            end
        end
    end

    %get the max value of most voted point
    BinW=50;BinH=50;
    [N,Xedges,Yedges] = histcounts2(intersect(:,1),intersect(:,2),...
        'BinWidth',[BinW BinH], 'XBinLimits', [0, 320], 'YBinLimits', [0,560]);
    maxPoint = max(N(:));
    [maxRow, maxCol] = find(N == maxPoint);

    %===DRAWING===%
    [img_h, img_w]= size(BW);
    iter_num_x = img_w/BinW; xL = xlim;
    for i = 0:iter_num_x
        line(xL, [BinH*i BinH*i], 'LineWidth', 1, 'Color', 'black');
    end

    iter_num_y = img_h/BinH; yL = ylim;
    for i = 0:iter_num_y
        line([BinW*i BinW*i],yL, 'LineWidth', 1, 'Color', 'black');
    end

    %get start point of the part among the grid
    start = [Xedges(maxRow), Yedges(maxCol)];
    rectangle('Position',[start(1) start(2) BinW BinH], 'FaceColor', [0 1 0 0.3])
    hold on;
end

function second(~)
    img = imread('image2.jpg');
    img = rgb2gray(img);

    %performing canny dection and hough transform
    BW = edge(img, 'canny');
    [H,theta,rho] = hough(BW,'RhoResolution',1 ,'Theta', -67:67);

    %find out hough peak and lines
    P = houghpeaks(H,4,'threshold',ceil(0.5*max(H(:))));
    lines = houghlines(BW,theta,rho,P,'FillGap',5,'MinLength',60);


    %draw lines
    figure, imshow(img), hold on
    li = zeros(length(lines), 2);
    x = get(gca,'XLim');
    for k = 1:length(lines) 
        %line
        x1  = [lines(k).point1(1) lines(k).point2(1)];
        y1  = [lines(k).point1(2) lines(k).point2(2)];

        %fit linear polynomial
        p1 = polyfit(x1,y1,1);
        li(k,:) = p1;
        plot(x, p1(1)*x+p1(2), 'LineWidth', 3, 'Color', 'green');
    end

    %calculate intersection
    intersect = [];
    for k = 1:length(li)
        for l=1:length(li)
            if l ~= k
                x_intersect = fzero(@(x) polyval(li(k,:)-li(l,:),x),3);
                y_intersect = polyval(li(k,:), x_intersect);
                intersect = [intersect;[x_intersect y_intersect]];
            end
        end
    end

    %get the max value of most voted point
    BinW=80;BinH=40;
    [N,Xedges,Yedges] = histcounts2(intersect(:,1),intersect(:,2),...
        'BinWidth',[BinW BinH], 'XBinLimits', [0, 320], 'YBinLimits', [0,560]);
    maxPoint = max(N(:));
    [maxRow, maxCol] = find(N == maxPoint);

    %===DRAWING===%
    [img_h, img_w]= size(BW);
    iter_num_x = img_w/BinW; xL = xlim;
    for i = 0:iter_num_x
        line(xL, [BinH*i BinH*i], 'LineWidth', 1, 'Color', 'black');
    end

    iter_num_y = img_h/BinH; yL = ylim;
    for i = 0:iter_num_y
        line([BinW*i BinW*i],yL, 'LineWidth', 1, 'Color', 'black');
    end

    %get start point of the part among the grid
    start = [Xedges(maxRow), Yedges(maxCol)];
    rectangle('Position',[start(1) start(2) BinW BinH], 'FaceColor', [0 1 0 0.3])
    hold on;
end




