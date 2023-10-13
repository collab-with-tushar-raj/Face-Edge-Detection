function face_edge_detection()
    showImage('photo.png');
    addSignatureOnPhoto('signature.png');
    detectEdgeUsingRGB();
    detectEdgeUsingHSV();
    detectEdgeUsingKMeans();
    detectEdgeUsingSVM();
end

function showImage(fileName)
    global myPhoto;
    myPhoto = imread(fileName);
    figure('Name', 'Face Edge Detection', 'NumberTitle', 'off'),
    subplot(5,14,1), imshow(uint8(myPhoto)); title('Original (RGB)');
end

function addSignatureOnPhoto(fileName)
    global myPhoto;
    mySign = imread(fileName);
    mySign_binary = mySign < 50;
    mySign_binary = mySign_binary * 255;    
    [row, col, ~] = size(mySign_binary);
    row_offset = 1;
    col_offset = 1;
    x0 = col_offset;
    x1 = col;
    y0 = row_offset;
    y1 = row;
    myPhoto_copy = myPhoto;
    myPhoto_copy(y0:y1,x0:x1,1:3) = myPhoto_copy(y0:y1,x0:x1,1:3) - uint8(mySign_binary(:,:,1:3));
    subplot(5,14,2), imshow(uint8(mySign)), title('Only Sign')
    subplot(5,14,3), imshow(uint8(myPhoto_copy)), title('With Sign');
end

function detectEdgeUsingRGB()
    global myPhoto;
    [~, ~, dim] = size(myPhoto);
    if(dim > 1)
        grayPhoto = rgb2gray(myPhoto);
    else
        grayPhoto = myPhoto;
    end
    subplot(5,14,15), imshow(grayPhoto), title('Grey');
    threshold_img = applyThreshold(grayPhoto, 130);
    subplot(5,14,16), imshow(threshold_img), title('Threshold (130)');  
    segmentedFace = segmentFace(myPhoto);
    subplot(5,14,17), imshow(segmentedFace), title('Face only');  
    detectAndDisplayEdges(rgb2gray(segmentedFace), 18, 19, 20, 21, 22, 23, 24);
end

function detectEdgeUsingHSV()
    global myPhoto;
    hsvPhoto = rgb2hsv(myPhoto);
    subplot(5,14,29), imshow(hsvPhoto), title('HSV'); 

    % hue, saturation and value thresholds found using 'improfile'
    ht = 0.9;
    st = 0.5;
    vt = 0.07;
    hsvThresholdImg = hsvPhoto(:,:,3) < ht & hsvPhoto(:,:,2) < st & hsvPhoto(:,:,1) < vt;
    hsvThresholdImg_255 = hsvThresholdImg * 255; % converting [0-1] to [0-255]
    subplot(5,14,30), imshow(hsvThresholdImg_255); title('Threshold(HSV)');

    % segment face using mask
    imFace(:,:,1) = myPhoto(:,:,1) .* uint8(hsvThresholdImg);
    imFace(:,:,2) = myPhoto(:,:,2) .* uint8(hsvThresholdImg);
    imFace(:,:,3) = myPhoto(:,:,3) .* uint8(hsvThresholdImg);
    subplot(5,14,31), imshow(imFace); title('Mask face');

    % remove noise below the chin using 'improfile'
    imFace(990:end, :, :) = 0;
    subplot(5,14,32), imshow(imFace); title('Face only');

    detectAndDisplayEdges(rgb2gray(imFace), 33, 34, 35, 36, 37, 38, 39);
end

function detectEdgeUsingKMeans()
    global myPhoto;
    hsvPhoto = rgb2hsv(myPhoto);
    hsvPhoto8Bit = uint8(hsvPhoto);

    % segment the image into k=2,3,5,7,10 regions
    [L, ~] = imsegkmeans(hsvPhoto8Bit, 2); % 2 regions
    B = labeloverlay(hsvPhoto8Bit, L);
    subplot(5,14,43), imshow(B); title('k=2');

    [L, ~] = imsegkmeans(hsvPhoto8Bit, 3); % 3 regions
    B = labeloverlay(hsvPhoto8Bit, L);
    subplot(5,14,44), imshow(B); title('k=3');

    [L, ~] = imsegkmeans(hsvPhoto8Bit, 5); % 5 regions
    B = labeloverlay(hsvPhoto8Bit, L);
    subplot(5,14,45), imshow(B); title('k=5');

    [L, ~] = imsegkmeans(hsvPhoto8Bit, 7); % 7 regions
    B = labeloverlay(hsvPhoto8Bit, L);
    subplot(5,14,46), imshow(B); title('k=7');

    [L, ~] = imsegkmeans(hsvPhoto8Bit, 10); % 10 regions
    B = labeloverlay(hsvPhoto8Bit, L);
    subplot(5,14,47), imshow(B); title('k=10');

    class_1_img = (L==1); % take only the face class

    hsv_class_1_img = double(hsvPhoto8Bit) .* double(class_1_img);
    subplot(5,14,48), imshow(mat2gray(hsv_class_1_img)); title('Class 1(HSV)');

    rgb_class_1_img = double(myPhoto) .* double(class_1_img);
    subplot(5,14,49), imshow(mat2gray(rgb_class_1_img)); title('Class 1(RGB)');

    segmentedFace = segmentFace(rgb_class_1_img);
    detectAndDisplayEdges(rgb2gray(uint8(segmentedFace)), 50, 51, 52, 53, ...
        54, 55, 56);
end

function detectEdgeUsingSVM()
    global myPhoto;
    segmentedFace = segmentFace(myPhoto);
    % resizing to make the process faster
    img_resized = imresize(segmentedFace, 0.25);
    [m, n, ~] = size(img_resized); 
    hsvPhoto = rgb2hsv(img_resized);
    hs = [
            reshape(hsvPhoto(:,:,1), 1,[]); 
            reshape(hsvPhoto(:,:,2),1,[]); 
            reshape(hsvPhoto(:,:,3),1,[])
         ];
    X = hs';
    Y = (X(:,1) < 0.2);
    svm = fitcsvm(X, Y);
    cv = crossval(svm);
    loss = kfoldLoss(cv);
    disp(loss); % to undestand the accuracy
    [~, score] = kfoldPredict(cv);
    predX = (score(:,2) > 0);
    predX = predX .* X;

    % reshape back to image resolution
    im_pred(:,:,1) = reshape(predX(:,1), m, n);
    im_pred(:,:,2) = reshape(predX(:,2), m, n);
    im_pred(:,:,3) = reshape(predX(:,3), m, n);

    im_pred_rgb = hsv2rgb(im_pred);
    subplot(5,14,57), imshow(hsvPhoto), title('HSV');
    subplot(5,14,58), imshow(im_pred), title('Pred (HSV)');
    subplot(5,14,59), imshow(im_pred_rgb), title('Pred (RGB)');    
    detectAndDisplayEdges(rgb2gray(im_pred_rgb), 60, 61, 62, 63, 64, 65, 66);
    withoutNoise = removeFaceNoise(rgb2gray(im_pred_rgb));
    subplot(5,14,67), imshow(withoutNoise), title('Without noise');
end

function threshold_img = applyThreshold(img, threshold)
    [row, col, dim] = size(img);
    threshold_img = zeros(row,col,dim);
    index = find(img > threshold);
    threshold_img(index) = img(index);
end

function segmentedFace = segmentFace(im)
    segmentedFace = im;
    % remove everything below the chin
    segmentedFace(1050:end, :, :) = 0; 
    % remove everything above the head
    segmentedFace(1:180,:,:) = 0;
    % remove everything to the left of the face
    segmentedFace(1:end,1:300,:) = 0;
    % remove everything to the right of the face
    segmentedFace(1:end,890:end,:) = 0;
    % remove right shoulder starting
    segmentedFace(950:end, 800:end, :) = 0;
    % remove left shoulder starting
    segmentedFace(950:end, 1:400, :) = 0;
end

function withoutNoise = removeFaceNoise(im)
    im(50:200,100:200) = 0; % remove eyes and nose
    im(200:240,120:180) = 0; % remove lips
    customEdgeImg = myEdge(im);
    [row, col, ~] = size(customEdgeImg);
    withoutNoise = zeros(row, col, 3);
    withoutNoise(:,:,2) = customEdgeImg(:,:);
end

function detectAndDisplayEdges(img, si, ri, ci, li, zi, pi, mi)
    sobelImg = edge(img);
    robertsImg = edge(img, 'roberts');
    cannyImg = edge(img, 'canny');
    logImg = edge(img, 'log');
    zeroCrossImg = edge(img, 'zerocross');
    prewittImg = edge(img, 'prewitt');
    customEdgeImg = myEdge(img);
    colorEdges(sobelImg, robertsImg, cannyImg, logImg, zeroCrossImg, ...
        prewittImg, customEdgeImg, si, ri, ci, li, zi, pi, mi);
end

function customEdgeImg = myEdge(img)
    h1 = [0, 1, 2; -1, 0, 1; -2, -1, 0];
    h2 = [2, 1, 0; 1, 0, -1; 0, -1, -2];
    customEdgeImg = conv2(img, h1) + conv2(img, h2);
end

function colorEdges(img_sobel, img_roberts, img_canny, img_log, ...
    img_zerocross, img_prewitt, customEdgeImg, si, ri, ci, li, zi, pi, mi)
    [row,col] = size(img_sobel);
    colouredSobelEdge = zeros(row, col, 3);
    colouredSobelEdge(:,:,2) = img_sobel(:,:);

    [row,col] = size(img_roberts);
    colouredRobertsEdge = zeros(row, col, 3);
    colouredRobertsEdge(:,:,2) = img_roberts(:,:);

    [row,col] = size(img_canny);
    colouredCannyEdge = zeros(row, col, 3);
    colouredCannyEdge(:,:,2) = img_canny(:,:);

    [row,col] = size(img_log);
    colouredLogEdge = zeros(row, col, 3);
    colouredLogEdge(:,:,2) = img_log(:,:);

    [row,col] = size(img_zerocross);
    colouredZeroCrossEdge = zeros(row, col, 3);
    colouredZeroCrossEdge(:,:,2) = img_zerocross(:,:);

    [row, col] = size(img_prewitt);
    colouredPrewittEdge = zeros(row, col, 3);
    colouredPrewittEdge(:,:,2) = img_prewitt(:,:);

    [row, col] = size(customEdgeImg);
    colouredCustomEdge = zeros(row, col, 3);
    colouredCustomEdge(:,:,2) = customEdgeImg(:,:);

    displayEdges(colouredSobelEdge, colouredRobertsEdge, colouredCannyEdge, ...
        colouredLogEdge, colouredZeroCrossEdge, colouredPrewittEdge, ...
        colouredCustomEdge, si, ri, ci, li, zi, pi, mi)
end

function displayEdges(sobelImg, robertsImg, cannyImg, logImg, zeroCrossImg, ...
    prewittImg, customEdge, si, ri, ci, li, zi, pi, mi)    
    subplot(5,14,si), imshow(sobelImg), title('Sobel');
    subplot(5,14,ri), imshow(robertsImg), title('Roberts');
    subplot(5,14,ci), imshow(cannyImg), title('Canny');
    subplot(5,14,li), imshow(logImg), title('Log');
    subplot(5,14,zi), imshow(zeroCrossImg), title('Zero Cross');
    subplot(5,14,pi), imshow(prewittImg), title('Prewitt');   
    subplot(5,14,mi), imshow(customEdge), title('My method');
end
    