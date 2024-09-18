
%% Code to create simulated vessels

% June 2020
% Author: Berkan and Luis

%% clean everything

close all;
clear all;
clc;

%% create meshgrid

height          = 512;
width           = 512;
Nx              = (1:height);
Ny              = (1:width);
[meshX, meshY]  = meshgrid(Nx, Ny);

%% ellipse parameters

numImages       = 15;
numEllipses     = randi([6 10], 1, numImages);
numCircles      = randi([6 10], 1, numImages);

%% create ellipses

% ellipse limits
maxEllipseX     = 240;
minEllipseX     = 180;
maxEllipseY     = 24;
minEllipseY     = 24;

% circle limits
maxCircleX      = 20;
minCircleX      = 12;
maxCircleY      = 20;
minCircleY      = 12;

% frangi filter parameters
sigmas          = 20;
beta            = 1;
cc              = 100;

imageShapes     = zeros(height, width, numImages);
imageVessels    = zeros(height, width, numImages);

savePath        = 'simulatedData';

for imageInd = 1:size(numEllipses,2)
    
    % define ellipse dimensions
    heightEllipse   = unifrnd(minEllipseX, maxEllipseX, [1 numEllipses(1,imageInd)]);
    widthEllipse    = unifrnd(minEllipseY, maxEllipseY, [1 numEllipses(1,imageInd)]);
    centerEllipseX  = unidrnd(height,[1 numEllipses(1,imageInd)]);
    centerEllipseY  = unidrnd(width,[1 numEllipses(1,imageInd)]);
    phiEllipses     = unifrnd(0,2*pi,[1 numEllipses(1,imageInd)]);
    
    % define circle dimensions
    heightCircle    = unifrnd(minCircleX, maxCircleX, [1 numCircles(1,imageInd)]);
    widthCircle     = unifrnd(minCircleY, maxCircleY, [1 numCircles(1,imageInd)]);
    centerCircleX   = unidrnd(height,[1 numCircles(1,imageInd)]);
    centerCircleY   = unidrnd(width,[1 numCircles(1,imageInd)]);
    phiCircles      = unifrnd(0,2*pi,[1 numCircles(1,imageInd)]);
    
    % define a matrix to store images
    compoundImage   = zeros(height, width);
    
    % create ellipses
    for ellipseInd = 1:numEllipses(imageInd)
        
        vecEllipseX     = 1*cos(phiEllipses(ellipseInd));
        vecEllipseY     = 1*sin(phiEllipses(ellipseInd));
        
        posEllipseX     = meshX - centerEllipseX(ellipseInd);
        posEllipseY     = meshY - centerEllipseY(ellipseInd);
        
        sqrtEllipse     = sqrt(posEllipseX.^2 + posEllipseY.^2);
        distEllipse     = posEllipseX.*vecEllipseX + posEllipseY.*vecEllipseY;
        sqrtDistEllipse = sqrt((sqrtEllipse).^2 - (distEllipse).^2);
        
        singleImage     = 1 - ((distEllipse/heightEllipse(ellipseInd)).^2 + (sqrtDistEllipse/widthEllipse(ellipseInd)).^2);
        singleImage(((distEllipse/heightEllipse(ellipseInd)).^2 + (sqrtDistEllipse/widthEllipse(ellipseInd)).^2) > 1 ) = 0;
        compoundImage   = compoundImage + singleImage;
        
    end
    
    % create circles
    for circleInd = 1:numCircles(1,imageInd)
        
        vecCircleX     = 1*cos(phiCircles(circleInd));
        vecCircleY     = 1*sin(phiCircles(circleInd));
        
        posCircleX     = meshX - centerCircleX(circleInd);
        posCircleY     = meshY - centerCircleY(circleInd);
        
        sqrtCircle     = sqrt(posCircleX.^2 + posCircleY.^2);
        distCircle     = posCircleX.*vecCircleX + posCircleY.*vecCircleY;
        sqrtDistCircle = sqrt((sqrtCircle).^2 - (distCircle).^2);
        
        singleImage     = 1 - ((distCircle/heightCircle(circleInd)).^2 + (sqrtDistCircle/widthCircle(circleInd)).^2);
        singleImage(((distCircle/heightCircle(circleInd)).^2 + (sqrtDistCircle/widthCircle(circleInd)).^2) > 1 ) = 0;
        compoundImage   = compoundImage + singleImage;
        
    end
    
    imageShapes(:,:,imageInd) = min(compoundImage,1);

    generatedImage  = imageShapes(:,:,imageInd);

%     generatedImage  = image_ellipses(:,:,4);
    
    histImage       = adapthisteq(mat2gray(generatedImage),'ClipLimit',0.005,'Range','full','Distribution','rayleigh','Alpha',0.5);
    
    grayImage       = 256.*mat2gray(histImage);
    
    %% frangi filter
    for sigmaInd = 1:length(sigmas)
        
        [Dxx,Dxy,Dyy]   = Hessian2D(grayImage, sigmas(sigmaInd));
        
        % Correct for scale
        Dxx = (sigmas(sigmaInd)^2)*Dxx;
        Dxy = (sigmas(sigmaInd)^2)*Dxy;
        Dyy = (sigmas(sigmaInd)^2)*Dyy;
        
        % Calculate eigenvalues and vectors and sort the eigenvalues
        [Lambda1,Lambda2]=eig2image(Dxx,Dxy,Dyy);
        
        % Compute the similarity measures that make sense in 2D
        % and the corresponding vesselness measure
        Rb  = abs(Lambda1)./abs(Lambda2);
        S   = sqrt(Lambda1.^2+Lambda2.^2);
        Vesselness = exp(-(Rb.^2)/(2*beta^2)).*(1-exp(-(S.^2)/(2*cc^2)));
        
        % Set to Vesselness to 0 if Lambda2 is positive
        Vesselness = Vesselness.*double(Lambda2<=0);
        
        % AllScale(:,:,sigmaInd) = (Vesselness(:,:)./max(Vesselness(:))).*(length(sigmas)+1-sigmaInd);
        % AllScale(:,:,sigmaInd) = adapthisteq(mat2gray(AllScale(:,:,sigmaInd)),'clipLimit',0.01,'Distribution','rayleigh','Alpha',0.5);
        % AllScale_bm(:,:,sigmaInd) = Vesselness>0.01*max(Vesselness(:));
        
    end
    
    imageVessels(:,:,imageInd) = Vesselness;
    
%     % plot generated image befre Frangi
%     figure;
%     imagesc(generatedImage);
%     axis square;
%     colormap gray;
%     axis off;
%     axis tight;
%     axis equal;
%     
%     % plot image after Frangi
%     figure;
%     imagesc(Vesselness);
%     axis square;
%     colormap gray;
%     axis off;
%     axis tight;
%     axis equal;
    
    % save generated image
    Vesselness = ((Vesselness - min(Vesselness(:)))/(max(Vesselness(:)) - min(Vesselness(:))))*255;
    saveName = fullfile(savePath, strcat('image_', int2str(imageInd),'.png'));
    imwrite(uint8(Vesselness), saveName);

    
end



