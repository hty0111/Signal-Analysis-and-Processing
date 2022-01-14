imagesDir = 'datasets';
% url = 'http://www-i6.informatik.rwth-aachen.de/imageclef/resources/iaprtc12.tgz';
% downloadIAPRTC12Data(url,imagesDir);
untar('datasets/iaprtc12.tgz', imagesDir);
%%
imagesDir = 'datasets';
addpath(genpath('myfunction'));
trainImagesDir = fullfile(imagesDir,'iaprtc12','images','02');
exts = {'.jpg','.bmp','.png'};
pristineImages = imageDatastore(trainImagesDir,'FileExtensions',exts);
numel(pristineImages.Files);
%%
upsampledDirName = [trainImagesDir filesep 'upsampledImages'];
residualDirName = [trainImagesDir filesep 'residualImages'];
scaleFactors = [2 3 4];
mycreateVDSRT(pristineImages,scaleFactors,upsampledDirName,residualDirName);
%%
upsampledImages = imageDatastore(upsampledDirName,'FileExtensions','.mat','ReadFcn',@matRead);
residualImages = imageDatastore(residualDirName,'FileExtensions','.mat','ReadFcn',@matRead);
augmenter = imageDataAugmenter( ...
    'RandRotation',@()randi([0,1],1)*90, ...
    'RandXReflection',true);
patchSize = [41 41];
patchesPerImage = 64;
dsTrain = randomPatchExtractionDatastore(upsampledImages,residualImages,patchSize, ...
    "DataAugmentation",augmenter,"PatchesPerImage",patchesPerImage);
inputBatch = preview(dsTrain);
disp(inputBatch)
%%
networkDepth = 20;
firstLayer = imageInputLayer([41 41 1],'Name','InputLayer','Normalization','none');
convLayer = convolution2dLayer(3,64,'Padding',1, ...
    'WeightsInitializer','he','BiasInitializer','zeros','Name','Conv1');
relLayer = reluLayer('Name','ReLU1');
middleLayers = [convLayer relLayer];
for layerNumber = 2:networkDepth-1
    convLayer = convolution2dLayer(3,64,'Padding',[1 1], ...
        'WeightsInitializer','he','BiasInitializer','zeros', ...
        'Name',['Conv' num2str(layerNumber)]);
    
    relLayer = reluLayer('Name',['ReLU' num2str(layerNumber)]);
    middleLayers = [middleLayers convLayer relLayer];    
end
convLayer = convolution2dLayer(3,1,'Padding',[1 1], ...
    'WeightsInitializer','he','BiasInitializer','zeros', ...
    'NumChannels',64,'Name',['Conv' num2str(networkDepth)]);
finalLayers = [convLayer regressionLayer('Name','FinalRegressionLayer')];
layers = [firstLayer middleLayers finalLayers];
%%
maxEpochs = 100;
epochIntervals = 1;
initLearningRate = 0.1;
learningRateFactor = 0.1;
l2reg = 0.0001;
miniBatchSize = 64;
options = trainingOptions('sgdm', ...
    'Momentum',0.9, ...
    'InitialLearnRate',initLearningRate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',10, ...
    'LearnRateDropFactor',learningRateFactor, ...
    'L2Regularization',l2reg, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThresholdMethod','l2norm', ...
    'GradientThreshold',0.01, ...
    'Plots','training-progress', ...
    'Verbose',false);
%%
doTraining = false;
if doTraining
    net = trainNetwork(dsTrain,layers,options);
    modelDateTime = string(datetime('now','Format',"yyyy-MM-dd-HH-mm-ss"));
    save('datasets/trainedModel/myTrainedNet','net');
else
    load('datasets/trainedModel/trainedVDSR-Epoch-100-ScaleFactors-234.mat');
end
%%
clear all;
originalImage = imread('original.jpg');
originalImage = im2double(originalImage);
load('myTrainedNet.mat');
%%
% Read image
originalImage_R = originalImage(:,:,1);                                     %拆分为R、G、B三个矩阵
originalImage_G = originalImage(:,:,2);
originalImage_B = originalImage(:,:,3);
originalImages = {originalImage_R,originalImage_G,originalImage_B};         %重新组合成三维矩阵，方便直接输出


% Simulate image
% originalImage=zeros(200,200);
% for mr=15:70
%     for nc=20:90
%         originalImage(mr,nc)=1;
%     end
% end

% get the rows and colums
[mrow,ncol]=size(originalImages{1,1});                                              %获取图像数据的横纵坐标
fftImage_R=fftshift(myfft2(originalImages{1,1}));                                   %对图像数据进行二维傅里叶变换,并利用fftshift函数进行频谱搬移
fftImage_G=fftshift(myfft2(originalImages{1,2})); 
fftImage_B=fftshift(myfft2(originalImages{1,3})); 
fftImage = cat(3,fftImage_R,fftImage_G,fftImage_B);                                 %傅里叶变换后重新组合成三维矩阵

% disk blur
r=5;
h1=fspecial('disk',r);                                                      % 模糊核半径为3
[m,n]=size(h1);
h2=padarray(h1,[mrow-m,ncol-n],'post');                                     % 填充后的模糊核
h3=fftshift(myfft2(h2));                                                    % 模糊核频域
fftBI=fftImage.*h3;                                                         % 频域乘积
blurImage_R=real(myifft2(ifftshift(fftBI(:,:,1))));
blurImage_G=real(myifft2(ifftshift(fftBI(:,:,2))));
blurImage_B=real(myifft2(ifftshift(fftBI(:,:,3))));
blurImage=cat(3,blurImage_R,blurImage_G,blurImage_B);
save('blurImage.mat', 'blurImage');

figure(1);
imshow(originalImage);
figure(2);
imshow(blurImage);
ImageYCbCr = rgb2ycbcr(blurImage);
Iy = ImageYCbCr(:,:,1);
Icb = ImageYCbCr(:,:,2);
Icr = ImageYCbCr(:,:,3);
Iresidual = activations(net,Iy,41);
Iresidual = double(Iresidual);
figure(3);
imshow(Iresidual);
Isr = Iy + Iresidual;
restoration = ycbcr2rgb(cat(3,Isr,Icb,Icr));
figure(4);
imshow(restoration);
title('High-Resolution Image Obtained Using VDSR');
PEAKSNR = psnr(restoration, originalImage);                                  % 计算PSNR
SSIM = ssim(restoration, originalImage);
