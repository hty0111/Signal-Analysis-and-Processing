%%
clear all;
imagesDir = 'datasets';
addpath(genpath('myfunction'));
trainImagesDir = fullfile(imagesDir,'train','images');
validateImagesDir = fullfile(imagesDir,'validate','images');
trainMatsDir = fullfile(imagesDir,'train','matsets','train.mat');
validateMatsDir = fullfile(imagesDir,'validate','matsets','validate.mat');
trainLableMatsDir = fullfile(imagesDir,'train','matsets','trainLable.mat');
validateLableMatsDir = fullfile(imagesDir,'validate','matsets','validateLable.mat');
load(trainMatsDir);
load(validateMatsDir);
load(trainLableMatsDir);
load(validateLableMatsDir);
%%
for i = 1 : size(trainMats)
    trainMats{i} = trainMats{i}';
end
for i = 1 : size(validateMats)
    validateMats{i} = validateMats{i}';
end
%%
trainLableMats = cell2mat(trainLableMats);
validateLableMats = cell2mat(validateLableMats);
%%
features_number = 1;
models = [ ...
    sequenceInputLayer(features_number)
    bilstmLayer(1024, "OutputMode", "last")
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];
%%
miniBatchSize = 64;
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'Shuffle','every-epoch', ...
    "ValidationData",{validateMats,validateLableMats}, ...
    'ValidationFrequency', 100, ...
    'ValidationPatience', 10, ...
    'MiniBatchSize',miniBatchSize, ...
    'Plots','training-progress');
net = trainNetwork(trainMats, trainLableMats, models, options);
%%
save('datasets/trainedModel/LSTMNet.mat', 'net');