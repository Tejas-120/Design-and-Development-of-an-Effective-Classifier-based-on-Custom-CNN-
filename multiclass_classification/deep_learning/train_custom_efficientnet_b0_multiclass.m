%% train_custom_efficientnet_b0_multiclass.m
% ---------------------------------------------------------------
% Multiclass Classification — Custom EfficientNet-B0
% Loads pretrained EfficientNet-B0, replaces the final FC and
% classification layers, applies augmentation (rotation, reflection,
% scale, translation), and trains with a piecewise learning-rate
% schedule + L2 regularization.
%
% Expects dataset at: <datasetRoot> (subfolders = class names)
% Output model file: EfficientNetB0_70_15_15_Model.mat
% (also saves the exact imdsTrain/imdsValidation/imdsTest split)
% ---------------------------------------------------------------

clc;
clear;
close all;
rng(1);   % Reproducible split

%% ================= DATASET =================
datasetRoot = 'C:\Users\SAMPATH REDDY\Downloads\BRAINCLASSIFICATION\BRAINCLASSIFICATION';

imds = imageDatastore(datasetRoot,...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

% -------- 70% training --------
[imdsTrain, imdsTemp] = splitEachLabel(imds,0.7,'randomized');

% -------- Remaining 30% split into 15% val + 15% test --------
[imdsValidation, imdsTest] = splitEachLabel(imdsTemp,0.5,'randomized');

disp("Training Data:");
countEachLabel(imdsTrain)

disp("Validation Data:");
countEachLabel(imdsValidation)

disp("Test Data:");
countEachLabel(imdsTest)

numClasses = numel(categories(imdsTrain.Labels));

%% ================= LOAD EFFICIENTNET-B0 =================
net = efficientnetb0;
lgraph = layerGraph(net);

%% ================= MODIFY FINAL LAYERS =================
oldFC = lgraph.Layers(end-2).Name;
oldClass = lgraph.Layers(end).Name;

newFC = fullyConnectedLayer(numClasses,...
    'Name','new_fc',...
    'WeightLearnRateFactor',20,...
    'BiasLearnRateFactor',20);

lgraph = replaceLayer(lgraph,oldFC,newFC);

newClass = classificationLayer('Name','new_class');
lgraph = replaceLayer(lgraph,oldClass,newClass);

%% ================= DATA AUGMENTATION =================
inputSize = net.Layers(1).InputSize;

augmenter = imageDataAugmenter(...
    'RandRotation',[-15 15],...
    'RandXReflection',true,...
    'RandScale',[0.9 1.1],...
    'RandXTranslation',[-10 10],...
    'RandYTranslation',[-10 10]);

augTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,...
    'DataAugmentation',augmenter);

augVal  = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

%% ================= TRAIN OPTIONS =================
options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augVal, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',7, ...      % early stopping
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',10, ...
    'L2Regularization',0.0005, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',true);

%% ================= TRAIN =================
disp("🔥 Training EfficientNet-B0 (70-15-15 Split)...");
trainedNet = trainNetwork(augTrain,lgraph,options);

%% ================= FINAL TEST =================
YPred = classify(trainedNet,augTest);
testAccuracy = mean(YPred == imdsTest.Labels)*100;

fprintf('\n🔥 FINAL TEST ACCURACY = %.2f%%\n',testAccuracy);

figure;
confusionchart(imdsTest.Labels,YPred);
title('Final Test Confusion Matrix');

%% ================= SAVE MODEL =================
classNames = categories(imdsTrain.Labels);
save('EfficientNetB0_70_15_15_Model.mat',...
    'trainedNet','classNames','inputSize',...
    'imdsTrain','imdsValidation','imdsTest');

disp("✅ MODEL + SPLIT SAVED");
