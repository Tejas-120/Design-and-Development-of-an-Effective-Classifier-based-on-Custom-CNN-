%% train_mobilenetv2_multiclass.m
% ---------------------------------------------------------------
% Multiclass Classification — Plain/'Ultra-fast' MobileNetV2
% Transfer learning with only the final layer replaced, no custom
% augmentation, larger batch size and few epochs for fast training.
%
% Expects dataset at: <datasetRoot> (subfolders = class names)
% Output model file: UltraFast_PlainMobileNet_70_15_15.mat
% ---------------------------------------------------------------

clc;
clear;
close all;
rng(1);

%% ================= ENABLE FIGURES =================
set(0,'DefaultFigureVisible','on');

%% ================= DATASET =================
datasetRoot = 'C:\Users\SAMPATH REDDY\Downloads\BRAINCLASSIFICATION\BRAINCLASSIFICATION';

imds = imageDatastore(datasetRoot,...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

% 70–15–15 split
[imdsTrain, imdsTemp] = splitEachLabel(imds,0.7,'randomized');
[imdsVal, imdsTest]   = splitEachLabel(imdsTemp,0.5,'randomized');

disp("Train Count:"); countEachLabel(imdsTrain)
disp("Validation Count:"); countEachLabel(imdsVal)
disp("Test Count:"); countEachLabel(imdsTest)

numClasses = numel(categories(imdsTrain.Labels));

%% ================= LOAD PRETRAINED MOBILENET =================
net = mobilenetv2;
lgraph = layerGraph(net);

%% ================= REPLACE FINAL LAYER =================
newFC = fullyConnectedLayer(numClasses,...
    'Name','Logits',...
    'WeightLearnRateFactor',10,...
    'BiasLearnRateFactor',10);

lgraph = replaceLayer(lgraph,'Logits',newFC);
lgraph = replaceLayer(lgraph,'ClassificationLayer_Logits',...
    classificationLayer('Name','ClassificationLayer_Logits'));

%% ================= RESIZE ONLY (NO AUGMENTATION) =================
inputSize = net.Layers(1).InputSize;

augTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augVal   = augmentedImageDatastore(inputSize(1:2),imdsVal);
augTest  = augmentedImageDatastore(inputSize(1:2),imdsTest);

%% ================= FAST TRAINING OPTIONS =================
options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'MiniBatchSize',64, ...      % Bigger batch = faster
    'MaxEpochs',8, ...           % Very few epochs
    'InitialLearnRate',5e-4, ... % Slightly higher LR for fast convergence
    'ValidationData',augVal, ...
    'ValidationPatience',2, ...  % Early stop fast
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress');

%% ================= TRAIN =================
disp("🔥 Ultra Fast Transfer Learning (MobileNet)...");
trainedNet = trainNetwork(augTrain,lgraph,options);

%% ================= TEST =================
YPred = classify(trainedNet,augTest);
accuracy = mean(YPred == imdsTest.Labels)*100;

fprintf('\n🔥 FINAL TEST ACCURACY = %.2f%%\n',accuracy);

figure;
confusionchart(imdsTest.Labels,YPred);
title('Ultra Fast MobileNet Test Results');

save('UltraFast_PlainMobileNet_70_15_15.mat','trainedNet');

disp("✅ MODEL SAVED");
