%% train_custom_mobilenet_multiclass.m
% ---------------------------------------------------------------
% Multiclass Classification — Custom MobileNet
% Loads pretrained MobileNetV2, replaces the classification head,
% and inserts a dropout layer before the new fully-connected layer.
% Uses moderate data augmentation (rotation, reflection, scale,
% translation).
%
% Expects dataset at: <datasetRoot> (subfolders = class names)
% Output model file: FINAL_CustomMobileNetV2.mat
% ---------------------------------------------------------------

clc;
clear;
close all;

%% ================= DATASET =================
datasetRoot = 'C:\BRAINCLASSIFICATION';

imds = imageDatastore(datasetRoot,...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

% 80% train, 20% validation
[imdsTrain, imdsTest] = splitEachLabel(imds,0.8,'randomized');

disp("Training Data:");
countEachLabel(imdsTrain)

disp("Validation Data:");
countEachLabel(imdsTest)

numClasses = numel(categories(imdsTrain.Labels));

%% ================= LOAD MOBILENETV2 =================
net = mobilenetv2;
lgraph = layerGraph(net);

%% ================= CUSTOM CLASSIFIER =================
dropLayer = dropoutLayer(0.5,'Name','dropout');

fcLayer = fullyConnectedLayer(numClasses,...
    'Name','new_fc',...
    'WeightLearnRateFactor',20,...
    'BiasLearnRateFactor',20);

classLayer = classificationLayer('Name','new_class');

% Disconnect original
lgraph = disconnectLayers(lgraph,'global_average_pooling2d_1','Logits');

% Replace layers
lgraph = replaceLayer(lgraph,'Logits',fcLayer);
lgraph = replaceLayer(lgraph,'ClassificationLayer_Logits',classLayer);

% Add dropout
lgraph = addLayers(lgraph,dropLayer);
lgraph = connectLayers(lgraph,'global_average_pooling2d_1','dropout');
lgraph = connectLayers(lgraph,'dropout','new_fc');

%% ================= AUGMENTATION =================
inputSize = net.Layers(1).InputSize;

augmenter = imageDataAugmenter(...
    'RandRotation',[-10 10],...
    'RandXReflection',true,...
    'RandScale',[0.95 1.05],...
    'RandXTranslation',[-5 5],...
    'RandYTranslation',[-5 5]);

augTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,...
    'DataAugmentation',augmenter);

augVal = augmentedImageDatastore(inputSize(1:2),imdsTest);

%% ================= TRAIN OPTIONS =================
options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',10, ...
    'ValidationData',augVal, ...
    'ValidationPatience',7, ...
    'Shuffle','every-epoch', ...
    'L2Regularization',0.0005, ...
    'Plots','training-progress', ...
    'Verbose',true);

%% ================= TRAIN =================
disp("🔥 Training MobileNetV2 Model...");
trainedNet = trainNetwork(augTrain,lgraph,options);

%% ================= SAVE MODEL =================
classNames = categories(imdsTrain.Labels);

save('C:\BRAINCLASSIFICATION\FINAL_CustomMobileNetV2.mat',...
    'trainedNet','classNames','inputSize');

disp("✅ TRAINING COMPLETED & MODEL SAVED IN C:\\BRAINCLASSIFICATION");

