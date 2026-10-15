%% train_custom_cnn_resnet18_binary.m
% ---------------------------------------------------------------
% Binary Classification (Tumour / No Tumour) — Custom CNN
% Fine-tunes a pretrained ResNet18 (final FC/softmax/classification
% layers replaced) with data augmentation (rotation, translation,
% reflection). Trains only if no saved model is found, otherwise
% loads the existing model and evaluates it.
%
% Expects dataset at: <datasetRoot>/training and <datasetRoot>/testing
% Output model file: resnet18BrainTumor.mat
% ---------------------------------------------------------------

%% brain_tumor_resnet18.m
clc; clear; close all;

% ================== USER SETTINGS ==================
datasetRoot = 'C:\BRAIN TUMOR PROJECT';   % adjust this path
outSize     = [224 224];                  % ResNet-18 input size
modelFile   = 'resnet18BrainTumor.mat';
% ===================================================

%% 1) Load datasets
imdsTrain = imageDatastore(fullfile(datasetRoot,'training'), ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest  = imageDatastore(fullfile(datasetRoot,'testing'), ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%% 2) Preprocess datasets (resizing + gray2rgb)
imdsTrain.ReadFcn = @(f) preprocessResNet(imread(f), outSize);
imdsTest.ReadFcn  = @(f) preprocessResNet(imread(f), outSize);

disp('Train label counts:'); disp(countEachLabel(imdsTrain));
disp('Test  label counts:'); disp(countEachLabel(imdsTest));

%% 3) Data augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-10 10], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5], ...
    'RandXReflection',true);

augTrain = augmentedImageDatastore(outSize, imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augTest  = augmentedImageDatastore(outSize, imdsTest);

%% 4) Check if trained model exists
if isfile(modelFile)
    fprintf('\n[INFO] Found existing trained model: %s\n', modelFile);
    load(modelFile, 'net');
    fprintf('[INFO] Model loaded successfully. Skipping training.\n');
else
    fprintf('\n[INFO] No trained model found. Starting training...\n');

    %% Load pretrained ResNet-18
    net = resnet18;
    lgraph = layerGraph(net);

    % Replace final layers for classification
    numClasses = numel(unique(imdsTrain.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc_new','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
        softmaxLayer('Name','softmax_new')
        classificationLayer('Name','classoutput_new')];

    lgraph = replaceLayer(lgraph,'fc1000',newLayers(1));
    lgraph = replaceLayer(lgraph,'prob',newLayers(2));
    lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newLayers(3));

    %% Training options
    options = trainingOptions('adam', ...
        'MaxEpochs',10, ...                 % can increase if needed
        'MiniBatchSize',16, ...
        'InitialLearnRate',1e-4, ...
        'ValidationData',augTest, ...
        'ValidationFrequency',30, ...
        'Verbose',false, ...
        'Plots','training-progress');

    %% Train Network
    net = trainNetwork(augTrain, lgraph, options);

    %% Save Trained Model
    save(modelFile, 'net');
    fprintf('\n[INFO] Training complete. Model saved as "%s".\n', modelFile);
end

%% 5) Evaluate Performance
fprintf('\n================ EVALUATING PERFORMANCE ================\n');
[YPred, scores] = classify(net, augTest);
YTest = imdsTest.Labels;

testAccuracy = mean(YPred == YTest) * 100;
fprintf('\n=== TESTING RESULTS ===\n');
fprintf('Total Test Images     : %d\n', numel(YTest));
fprintf('Correct Predictions   : %d\n', sum(YPred == YTest));
fprintf('Incorrect Predictions : %d\n', sum(YPred ~= YTest));
fprintf('--------------------------------------------------\n');
fprintf('ResNet-18 Test Accuracy = %.2f %%\n', testAccuracy);
fprintf('==================================================\n');

classes = unique(YTest);
for i = 1:numel(classes)
    idx = (YTest == classes(i));
    classAcc = mean(YPred(idx) == YTest(idx)) * 100;
    fprintf('Class "%s" Accuracy = %.2f %%\n', string(classes(i)), classAcc);
end

%% 6) Confusion Matrix
figure('Name','Confusion Matrix (ResNet-18)');
confusionchart(YTest, YPred);
title(sprintf('ResNet-18 Confusion Matrix (Accuracy %.2f%%)', testAccuracy));

%% ================= Helper Function =================
function Iout = preprocessResNet(I, outSize)
    % Convert grayscale to RGB if needed
    if size(I,3)==1
        I = cat(3,I,I,I);
    end
    % Resize to network input
    Iout = imresize(I, outSize);
end
