%% test_custom_cnn_resnet18_binary.m
% ---------------------------------------------------------------
% Loads a previously trained fine-tuned ResNet18 model and
% evaluates it on a binary (Tumour / No Tumour) test set.
%
% NOTE: This script loads 'bestModel_resnet18_final.mat', while the
% training script (train_custom_cnn_resnet18_binary.m) saves its
% model as 'resnet18BrainTumor.mat'. Make sure the filenames match
% before running this script standalone, e.g. rename the saved file
% or update 'modelPath' below.
% ---------------------------------------------------------------

%% 🚀 Brain Tumor Classification Testing Script (Fine-tuned ResNet18)
clc; clear; close all;

disp("🚀 Loading saved Brain Tumor Classification model...");

%% Step 1: Load saved model
modelPath = fullfile(pwd, 'bestModel_resnet18_final.mat');
if isfile(modelPath)
    loadedData = load(modelPath);
    netTransfer = loadedData.netTransfer;
    disp("✅ Model loaded successfully.");
else
    error("❌ Model file not found: " + modelPath);
end

%% Step 2: Load test dataset
testDir = "C:\BRAIN TUMOR PROJECT"; % update path if needed
imdsTest = imageDatastore(testDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% If you have a dedicated test split, subset can be used:
% imdsTest = subset(imdsTest, testIndices);

%% Step 3: Prepare test images
inputSize = netTransfer.Layers(1).InputSize; % typically [224 224 3]
augTest = augmentedImageDatastore(inputSize, imdsTest);

%% Step 4: Classify test images
[YPred, scores] = classify(netTransfer, augTest);
YTest = imdsTest.Labels;

%% Step 5: Compute accuracy
accuracy = mean(YPred == YTest) * 100;
fprintf('\n✅ Test Accuracy: %.2f%%\n', accuracy);
numTestImages = numel(imdsTest.Files)
numPred = numel(YPred)
numTrue = numel(YTest)

%% Step 6: Confusion Matrix
figure;
plotconfusion(YTest, YPred);
title(sprintf('Confusion Matrix (%.2f%% Accuracy)', accuracy));

%% Step 7: Display example predictions
figure;
numExamples = 8;
perm = randperm(numel(imdsTest.Files), numExamples);
for i = 1:numExamples
    subplot(2,4,i);
    img = readimage(imdsTest, perm(i));
    imshow(img);
    label = YPred(perm(i));
    title(string(label));
end

