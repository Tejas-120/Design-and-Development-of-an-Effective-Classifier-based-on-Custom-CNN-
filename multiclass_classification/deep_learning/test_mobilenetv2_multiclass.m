%% test_mobilenetv2_multiclass.m
% ---------------------------------------------------------------
% Loads the trained plain MobileNetV2 model, reproduces the
% 70/15/15 split, evaluates on the test split, and reports
% micro-averaged accuracy/precision/recall/F1.
% ---------------------------------------------------------------

clc;
clear;
close all;

%% ================= LOAD MODEL =================
load('UltraFast_PlainMobileNet_70_15_15.mat');  
% loads: trainedNet

disp("✅ Model loaded successfully");

%% ================= GET INPUT SIZE =================
inputSize = trainedNet.Layers(1).InputSize;

%% ================= LOAD DATASET =================
datasetRoot = 'C:\BRAINCLASSIFICATION';

imds = imageDatastore(datasetRoot,...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

%% ================= DATASET COUNT =================
disp("======================================");
disp("📊 FULL DATASET COUNT");
disp("======================================");

totalImages = numel(imds.Files);
fprintf("Total Images: %d\n\n", totalImages);

countEachLabel(imds);

%% ================= SPLIT (70-15-15) =================
rng(1);  % must match training

[imdsTrain, imdsTemp] = splitEachLabel(imds,0.7,'randomized');
[imdsValidation, imdsTest] = splitEachLabel(imdsTemp,0.5,'randomized');

%% ================= SPLIT COUNT =================
disp("======================================");
disp("📊 TRAIN DATA COUNT (70%)");
disp("======================================");
countEachLabel(imdsTrain);

disp("======================================");
disp("📊 VALIDATION DATA COUNT (15%)");
disp("======================================");
countEachLabel(imdsValidation);

disp("======================================");
disp("📊 TEST DATA COUNT (15%)");
disp("======================================");
countEachLabel(imdsTest);

%% ================= RESIZE =================
augTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

%% ================= PREDICT =================
disp("🔥 Testing MobileNet model...");

YPred = classify(trainedNet,augTest);
YTrue = imdsTest.Labels;

%% ================= CONFUSION MATRIX =================
figure;
confusionchart(YTrue,YPred);
title('Test Confusion Matrix');

%% ================= OVERALL METRICS =================
disp("======================================");
disp("📊 OVERALL PERFORMANCE");
disp("======================================");

confMat = confusionmat(YTrue, YPred);

% Total samples
totalSamples = sum(confMat(:));

% True Positives (correct predictions)
TP = sum(diag(confMat));

% False Positives + False Negatives (all wrong predictions)
FP = totalSamples - TP;
FN = FP;

% Accuracy
accuracy = TP / totalSamples;

% Precision, Recall, F1 (micro-average)
precision = TP / (TP + FP + eps);
recall    = TP / (TP + FN + eps);
f1score   = 2 * (precision * recall) / (precision + recall + eps);

%% ================= PRINT RESULTS =================
fprintf('\n🔥 FINAL RESULTS\n');
fprintf('--------------------------------------\n');
fprintf('Accuracy   : %.2f%%\n', accuracy * 100);
fprintf('Precision  : %.4f\n', precision);
fprintf('Recall     : %.4f\n', recall);
fprintf('F1 Score   : %.4f\n', f1score);
fprintf('--------------------------------------\n');

