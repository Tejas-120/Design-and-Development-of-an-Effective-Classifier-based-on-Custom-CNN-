%% hybrid_soft_voting_multiclass.m
% ---------------------------------------------------------------
% Multiclass Soft Voting Ensemble (EfficientNet-B0 + Custom MobileNet)
% Loads both trained models, evaluates each on the validation split
% to compute adaptive accuracy-based weights, then on the test split:
%   - if either model's top-class confidence > 0.85, uses that
%     model's prediction directly (high-confidence override)
%   - otherwise, fuses probabilities using the adaptive weights
%
% Requires: EfficientNetB0_70_15_15_Model.mat,
%           CustomMobileNet_SingleStage_70_15_15.mat
% Expects dataset at: <datasetRoot> (subfolders = class names)
% ---------------------------------------------------------------

clc;
clear;
close all;

disp("🔥 STARTING HYBRID SOFT VOTING...");

%% ================= LOAD MODELS =================

data1 = load('EfficientNetB0_70_15_15_Model.mat');
trainedNet_EfficientNet = data1.trainedNet;

data3 = load('CustomMobileNet_SingleStage_70_15_15.mat');
trainedNet_CustomMobileNet = data3.trainedNet;

disp("✅ Models Loaded");

%% ================= DATASET =================

datasetRoot = 'C:\BRAINCLASSIFICATION';

imds = imageDatastore(datasetRoot,...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

rng(1);
[imdsTrain, imdsTemp] = splitEachLabel(imds,0.7,'randomized');
[imdsValidation, imdsTest] = splitEachLabel(imdsTemp,0.5,'randomized');

disp("✅ Dataset Split Done");

%% ================= INPUT SIZE CHECK =================

inputSize1 = data1.inputSize;

% (Optional but safer)
if isfield(data3,'inputSize')
    inputSize3 = data3.inputSize;
else
    inputSize3 = inputSize1;
end

inputSize = inputSize1; % force common size

classNames = categories(imdsTrain.Labels);
numClasses = numel(classNames);

%% ================= VALIDATION =================

augVal = augmentedImageDatastore(inputSize(1:2), imdsValidation);

disp("🔍 Validation evaluation...");

valProb1 = predict(trainedNet_EfficientNet, augVal);
valProb3 = predict(trainedNet_CustomMobileNet, augVal);

[~, idx1] = max(valProb1, [], 2);
pred1 = categorical(idx1, 1:numClasses, classNames);

[~, idx3] = max(valProb3, [], 2);
pred3 = categorical(idx3, 1:numClasses, classNames);

acc1 = mean(pred1 == imdsValidation.Labels);
acc3 = mean(pred3 == imdsValidation.Labels);

fprintf("EffNet Val Acc: %.4f\n", acc1);
fprintf("MobileNet Val Acc: %.4f\n", acc3);

%% ================= AUTO WEIGHTS =================

w1 = acc1 / (acc1 + acc3);
w3 = acc3 / (acc1 + acc3);

fprintf("Weights -> EffNet: %.2f | MobileNet: %.2f\n", w1, w3);

%% ================= TEST DATA =================

augTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

disp("🔥 Running Hybrid Soft Voting...");

prob1 = predict(trainedNet_EfficientNet, augTest);
prob3 = predict(trainedNet_CustomMobileNet, augTest);

%% ================= HYBRID LOGIC =================

finalProb = zeros(size(prob1));

for i = 1:size(prob1,1)
    
    p1 = prob1(i,:);
    p3 = prob3(i,:);
    
    % Confidence (max probability)
    c1 = max(p1);
    c3 = max(p3);
    
    % -------- RULE 1: HIGH CONFIDENCE OVERRIDE --------
    if c3 > 0.85
        finalProb(i,:) = p3;
        
    elseif c1 > 0.85
        finalProb(i,:) = p1;
        
    % -------- RULE 2: NORMAL WEIGHTED --------
    else
        finalProb(i,:) = (w1 * p1) + (w3 * p3);
    end
end

%% ================= FINAL PREDICTION =================

[~, idx] = max(finalProb, [], 2);
finalPred = categorical(idx, 1:numClasses, classNames);

%% ================= ACCURACY =================

accuracy = mean(finalPred == imdsTest.Labels) * 100;
fprintf('\n🔥 FINAL HYBRID ACCURACY = %.2f%%\n', accuracy);

%% ================= CONFUSION MATRIX =================

figure;
confusionchart(imdsTest.Labels, finalPred);
title('Hybrid Soft Voting Confusion Matrix');

disp("✅ DONE");
