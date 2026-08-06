%% test_custom_efficientnet_b0_multiclass.m
% ---------------------------------------------------------------
% Loads the trained EfficientNet-B0 model, reproduces the 70/15/15
% split, evaluates on the test split, shows sample predictions, and
% reports per-class + macro-average precision/recall/F1.
% ---------------------------------------------------------------

clc;
clear;
close all;

%% ================= LOAD MODEL =================
load('EfficientNetB0_70_15_15_Model.mat');  
% loads: trainedNet, classNames, inputSize

disp("✅ Model Loaded Successfully");

%% ================= LOAD FULL DATASET =================
datasetPath = 'C:\BRAINCLASSIFICATION';

imds = imageDatastore(datasetPath,...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

%% ================= BEFORE SPLIT =================
disp("======================================");
disp("📊 FULL DATASET COUNT (Before Split)");
disp("======================================");

totalImages = numel(imds.Files);
fprintf("Total Images: %d\n\n", totalImages);

countFull = countEachLabel(imds);
disp(countFull);

%% ================= SPLIT 70-15-15 =================
rng(1);   % ensure reproducibility

[imdsTrain, imdsTemp] = splitEachLabel(imds,0.7,'randomized');
[imdsValidation, imdsTest] = splitEachLabel(imdsTemp,0.5,'randomized');

%% ================= AFTER SPLIT =================
disp("======================================");
disp("📊 TRAINING DATA COUNT (70%)");
disp("======================================");
countEachLabel(imdsTrain)

disp("======================================");
disp("📊 VALIDATION DATA COUNT (15%)");
disp("======================================");
countEachLabel(imdsValidation)

disp("======================================");
disp("📊 TEST DATA COUNT (15%)");
disp("======================================");
countEachLabel(imdsTest)

%% ================= RESIZE TEST =================
augTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

%% ================= PREDICT =================
disp("Testing EfficientNet model...");

YPred = classify(trainedNet,augTest);
YTrue = imdsTest.Labels;

accuracy = mean(YPred == YTrue)*100;
fprintf('\n🔥 FINAL TEST ACCURACY = %.2f%%\n',accuracy);

%% ================= CONFUSION MATRIX =================
figure;
confusionchart(YTrue,YPred);
title('Test Confusion Matrix');

%% ================= SHOW SAMPLE RESULTS =================
figure;
perm = randperm(numel(imdsTest.Files),9);

for i = 1:9
    subplot(3,3,i)
    img = readimage(imdsTest,perm(i));
    imshow(img)
    pred = string(YPred(perm(i)));
    trueLabel = string(YTrue(perm(i)));
    title("Pred: "+pred+" | True: "+trueLabel)
end
%% ================= PRECISION RECALL F1 =================

disp("======================================");
disp("📊 DETAILED EVALUATION METRICS");
disp("======================================");

confMat = confusionmat(YTrue,YPred);

numClasses = size(confMat,1);

precision = zeros(numClasses,1);
recall    = zeros(numClasses,1);
f1score   = zeros(numClasses,1);

for i = 1:numClasses
    TP = confMat(i,i);
    FP = sum(confMat(:,i)) - TP;
    FN = sum(confMat(i,:)) - TP;

    precision(i) = TP/(TP+FP+eps);
    recall(i)    = TP/(TP+FN+eps);
    f1score(i)   = 2*((precision(i)*recall(i))/(precision(i)+recall(i)+eps));
end

classNames = categories(YTrue);

for i = 1:numClasses
    fprintf('\nClass: %s\n',string(classNames(i)));
    fprintf('Precision: %.3f\n',precision(i));
    fprintf('Recall: %.3f\n',recall(i));
    fprintf('F1-Score: %.3f\n',f1score(i));
end

%% Macro average
fprintf('\n======================================\n');
fprintf('📊 OVERALL METRICS\n');
fprintf('======================================\n');

fprintf('Average Precision: %.3f\n',mean(precision));
fprintf('Average Recall: %.3f\n',mean(recall));
fprintf('Average F1-score: %.3f\n',mean(f1score));

