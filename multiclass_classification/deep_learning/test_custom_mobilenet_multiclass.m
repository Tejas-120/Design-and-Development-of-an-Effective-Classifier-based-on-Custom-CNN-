%% test_custom_mobilenet_multiclass.m
% ---------------------------------------------------------------
% Loads a trained Custom MobileNet model, reproduces the 70/15/15
% train/val/test split (rng(1) must match training), evaluates on
% the test split, and reports per-class + weighted-average
% precision/recall/F1.
%
% NOTE: This script loads 'CustomMobileNet_Singlestage_70_15_15.mat'
% — make sure this matches the filename actually saved by your
% training run (the training script above saves
% 'FINAL_CustomMobileNetV2.mat'; rename as needed, or update the
% training script's save name to match).
% ---------------------------------------------------------------

clc;
clear;
close all;

%% ================= LOAD TRAINED MODEL =================
load('CustomMobileNet_Singlestage_70_15_15.mat');  
% loads: trainedNet, inputSize

disp("✅ Model Loaded Successfully");

%% ================= LOAD DATASET =================
datasetRoot = 'C:\BRAINCLASSIFICATION';

imds = imageDatastore(datasetRoot,...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

%% ================= FULL DATASET INFO =================
disp("======================================");
disp("📊 FULL DATASET");
disp("======================================");

totalImages = numel(imds.Files);
fprintf("Total Images: %d\n\n", totalImages);
countEachLabel(imds);

%% ================= SPLIT (70-15-15) =================
rng(1);   % reproducibility

[imdsTrain, imdsTemp] = splitEachLabel(imds,0.7,'randomized');
[imdsValidation, imdsTest] = splitEachLabel(imdsTemp,0.5,'randomized');

%% ================= SPLIT SUMMARY =================
disp("======================================");
disp("📊 DATA SPLIT SUMMARY");
disp("======================================");

trainCount = numel(imdsTrain.Files);
valCount   = numel(imdsValidation.Files);
testCount  = numel(imdsTest.Files);

totalCount = trainCount + valCount + testCount;

fprintf('Train       : %d (%.2f%%)\n', trainCount, (trainCount/totalCount)*100);
fprintf('Validation  : %d (%.2f%%)\n', valCount, (valCount/totalCount)*100);
fprintf('Test        : %d (%.2f%%)\n', testCount, (testCount/totalCount)*100);

%% ================= CLASS DISTRIBUTION =================
disp("======================================");
disp("📊 CLASS DISTRIBUTION");
disp("======================================");

disp("Train:"); countEachLabel(imdsTrain);
disp("Validation:"); countEachLabel(imdsValidation);
disp("Test:"); countEachLabel(imdsTest);

%% ================= PREPROCESS TEST DATA =================
augTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

%% ================= PREDICTION =================
disp("🔥 Testing Custom MobileNet...");

YPred = classify(trainedNet,augTest);
YTrue = imdsTest.Labels;

%% ================= CONFUSION MATRIX =================
figure;
confusionchart(YTrue,YPred);
title('Test Confusion Matrix');

%% ================= SAMPLE PREDICTIONS =================
figure;
perm = randperm(numel(imdsTest.Files),9);

for i = 1:9
    subplot(3,3,i)
    img = readimage(imdsTest,perm(i));
    imshow(img)
    pred = string(YPred(perm(i)));
    trueLabel = string(YTrue(perm(i)));
    title("Pred: " + pred + " | True: " + trueLabel)
end

%% ================= MULTI-CLASS METRICS =================
disp("======================================");
disp("📊 FINAL AGGREGATED METRICS");
disp("======================================");

confMat = confusionmat(YTrue, YPred);
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
    f1score(i)   = 2*(precision(i)*recall(i))/(precision(i)+recall(i)+eps);
end

%% ================= FINAL SINGLE VALUES =================

% Accuracy
accuracy = sum(diag(confMat)) / sum(confMat(:));

% Weighted average (correct for multi-class)
weights = sum(confMat,2) / sum(confMat(:));

precision_final = sum(precision .* weights);
recall_final    = sum(recall .* weights);
f1_final        = sum(f1score .* weights);

%% ================= PRINT RESULTS =================
fprintf('\n🔥 FINAL RESULTS\n');
fprintf('--------------------------------------\n');
fprintf('Accuracy   : %.2f%%\n', accuracy*100);
fprintf('Precision  : %.2f\n', precision_final*100);
fprintf('Recall     : %.2f\n', recall_final*100);
fprintf('F1 Score   : %.2f\n', f1_final*100);
fprintf('--------------------------------------\n');

%% ================= SAVE RESULTS =================
metrics.accuracy  = accuracy;
metrics.precision = precision_final;
metrics.recall    = recall_final;
metrics.f1score   = f1_final;

splitInfo.trainCount = trainCount;
splitInfo.valCount   = valCount;
splitInfo.testCount  = testCount;

save('model_metrics.mat','metrics');
save('dataset_split.mat','splitInfo');
 
