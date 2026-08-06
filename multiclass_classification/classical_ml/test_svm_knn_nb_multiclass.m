%% test_svm_knn_nb_multiclass.m
% ---------------------------------------------------------------
% Evaluates the SVM / KNN / Naive Bayes multiclass models saved by
% train_svm_knn_nb_multiclass.m against a held-out test set.
% Requires: BrainTumorModels.mat (produced by the training script)
% Expects dataset at: <datasetRoot>/Brain_testing
% ---------------------------------------------------------------

clc; clear; close all;

%% ================= LOAD MODELS =================
load BrainTumorModels

datasetRoot = 'C:\BRAINCLASSIFICATION';

%% ================= LOAD TEST DATA =================
imdsTest = imageDatastore(fullfile(datasetRoot,'Brain_testing'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

disp("TEST LABELS:");
disp(categories(imdsTest.Labels));
countEachLabel(imdsTest)

%% ================= FEATURE EXTRACTION =================
[Xtest,Ytest] = extractDatasetFeatures(imdsTest,outSize,sigma);

%% ================= NORMALIZATION =================
Xtest = (Xtest - muTrain) ./ sigmaTrain;

%% ================= PCA =================
XtestP = (Xtest - muPCA) * coeff(:,1:k);

%% ================= EVALUATE MODELS =================
[a1,p1,r1,f1] = evaluateModel(SVMModel,XtestP,Ytest,"SVM");
[a2,p2,r2,f2] = evaluateModel(KNNModel,XtestP,Ytest,"KNN");
[a3,p3,r3,f3] = evaluateModel(NBModel,XtestP,Ytest,"Naive Bayes");

%% ================= FINAL RESULT TABLE =================
fprintf('\n\n============================================================\n');
fprintf('   Medical Image Classification Results\n');
fprintf('============================================================\n');
fprintf('%-15s %-10s %-12s %-12s %-12s\n',...
    'Algorithm','Accuracy','Precision','Recall','F1-Score');
fprintf('------------------------------------------------------------\n');
fprintf('%-15s %-10.2f %-12.2f %-12.2f %-12.2f\n','SVM',a1,p1,r1,f1);
fprintf('%-15s %-10.2f %-12.2f %-12.2f %-12.2f\n','KNN',a2,p2,r2,f2);
fprintf('%-15s %-10.2f %-12.2f %-12.2f %-12.2f\n','Naive Bayes',a3,p3,r3,f3);
fprintf('============================================================\n');

disp("🎯 TESTING COMPLETE!");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [accuracy,precision,recall,f1] = evaluateModel(model,Xtest,Ytest,name)

YPred = predict(model,Xtest);

accuracy = mean(YPred==Ytest)*100;

confMat = confusionmat(Ytest,YPred);

% binary classification
TP = confMat(2,2);
TN = confMat(1,1);
FP = confMat(1,2);
FN = confMat(2,1);

precision = TP/(TP+FP+eps)*100;
recall    = TP/(TP+FN+eps)*100;
f1        = 2*((precision*recall)/(precision+recall+eps));

% show confusion matrix
figure;
confusionchart(Ytest,YPred);
title(name);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [features,labels] = extractDatasetFeatures(imds,outSize,sigma)

numImages = numel(imds.Files);

sample = preprocessMRI(imread(imds.Files{1}),outSize,sigma);
hog = extractHOGFeatures(sample,'CellSize',[8 8]);

features = zeros(numImages,length(hog));
labels = imds.Labels;

for i = 1:numImages
    img = imread(imds.Files{i});
    img = preprocessMRI(img,outSize,sigma);
    features(i,:) = extractHOGFeatures(img,'CellSize',[8 8]);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function imgOut = preprocessMRI(I,outSize,sigma)

if size(I,3)==3
    I = rgb2gray(I);
end

I = imresize(I,outSize);
I = imgaussfilt(I,sigma);
I = histeq(I);

pixels = double(I(:));
pixels = (pixels-min(pixels))/(max(pixels)-min(pixels)+eps);

idx = kmeans(pixels,2,'Replicates',3);
seg = reshape(idx,size(I));

if mean(I(seg==1)) > mean(I(seg==2))
    tumorMask = seg==1;
else
    tumorMask = seg==2;
end

imgOut = uint8(double(I).*double(tumorMask));

end
