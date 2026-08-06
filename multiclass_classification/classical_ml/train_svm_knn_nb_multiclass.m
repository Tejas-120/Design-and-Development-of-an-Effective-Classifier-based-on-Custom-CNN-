%% train_svm_knn_nb_multiclass.m
% ---------------------------------------------------------------
% Multiclass Classification (Glioma / Meningioma / Pituitary) — Classical ML
% Pipeline: Gaussian smoothing -> Histogram Equalization ->
%           K-Means segmentation -> HOG feature extraction ->
%           Z-score normalization -> PCA (98% variance)
% Trains: Multiclass SVM (ECOC, one-vs-one, RBF kernel),
%         Weighted KNN (k=7), Kernel Naive Bayes
%
% Expects dataset at: <datasetRoot>/Brain_training
% Output: BrainTumorModels.mat (models + PCA/normalization params)
% ---------------------------------------------------------------

clc; clear; close all;
rng(1);

%% ================= SETTINGS =================
datasetRoot = 'C:\BRAINCLASSIFICATION';

outSize = [128 128];   
sigma = 1;
pcaVariance = 98;

%% ================= LOAD TRAINING =================
imdsTrain = imageDatastore(fullfile(datasetRoot,'Brain_training'),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

disp("TRAINING LABELS:");
disp(categories(imdsTrain.Labels));
countEachLabel(imdsTrain)

%% ================= FEATURE EXTRACTION =================
[Xtrain,Ytrain] = extractDatasetFeatures(imdsTrain,outSize,sigma);

%% ================= NORMALIZATION =================
[Xtrain, muTrain, sigmaTrain] = zscore(Xtrain);

%% ================= PCA =================
[coeff,scoreTrain,~,~,explained,muPCA] = pca(Xtrain);

k = find(cumsum(explained)>=pcaVariance,1);
XtrainP = scoreTrain(:,1:k);

fprintf("PCA retained %d features\n",k);

%% ================= TRAIN CLASSIFIERS =================

disp("Training Multiclass SVM...");
SVMModel = fitcecoc(XtrainP,Ytrain,...
    'Learners',templateSVM('KernelFunction','rbf','Standardize',true),...
    'Coding','onevsone');

disp("Training Weighted KNN...");
KNNModel = fitcknn(XtrainP,Ytrain,...
    'NumNeighbors',7,...
    'DistanceWeight','squaredinverse',...
    'Standardize',1);

disp("Training Kernel Naive Bayes...");
NBModel = fitcnb(XtrainP,Ytrain,...
    'DistributionNames','kernel');

%% ================= SAVE =================
save BrainTumorModels ...
    SVMModel KNNModel NBModel ...
    coeff muPCA k ...
    muTrain sigmaTrain ...
    outSize sigma

disp("✅ TRAINING COMPLETE — MODELS SAVED!");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% LOCAL FUNCTIONS BELOW %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


function imgOut = preprocessMRI(I,outSize,sigma)

if size(I,3)==3
    I = rgb2gray(I);
end

I = imresize(I,outSize);

% Gaussian
I = imgaussfilt(I,sigma);

% Histogram equalization
I = histeq(I);

% KMeans segmentation
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
