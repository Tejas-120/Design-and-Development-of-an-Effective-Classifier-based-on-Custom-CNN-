%% train_test_svm_knn_nb_binary.m
% ---------------------------------------------------------------
% Binary Classification (Tumour / No Tumour) — Classical ML
% Pipeline: Gaussian smoothing -> Histogram Equalization ->
%           K-Means segmentation -> PCA (98% variance) ->
%           SVM (RBF) / KNN (k=5) / Naive Bayes
%
% Expects dataset at: <datasetRoot>/training and <datasetRoot>/testing
% (update 'datasetRoot' below before running)
%
% Outputs: svmModel.mat, knnModel.mat, nbModel.mat, pcaParams.mat
% ---------------------------------------------------------------

%% brain_tumor_pca_histeq_kmeans_full.m
clc; clear; close all;
rng(1);  % reproducibility

% ================== USER SETTINGS ==================
datasetRoot = 'C:\BRAIN TUMOR PROJECT';   % adjust path
outSize     = [64 64];
sigma       = 1.0;
filterSize  = 0;
% ===================================================

%% 1) Load dataset
imdsTrain = imageDatastore(fullfile(datasetRoot,'training'), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTest  = imageDatastore(fullfile(datasetRoot,'testing'), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% 2) Preprocessing: Gaussian + HistEq + K-means segmentation
imdsTrainPre = imageDatastore(imdsTrain.Files, 'Labels', imdsTrain.Labels, ...
    'ReadFcn', @(f) preprocessKMeans(imread(f), outSize, sigma, filterSize));
imdsTestPre  = imageDatastore(imdsTest.Files,  'Labels', imdsTest.Labels, ...
    'ReadFcn', @(f) preprocessKMeans(imread(f), outSize, sigma, filterSize));

%% 3) Vectorize
[Xtrain, Ytrain] = vectorize(imdsTrainPre, outSize);
[Xtest,  Ytest ] = vectorize(imdsTestPre,  outSize);

%% 4) PCA (retain 98% variance)
[coeff, scoreTrain, ~, ~, explained, mu] = pca(Xtrain);
k = find(cumsum(explained) >= 98, 1);
YtrainP = scoreTrain(:,1:k);
YtestP  = (Xtest - mu) * coeff(:,1:k);

%% 5) Train Classifiers
% ---- Tuned SVM 
svmModel = fitcsvm(YtrainP, Ytrain, ...
    'KernelFunction','rbf', 'KernelScale',0.7, 'BoxConstraint',6, ...
    'Standardize',true);

% ---- KNN (k=5)
knnModel = fitcknn(YtrainP, Ytrain, 'NumNeighbors',5, 'Distance','euclidean');

% ---- Naive Bayes
nbModel = fitcnb(YtrainP, Ytrain);

%% 6) Save models and PCA parameters
save('svmModel.mat','svmModel');
save('knnModel.mat','knnModel');
save('nbModel.mat','nbModel');
save('pcaParams.mat','coeff','mu','k');

%% 7) Load models immediately (optional, just to simulate separate session)
load('svmModel.mat','svmModel');
load('knnModel.mat','knnModel');
load('nbModel.mat','nbModel');
load('pcaParams.mat','coeff','mu','k');

%% 8) Predict on test data
YPredSVM = predict(svmModel, YtestP);
YPredKNN = predict(knnModel, YtestP);
YPredNB  = predict(nbModel, YtestP);

%% 9) Compute test accuracies
accSVM = mean(YPredSVM == Ytest);
accKNN = mean(YPredKNN == Ytest);
accNB  = mean(YPredNB == Ytest);

%% 10) Print accuracies in command window
fprintf('\n=== Classifier Accuracies ===\n');
fprintf('SVM (tuned RBF)   = %.2f %%\n', accSVM*100);
fprintf('KNN (k=5)         = %.2f %%\n', accKNN*100);
fprintf('Naive Bayes       = %.2f %%\n', accNB*100);

%% 11) Performance metrics for each classifier
fprintf('\n=== Performance Metrics ===\n');
computeMetrics(Ytest, YPredSVM, 'SVM');
computeMetrics(Ytest, YPredKNN, 'KNN');
computeMetrics(Ytest, YPredNB, 'Naive Bayes');

%% 12) Confusion matrix (SVM only)
figure('Name','Confusion Matrix (SVM)');
confusionchart(Ytest, YPredSVM);
title(sprintf('SVM Confusion Matrix (Accuracy %.2f%%)', accSVM*100));

%% 13) Bar graph of all accuracies
figure('Name','Accuracy Comparison');
bar([accSVM, accKNN, accNB]*100);
set(gca,'XTickLabel',{'SVM','KNN','Naive Bayes'});
ylabel('Accuracy (%)'); ylim([0 100]);
title('Classifier Accuracy Comparison');

%% ================= Helper Functions =================
function Iout = preprocessKMeans(I, outSize, sigma, filterSize)
    if size(I,3)==3, I = rgb2gray(I); end
    I = imresize(I, outSize);

    % Gaussian smoothing
    if filterSize && filterSize>0
        I = imgaussfilt(I, sigma, 'FilterSize', filterSize);
    else
        I = imgaussfilt(I, sigma);
    end

    % Histogram equalisation
    I = histeq(I);

    % K-means segmentation (2 clusters)
    pixels = double(I(:));
    pixels = (pixels - min(pixels)) / (max(pixels)-min(pixels)+eps);
    idx = kmeans(pixels, 2, 'MaxIter', 200, 'Replicates', 3, 'Distance','sqeuclidean');
    segImg = reshape(idx, size(I));

    % Keep bright cluster as tumor
    if mean(I(segImg==1)) > mean(I(segImg==2))
        segImg = segImg==1;
    else
        segImg = segImg==2;
    end

    Iout = im2uint8(segImg);
end

function [X, Y] = vectorize(imds, outSize)
    N = numel(imds.Files); D = prod(outSize);
    X = zeros(N,D,'double'); Y = imds.Labels;
    for i=1:N
        I = readimage(imds,i);
        X(i,:) = double(I(:))';
    end
end

function computeMetrics(trueLabels, predLabels, name)
    if ~isa(predLabels,'categorical')
        predLabels = categorical(predLabels, categories(trueLabels));
    end
    C = confusionmat(trueLabels, predLabels);
    TP = C(2,2); TN = C(1,1); FP = C(1,2); FN = C(2,1);
    acc = (TP+TN)/sum(C(:));
    sens = safeDivide(TP,TP+FN);
    spec = safeDivide(TN,TN+FP);
    prec = safeDivide(TP,TP+FP);
    F1 = safeDivide(2*(prec*sens),(prec+sens));
    fprintf('%s: Acc=%.2f%%, Sens=%.2f%%, Spec=%.2f%%, Prec=%.2f%%, F1=%.2f%%\n', ...
        name, acc*100, sens*100, spec*100, prec*100, F1*100);
end

function v = safeDivide(a,b)
    if b==0, v=0; else, v=a/b; end
end

