%% soft_voting_binary.m
% ---------------------------------------------------------------
% Binary Classification (Tumour / No Tumour) — 5-Model Soft Voting Ensemble
%
% Trains 5 pretrained CNNs (MobileNetV2, ResNet18, ResNet50,
% DenseNet201, EfficientNet-B0) end-to-end with data augmentation,
% GPU support, and early stopping, then fuses their test-set
% predictions three ways:
%   - Average Fusion (soft voting)  — the method reported in the thesis
%   - Majority Vote Fusion
%   - Test-Time Augmentation (TTA), applied to the first model only
%
% Corresponds to Section 5.3 / Table 6.5.1 of the project report
% (Soft Voting Ensemble — 99.49% accuracy).
%
% NOTE — dataset loading differs from the other binary scripts:
% this script points 'datasetRoot' straight at the top-level
% 'BRAIN TUMOR PROJECT' folder and loads ALL images recursively
% (IncludeSubfolders = true), using each image's immediate parent
% folder name ('tumor' / 'no_tumor') as its label — so the existing
% Training/Testing split is ignored and a fresh stratified
% 70% train / (of that, 80% train / 20% val) / 30% test split is
% created at runtime via splitEachLabel(rng(0) for reproducibility).
% This differs from train_test_svm_knn_nb_binary.m and
% train_custom_cnn_resnet18_binary.m, which use the pre-existing
% training/ and testing/ folders directly. Keep this in mind if you
% want all binary scripts to evaluate on an identical held-out set.
%
% Expects dataset at: <datasetRoot> containing (recursively) folders
% named 'tumor' and 'no_tumor' — e.g. point it at 'BRAIN TUMOR PROJECT'
% (update 'datasetRoot' below — currently set to 'E:\BRAIN TUMOR PROJECT')
%
% Outputs:
%   trained_MobileNetV2.mat, trained_ResNet18.mat, trained_ResNet50.mat,
%   trained_DenseNet201.mat, trained_EfficientNetB0.mat  (one per model)
%   ensemble_5models_results_final.mat  (predictions, accuracies, models)
%
% Requires Deep Learning Toolbox Model support packages for:
%   MobileNetV2, ResNet18, ResNet50, DenseNet201, EfficientNet-B0
% ---------------------------------------------------------------

clc; clear; close all;

%% ========== SETTINGS ==========
datasetRoot = 'E:\BRAIN TUMOR PROJECT'; % update path
inputSize = [224 224 3];
numClasses = 2;
maxEpochs = 20;
miniBatchSize = 16;
initialLR = 3e-4;
patience = 5; % early stopping
useGPU = true;

rng(0); % reproducibility

%% ========== LOAD DATA ==========
imdsAll = imageDatastore(datasetRoot,'IncludeSubfolders',true,'LabelSource','foldernames');
labels = imdsAll.Labels;

% Stratified split: 70% train, 30% test
[imdsTrainVal, imdsTest] = splitEachLabel(imdsAll,0.7,'randomized');

% Further split Train into Train + Validation (80%-20%)
[imdsTrain, imdsVal] = splitEachLabel(imdsTrainVal,0.8,'randomized');

fprintf('Images: Train=%d | Val=%d | Test=%d\n', numel(imdsTrain.Files), numel(imdsVal.Files), numel(imdsTest.Files));

classNames = categories(imdsTrain.Labels); % explicit class names

%% ========== DATA AUGMENTATION ==========
augmenter = imageDataAugmenter( ...
    'RandRotation',[-15 15], ...
    'RandXReflection', true, ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5], ...
    'RandScale',[0.95 1.05]);

augTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augmenter);
augVal   = augmentedImageDatastore(inputSize, imdsVal);
augTest  = augmentedImageDatastore(inputSize, imdsTest);

%% ========== TRAINING OPTIONS ==========
execEnv = 'cpu';
if useGPU && gpuDeviceCount>0
    g = gpuDevice;
    execEnv = 'gpu';
    fprintf('Using GPU: %s\n', g.Name);
end

options = trainingOptions('adam', ...
    'InitialLearnRate', initialLR, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 20, ...
    'Verbose', false, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment', execEnv, ...
    'ValidationPatience', patience);

%% ========== PRETRAINED MODELS ==========
nets = {@mobilenetv2, @resnet18, @resnet50, @densenet201, @efficientnetb0};
modelNames = {'MobileNetV2','ResNet18','ResNet50','DenseNet201','EfficientNetB0'};
numModels = numel(nets);
trainedNets = cell(numModels,1);

for i = 1:numModels
    fprintf('\n=== Training model %d/%d: %s ===\n', i,numModels,modelNames{i});
    net = nets{i}();
    
    % Replace final layers for binary classification
    lgraph = layerGraph(net);
    [learnableLayer, classLayer] = findLayersToReplace(lgraph);
    newFCLayer = fullyConnectedLayer(numClasses,'Name','fc_out',...
        'WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
    newClassLayer = classificationLayer('Name','classoutput');
    lgraph = replaceLayer(lgraph,learnableLayer.Name,newFCLayer);
    lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
    
    % Train
    trainedNets{i} = trainNetwork(augTrain, lgraph, options);
    
    % Save model
    netToSave = trainedNets{i};
    save(fullfile(pwd,['trained_' modelNames{i} '.mat']),'netToSave','-v7.3');
end

%% ========== ENSEMBLE PREDICTIONS ==========
numTest = numel(imdsTest.Files);
allScores = zeros(numTest,numClasses,numModels);
predLabelsMatrix = categorical(zeros(numTest,numModels));

for i = 1:numModels
    [predLabels,scores] = classify(trainedNets{i}, augTest);
    allScores(:,:,i) = scores;
    predLabelsMatrix(:,i) = predLabels;
end

% Average Fusion
avgScores = mean(allScores,3);
[~, idxAvg] = max(avgScores,[],2);
YPredAvg = categorical(classNames(idxAvg));
accAvg = mean(YPredAvg == imdsTest.Labels)*100;

% Majority Vote Fusion
YPredMaj = categorical(mode(predLabelsMatrix,2));
accMaj = mean(YPredMaj == imdsTest.Labels)*100;

% Test-Time Augmentation (TTA) on first model
ttaScores = zeros(numTest,numClasses,5);
for k=1:5
    ttaAug = imageDataAugmenter('RandRotation',[-5 5],'RandXReflection',true);
    augTTA = augmentedImageDatastore(inputSize, imdsTest, 'DataAugmentation',ttaAug);
    [~, scoreTTA] = classify(trainedNets{1},augTTA);
    ttaScores(:,:,k) = scoreTTA;
end
ttaScoreAvg = mean(ttaScores,3);
[~, idxTTA] = max(ttaScoreAvg,[],2);
YPredTTA = categorical(classNames(idxTTA));
accTTA = mean(YPredTTA == imdsTest.Labels)*100;

%% ========== DISPLAY RESULTS ==========
fprintf('\n=== FINAL ENSEMBLE RESULTS ===\n');
fprintf('Average Fusion Accuracy : %.2f %%\n', accAvg);
fprintf('Majority Vote Accuracy  : %.2f %%\n', accMaj);
fprintf('TTA-Enhanced Accuracy   : %.2f %%\n', accTTA);

figure('Name','Confusion - Average Fusion'); confusionchart(imdsTest.Labels,YPredAvg);
figure('Name','Confusion - Majority Vote'); confusionchart(imdsTest.Labels,YPredMaj);

%% ========== SAVE RESULTS ==========
save('ensemble_5models_results_final.mat','YPredAvg','YPredMaj','YPredTTA','accAvg','accMaj','accTTA','trainedNets','modelNames','classNames','-v7.3');

%% ========== HELPER FUNCTION ==========
function [learnableLayer, classLayer] = findLayersToReplace(lgraph)
layers = lgraph.Layers;
learnableLayer = [];
classLayer = [];
for i = numel(layers):-1:1
    if isa(layers(i),'nnet.cnn.layer.ClassificationOutputLayer')
        classLayer = layers(i); break;
    end
end
for i = numel(layers):-1:1
    if isa(layers(i),'nnet.cnn.layer.FullyConnectedLayer')
        learnableLayer = layers(i); break;
    end
end
if isempty(learnableLayer)
    for i = numel(layers):-1:1
        if isa(layers(i),'nnet.cnn.layer.Convolution2DLayer')
            learnableLayer = layers(i); break;
        end
    end
end
end
