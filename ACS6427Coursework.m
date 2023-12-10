% ACS6427 Data Modelling and Machine Intelligence - Master Script %
%   Author - Charlie Baldwin (200138183)                          %
%                                                                 %
%   REQUIRED TOOLBOXES:                                           %
%       - Statistics and Machine Learning Toolbox                 %

clear;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%
% VARIABLE DECLARATION %
%%%%%%%%%%%%%%%%%%%%%%%%

%Hyperparameters - These change how the model is trained
alpha = 0.1;        % Learning rate
iterations = 1000;  % Number of iterations
lambda = 0.1;       % Regularization parameter
k = 10;             % Number of folds for CV

%Initialize variables - Numerical values and arrays generated after the
%model is trained
accuracy = zeros(k, 1);
precision = zeros(k, 1);
recall = zeros(k, 1);
f1Score = zeros(k, 1);
rocLabels = [];    % Array for ROC labels
rocScores = [];    % Array for ROC scores

%%%%%%%%%%%%%
% LOAD DATA %
%%%%%%%%%%%%%

%Load data file
load('QSAR_data.mat');

%Assign features and labels
X = QSAR_data(:, 1:end-1);  %features
Y = QSAR_data(:, end);      %labels

%%%%%%%%%%%%%%%%%%%%%%%
% DATA PRE-PROCESSING %
%%%%%%%%%%%%%%%%%%%%%%%

%Remove duplicate rows
[X, uniqueVector, ~] = unique(X, 'rows');
Y = Y(uniqueVector);

%Standardize features
X = zscore(X);

%Remove outliers
outliers = any(abs(X) > 3, 2);
X = X(~outliers, :);
Y = Y(~outliers);

%Shuffle data so it is not in order after zscore
samples = size(X, 1);
shuffle = randperm(samples);
X = X(shuffle, :);
Y = Y(shuffle);

%Get size of each fold
foldSize = floor(samples / k);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODEL TRAINING AND TESTING %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for fold = 1:k

    %Get testing and training data for each fold
    testStart = (fold - 1) * foldSize + 1;
    testEnd = fold * foldSize;

    %Code to determine last fold
    if fold == k
        testEnd = samples;
    end

    %Get testing and training indices
    testIndices = testStart:testEnd;
    trainIndices = setdiff(1:samples, testIndices);

    %Split data into testing and training sets
    Xtest = X(testIndices, :);
    Ytest = Y(testIndices);
    Xtrain = X(trainIndices, :);
    Ytrain = Y(trainIndices);

    %Add bias term to the features
    Xtest = [ones(length(testIndices), 1), Xtest];
    Xtrain = [ones(length(trainIndices), 1), Xtrain];

    %Initialize theta
    theta = zeros(size(Xtrain, 2), 1);

    %%%%%%%%%%%%%%%
    % TRAIN MODEL %
    %%%%%%%%%%%%%%%

    %Train the logistic regression model using gradient descent with regularization
    for i = 1:iterations
        %Calculate the predicted probabilities
        h = 1 ./ (1 + exp(-Xtrain * theta));

        %Calculate gradient with regularization
        gradient = (Xtrain' * (h - Ytrain) + lambda * theta) / length(Ytrain);

        %Update parameters using gradient descent
        theta = theta - alpha * gradient;
    end

    %%%%%%%%%%%%%%%%%
    % FOLD ANALYSIS %
    %%%%%%%%%%%%%%%%%

    %Make predictions on the test set
    pred = 1 ./ (1 + exp(-Xtest * theta));
    %Round the test set up or down for binary 1 or 0 (> 0.5 or < 0.5)
    Ypred = round(1 ./ (1 + exp(-Xtest * theta)));
    
    %Calculate performance of current fold
    accuracy(fold) = sum(Ypred == Ytest) / length(Ytest);
    precision(fold) = sum(Ypred & Ytest) / sum(Ypred);
    recall(fold) = sum(Ypred & Ytest) / sum(Ytest);
    f1Score(fold) = 2 * (precision(fold) * recall(fold)) / (precision(fold) + recall(fold));

    % CONFUSION MATRIX %
    %Create confusion matrix for fold
    confusionFig = figure(fold);
    confusionchart(confusionmat(Ytest, Ypred), {'Negative', 'Positive'});
    grid on;

    % ROC CURVE %
    %Update array by concatenating existing array with new data
    rocLabels = [rocLabels; Ytest]; %testing data
    rocScores = [rocScores; pred];  %predicted data
    %Calculate ROC Curve
    [falsepos, truepos, thresholds, ~] = perfcurve(rocLabels, rocScores, 1);  
    %Plot ROC curve
    figure;
    plot(falsepos, truepos);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('ROC Curve of Logisitic Regression Model');
    grid on;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FINAL NUMERICAL OUTPUTS %
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate mean accuracy accross all folds
accuracy = mean(accuracy);
disp(['Mean Accuracy: ', num2str(accuracy)]);
precision = mean(precision);
disp(['Mean Precision: ', num2str(precision)]);
recall = mean(recall);
disp(['Mean Recall: ', num2str(recall)]);
f1Score = mean(f1Score);
disp(['Mean F1 Score: ', num2str(f1Score)]);