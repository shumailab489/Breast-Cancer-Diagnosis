clc
clear all
data = readtable("D:\Research Work\BC data\data.csv");

% Convert the table to an array
dataArray = table2array(data(:, 3:end));

% Split data into features and labels
X = double(dataArray);  % Convert X to double
y = double(strcmp(data.diagnosis, 'M')); % Convert 'M' to 1 and 'B' to 0

% Data preprocessing
X = zscore(X); % Standardize input features

% Split the dataset into training and testing sets
trainRatio = 0.8;
numSamples = size(X, 1);
numTrain = round(trainRatio * numSamples);

X_train = X(1:numTrain, :);
y_train = y(1:numTrain);

X_test = X(numTrain+1:end, :);
y_test = y(numTrain+1:end);

% Neural Network parameters
N1 = 20;                % Middle(Hidden) Layer Neurons
N2 = 1;                 % Output Layer Neurons 
N0 = size(X_train, 2);  % Input Layer Neurons 

% Training parameters
initial_eta = 0.01;     % Initial Learning Rate
eta_decay = 0.95;       % Learning Rate Decay
epoch = 50;             % Training iterations

% Initialization of weights using Xavier initialization
w1 = sqrt(2/N0) * randn(N1, N0); % Input to Hidden
w2 = sqrt(2/N1) * randn(N2, N1); % Hidden to Output

% Arrays to store results
MSE = zeros(1, epoch);
TCE = zeros(1, epoch);

for j = 1:epoch
    ind = randperm(numTrain);  % Randomize training data
    totalError = 0;
    correctClassifications = 0;
    
    for k = 1:numTrain
        Input = X_train(ind(k), :)';
        
        % Forward Propagation
        a1 = tansig(w1 * Input);
        a2 = logsig(w2 * a1); %Logistic Sigmoidal activation function
        
        % Calculate error
        e = y_train(ind(k)) - a2;
        totalError = totalError + e^2;
        
        % Backpropagation
        Y2 = 2 * dlogsig(w2 * a1, a2) * e;
        Y1 = diag(dtansig(w1 * Input, a1), 0) * w2' * Y2; %d shows derivative
        
        % Update weights
        w1 = w1 + initial_eta * Y1 * Input';
        w2 = w2 + initial_eta * Y2 * a1';
        
        if round(a2) == y_train(ind(k))
            correctClassifications = correctClassifications + 1;
        end
    end
    
    % Calculate and store MSE and TCE
    MSE(j) = totalError / numTrain;
    TCE(j) = correctClassifications * 100 / numTrain;
    
    % Decay the learning rate
    initial_eta = initial_eta * eta_decay;
end

% Calculate Output_train
Output_train = zeros(numTrain, 1);
for k = 1:numTrain
    Input = X_train(ind(k), :)';
    a1 = tansig(w1 * Input);
    a2 = logsig(w2 * a1);
    Output_train(k) = a2; % Store predicted output
end

% Calculate Output_test and Testing Accuracy
Output_test = zeros(length(X_test), 1);
correctTestClassifications = 0;
for k = 1:length(X_test)
    Input = X_test(k, :)';
    a1 = tansig(w1 * Input);
    a2 = logsig(w2 * a1);
    Output_test(k) = a2; % Store predicted output
    
    if round(a2) == y_test(k)
        correctTestClassifications = correctTestClassifications + 1;
    end
end
Testing_Accuracy = correctTestClassifications * 100 / length(X_test);

% Display final results
fprintf('Training Accuracy: %.2f%%\n', TCE(end));
fprintf('Testing Accuracy: %.2f%%\n', Testing_Accuracy);

% Plots
figure
semilogy(MSE)
xlabel('Training epochs')
ylabel('MSE (log scale)')
title('Mean Squared Error')
grid on
figure
plot(TCE)
xlabel('Training epochs')
ylabel('Classification accuracy (%)')
title('Classification Performance (Training)')
grid on

% Figure (3): Actual vs. Predicted Classes for Training Data
figure
plot(y_train, 'ob')
hold on
plot(round(Output_train))
legend('Actual class', 'Predicted class using MLP')
xlabel('Training sample')
ylabel('Class Label')
title('Classification Performance (Training)')
grid on

% Figure (4): Actual vs. Predicted Classes for Testing Data
figure
plot(y_test, 'ob')
hold on
plot(round(Output_test))
legend('Actual class', 'Predicted class using MLP')
xlabel('Test sample')
ylabel('Class Label')
title('Classification Performance (Test)')
grid on
