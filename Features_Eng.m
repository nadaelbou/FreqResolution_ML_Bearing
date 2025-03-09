tic
clear all
clc;
close all

%% Interesting Sources

% https://se.mathworks.com/help/predmaint/ref/bearingfaultbands.html
% https://www.google.com/search?q=cage+fault+signature+&tbm=isch&ved=2ahUKEwjusoX1tumDAxXZGxAIHQwgBpEQ2-cCegQIABAA&oq=cage+fault+signature+&gs_lcp=CgNpbWcQAzoECCMQJ1DGA1j6C2DDDGgAcAB4AIABTogBiQaSAQIxMpgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=0WeqZa6GEtm3wPAPjMCYiAk&bih=1086&biw=1912#imgrc=8a3VIVJr43aUkM&imgdii=5X_QDbL7rc4rRM

%% Import the Dataset (remember to change path to where you store the data)

% Healthy 

load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Healthy\0.mat')
Normal_0_x = ch_AI_7_1;
Normal_0_x_TIME = ch_AI_7_1_TIME;
Normal_0_y = ch_AI_7_2;
Normal_0_y_TIME = ch_AI_7_2_TIME;
Normal_0_z = ch_AI_7_3;
Normal_0_z_TIME = ch_AI_7_3_TIME;
load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Healthy\25.mat')
Normal_25_x = ch_AI_7_1;
Normal_25_x_TIME = ch_AI_7_1_TIME;
Normal_25_y = ch_AI_7_2;
Normal_25_y_TIME = ch_AI_7_2_TIME;
Normal_25_z = ch_AI_7_3;
Normal_25_z_TIME = ch_AI_7_3_TIME;
load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Healthy\50.mat')
Normal_50_x = ch_AI_7_1;
Normal_50_x_TIME = ch_AI_7_1_TIME;
Normal_50_y = ch_AI_7_2;
Normal_50_y_TIME = ch_AI_7_2_TIME;
Normal_50_z = ch_AI_7_3;
Normal_50_z_TIME = ch_AI_7_3_TIME;
load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Healthy\75.mat')
Normal_75_x = ch_AI_7_1;
Normal_75_x_TIME = ch_AI_7_1_TIME;
Normal_75_y = ch_AI_7_2;
Normal_75_y_TIME = ch_AI_7_2_TIME;
Normal_75_z = ch_AI_7_3;
Normal_75_z_TIME = ch_AI_7_3_TIME;
load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Healthy\100.mat')
Normal_100_x = ch_AI_7_1;
Normal_100_x_TIME = ch_AI_7_1_TIME;
Normal_100_y = ch_AI_7_2;
Normal_100_y_TIME = ch_AI_7_2_TIME;
Normal_100_z = ch_AI_7_3;
Normal_100_z_TIME = ch_AI_7_3_TIME;

% Cage 

load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Cage\0.mat')
Cage_0_x = ch_AI_7_1;
Cage_0_x_TIME = ch_AI_7_1_TIME;
Cage_0_y = ch_AI_7_2;
Cage_0_y_TIME = ch_AI_7_2_TIME;
Cage_0_z = ch_AI_7_3;
Cage_0_z_TIME = ch_AI_7_3_TIME;
load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Cage\25.mat')
Cage_25_x = ch_AI_7_1;
Cage_25_x_TIME = ch_AI_7_1_TIME;
Cage_25_y = ch_AI_7_2;
Cage_25_y_TIME = ch_AI_7_2_TIME;
Cage_25_z = ch_AI_7_3;
Cage_25_z_TIME = ch_AI_7_3_TIME;
load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Cage\50.mat')
Cage_50_x = ch_AI_7_1;
Cage_50_x_TIME = ch_AI_7_1_TIME;
Cage_50_y = ch_AI_7_2;
Cage_50_y_TIME = ch_AI_7_2_TIME;
Cage_50_z = ch_AI_7_3;
Cage_50_z_TIME = ch_AI_7_3_TIME;
load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Cage\75.mat')
Cage_75_x = ch_AI_7_1;
Cage_75_x_TIME = ch_AI_7_1_TIME;
Cage_75_y = ch_AI_7_2;
Cage_75_y_TIME = ch_AI_7_2_TIME;
Cage_75_z = ch_AI_7_3;
Cage_75_z_TIME = ch_AI_7_3_TIME;
load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Cage\100.mat')
Cage_100_x = ch_AI_7_1;
Cage_100_x_TIME = ch_AI_7_1_TIME;
Cage_100_y = ch_AI_7_2;
Cage_100_y_TIME = ch_AI_7_2_TIME;
Cage_100_z = ch_AI_7_3;
Cage_100_z_TIME = ch_AI_7_3_TIME;

% IR Fault 

load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\IR\0.mat')
IR_0_x = ch_AI_7_3;
IR_0_x_TIME = ch_AI_7_3_TIME;
IR_0_y = ch_AI_7_3;
IR_0_y_TIME = ch_AI_7_3_TIME;
IR_0_z = ch_AI_7_3;
IR_0_z_TIME = ch_AI_7_3_TIME;
load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\IR\25.mat')
IR_25_x = ch_AI_7_3;
IR_25_x_TIME = ch_AI_7_3_TIME;
IR_25_y = ch_AI_7_3;
IR_25_y_TIME = ch_AI_7_3_TIME;
IR_25_z = ch_AI_7_3;
IR_25_z_TIME = ch_AI_7_3_TIME;
load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\IR\50.mat')
IR_50_x = ch_AI_7_3;
IR_50_x_TIME = ch_AI_7_3_TIME;
IR_50_y = ch_AI_7_3;
IR_50_y_TIME = ch_AI_7_3_TIME;
IR_50_z = ch_AI_7_3;
IR_50_z_TIME = ch_AI_7_3_TIME;
load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\IR\75.mat')
IR_75_x = ch_AI_7_3;
IR_75_x_TIME = ch_AI_7_3_TIME;
IR_75_y = ch_AI_7_3;
IR_75_y_TIME = ch_AI_7_3_TIME;
IR_75_z = ch_AI_7_3;
IR_75_z_TIME = ch_AI_7_3_TIME;
load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\IR\100.mat')
IR_100_x = ch_AI_7_3;
IR_100_x_TIME = ch_AI_7_3_TIME;
IR_100_y = ch_AI_7_3;
IR_100_y_TIME = ch_AI_7_3_TIME;
IR_100_z = ch_AI_7_3;
IR_100_z_TIME = ch_AI_7_3_TIME;

% OR Fault 

load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\OR\0.mat')
OR_0_x = ch_AI_7_3;
OR_0_x_TIME = ch_AI_7_3_TIME;
OR_0_y = ch_AI_7_3;
OR_0_y_TIME = ch_AI_7_3_TIME;
OR_0_z = ch_AI_7_3;
OR_0_z_TIME = ch_AI_7_3_TIME;
load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\OR\25.mat')
OR_25_x = ch_AI_7_3;
OR_25_x_TIME = ch_AI_7_3_TIME;
OR_25_y = ch_AI_7_3;
OR_25_y_TIME = ch_AI_7_3_TIME;
OR_25_z = ch_AI_7_3;
OR_25_z_TIME = ch_AI_7_3_TIME;
load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\OR\50.mat')
OR_50_x = ch_AI_7_3;
OR_50_x_TIME = ch_AI_7_3_TIME;
OR_50_y = ch_AI_7_3;
OR_50_y_TIME = ch_AI_7_3_TIME;
OR_50_z = ch_AI_7_3;
OR_50_z_TIME = ch_AI_7_3_TIME;
load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\OR\75.mat')
OR_75_x = ch_AI_7_3;
OR_75_x_TIME = ch_AI_7_3_TIME;
OR_75_y = ch_AI_7_3;
OR_75_y_TIME = ch_AI_7_3_TIME;
OR_75_z = ch_AI_7_3;
OR_75_z_TIME = ch_AI_7_3_TIME;
load('\Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\OR\100.mat')
OR_100_x = ch_AI_7_3;
OR_100_x_TIME = ch_AI_7_3_TIME;
OR_100_y = ch_AI_7_3;
OR_100_y_TIME = ch_AI_7_3_TIME;
OR_100_z = ch_AI_7_3;
OR_100_z_TIME = ch_AI_7_3_TIME;

% Set the same length to all the signals

min_size = min([size(Normal_0_x, 1) size(Normal_25_x, 1) size(Normal_50_x, 1) size(Normal_75_x, 1) size(Normal_100_x, 1) ...
                size(Normal_0_y, 1) size(Normal_25_y, 1) size(Normal_50_y, 1) size(Normal_75_y, 1) size(Normal_100_y, 1) ...
                size(Normal_0_z, 1) size(Normal_25_z, 1) size(Normal_50_z, 1) size(Normal_75_z, 1) size(Normal_100_z, 1) ...
                size(Cage_0_x, 1) size(Cage_25_x, 1) size(Cage_50_x, 1) size(Cage_75_x, 1) size(Cage_100_x, 1) ...
                size(Cage_0_y, 1) size(Cage_25_y, 1) size(Cage_50_y, 1) size(Cage_75_y, 1) size(Cage_100_y, 1) ...
                size(Cage_0_z, 1) size(Cage_25_z, 1) size(Cage_50_z, 1) size(Cage_75_z, 1) size(Cage_100_z, 1) ...
                size(IR_0_x, 1) size(IR_25_x, 1) size(IR_50_x, 1) size(IR_75_x, 1) size(IR_100_x, 1) ... 
                size(IR_0_y, 1) size(IR_25_y, 1) size(IR_50_y, 1) size(IR_75_y, 1) size(IR_100_y, 1) ...
                size(IR_0_z, 1) size(IR_25_z, 1) size(IR_50_z, 1) size(IR_75_z, 1) size(IR_100_z, 1) ...
                size(OR_0_x, 1) size(OR_25_x, 1) size(OR_50_x, 1) size(OR_75_x, 1) size(OR_100_x, 1) ...
                size(OR_0_y, 1) size(OR_25_y, 1) size(OR_50_y, 1) size(OR_75_y, 1) size(OR_100_y, 1) ...
                size(OR_0_z, 1) size(OR_25_z, 1) size(OR_50_z, 1) size(OR_75_z, 1) size(OR_100_z, 1)]); 

% Extract the required samples from each category

Normal_0_x = [Normal_0_x(6962:min_size, :)];
Normal_0_y = Normal_0_y(6962:min_size, :);
Normal_0_z = Normal_0_z(6962:min_size, :);
Normal_25_x = Normal_25_x(6962:min_size, :);
Normal_25_y = Normal_25_y(6962:min_size, :);
Normal_25_z = Normal_25_z(6962:min_size, :);
Normal_50_x = Normal_50_x(6962:min_size, :);
Normal_50_y = Normal_50_y(6962:min_size, :);
Normal_50_z = Normal_50_z(6962:min_size, :);
Normal_75_x = Normal_75_x(6962:min_size, :);
Normal_75_y = Normal_75_y(6962:min_size, :);
Normal_75_z = Normal_75_z(6962:min_size, :);
Normal_100_x = Normal_100_y(6962:min_size, :);
Normal_100_y = Normal_100_y(6962:min_size, :);
Normal_100_z = Normal_100_z(6962:min_size, :);

Cage_0_x = Cage_0_x(6962:min_size, :);
Cage_0_y = Cage_0_y(6962:min_size, :);
Cage_0_z = Cage_0_z(6962:min_size, :);
Cage_25_x = Cage_25_x(6962:min_size, :);
Cage_25_y = Cage_25_y(6962:min_size, :);
Cage_25_z = Cage_25_z(6962:min_size, :);
Cage_50_x = Cage_50_x(6962:min_size, :);
Cage_50_y = Cage_50_y(6962:min_size, :);
Cage_50_z = Cage_50_z(6962:min_size, :);
Cage_75_x = Cage_75_x(6962:min_size, :);
Cage_75_y = Cage_75_y(6962:min_size, :);
Cage_75_z = Cage_75_z(6962:min_size, :);
Cage_100_x = Cage_100_y(6962:min_size, :);
Cage_100_y = Cage_100_y(6962:min_size, :);
Cage_100_z = Cage_100_z(6962:min_size, :);

IR_0_x = IR_0_x(6962:min_size, :);
IR_0_y = IR_0_y(6962:min_size, :);
IR_0_z = IR_0_z(6962:min_size, :);
IR_25_x = IR_25_x(6962:min_size, :);
IR_25_y = IR_25_y(6962:min_size, :);
IR_25_z = IR_25_z(6962:min_size, :);
IR_50_x = IR_50_x(6962:min_size, :);
IR_50_y = IR_50_y(6962:min_size, :);
IR_50_z = IR_50_z(6962:min_size, :);
IR_75_x = IR_75_x(6962:min_size, :);
IR_75_y = IR_75_y(6962:min_size, :);
IR_75_z = IR_75_z(6962:min_size, :);
IR_100_x = IR_100_y(6962:min_size, :);
IR_100_y = IR_100_y(6962:min_size, :);
IR_100_z = IR_100_z(6962:min_size, :);

OR_0_x = OR_0_x(6962:min_size, :);
OR_0_y = OR_0_y(6962:min_size, :);
OR_0_z = OR_0_z(6962:min_size, :);
OR_25_x = OR_25_x(6962:min_size, :);
OR_25_y = OR_25_y(6962:min_size, :);
OR_25_z = OR_25_z(6962:min_size, :);
OR_50_x = OR_50_x(6962:min_size, :);
OR_50_y = OR_50_y(6962:min_size, :);
OR_50_z = OR_50_z(6962:min_size, :);
OR_75_x = OR_75_x(6962:min_size, :);
OR_75_y = OR_75_y(6962:min_size, :);
OR_75_z = OR_75_z(6962:min_size, :);
OR_100_x = OR_100_y(6962:min_size, :);
OR_100_y = OR_100_y(6962:min_size, :);
OR_100_z = OR_100_z(6962:min_size, :);

%%%%% All signals are of length 600000

%% Time Span 

fs= 20*1e3; % [Hz] Sampling Frequency 
Ts = 1/fs; 

%% FFT on all signal (ALL)

% Define parameters

window_size = 2^19;             % [-]Replace with your desired window size
Fs = 20*1e3;                    % [Hz] Sampling Frequency 
overlap = 0;                    % [- ]Replace with your desired overlap
type_window = 1; 

batchSize = 2^10;     % Replace with your desired batch size
overlapSize = 0;   % Replace with your desired overlap size

% Iterate through each set of variables and apply the function
variable_prefixes = {'Normal_0', 'Cage_0', 'IR_0', 'OR_0','Normal_25', 'Cage_25', 'IR_25', 'OR_25','Normal_50', 'Cage_50', 'IR_50', 'OR_50','Normal_75', 'Cage_75', 'IR_75', 'OR_75','Normal_100', 'Cage_100', 'IR_100', 'OR_100'};  % Add other prefixes as needed

 
for i = 1:length(variable_prefixes)
    prefix = variable_prefixes{i};
   
    % Create a struct to store the results
    fft_results.(prefix) = struct();
    
    % Iterate through x, y, and z
    %for axis = {'x', 'y', 'z'}
    for axis = {'x'}
        variable_name = sprintf('%s_%s', prefix, axis{1});
        fft_variable_name = sprintf('%s_FFT_%s', prefix, axis{1});
        
        % Extract the variable
        data = eval(variable_name);
        
        % Apply the fft_sliding_window function
        [frequencies, fft_results.(prefix).(fft_variable_name)]= fft_sliding_window(data, window_size, type_window, Fs);
    end
end

%%
%%%
%%%%% ----------------- Training set------------------------
%%%

%%  Dataset freq domain features without labels  (TRAINING)

variable_prefixes_train = {'Normal_0', 'Cage_0', 'IR_0', 'OR_0','Normal_100', 'Cage_100', 'IR_100', 'OR_100'};  % Add other prefixes as needed

% Iterate through each set of variables and apply the function
for i = 1:length(variable_prefixes_train)
    prefix = variable_prefixes_train{i};
    
    % Access the struct for the specific variable set
    fft_struct = fft_results.(prefix);
    fft_freq.(prefix) = struct();
    % Iterate through x, y, and z
    for axis = {'x'}
        field_name = sprintf('%s_FFT_%s', prefix, axis{1});
        
        % Extract the variable
        data = fft_struct.(field_name);
        % Apply the calculateFreqDomainFeatures function
        features_freq_domain_train.(prefix).(field_name) = calculateFreqDomainFeatures(data, batchSize, overlapSize);
        
    end
end

%% Dataset freq domain features with labels (Training)

% Step 1: Define labels
labels = {'Normal', 'Cage', 'IR', 'OR'};
labelsnum = {1, 2, 3, 4};
dim = {'_FFT_x'};

% Step 2-4: Iterate through the structure and extract data and labels
dataset_freqdomain_train = [];

for labelIdx = 1:length(labels)
    currentLabel = labels{labelIdx};
    for dimIdx = 1:length(dim)
        currentDim = dim{dimIdx};
        for sampleIdx = [0, 100] %changed for training 
            fieldName1 = [currentLabel '_' num2str(sampleIdx)];
            fieldName2 = [currentLabel '_' num2str(sampleIdx) currentDim];
            currentData = features_freq_domain_train.(fieldName1).(fieldName2);
        % Concatenate data
            currentData_new =  [currentData repmat(labelsnum{labelIdx}, length(currentData), 1)];
            dataset_freqdomain_train = [dataset_freqdomain_train; currentData_new];
        end
    end 
end

%% Time domain features (Training)

for i = 1:length(variable_prefixes_train)
    prefix = variable_prefixes_train{i};
    
    % Create a struct to store the results
    time_domain_feat_train.(prefix) = struct();
    
    % Iterate through x, y, and z
    for axis = {'x'}
        time_variable_name = sprintf('%s_%s', prefix, axis{1});
        
        % Extract the variable
        data = eval(variable_name);
        numBatches = ceil(length(data) / (batchSize * (1 - overlap)));
        
        % Apply the fft_sliding_window function
        time_domain_feat_train.(prefix).(time_variable_name) = calculateTimeDomainFeatures(data(1:300000,1), numBatches, batchSize, overlapSize);
        
    end
end

%% Dataset time domain with labels (Training)

% Step 1: Define labels
labels = {'Normal', 'Cage', 'IR', 'OR'};
labelsnum = {1, 2, 3, 4};
dim = {'_x'};

% Step 2-4: Iterate through the structure and extract data and labels
dataset_timedomain_train = [];

for labelIdx = 1:length(labels)
    currentLabel = labels{labelIdx};
    for dimIdx = 1:length(dim)
        currentDim = dim{dimIdx};
        for sampleIdx = [0, 100] %changed for training 
            fieldName1 = [currentLabel '_' num2str(sampleIdx)];
            fieldName2 = [currentLabel '_' num2str(sampleIdx) currentDim];
            currentData = time_domain_feat_train.(fieldName1).(fieldName2);
        
            currentData_new =  [currentData repmat(labelsnum{labelIdx}, length(currentData), 1)];% Concatenate data
            dataset_timedomain_train = [dataset_timedomain_train; currentData_new];
        end
    end 
end

%%
%%%
%%%%% ----------------- Evaluation set------------------------
%%%

%% Dataset freq domain features without labels (Eval) 

variable_prefixes_eval = {'Normal_50', 'Cage_50', 'IR_50', 'OR_50'};  % In this case, I choose to evaluate on 50% loading level data. 

% Iterate through each set of variables and apply the function
for i = 1:length(variable_prefixes_eval)
    prefix = variable_prefixes_eval{i};
    
    % Access the struct for the specific variable set
    fft_struct = fft_results.(prefix);
    fft_freq.(prefix) = struct();
    % Iterate through x, y, and z
    for axis = {'x'}
        field_name = sprintf('%s_FFT_%s', prefix, axis{1});
        
        % Extract the variable
        data = fft_struct.(field_name);
        
        % Apply the calculateFreqDomainFeatures function
        features_freq_domain_eval.(prefix).(field_name) = calculateFreqDomainFeatures(data, batchSize, overlapSize);
    end
end

%% Dataset freq domain features with labels (Eval)

% Step 1: Define labels
labels = {'Normal', 'Cage', 'IR', 'OR'};
labelsnum = {1, 2, 3, 4};
dim = {'_FFT_x'};

% Step 2-4: Iterate through the structure and extract data and labels
dataset_freqdomain_eval = [];

for labelIdx = 1:length(labels)
    currentLabel = labels{labelIdx};
    for dimIdx = 1:length(dim)
        currentDim = dim{dimIdx};
        for sampleIdx = 50 %changed for evaluation
            fieldName1 = [currentLabel '_' num2str(sampleIdx)];
            fieldName2 = [currentLabel '_' num2str(sampleIdx) currentDim];
            currentData = features_freq_domain_eval.(fieldName1).(fieldName2);
        % Concatenate data
            currentData_new =  [currentData repmat(labelsnum{labelIdx}, length(currentData), 1)];
            dataset_freqdomain_eval = [dataset_freqdomain_eval; currentData_new];
        end
    end 
end

%% Dataset time domain features without labels (Eval)

for i = 1:length(variable_prefixes_eval)
    prefix = variable_prefixes_eval{i};
    
    % Create a struct to store the results
    time_domain_feat_eval.(prefix) = struct();
    
    % Iterate through x, y, and z
    for axis = {'x'}
        time_variable_name = sprintf('%s_%s', prefix, axis{1});
        
        % Extract the variable
        data = eval(variable_name);
        numBatches = ceil(length(data) / (batchSize * (1 - overlap)));
        
        % Apply the fft_sliding_window function
        time_domain_feat_eval.(prefix).(time_variable_name) = calculateTimeDomainFeatures(data(1:300000,1), numBatches, batchSize, overlapSize);
        
    end
end

%% Dataset time domain features with labels (Eval)

% Step 1: Define labels
labels = {'Normal', 'Cage', 'IR', 'OR'};
labelsnum = {1, 2, 3, 4};
dim = {'_x'};

% Step 2-4: Iterate through the structure and extract data and labels
dataset_timedomain_eval = [];

for labelIdx = 1:length(labels)
    currentLabel = labels{labelIdx};
    for dimIdx = 1:length(dim)
        currentDim = dim{dimIdx};
        for sampleIdx = 50 %changed for training 
            fieldName1 = [currentLabel '_' num2str(sampleIdx)];
            fieldName2 = [currentLabel '_' num2str(sampleIdx) currentDim];
            currentData = time_domain_feat_eval.(fieldName1).(fieldName2);
        
            currentData_new =  [currentData repmat(labelsnum{labelIdx}, length(currentData), 1)];% Concatenate data
            dataset_timedomain_eval = [dataset_timedomain_eval; currentData_new];
        end
    end 
end

%% DATASET FOR MACHINE LEARNING PART

% Combine time-domain and frequency-domain datasets for training and evaluation
TOTAL_DATA_TRAIN = [dataset_timedomain_train(1:size(dataset_freqdomain_train, 1), 1:end-1), dataset_freqdomain_train];
TOTAL_DATA_EVAL = [dataset_timedomain_eval(1:size(dataset_freqdomain_eval, 1), 1:end), dataset_freqdomain_eval];

%% Classifiers and ML Part 

% Split data into training and evaluation sets
trainData = TOTAL_DATA_TRAIN;         % Training data
evalData = TOTAL_DATA_EVAL;           % Evaluation data

% Set seed for reproducibility
seed = 42;
rng(seed);

% Shuffle the training and evaluation data
shuffledTrainData = trainData(randperm(size(trainData, 1)), :);
shuffledEvalData = evalData(randperm(size(evalData, 1)), :);

% Separate features and labels for training and evaluation
X_train = shuffledTrainData(:, 1:end-1);   % Features for training
y_train = shuffledTrainData(:, end);       % Labels for training

X_eval = shuffledEvalData(:, 1:end-1);     % Features for evaluation
y_eval = shuffledEvalData(:, end);         % Labels for evaluation

%% Split training data into train and test sets
splitRatio = 0.8;  % 80% for training, 20% for testing
splitIdx = round(splitRatio * size(X_train, 1));

X_trainFinal = X_train(1:splitIdx, :);  % Final training features
y_trainFinal = y_train(1:splitIdx);    % Final training labels

X_test = X_train(splitIdx+1:end, :);   % Test features
y_test = y_train(splitIdx+1:end);      % Test labels

%% Train a multiclass SVM classifier
svmModel = fitcecoc(X_trainFinal, y_trainFinal);

%% Make predictions
trainPredictions = predict(svmModel, X_trainFinal);  % Predictions on training data
testPredictions = predict(svmModel, X_test);         % Predictions on test data

%% Calculate and display accuracy
trainAccuracy = sum(trainPredictions == y_trainFinal) / numel(y_trainFinal);
testAccuracy = sum(testPredictions == y_test) / numel(y_test);

disp(['Train Accuracy: ', num2str(trainAccuracy * 100), '%']);
disp(['Test Accuracy: ', num2str(testAccuracy * 100), '%']);
