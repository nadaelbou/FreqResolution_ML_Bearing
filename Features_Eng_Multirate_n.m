%%% Resampling and FFT

tic
clear all
clc;
close all

%% Interesting Source 

% https://se.mathworks.com/help/predmaint/ref/bearingfaultbands.html
% https://www.google.com/search?q=cage+fault+signature+&tbm=isch&ved=2ahUKEwjusoX1tumDAxXZGxAIHQwgBpEQ2-cCegQIABAA&oq=cage+fault+signature+&gs_lcp=CgNpbWcQAzoECCMQJ1DGA1j6C2DDDGgAcAB4AIABTogBiQaSAQIxMpgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=0WeqZa6GEtm3wPAPjMCYiAk&bih=1086&biw=1912#imgrc=8a3VIVJr43aUkM&imgdii=5X_QDbL7rc4rRM
% Tallinn University Dataset 

%% Import 

% Healthy 

load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Healthy\0 checked.mat')
Normal_0_x = ch_AI_7_1;
Normal_0_x_TIME = ch_AI_7_1_TIME;
Normal_0_y = ch_AI_7_2;
Normal_0_y_TIME = ch_AI_7_2_TIME;
Normal_0_z = ch_AI_7_3;
Normal_0_z_TIME = ch_AI_7_3_TIME;
load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Healthy\25.mat')
Normal_25_x = ch_AI_7_1;
Normal_25_x_TIME = ch_AI_7_1_TIME;
Normal_25_y = ch_AI_7_2;
Normal_25_y_TIME = ch_AI_7_2_TIME;
Normal_25_z = ch_AI_7_3;
Normal_25_z_TIME = ch_AI_7_3_TIME;
load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Healthy\50.mat')
Normal_50_x = ch_AI_7_1;
Normal_50_x_TIME = ch_AI_7_1_TIME;
Normal_50_y = ch_AI_7_2;
Normal_50_y_TIME = ch_AI_7_2_TIME;
Normal_50_z = ch_AI_7_3;
Normal_50_z_TIME = ch_AI_7_3_TIME;
load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Healthy\75.mat')
Normal_75_x = ch_AI_7_1;
Normal_75_x_TIME = ch_AI_7_1_TIME;
Normal_75_y = ch_AI_7_2;
Normal_75_y_TIME = ch_AI_7_2_TIME;
Normal_75_z = ch_AI_7_3;
Normal_75_z_TIME = ch_AI_7_3_TIME;
load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Healthy\100.mat')
Normal_100_x = ch_AI_7_1;
Normal_100_x_TIME = ch_AI_7_1_TIME;
Normal_100_y = ch_AI_7_2;
Normal_100_y_TIME = ch_AI_7_2_TIME;
Normal_100_z = ch_AI_7_3;
Normal_100_z_TIME = ch_AI_7_3_TIME;

% Cage 

load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Cage\0.mat')
Cage_0_x = ch_AI_7_1;
Cage_0_x_TIME = ch_AI_7_1_TIME;
Cage_0_y = ch_AI_7_2;
Cage_0_y_TIME = ch_AI_7_2_TIME;
Cage_0_z = ch_AI_7_3;
Cage_0_z_TIME = ch_AI_7_3_TIME;
load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Cage\25.mat')
Cage_25_x = ch_AI_7_1;
Cage_25_x_TIME = ch_AI_7_1_TIME;
Cage_25_y = ch_AI_7_2;
Cage_25_y_TIME = ch_AI_7_2_TIME;
Cage_25_z = ch_AI_7_3;
Cage_25_z_TIME = ch_AI_7_3_TIME;
load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Cage\50.mat')
Cage_50_x = ch_AI_7_1;
Cage_50_x_TIME = ch_AI_7_1_TIME;
Cage_50_y = ch_AI_7_2;
Cage_50_y_TIME = ch_AI_7_2_TIME;
Cage_50_z = ch_AI_7_3;
Cage_50_z_TIME = ch_AI_7_3_TIME;
load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Cage\75.mat')
Cage_75_x = ch_AI_7_1;
Cage_75_x_TIME = ch_AI_7_1_TIME;
Cage_75_y = ch_AI_7_2;
Cage_75_y_TIME = ch_AI_7_2_TIME;
Cage_75_z = ch_AI_7_3;
Cage_75_z_TIME = ch_AI_7_3_TIME;
load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\Cage\100.mat')
Cage_100_x = ch_AI_7_1;
Cage_100_x_TIME = ch_AI_7_1_TIME;
Cage_100_y = ch_AI_7_2;
Cage_100_y_TIME = ch_AI_7_2_TIME;
Cage_100_z = ch_AI_7_3;
Cage_100_z_TIME = ch_AI_7_3_TIME;

% IR Fault 

load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\IR\0.mat')
IR_0_x = ch_AI_7_3;
IR_0_x_TIME = ch_AI_7_3_TIME;
IR_0_y = ch_AI_7_3;
IR_0_y_TIME = ch_AI_7_3_TIME;
IR_0_z = ch_AI_7_3;
IR_0_z_TIME = ch_AI_7_3_TIME;
load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\IR\25.mat')
IR_25_x = ch_AI_7_3;
IR_25_x_TIME = ch_AI_7_3_TIME;
IR_25_y = ch_AI_7_3;
IR_25_y_TIME = ch_AI_7_3_TIME;
IR_25_z = ch_AI_7_3;
IR_25_z_TIME = ch_AI_7_3_TIME;
load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\IR\50.mat')
IR_50_x = ch_AI_7_3;
IR_50_x_TIME = ch_AI_7_3_TIME;
IR_50_y = ch_AI_7_3;
IR_50_y_TIME = ch_AI_7_3_TIME;
IR_50_z = ch_AI_7_3;
IR_50_z_TIME = ch_AI_7_3_TIME;
load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\IR\75.mat')
IR_75_x = ch_AI_7_3;
IR_75_x_TIME = ch_AI_7_3_TIME;
IR_75_y = ch_AI_7_3;
IR_75_y_TIME = ch_AI_7_3_TIME;
IR_75_z = ch_AI_7_3;
IR_75_z_TIME = ch_AI_7_3_TIME;
load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\IR\100.mat')
IR_100_x = ch_AI_7_3;
IR_100_x_TIME = ch_AI_7_3_TIME;
IR_100_y = ch_AI_7_3;
IR_100_y_TIME = ch_AI_7_3_TIME;
IR_100_z = ch_AI_7_3;
IR_100_z_TIME = ch_AI_7_3_TIME;

% OR Fault 

load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\OR\0.mat')
OR_0_x = ch_AI_7_3;
OR_0_x_TIME = ch_AI_7_3_TIME;
OR_0_y = ch_AI_7_3;
OR_0_y_TIME = ch_AI_7_3_TIME;
OR_0_z = ch_AI_7_3;
OR_0_z_TIME = ch_AI_7_3_TIME;
load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\OR\25.mat')
OR_25_x = ch_AI_7_3;
OR_25_x_TIME = ch_AI_7_3_TIME;
OR_25_y = ch_AI_7_3;
OR_25_y_TIME = ch_AI_7_3_TIME;
OR_25_z = ch_AI_7_3;
OR_25_z_TIME = ch_AI_7_3_TIME;
load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\OR\50.mat')
OR_50_x = ch_AI_7_3;
OR_50_x_TIME = ch_AI_7_3_TIME;
OR_50_y = ch_AI_7_3;
OR_50_y_TIME = ch_AI_7_3_TIME;
OR_50_z = ch_AI_7_3;
OR_50_z_TIME = ch_AI_7_3_TIME;
load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\OR\75.mat')
OR_75_x = ch_AI_7_3;
OR_75_x_TIME = ch_AI_7_3_TIME;
OR_75_y = ch_AI_7_3;
OR_75_y_TIME = ch_AI_7_3_TIME;
OR_75_z = ch_AI_7_3;
OR_75_z_TIME = ch_AI_7_3_TIME;
load('Desktop\Conference & Journal\ICEM (IEEE) - Stray Flux\Dataset\OR\100.mat')
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

IR_0_x = IR_0_x(6962:min_size, :);s
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

%% Global variables 

variable_prefixes_train = {'Normal_0', 'Cage_0', 'IR_0', 'OR_0','Normal_100', 'Cage_100', 'IR_100', 'OR_100'}; 

%% Calculate features train

M = 1; % [CHOICE] Decimation 
L = 1; % [CHOICE] Interpolation 

batchSize = length(6962:min_size)*M/L;

fs = 20*1e3;  % [Hz] Sampling Frequency 

ep = 19; % Maximum sequence length acknowledging the size of our signal/to be adjusted to your signal length - Based on floor(log2(batchSize));

for exponent = 7: ep
    tic
    
    % Define batch size
    batchSize = pow2(exponent); % You can adjust this if needed/fix its value if you want. 
    
    %% Training Phase
    for i = 1:length(variable_prefixes_train)
        prefix = variable_prefixes_train{i};
        
        % Create a struct to store the results for training
        time_domain_feat_train.(prefix) = struct();
        
        % Iterate through x (other axes could be added here)
        for axis = {'x'}
            variable_name = sprintf('%s_%s', prefix, axis{1});
            
            % Extract the variable
            data = eval(variable_name);
            
            % Resample the data
            data_res = resample(data, L, M);
            
            % Apply the feature extraction function
            feat_train.(prefix).(variable_name) = calculateFeatures_fs(data_res, batchSize, fs * L / M);
        end
    end
    
    %% Convert struct to matrix for training data
    fieldnames_feat = fieldnames(feat_train);
    data_train_labelled = [];
    
    for i = 1:length(fieldnames_feat)
        current_field = feat_train.(fieldnames_feat{i});
        
        % Extract the data from the current field
        field_name = strcat(fieldnames_feat{i}, '_x');
        current_data = current_field.(field_name); % Assuming there is a 'data_x' field
        
        % Determine the label based on the field name (Normal, OR, IR, Cage)
        if contains(fieldnames_feat{i}, 'Normal')
            label = 1;
        elseif contains(fieldnames_feat{i}, 'OR')
            label = 2;
        elseif contains(fieldnames_feat{i}, 'IR')
            label = 3;
        elseif contains(fieldnames_feat{i}, 'Cage')
            label = 4;
        else
            label = 0; % Handle other cases if needed
        end
        
        % Append the label and add the data to the matrix
        data_train_labelled = [data_train_labelled; [current_data, repmat(label, [size(current_data, 1), 1])]];
    end
    
    %% Evaluation Phase
    variable_prefixes_eval = {'Normal_50', 'Cage_50', 'IR_50', 'OR_50'};
    fs = 20 * 1e3;  % Sampling frequency
    
    for i = 1:length(variable_prefixes_eval)
        prefix = variable_prefixes_eval{i};
        
        % Create a struct to store the results for evaluation
        feat_eval.(prefix) = struct();
        
        % Iterate through x (other axes could be added here)
        for axis = {'x'}
            variable_name = sprintf('%s_%s', prefix, axis{1});
            
            % Extract the variable
            data = eval(variable_name);
            
            % Resample the data
            data_res = resample(data, L, M);
            
            % Apply the feature extraction function
            feat_eval.(prefix).(variable_name) = calculateFeatures_fs(data_res, batchSize, fs);
        end
    end
    
    %% Convert struct to matrix for evaluation data
    fieldnames_feat = fieldnames(feat_eval);
    data_eval_labelled = [];
    
    for i = 1:length(fieldnames_feat)
        current_field = feat_eval.(fieldnames_feat{i});
        
        % Extract the data from the current field
        field_name = strcat(fieldnames_feat{i}, '_x');
        current_data = current_field.(field_name); % Assuming there is a 'data_x' field
        
        % Determine the label based on the field name (Normal, OR, IR, Cage)
        if contains(fieldnames_feat{i}, 'Normal')
            label = 1;
        elseif contains(fieldnames_feat{i}, 'OR')
            label = 2;
        elseif contains(fieldnames_feat{i}, 'IR')
            label = 3;
        elseif contains(fieldnames_feat{i}, 'Cage')
            label = 4;
        else
            label = 0; % Handle other cases if needed
        end
        
        % Append the label and add the data to the matrix
        data_eval_labelled = [data_eval_labelled; [current_data, repmat(label, [size(current_data, 1), 1])]];
    end
    
    %% Machine Learning - Dataset (Label - Features)
    seed = 3;
    rng(seed);
    
    dataWithLabels_train = data_train_labelled;  % Train
    dataWithLabels_eval = data_eval_labelled;    % Test
    
    % Randomize the data order
    selected_data_train = dataWithLabels_train(randperm(size(dataWithLabels_train, 1)), :);
    selected_data_eval = dataWithLabels_eval(randperm(size(dataWithLabels_eval, 1)), :);
    
    % Split data into features and labels
    selected_X_train = selected_data_train(:, 1:end-1);
    selectedy_train = selected_data_train(:, end);
    
    selected_X_eval = selected_data_eval(:, 1:end-1);
    selectedy_eval = selected_data_eval(:, end);
    
    %% SVM Classifier
    X_train = selected_X_train;
    y_train = selectedy_train;
    X_test = selected_X_eval;
    y_test = selectedy_eval;
    
    svmModel = fitcecoc(X_train, y_train);
    
    predictions_train = predict(svmModel, X_train);
    predictions = predict(svmModel, X_test);
    
    % Calculate accuracy
    accuracy = sum(predictions == y_test) / numel(y_test);
    accuracy_train = sum(predictions_train == y_train) / numel(y_train);
    
    disp(['Train Accuracy (SVM): ', num2str(accuracy_train * 100), '%']);
    disp(['Test Accuracy (SVM): ', num2str(accuracy * 100), '%']);
    
    %% Random Forest Classifier
    numTrees = 100;  % Adjust this as needed
    rf = TreeBagger(numTrees, X_train, y_train, 'Method', 'classification');
    
    % Make predictions on the test set
    y_pred_cell_test = predict(rf, X_test);
    y_pred_RF_test = categorical(y_pred_cell_test); % Convert to categorical array
    
    % Make predictions on the train set
    y_pred_cell_train = predict(rf, X_train);
    y_pred_RF_train = categorical(y_pred_cell_train); % Convert to categorical array
    
    % Convert Y_test to categorical if not already
    y_test_RF = categorical(y_test);
    y_train_RF = categorical(y_train);
    
    % Evaluate accuracy of Random Forest classifier
    accuracy_test_RF = sum(y_pred_RF_test == y_test_RF) / numel(y_test_RF);
    accuracy_train_RF = sum(y_pred_RF_train == y_train_RF) / numel(y_train_RF);
    
    disp(['Train Accuracy (RF): ', num2str(accuracy_train_RF * 100), '%']);
    fprintf('Test Accuracy (RF): %.2f%%\n', accuracy_test_RF * 100);
    
    toc
end
