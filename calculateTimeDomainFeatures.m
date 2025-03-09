function features_time_domain = calculateTimeDomainFeatures(data, numBatches, batchSize, overlapSize)
    % Initialize the features_time_domain vector
    features_time_domain = [];
    
    signal_length = length(data);

    step_size = round((1 - overlapSize) * batchSize);
    
   % for i = 1:numBatches-2
   
    for startIdx = 1:step_size:(signal_length - batchSize + 1)
        endIdx = startIdx + batchSize - 1;
        % Calculate start and end indices for the current batch

        % Extract the current batch of data
        batchData = data(startIdx:endIdx, 1);

        % Calculate feature of the batch
        sig = batchData; 

        TD1 = max(sig);
        TD2 = rms(sig);
        TD3 = kurtosis(sig);
        TD4 = skewness(sig);
        TD5 = min(sig);
        TD6 = var(sig);
        TD7 = peak2peak(sig);
        TD8 = mad(sig);             % median absolute deviation
        TD9 = peak2rms(sig);
        TD10 = max(abs(sig));
        TD11 = mean(sig);
        TD12 = TD2 / TD11;
        TD13 = TD10 / TD2;
        TD14 = TD10 / TD11;
        TD15 = (1 / length(sig)) * sum(abs(sig).^(1/2))^2;
        TD16 = TD1 / TD16;
        TD17 = (1 / length(sig)) * sum(abs(sig).^(2));
        TD18 = sqrt(sum(abs(sig).^(2)));
        TD19 = TD10 / TD16;

        % Store the batched spectrum
        batchedSpectrums_new = [TD1, TD2, TD3, TD4, TD5, TD6, TD7, TD8, TD9, TD10, TD11, TD12, TD13, TD14, TD15, TD16, TD17, TD18, TD19];

        % Time Features Vector Update
        features_time_domain = [features_time_domain; batchedSpectrums_new];
    end 
end
