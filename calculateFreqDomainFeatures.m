function features_freq_domain = calculateFreqDomainFeatures(data,  batchSize, overlapSize)
    % Initialize the features_freq_domain vector

    features_freq_domain = [];

    signal_length = length(data);

    step_size = round((1 - overlapSize) * batchSize);
    
    for startIdx = 1:step_size:(signal_length - batchSize + 1)
        endIdx = startIdx + batchSize - 1;
        % Extract the current batch of data
        batchData = data(startIdx:endIdx, 1);
        
        sig_f = batchData;    % Rename 

        % Alireza's paper - Frequency 
        FD1 = sum(sig_f) / length(sig_f);
        FD2 = sum((sig_f - FD1).^2) / (length(sig_f) - 1);
        FD3 = sum((sig_f - FD1).^3) / (length(sig_f) * (sqrt(FD2).^3));
        FD4 = sum((sig_f - FD1).^4) / (length(sig_f) * (FD2).^2);
        FD5 = sum((sig_f(batchSize) .* sig_f)) / sum(sig_f);
        FD6 = sum((sig_f(batchSize) - FD5)) / length(sig_f);
        FD7 = sqrt(sum(sig_f(batchSize)^2 * sig_f) / sum(sig_f));
        FD8 = sqrt(sum(sig_f(batchSize)^4 * sig_f) / sum(sig_f(batchSize)^2 * sig_f));
        FD9 = sum(sig_f(batchSize)^2 * sig_f) / sqrt(sum(sig_f) * sum(sig_f(batchSize)^4 * sig_f));
        FD10 = FD6 / FD5;
        FD11 = sum((sig_f(batchSize) - FD5).^3 .* sig_f) / length(sig_f) .* FD6^3;
        FD12 = sum((sig_f(batchSize) - FD5).^4 .* sig_f) / length(sig_f) .* FD6^4;

        % Store the batched spectrum
        batchedSpectrums_new = [FD1, FD2, FD3, FD4, FD5, FD6, FD7, FD8, FD9, FD10, FD11, FD12];

        % Freq Features Vector Update 
        features_freq_domain = [features_freq_domain; batchedSpectrums_new];
    end 
end
