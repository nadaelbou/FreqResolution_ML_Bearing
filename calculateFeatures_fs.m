function features = calculateFeatures_fs(data,  batchSize, fs)


    features = [];

    signal_length = length(data);

    step_size = round(batchSize);
    
    for startIdx = 1:step_size:(signal_length - batchSize + 1)
        endIdx = round(startIdx + batchSize - 1);
        
        batchData = data(startIdx:endIdx, 1);

        TD1 = max(batchData);
        TD2 = rms(batchData);
        TD3 = kurtosis(batchData);
        TD4 = skewness(batchData);
        TD5 = min(batchData);
        TD6 = var(batchData);
        TD7 = peak2peak(batchData);
        TD8 = mad(batchData);             % median absolute deviation
        TD9 = peak2rms(batchData);
        TD10 = max(abs(batchData));
        TD11 = mean(batchData);
        TD12 = TD2 / TD11;
        TD13 = TD10 / TD2;
        TD14 = TD10 / TD11;
        TD15 = (1 / length(batchData)) * sum(abs(batchData).^(1/2))^2;
        TD16 = TD1 / TD15;
        TD17 = (1 / length(batchData)) * sum(abs(batchData).^(2));
        TD18 = sqrt(sum(abs(batchData).^(2)));
        TD19 = TD10 / TD15;

        % % Store the batched spectrum
        batchedSpectrums_TD = [TD1, TD2, TD3, TD4, TD5, TD6, TD7, TD8, TD9, TD10, TD11, TD12, TD13, TD14, TD15, TD16, TD17, TD18, TD19];
        

        % 2. Frequency-Domain Features 
        % 2.1 FFT 
        n = length(batchData);
        if rem(n,2) ~= 0
           batchData(end) = [];
           n = n-1;
        end

        x = batchData; 
        
        f = 0: fs/n :(fs/2)-fs/n ; 

        X  = fft( x, n );
        X  = abs(X);
        X  = X(1:n/2);
        X  = X/n;
        
        sig_f = X;    % Rename 

        batchSize_freq = length(X);

        % 2.2. Features (Alireza's paper)
        FD1 = sum(sig_f) / length(sig_f);
        FD2 = sum((sig_f - FD1).^2) / (length(sig_f) - 1);
        FD3 = sum((sig_f - FD1).^3) / (length(sig_f) * (sqrt(FD2).^3));
        FD4 = sum((sig_f - FD1).^4) / (length(sig_f) * (FD2).^2);
        FD5 = sum((sig_f(batchSize_freq) .* sig_f)) / sum(sig_f);
        FD6 = sum((sig_f(batchSize_freq) - FD5)) / length(sig_f);
        FD7 = sqrt(sum(sig_f(batchSize_freq)^2 * sig_f) / sum(sig_f));
        FD8 = sqrt(sum(sig_f(batchSize_freq)^4 * sig_f) / sum(sig_f(batchSize_freq)^2 * sig_f));
        FD9 = sum(sig_f(batchSize_freq)^2 * sig_f) / sqrt(sum(sig_f) * sum(sig_f(batchSize_freq)^4 * sig_f));
        FD10 = FD6 / FD5;
        FD11 = sum((sig_f(batchSize_freq) - FD5).^3 .* sig_f) / length(sig_f) .* FD6^3;
        FD12 = sum((sig_f(batchSize_freq) - FD5).^4 .* sig_f) / length(sig_f) .* FD6^4;

        % Store the batched spectrum
        batchedSpectrums_FD = [FD1, FD2, FD3, FD4, FD5, FD6, FD7, FD8, FD9, FD10, FD11, FD12];

        % Freq Features Vector Update 
        features_new = [batchedSpectrums_FD, batchedSpectrums_TD];
        features = [features; features_new];
    end 
end
