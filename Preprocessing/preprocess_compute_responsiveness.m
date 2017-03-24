parpool(8)

% variables
ei = exist('indata') == 1;
if ei ~= 1
    disp('Required varibles are not set! Terminating')
    exit
end

% load third party code
addpath('spectra')

% parameters
freqlist = 5:5:149;
windows = [[50 250]; [100 300]; [150 350]; [200 400]; [250 450]; [300 500]];
outdata = ['responsiveness_' indata];
nfreqs = length(freqlist);
nwin = length(windows);

% load subject list
listing = dir(['../../Data/Intracranial/Processed/' indata '/*.mat']);

% matrix to store the final results
results = {};
for wid = 1:nwin
    for fid = 1:nfreqs
        results{wid, fid} = zeros(0, 6);
    end
end

% for each subject
for si = 1:length(listing)
    sfile = listing(si);
    disp(['Processing ' num2str(si) ' ' sfile.name])
    
    % load the data
    load(['../../Data/Intracranial/Processed/' indata '/' sfile.name]);
    
    % for each probe
    nprobes = length(s.probes.probe_ids);
    for probe = 1:nprobes
        
        % display progress
        fprintf('\r%d / %d', probe, nprobes);
         
        nstim = length(s.stimseq);
        baseline_band_means = zeros(nwin, nfreqs, nstim);
        fqsignal_band_means = zeros(nwin, nfreqs, nstim);

        % for each stimulus
        parfor stimulus = 1:length(s.stimseq)
            
            % take the signal
            signal = detrend(squeeze(s.data(stimulus, probe, :)));
            
            % filter the signal
            for f = [50, 100, 150, 200, 250]
                Wo = f / (512 / 2);
                BW = Wo / 50;
                [b,a] = iirnotch(Wo, BW);
                signal = filter(b, a, signal);
            end
            
            % wavelet transform
            [power_c3, faxis, times, period] = waveletspectrogram(signal', 512, 'freqlimits', [5 9], 'ncycles', 3);
            [power_c4, faxis, times, period] = waveletspectrogram(signal', 512, 'freqlimits', [10 14], 'ncycles', 4);
            [power_c5, faxis, times, period] = waveletspectrogram(signal', 512, 'freqlimits', [15 29], 'ncycles', 5);
            [power_c6, faxis, times, period] = waveletspectrogram(signal', 512, 'freqlimits', [30 150], 'ncycles', 6);
            
            % take baseline for later normalization
            baseline_at = 205; % baseline from -500 to -100
            
            for wid = 1:nwin
                w_sta_ms = windows(wid, 1);
                w_end_ms = windows(wid, 2);
                w_sta_t = round(w_sta_ms / 1000 * 512);
                w_end_t = round(w_end_ms / 1000 * 512);   

                % take only part of the signal
                stimulus_at = 256;
                from = stimulus_at + w_sta_t;
                till = stimulus_at + w_end_t;

                for fid = 1:nfreqs  
                    freq = freqlist(fid);
                    if freq == 5
                        fqsignal = power_c3(:, from:till);
                        baseline = power_c3(:, 1:baseline_at);
                    elseif freq == 10
                        fqsignal = power_c4(:, from:till);
                        baseline = power_c4(:, 1:baseline_at);
                    elseif freq == 15
                        fqsignal = power_c5(1:5, from:till);
                        baseline = power_c5(1:5, 1:baseline_at);
                    elseif freq == 20
                        fqsignal = power_c5(6:10, from:till);
                        baseline = power_c5(6:10, 1:baseline_at);
                    elseif freq == 25
                        fqsignal = power_c5(11:15, from:till);
                        baseline = power_c5(11:15, 1:baseline_at);
                    else
                        fqsignal = power_c6(freq - 30 + 1:freq - 30 + 5, from:till);
                        baseline = power_c6(freq - 30 + 1:freq - 30 + 5, 1:baseline_at);
                    end
                
                    % store frequency means
                    baseline_band_means(wid, fid, stimulus) = mean2(baseline);
                    fqsignal_band_means(wid, fid, stimulus) = mean2(fqsignal);

                end
                
            end
            
        end 

        % test the null hypothesis that baseline = signal in band means
        for wid = 1:nwin
            for fid = 1:nfreqs
                bbm = reshape(baseline_band_means(wid, fid, :), 1, []);
                fqbm = reshape(fqsignal_band_means(wid, fid, :), 1, []);
                p = signrank(bbm, fqbm);
                results{wid, fid} = [results{wid, fid}; range(1)-1+si, probe, p, 0, mean(bbm), mean(fqbm)];
            end
        end
    
    end

    % clear all subject-specific variables
    clearvars -except freqlist windows indata outdata nfreqs nwin listing results
    fprintf('\n');
    
end

% store the results
save(['../../Outcome/Probe responsiveness/' outdata], 'results')
