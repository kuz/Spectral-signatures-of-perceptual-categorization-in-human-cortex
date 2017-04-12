% variables
ei = exist('indata') == 1;
ef = exist('freqlimits') == 1;
ec = exist('ncycles') == 1;
ew = exist('window') == 1;
if ei + ef + ec + ew ~= 4
    disp('Required varibles are not set! Terminating')
    exit
end

% parameters
parpool(20);
w_sta_ms = window(1);
w_end_ms = window(2);
w_sta_t = round(w_sta_ms / 1000 * 512);
w_end_t = round(w_end_ms / 1000 * 512);
outdata = ['mean_' num2str(freqlimits(1)) 'hz' num2str(freqlimits(2)) '_' num2str(window(1)) 'ms' num2str(window(2)) '_' indata];
if exist(['../../Data/Intracranial/Processed/' outdata], 'dir') == 7
    disp(['WARNING: Directory exists: ' outdata])
else
    mkdir(['../../Data/Intracranial/Processed/' outdata])
end

% load third party code
addpath('spectra')

% load subject list
listing = dir(['../../Data/Intracranial/Processed/' indata '/*.mat']);
nsubject = length(listing);

% for each subject
parfor sid = 1:nsubject
    
    sfile = listing(sid);
    disp(['Processing ' sfile.name])
    
    % load the data
    s = load(['../../Data/Intracranial/Processed/' indata '/' sfile.name]);
    s = s.s;
    
    % output data structure
    meanband = zeros(length(s.stimseq), length(s.probes.probe_ids));
    
    % for each stimulus
    nstim = length(s.stimseq);
    for stimulus = 1:length(s.stimseq)
    
        % display progress
        fprintf('\r%d / %d', stimulus, nstim);
        
        % for each probe
        for probe = 1:length(s.probes.probe_ids)
    
            % take the signal
            signal = detrend(squeeze(s.data(stimulus, probe, :)));
            
            % to do artifact rejection we have dropped some number of
            % images from each of the probes, now each proble has varying
            % number of "trials" (images), to keep the data in matrix
            % format we inroduce a "poison pill" value of -123456 -- the
            % images with this values as a response should be excluded
            % from further analysis
            if sum(signal) == 0.0
                meanband(stimulus, probe) = -123456;
                continue
            end
            
            % filter the signal
            for f = [50, 100, 150, 200, 250]
                Wo = f / (512 / 2);
                BW = Wo / 50;
                [b,a] = iirnotch(Wo, BW);
                signal = filter(b, a, signal);
            end
            
            % wavelet transform
            [power, faxis, times, period] = waveletspectrogram(signal', 512, 'freqlimits', freqlimits, 'ncycles', ncycles);

            % take baseline for later normalization
            baseline_at = 205; % baseline from -500 to -100
            baseline = power(:, 1:baseline_at);

            % take only part of the signal
            stimulus_at = 256;
            from = stimulus_at + w_sta_t;
            till = stimulus_at + w_end_t;
            fqsignal = power(:, from:till);

            % compute baseline normalized band response
            %lhz = freqlimits(1);
            %hhz = freqlimits(2);
            %baseline = baseline(lhz:hhz, :);
            %signal = fqsignal(lhz:hhz, :);
            %meanband(stimulus, probe) = mean2(signal) / mean2(baseline)
            meanband(stimulus, probe) = mean2(fqsignal) / mean2(baseline);
            
        end
    end
    
    % store extracted features
    s.data = meanband;
    parsave(['../../Data/Intracranial/Processed/' outdata '/' sfile.name], s);
    
    % clear all subject-specific variables
    %clearvars -except listing indata outdata freqlimits ncycles window w_sta_ms w_end_ms w_sta_t w_end_t
    fprintf('\n')

end

