% variables
ei = exist('indata') == 1;
es = exist('sid') == 1;
if ei + es ~= 2
    disp('Required varibles are not set! Terminating')
    exit
end

% parameters
outdata = ['ft_4hz150_' indata];
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
nfreq = 146; % from 4Hz to 149Hz inclusive
    
sfile = listing(sid);
disp(['Processing ' sfile.name])

% load the data
s = load(['../../Data/Intracranial/Processed/' indata '/' sfile.name]);
s = s.s;

nstim = length(s.stimseq);
nprobes = length(s.probes.probe_ids);

% for each probe
for probe = 1:nprobes

    ft = zeros(length(s.stimseq), nfreq, 48);

    % display progress
    fprintf('\r%d / %d', probe, nprobes);

    % for each stimulus
    for stimulus = 1:nstim

        % take the signal
        signal = detrend(squeeze(s.data(stimulus, probe, :)));
        
        % to do artifact rejection we have dropped some number of
        % images from each of the probes, now each proble has varying
        % number of "trials" (images), to keep the data in matrix
        % format we inroduce a "poison pill" value of -123456 -- the
        % images with this values as a response should be excluded
        % from further analysis
        if sum(signal) == 0.0
            ft(stimulus, :, :) = zeros(nfreq, 48);
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
        [power3, faxis, times, period] = waveletspectrogram(signal', 512, 'freqlimits', [4 8], 'ncycles', 3);
        [power4, faxis, times, period] = waveletspectrogram(signal', 512, 'freqlimits', [9 14], 'ncycles', 4);
        [power5, faxis, times, period] = waveletspectrogram(signal', 512, 'freqlimits', [15 29], 'ncycles', 5);
        [power6, faxis, times, period] = waveletspectrogram(signal', 512, 'freqlimits', [30 149], 'ncycles', 6);

        power3 = flipud(power3);
        power4 = flipud(power4);
        power5 = flipud(power5);
        power6 = flipud(power6);

        for i = 1:16:768
            ft(stimulus, 1:5,    (i+15)/16) = mean(power3(:, i:i+15), 2);
            ft(stimulus, 6:11,   (i+15)/16) = mean(power4(:, i:i+15), 2);
            ft(stimulus, 12:26,  (i+15)/16) = mean(power5(:, i:i+15), 2);
            ft(stimulus, 27:146, (i+15)/16) = mean(power6(:, i:i+15), 2);
        end
    end

    save(['../../Data/Intracranial/Processed/' outdata '/' s.name '-' num2str(probe) '.mat'], 'ft');
end
