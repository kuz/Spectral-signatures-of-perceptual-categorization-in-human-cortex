% parameters
ei = exist('indata') == 1;
if ei ~= 1
    disp('Required varibles are not set! Terminating.')
    exit
end
freqlist = 5:5:149;
windows = [[50 250]; [100 300]; [150 350]; [200 400]; [250 450]; [300 500]];
nfreqs = length(freqlist);
nwin = length(windows);

% create output directory
outdata = [indata '_responsive'];
if exist(['../../Data/Intracranial/Processed/' outdata], 'dir') == 7
    disp(['Directory exists: ' outdata ', exiting...']);
    exit()
end
mkdir(['../../Data/Intracranial/Processed/' outdata]);

% load data
load(['../../Outcome/Probe responsiveness/responsiveness_' indata '.mat']);
listing = dir(['../../Data/Intracranial/Processed/' indata '/*.mat']);

% mark for dropping the probes that are not significant or their response ratio is less that 0.2 away from 1.0
for wid = 1:nwin
    for fid = 1:nfreqs
        pID = fdr(results{wid, fid}(:, 3), 0.05);
        ratio = results{wid, fid}(:, 6) ./ results{wid, fid}(:, 5);
        results{wid, fid}(:, 4) = (ratio > 0.8 & ratio < 1.2) | results{wid, fid}(:, 3) >= pID;
    end
end

% drop only the probes that were not responsive in any of the regions of interest
nprobes = size(results{1, 1}, 1);
todrop = zeros(nprobes, 2);
todrop(:, 1) = results{1, 1}(:, 1);

% for each probe
for pid = 1:nprobes
    
    % count how many time this probe was not dropped (e.g. was useful)
    times_not_dropped = 0;
    for wid = 1:nwin
        for fid = 1:nfreqs
            if results{wid, fid}(pid, 4) == 0
                times_not_dropped = times_not_dropped + 1;
            end
        end
    end

    % if it was useless most of the times useless we mark it for dropping
    if times_not_dropped < 3
        todrop(pid, 2) = 1;
    end
end


% drop all of the probes marked for dropping
nsurvivors = 0;
for si = 1:length(listing)
    sfile = listing(si);
    disp(['Cleaning ' sfile.name ': dropping ' ...
          num2str(length(todrop(todrop(:, 1) == si & todrop(:, 2) == 1, 2))) ...
          ' (out of ' num2str(length(todrop(todrop(:, 1) == si))) ') probes'])
    
    % load the data
    load(['../../Data/Intracranial/Processed/' indata '/' sfile.name]);
    
    % drop discarded probes
    keepidx = results{1, 1}(todrop(:, 1) == si & todrop(:, 2) == 0, 2);
    s.probes.rod_names = s.probes.rod_names(keepidx);
    s.probes.probe_ids = s.probes.probe_ids(keepidx);
    s.probes.mni = s.probes.mni(keepidx, :);
    s.probes.areas = s.probes.areas(keepidx);
    s.data = s.data(:, keepidx, :);
    nsurvivors = nsurvivors + length(keepidx);

    % store the data
    save(['../../Data/Intracranial/Processed/' outdata '/' sfile.name], 's');
    clearvars -except indata nfreqs nwin outdata listing results nsurvivors todrop
    
end

disp(['In total survived ' num2str(nsurvivors) ' probes'])
