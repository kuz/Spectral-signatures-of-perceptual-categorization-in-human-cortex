% variables
ei = exist('indata') == 1;
if ei ~= 1
    disp('Required varibles are not set! Terminating')
    exit
end

% parameters
outdata = [indata '_BA'];

% make output directory
mkdir(['../../Data/Intracranial/Processed/' outdata]);

% load libraries
addpath('mni2name')
addpath('nifti')
db = load_nii('mni2name/brodmann.nii');

% load subject list
listing = dir(['../../Data/Intracranial/Processed/' indata '/*.mat']);

% for each subject
for si = 1:length(listing)
    sfile = listing(si);
    disp(['Processing ' sfile.name])
    
    % load the data
    load(['../../Data/Intracranial/Processed/' indata '/' sfile.name]);
    
    % assign areas
    [~, areas] = mni2name_brodmann(s.probes.mni, db);
    areas = cell2mat(areas)';
    s.probes.areas = areas;
    
    % store the data
    save(['../../Data/Intracranial/Processed/' outdata '/' sfile.name], 's');
    clearvars -except indata outdata db listing
end
