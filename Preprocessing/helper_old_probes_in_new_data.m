%
% Given two featuresets find which probes in 2nd are present in 1st
%

indata_old = 'meanhighgamma_LFP_bipolar_noscram_artif_brodmann_w50_highgamma_resppositive';
indata_new = 'mean_70hz150_50ms250_LFP_8c_artif_bipolar_BA_resppositive';
listing_old = dir(['../../Data/Intracranial/Processed/' indata_old '/*.mat']);
listing_new = dir(['../../Data/Intracranial/Processed/' indata_new '/*.mat']);

total_new = 0;
total_old = 0;
total_match = 0;

si_old = 0;
disp('{')
for si_new = 1:length(listing_new)
  si_old = si_old + 1;
  sfile_old = listing_old(si_old);
  sfile_new = listing_new(si_new);

  if ~all(sfile_old.name == sfile_new.name)
    si_old = si_old + 1;
    sfile_old = listing_old(si_old);
  end

  s_old = load(['../../Data/Intracranial/Processed/' indata_old '/' sfile_old.name]);
  s_new = load(['../../Data/Intracranial/Processed/' indata_new '/' sfile_new.name]);
  total_new = total_new + size(s_new.s.probes.mni, 1);
  total_old = total_old + size(s_old.s.probes.mni, 1);

  results = [];
  for mni = s_old.s.probes.mni'
    if sum(mni) ~= 0
      pid = find(ismember(s_new.s.probes.mni, mni', 'rows')) - 1;
      if ~isempty(pid)
        total_match = total_match + 1;
        results = [results find(ismember(s_new.s.probes.mni, mni', 'rows')) - 1];
      end
    end
  end

  if length(results) > 0
    string_results = {};
    for r = results
      string_results{end + 1} = num2str(r);
    end
    disp(['"' sfile_new.name '": [' strjoin(string_results, ',') '],'])
   end 

end
disp('}')
disp(' ')
disp(['Total Old: ' num2str(total_old)])
disp(['Total New: ' num2str(total_new)])
disp(['Match: ' num2str(total_match)])