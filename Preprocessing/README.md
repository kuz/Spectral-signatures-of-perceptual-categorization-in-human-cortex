## Order of execution

1. `matlab -nojvm -nodisplay -nosplash -r "structured_to_processed.m; exit"` to extract chunks of LFP recordings for each of the stimuli.  
2. Drop trials with artifacts: `matlab -nojvm -nodisplay -nosplash -r "indata = 'LFP_8c'; preprocess_lfp_artifacts; exit"`  
3. Bipolar referencing: `matlab -nojvm -nodisplay -nosplash -r "indata = 'LFP_8c_artif'; preprocess_lfp_to_bipolar; exit"`  
4. Supply with area number: `matlab -nojvm -nodisplay -nosplash -r "indata = 'LFP_8c_artif_bipolar'; preprocess_add_area; exit"`  
5. Compute electrode responsiveness `matlab -nodisplay -nosplash -r "indata='LFP_8c_artif_bipolar_BA'; preprocess_compute_responsiveness; exit"`  
6. Filter out non-responsive probes `matlab -nojvm -nodisplay -nosplash -r "indata='LFP_8c_artif_bipolar_BA'; preprocess_filter_unresponsive; exit"`  
7. Extract features `./extract_features.sh` (look inside if you wish to change some parameters)  



## List of stimuli categories

    * H = house            10
    * V = visage           20
    * AA = animal          30
    * SCEE = scene         40
    * T = tool             50
    * tambulo = pseudoword 60
    * hgfjh = characters   70
    * F = target fruit     80 -- excluded from the analysis
    * SCR = scrambled      90

