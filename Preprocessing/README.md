## Order of execution

1. `matlab -nojvm -nodisplay -nosplash -r "structured_to_processed.m; exit"` to extract chunks of LFP recordings for each of the stimuli.  
2. Drop trials with artifacts: `matlab -nojvm -nodisplay -nosplash -r "indata = 'LFP_8c'; preprocess_lfp_artifacts; exit"`  
3. Bipolar referencing: `matlab -nojvm -nodisplay -nosplash -r "indata = 'LFP_8c_artif'; preprocess_lfp_to_bipolar; exit"`  
4. Supply with area number: `matlab -nojvm -nodisplay -nosplash -r "indata = 'LFP_8c_artif_bipolar'; preprocess_add_area; exit"`  
5. Compute electrode responsiveness `matlab -nodisplay -nosplash -r "indata='LFP_8c_artif_bipolar_BA'; preprocess_compute_responsiveness; exit"`  
6. Filter out non-responsive probes `matlab -nojvm -nodisplay -nosplash -r "indata='LFP_8c_artif_bipolar_BA'; preprocess_keep_responsive; exit"`  
7. Extract features `./extract_features.sh` (look inside if you wish to change some parameters)  
8. `python DataBuilder.py -f mean_5hz9_0ms200_LFP_8c_artif_bipolar_BA_responsive`

This concludes data preprocessing. Proceed to `Classification`.

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

## Subjects with non-512 Hz sampling rate

```
1024 Y GRE_2014_AUDe
1024 Y GRE_2014_BARf
1024 Y GRE_2014_BASc
1024 Y GRE_2014_LESs
2048 Y LYONNEURO_2014_CHOK
2048 Y LYONNEURO_2014_DESJ
2048 Y LYONNEURO_2014_FEPK
2048 Y LYONNEURO_2014_FRAP
2048 Y LYONNEURO_2014_PERRa
2048 Y LYONNEURO_2014_REID
2048 Y LYONNEURO_2015_BOUC
2048 Y LYONNEURO_2015_BOUc1
2048 Y LYONNEURO_2015_FEPK
2048 Y LYONNEURO_2015_GAUC
2048 Y LYONNEURO_2015_JIMa
2048 Y LYONNEURO_2015_MENj
2048 Y LYONNEURO_2015_PASj
2048 Y LYONNEURO_2015_PELS
2048 Y LYONNEURO_2015_SAAY
2048 Y LYONNEURO_2015_SCAl
```