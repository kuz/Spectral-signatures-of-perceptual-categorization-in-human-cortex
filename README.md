Spectral signatures of perceptual categorization in human cortex
=====================

The code to support the findings in _Identifying task-relevant spectral signatures of perceptual categorization in the human cortex_ by Ilya Kuzovkin, Juan R. Vidal, Marcela Perrone-Bertlotti, Philippe Kahane, Sylvain Rheims, Jaan Aru, Jean-Philippe Lachaux, Raul Vicente  
https://www.biorxiv.org/content/early/2018/12/04/483487

Start with preprocessing by following the steps outlined in the `Preprocessing` directory.  
After that perform the decoding as explained in `Classification`.  
To obtain final figures and analysis results follow the `Analysis` section.

Data
----
The final spectrotemproal data is available from 
https://web.gin.g-node.org/ilyakuzovkin/Spectral-Signatures-of-Perceptual-Categorization-in-Human-Cortex.

Running
-------
1. Follow the instructions in [Preprocessing](Preprocessing) to prepare the data,
2. then train Random Forests as explained in [Classification](Classification) and
3. finally do the [Analysis](Analysis) to get to the conclusions!

Hints
-----
Initialize FreeSurfer
```
export FREESURFER_HOME=/Applications/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```
