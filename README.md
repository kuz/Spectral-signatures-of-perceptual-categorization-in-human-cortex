Intracranial Decoding
=====================

Start with preprocessing by following the steps outlined in the `Preprocessing` directory.  
After that perform the decoding as explained in `Classification`.  
To obtain final figures and analysis results follow the `Analysis` section.

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