# TOPA
official implementation of the paper TOPA: Target Oriented Prototype Adaptation for Cross-Subject Motor Imagery EEG Decoding

![[Abstract]([https://github.com/yulom/TOPA/edit/main/abstract.png](https://github.com/yulom/TOPA/blob/main/abstract.png))](https://github.com/yulom/TOPA/blob/main/abstract.png)

# Data download and preprocessing
get BCI competition IV dataset from https://www.bbci.de/competition/iv/

get SHU 3C dataset from https://plus.figshare.com/articles/dataset/Brain_Computer_Interface_Motor_Imagery-EEG_Dataset/22671172

a demo eeg preprocess pipeline code is get_raw_data.m

# EEGEncoder
three architecture included in the paper can be found in:

SST-DPN: https://github.com/hancan16/SST-DPN

EEG-Conformer: https://github.com/eeyhsong/EEG-Conformer

EEGNET: https://github.com/aliasvishnu/EEGNet

# Demo code
a simple demo code to compute TOPA loss is TOPA_demo.py

a implementation for EEG cross-subjects classification is TOPA_main.ipynb
