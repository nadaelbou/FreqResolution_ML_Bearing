# Multi-Rate Vibration Signal Analysis for Enhanced Data-Driven Monitoring of Bearing Faults in Induction Machines

In this repository, the codes related to the paper *"Multi-Rate Vibration Signal Analysis for Enhanced Data-Driven Monitoring of Bearing Faults in Induction Machines"* are provided. This work has been presented at the International Conference in Electrical Machines Conference (ICEM) in Turin, Italy, 2024. 

DOI: [10.1109/ICEM60801.2024.10700488](https://doi.org/10.1109/ICEM60801.2024.10700488)

## Dataset
The data supporting this study's findings have been obtained thanks to a collaboration with Tallinn University of Technology. Due to privacy and ethical restrictions, the data are not publicly available. However, any current signal measurements from a drive tested with and without bearing faults should be sufficient for fine-tuning. In the present case, the data have been acquired at 20kHz.

The experimental setup includes a tested induction motor (IM), a load machine, and a torque transducer with an encoder. The test motor is a 7.5 kW machine operating at a grid voltage of 50 Hz. An ABB ACS600 industrial drive controls the load motor, allowing for the application and regulation of variable load torque to the test IM. Further details can be find in the paper.

For the purpose of this study, I trained the models using data from 0% and 100% loading levels and evaluated them using data from the 50% loading level.

## Codes 
The codes are written on MATLAB as I prefer this language when it comes to signal processing. The ML part is also performed on MALTAB, despite the fact that Python is definitely a better choice for this part. If preferred, I suggest saving the features extracted and processing the data on Python. 

In this repository, find two types of codes: 
1. Features_Eng.m: extract time and frequency domain features for vibration signals at a given sampling rate (fixed frequency resolution). This function calls calculateFreqDomainFeatures.m and calculateTimeDomainFeatures.m.
2. Features_Eng_MultiRate_fs.m: Extracts time and frequency domain features for vibration signals for difference batchsize at different sampling rates (fixed resolution) and then trains SVM/RF classifier to assess the feature quality for different frequency resolution. This function calls calculateFeatures_fs.m, which takes parameters N and M to interpolate and decimate the signals, allowing you to select the desired signal rates for analysis.
3. Features_Eng_MultiRate_n.m: Extracts time and frequency domain features for vibration signals for difference batchsize at a fixed sampling rate (varied resolutions) and then trains SVM/RF classifier to assess the feature quality for different frequency resolution. This function calls calculateFeatures_fs.m, which takes parameters N and M to interpolate and decimate the signals, allowing you to select the desired signal rates for analysis.


📄 **Citation:**  
```
N. El Bouharrouti et al., "Multi-Rate Vibration Signal Analysis for Enhanced Data-Driven Monitoring of Bearing Faults in Induction Machines," 2024 International Conference on Electrical Machines (ICEM), Torino, Italy, pp. 1-7, 2024.
```

