This is part of bachelor thesis “The Brain-Computer Interface for Rehabilitation After Stroke Using EEG Signals and Virtual Reality Glasses”

Author(s)			Anete Savicka, CSOT2
Supervisors		Jānis Lazovskis, Ph.D
				      Maksims Ivanovs, Mg. sc. cogn.

This thesis investigates the potential of EEG-based BCIs in stroke rehabilitation. The study focuses on improving rehabilitation effectiveness by incorporating virtual reality (VR) glasses, aiming to provide stroke patients with a more realistic experience. 
A key focus is identifying the optimal combination of data preprocessing, normalization, and classification techniques to achieve the most accurate machine learning model for EEG signal processing. The research evaluated the performance of the developed model in classifying EEG signals associated with right-hand movements using data from multiple file recordings and real-time data. While accuracy varied across individual recordings, the average classification accuracy reached 88.3% when external factors were similar (range: 72.15% - 95.86%). This was achieved by a combination of a notch filter of 50Hz, a bandpass filter from 5 to 50Hz, and z-score normalization.
However, external factors such as variations in recording conditions, electromagnetic fields, and connections between the head and electrodes introduce inconsistencies in the recorded data across different sessions. While z-score normalization achieved promising results when these factors were similar, further refinement is necessary. Normalization could be the step where data could be adjusted to become comparable session-to-session, which is crucial for improvement. Various techniques, including common average referencing, adaptive filtering, and calibration periods, were explored to address this issue, but none achieved improvements in accuracy across all types of data influenced by external factors. This remains the primary challenge and area requiring further investigation and improvement.

Setup for experiment using OpenBCI EEG and Cyton Daisy board
![image](https://github.com/AneteSavicka/OpenBCI_EEG_Stroke_Rehab/assets/71130798/09abf6e0-9754-4dde-877f-143121862ff1)

EEG electrodes used in the experiment based on 10-20 system
![image](https://github.com/AneteSavicka/OpenBCI_EEG_Stroke_Rehab/assets/71130798/349db510-ad0b-4b2e-854a-9eed82b17c5b)

File structure for data recording, preprocessing, and validation with file and real-time
![image](https://github.com/AneteSavicka/OpenBCI_EEG_Stroke_Rehab/assets/71130798/5dd17872-4594-4318-95bd-5dd6f1424dc8)
