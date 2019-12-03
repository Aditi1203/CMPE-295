### Database Used:

MIT BIH dataset




### Technique Used:

##### Package: biosppy

```
    biosppy.signals.ecg.christov_segmenter(signal=None, sampling_rate=1000.0)
```


This is known as ECG R-peak segmentation algorithm.

Parameters:	
1. signal (array) – Input filtered ECG signal.
2. sampling_rate (int, float, optional) – Sampling frequency (Hz).
3. Returns:	
   rpeaks (array) – R-peak location indices.




### Steps to run the file

```
    python3 preprocess.py
```
