### Database Used:

MIT BIH dataset


### Technique Used:

##### Package: wfdb

```
      class wfdb.processing.XQRS(sig, fs, conf=None)
```

The XQRS.Conf class is the configuration class that stores initial parameters for the detection.

The XQRS.detect method runs the detection algorithm.

The process works as follows:

1. Load the signal and configuration parameters.
2. Bandpass filter the signal between 5 and 20 Hz, to get the filtered signal.
3. Apply moving wave integration (mwi) with a ricker (Mexican hat) wavelet onto the filtered signal, and save the square of the integrated signal.
4. Conduct learning if specified, to initialize running parameters of noise and qrs amplitudes, the qrs detection threshold, and recent rr intervals. If learning is unspecified or fails, use default parameters. See the docstring for the _learn_init_params method of this class for details.
Run the main detection. Iterate through the local maxima of the mwi signal. 

5. For each local maxima: check if it is a qrs complex.
To be classified as a qrs, it must come after the refractory period, cross the qrs detection threshold, and not be classified 
as a t-wave if it comes close enough to the previous qrs. 

6. If successfully classified, update running detection threshold and heart rate parameters.
If not a qrs, classify it as a noise peak and update running parameters.

7. Before continuing to the next local maxima, if no qrs was detected within 1.66 times the recent rr interval, 
perform backsearch qrs detection. This checks previous peaks using a lower qrs detection threshold.


### Steps to run the file:

Run the following command:
```
  python3 data_preprocess.py
```

It will perform the following operations:

1. Read the data from binary format
2. Identify the indexes of QRS complex from the recordings
3. Segment the record
4. Save the segmented record in form of images with encoding
5. Save the csv file in form of fixed chunks to be used for model training
