# CMPE-295

### Dataset Used:

MIT BIH dataset from physio.net

### Steps to convert the Binary data to CSV files:

Run the following command:

```
    python3 pre_process.py
```

It will perform the following operations:

1. Navigate to the data folder and fetch all the dat files from the each of the Person\_\* folder
2. Import the data into the code
3. Using the WFDB module in python, converts the data into a readable format.
4. Saves the data as a CSV file onto the disk

### Steps to run the Butterworth Band pass filter:

Run the following command to execute:

```
    python3 butterworth.py
```

It will perform the following operations:

1. Read the CSV file into a 1D array
2. Imports packages necessary to perform Butterworth band pass filter in the SciPy.signal package
3. Filters out the noise using a Butterworth band pass filter with the order of 4.
4. Plot the graph between the noisy data and the filtered data.
