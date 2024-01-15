import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

dataFrame = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

predictorColumns = list(dataFrame.columns[:6])

# Plot settings

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictorColumns:
    dataFrame[col] = dataFrame[col].interpolate()


# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------


for s in dataFrame["set"].unique():
    start = dataFrame[dataFrame["set"] == s].index[0]
    stop = dataFrame[dataFrame["set"] == s].index[-1]

    duration = stop - start
    dataFrame.loc[(dataFrame["set"] == s), "duration"] = duration.seconds

durationDataFrame = dataFrame.groupby(["category"])["duration"].mean()
durationDataFrame.iloc[0] / 5
durationDataFrame.iloc[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
dataFrameLowPass = dataFrame.copy()
lowPass = LowPassFilter()

fs = 1000 / 200
cutoffFrequency = 1.3
dataFrameLowPass = lowPass.low_pass_filter(
    dataFrameLowPass, "acc_y", fs, cutoffFrequency, order=5
)

subset = dataFrameLowPass[dataFrameLowPass["set"] == 58]
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw_data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)


# Looping through all the columns and applying a low pass filter
dataFrameLowPass = dataFrame.copy()
for col in predictorColumns:
    dataFrameLowPass = lowPass.low_pass_filter(
        dataFrameLowPass, col, fs, cutoffFrequency, order=5
    )
    dataFrameLowPass[col] = dataFrameLowPass[col + "_lowpass"]
    del dataFrameLowPass[col + "_lowpass"]
# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
