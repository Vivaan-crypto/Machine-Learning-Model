import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from LearningAlgorithms import ClassificationAlgorithms
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


dataFrame = pd.read_pickle("../../data/interim/03_data_features.pkl")

# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
dataFrameTraining = dataFrame.drop(["participant", "category", "set"], axis=1)

x = dataFrameTraining.drop(["label"], axis=1)
y = dataFrameTraining["label"]

xTraining, xTest, yTraining, yTest = train_test_split(
    x, y, test_size=0.25, random_state=42, stratify=y
)

fig, ax = plt.subplots(figsize=(10, 5))
dataFrameTraining["label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)
yTraining.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
yTraining.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

basicFeatures = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
squareFeatures = ["acc_r", "gyr_r"]
pcaFeatures = ["pca_1", "pca_2", "pca_3"]
timeFeatures = [f for f in dataFrameTraining.columns if "_temp_" in f]
freqFeatures = [f for f in dataFrameTraining.columns if ("_freq" in f) or ("_pse" in f)]
clusterFeatures = ["cluster"]

print("Basic features", len(basicFeatures))
print("Square features", len(squareFeatures))
print("PCA features", len(pcaFeatures))
print("Time features", len(timeFeatures))
print("Frequency features", len(freqFeatures))
print("Cluster features", len(clusterFeatures))

featureSetOne = list(set(basicFeatures))  # Only sensor values
featureSetTwo = list(set(featureSetOne + squareFeatures + pcaFeatures))
featureSetThree = list(set(featureSetTwo + timeFeatures))
featureSetFour = list(
    set(featureSetThree + freqFeatures + clusterFeatures)
)  # All the additional columns

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------

learner = ClassificationAlgorithms()
maxFeatures = 10

selected_features, ordered_features, ordered_scores = learner.forward_selection(
    maxFeatures, xTraining, yTraining
)

selected_features = [
    "pca_1",
    "duration",
    "acc_z_freq_0.0_Hz_ws_14",
    "acc_y_temp_mean_ws_5",
    "gyr_x_freq_1.071_Hz_ws_14",
    "gyr_r_freq_0.357_Hz_ws_14",
    "acc_x_freq_2.143_Hz_ws_14",
    "acc_x_freq_1.786_Hz_ws_14",
    "acc_r_freq_2.143_Hz_ws_14",
    "acc_x_freq_weighted",
]

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, maxFeatures + 1, 1), ordered_scores)
plt.xlabel("# of features")
plt.ylabel("Accuraccy")
plt.xticks(np.arange(1, maxFeatures + 1, 1))
plt.show()

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------

possibleFeatureSets = {
    featureSetOne,
    featureSetTwo,
    featureSetThree,
    featureSetFour,
    selected_features,
}
featureNames = {
    "Feature Set One",
    "Feature Set Two",
    "Feature Set Three",
    "Feature Set Four",
    "Selected Features",
}
iterations = 1
scoreDataFrame = pd.DataFrame()

for i, f in zip(range(len(possibleFeatureSets)), featureNames):
    print("Feature set:", i)
    selected_train_X = xTraining[possibleFeatureSets[i]]
    selected_test_X = xTest[possibleFeatureSets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            yTraining,
            selected_test_X, 
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(yTest, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, yTraining, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(yTest, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, yTraining, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(yTest, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, yTraining, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(yTest, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, yTraining, selected_test_X)

    performance_test_nb = accuracy_score(yTest, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    scoreDataFrame = pd.concat([scoreDataFrame, new_scores])
# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------
