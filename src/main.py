import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Gathering the data
dataFrame = pd.read_csv("../data/delaney_solubility_with_descriptors.csv")

# Splitting the x and y variables (Normally the rightmost column is the y)
yVal = dataFrame["logS"]
xVal = dataFrame.drop("logS", axis=1)

# Splitting the data into training and testing datasets(80/20 split)
xTrain, xTest, yTrain, yTest = train_test_split(
    xVal, yVal, test_size=0.2, random_state=100
)

# Building/Training the models


# ------------------ Linear Regression Model--------------------------#


# Training the model
linReg = LinearRegression()
linReg.fit(xTrain, yTrain)
LinearRegression()

# Predicting with the trained model
YLinearTrainingPrediction = linReg.predict(xTrain)
YLinearTestPrediction = linReg.predict(xTest)


# Evaluating Model Performance
linearRegressionTrainMSE = mean_squared_error(yTrain, YLinearTrainingPrediction)
linearRegressionTrainR2 = r2_score(yTrain, YLinearTrainingPrediction)
linearRegressionTestMSE = mean_squared_error(yTest, YLinearTestPrediction)
linearRegressionTestR2 = r2_score(yTest, YLinearTestPrediction)

# Outputting the evaluations in a tabular form
linearRegressionData = pd.DataFrame(
    [
        "Linear Regression",
        linearRegressionTrainMSE,
        linearRegressionTrainR2,
        linearRegressionTestMSE,
        linearRegressionTestR2,
    ]
).transpose()
linearRegressionData.columns = [
    "Method",
    "Training_MSE",
    "Training_R2",
    "Test_MSE",
    "Test_R2",
]


# ------------------ Random Forest Model --------------------------#

# Training the model
randomForest = RandomForestRegressor(max_depth=2, random_state=100)
randomForest.fit(xTrain, yTrain)

# Predicting the model
YRandomForestTrainingPrediction = randomForest.predict(xTrain)
YRandomForestTestPrediction = randomForest.predict(xTest)

# Evaluating the model
randomForestTrainMSE = mean_squared_error(yTrain, YRandomForestTrainingPrediction)
randomForestTrainR2 = r2_score(yTrain, YRandomForestTrainingPrediction)
randomForestTestMSE = mean_squared_error(yTest, YRandomForestTestPrediction)
randomForestTestR2 = r2_score(yTest, YRandomForestTestPrediction)

# Outputting the evaluations in a tabular form
randomForestData = pd.DataFrame(
    [
        "Random Forest",
        randomForestTrainMSE,
        randomForestTrainR2,
        randomForestTestMSE,
        randomForestTestR2,
    ]
).transpose()
randomForestData.columns = [
    "Method",
    "Training_MSE",
    "Training_R2",
    "Test_MSE",
    "Test_R2",
]
