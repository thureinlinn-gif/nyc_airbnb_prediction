# Capstone: Define and Solve an ML Problem
# Airbnb NYC Listings — predict whether a listing is high- or low-priced
# Extracted code from Capstone.ipynb

import pandas as pd
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler
import time


# ----------------------------------------------------------------------
# Part 1: Load the data set and build the DataFrame
# ----------------------------------------------------------------------

# File paths for both data sets
# census_filename = os.path.join(os.getcwd(), "data_capstone", "censusData.csv")
airbnb_filename = os.path.join(os.getcwd(), "data_capstone", "airbnbListingsData.csv")

# Load your chosen dataset and save it to df
df = pd.read_csv(airbnb_filename)

df.head()


# ----------------------------------------------------------------------
# Part 3: Understand your data (EDA)
# ----------------------------------------------------------------------

# Class imbalance: count how many of each class
print(df['price_category'].value_counts())

# Visualizing the class distribution
sns.countplot(x='price_category', data=df)
plt.title('Class Distribution')
plt.show()


print(df.dtypes)

# Summary statistics for number columns
print(df.describe())

# Visualizing Relationship between a feature and the label
sns.boxplot(x='price_category', y='bedrooms', data=df)
plt.title('Bedrooms by Price Category')
plt.show()


# ----------------------------------------------------------------------
# Part 4: Prepare your data
# ----------------------------------------------------------------------

# Making a copy to prevent from changing the original data
df_prep = df.copy()

# Changing the label into numbers; "high" becomes 1, "low" becomes 0
df_prep['price_category'] = df_prep['price_category'].map({'high': 1, 'low': 0})

# Dropping columns which are not required for prediction
text_cols = ['name', 'description', 'neighborhood_overview', 'host_name',
             'host_location', 'host_about', 'amenities']
df_prep = df_prep.drop(columns=text_cols)

# Dropping columns with many missing values
df_prep = df_prep.drop(columns=['host_response_rate', 'host_acceptance_rate'])

# Dropping the raw price
df_prep = df_prep.drop(columns=['price'])

# Filling in remaining missing values with the median to save from outliers
df_prep['bedrooms'] = df_prep['bedrooms'].fillna(df_prep['bedrooms'].median())
df_prep['beds'] = df_prep['beds'].fillna(df_prep['beds'].median())

# Changing text categories into numbers by creating a new 0/1 column for each category
df_prep = pd.get_dummies(df_prep, columns=['room_type', 'neighbourhood_group_cleansed'])

# Converting any True/False columns to 1/0
df_prep = df_prep.astype({col: int for col in df_prep.select_dtypes('bool').columns})

# Checking the result
df_prep.head()


# ----------------------------------------------------------------------
# Part 5: Train, test, evaluate, and improve a traditional ML model
# ----------------------------------------------------------------------

# Create labeled examples from the dataset
# X = features (all input columns), y = label (the answer we predict)

# everything except the label
X = df_prep.drop(columns=['price_category'])

# just the label
y = df_prep['price_category']


# Create training and test sets out of the labeled examples
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


# Train, test and evaluate your model
# Scaling features so that all columns are on the same range for Logistic Regression
scaler = StandardScaler()

# learning scale from training data & apply
X_train_scaled = scaler.fit_transform(X_train)

# Applying same scale to test data
X_test_scaled = scaler.transform(X_test)

# Creating and training the model
# max_iter higher to finish learning
model = LogisticRegression(max_iter=1000)

# learning step
model.fit(X_train_scaled, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test_scaled)

# Scoring the model
acc = accuracy_score(y_test, y_pred)

# balanced score for imbalanced data
f1 = f1_score(y_test, y_pred, average='binary')

print("Accuracy:", acc)
print("F1 Score:", f1)


# Perform model selection through grid search cross-validation (GridSearchCV)
# to identify optimal hyperparameter values for your model
# Trying different settings of "C" (controls how strict the model is) and pick the best
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

# cv=5 splits training data 5 ways to test each setting fairly; scoring by F1
grid = GridSearchCV(LogisticRegression(max_iter=1000),
                    param_grid, cv=5, scoring='f1')
grid.fit(X_train_scaled, y_train)

print("Best C:", grid.best_params_)  # shows the winning setting


# Train, test and evaluate a final version of your model using the optimal hyperparameter values.
# Using the best model grid search found
best_model = grid.best_estimator_

# Testing it on the unseen test set
y_pred_final = best_model.predict(X_test_scaled)

# Saving final scores for later comparison to the neural network
acc_final = accuracy_score(y_test, y_pred_final)
f1_final = f1_score(y_test, y_pred_final, average='binary')

print("Final Accuracy:", acc_final)
print("Final F1 Score:", f1_final)


# Interpret your model's outputs
# Logistic Regression: coefficients show which features push toward "high price"
coefficients = pd.DataFrame({
    'feature': X.columns,
    # one number per feature
    'coefficient': best_model.coef_[0]
}).sort_values('coefficient', ascending=False)  # biggest positive at top

print(coefficients.head(10))   # top features pushing toward high price
print(coefficients.tail(10))   # features pushing toward low price


# ----------------------------------------------------------------------
# Part 6: Train, test, evaluate, and improve a neural network
# ----------------------------------------------------------------------

# Scale your data for the neural network

# Create the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler to transform the test data
X_test_scaled = scaler.transform(X_test)


# Step 1: Define your model architecture
# Get the number of features in your training data
n_features = X_train_scaled.shape[1]

# Create the neural network model
nn_model = keras.Sequential()

# Create the input layer and add the input layer to the 'nn_model' object
nn_model.add(keras.layers.InputLayer(input_shape=(n_features,)))

# Decision: I chose 2 hidden layers (64 and 32 units) with 'relu' activation.
# Why: 2 layers is enough for this size of data without overfitting; shrinking
# units (64 -> 32) is a common pattern; 'relu' is fast and works well in practice.
nn_model.add(keras.layers.Dense(64, activation='relu'))
nn_model.add(keras.layers.Dense(32, activation='relu'))

# Output layer for binary classification
nn_model.add(keras.layers.Dense(1, activation='sigmoid'))

# Print a summary of your model
nn_model.summary()


# Step 2: Define the optimization function
# Decision: learning rate = 0.01.
# A moderate rate which is big enough to learn at a reasonable speed, small enough
# to avoid overshooting and becoming unstable.
sgd_optimizer = keras.optimizers.SGD(learning_rate=0.01)


# Step 3: Define the loss function
# Measuring how wrong the model is; the model tries to make this number small.
loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)


# Step 4: Compile the model
# compiling how to improve (optimizer), how to measure error (loss), and what to report (accuracy).
nn_model.compile(optimizer=sgd_optimizer, loss=loss_fn, metrics=['accuracy'])


# Step 5: Fit the model to the training data
# Callback to output information from the model while it is training
class ProgBarLoggerNEpochs(keras.callbacks.Callback):

    def __init__(self, num_epochs: int, every_n: int = 50):
        self.num_epochs = num_epochs
        self.every_n = every_n

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n == 0:
            s = 'Epoch [{}/ {}]'.format(epoch + 1, self.num_epochs)
            logs_s = ['{}: {:.4f}'.format(k.capitalize(), v)
                      for k, v in logs.items()]
            s_list = [s] + logs_s
            print(', '.join(s_list))


t0 = time.time()  # start time

# Decision: 100 epochs to be enough passes for the model to learn without training forever
num_epochs = 100

history = nn_model.fit(
    X_train_scaled, y_train,        # training data and answers
    epochs=num_epochs,              # how many times to go through the data
    validation_split=0.2,           # hold out 20% of training data to check progress
    callbacks=[ProgBarLoggerNEpochs(num_epochs, every_n=20)],  # print every 20 epochs
    verbose=0                       # silence default output (our logger handles it)
)

t1 = time.time()  # stop time

print('Elapsed time: %.2fs' % (t1 - t0))


# Step 6: Visualize training performance
# Plot training loss and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')        # error on training data
plt.plot(history.history['val_loss'], label='Validation Loss')  # error on held-out data
plt.xlabel('Epoch')          # x-axis for training rounds
plt.ylabel('Loss')           # y-axis for how wrong the model is
plt.title('Loss Over Epochs')
plt.legend()                 # show which line is which
plt.show()

# Plot training accuracy and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.show()


# Step 7: Evaluate the model's performance on the test set
# Getting predictions as probabilities between 0 and 1
y_pred_probs = nn_model.predict(X_test_scaled)

# Converting probabilities to 0/1; anything 0.5 or above becomes "high price" (1)
y_pred_nn = (y_pred_probs >= 0.5).astype(int)


# Compute accuracy and F1 score for the neural network and print the results
nn_accuracy = accuracy_score(y_test, y_pred_nn)
nn_f1 = f1_score(y_test, y_pred_nn, average='binary')

print("Neural Network Accuracy:", nn_accuracy)
print("Neural Network F1 Score:", nn_f1)


# ----------------------------------------------------------------------
# Part 7: Compare your models
# ----------------------------------------------------------------------

# Side-by-side comparison table
results = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 Score'],
    'Logistic Regression': [acc_final, f1_final],  # traditional model's scores
    'Neural Network': [nn_accuracy, nn_f1]          # neural network's scores
})

print(results.to_string(index=False))
