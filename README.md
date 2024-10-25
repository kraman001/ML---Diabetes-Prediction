# Diabetes Prediction Model

This repository contains a machine learning model built to predict whether a person has diabetes or not based on specific health parameters using the PIMA Diabetes Dataset.

## Table of Contents
- [Dataset](#dataset)
- [Features](#features)
- [Model](#model)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Code Structure](#code-structure)

## Dataset

The dataset used is the **PIMA Diabetes Dataset**, which contains health records of 768 individuals. Each record includes various health metrics such as glucose level, BMI, insulin level, and more. The target variable is `Outcome` which indicates if the person is diabetic (`1`) or non-diabetic (`0`).

## Features

The dataset consists of the following features:
- **Pregnancies**: Number of times the person has been pregnant.
- **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skin fold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg/(height in m)^2).
- **DiabetesPedigreeFunction**: A function which scores likelihood of diabetes based on family history.
- **Age**: Age of the person in years.
- **Outcome**: Class variable (0 for non-diabetic, 1 for diabetic).

## Model

The model uses the **Support Vector Machine (SVM)** algorithm for classification. The following steps are performed:
1. **Data Preprocessing**: Scaling of features using `StandardScaler` to ensure that all variables are on the same scale.
2. **Model Training**: The SVM model is trained on 80% of the data.
3. **Model Testing**: The model's performance is evaluated on the remaining 20% of the data.

## Technologies Used

- **Python**: Programming language.
- **Pandas & NumPy**: Libraries for data manipulation.
- **Scikit-Learn**: Machine learning library.

## Usage

To use the model for prediction:
1. Run the notebook or Python script to load the dataset and train the model.
2. Modify the input parameters in the script or notebook to predict whether a person is diabetic.

Example usage in the notebook:
```python
# Example input data
input_data = [5, 166, 72, 19, 175, 25.8, 0.587, 51]

# Reshaping the input for a single prediction
input_data_reshaped = np.array(input_data).reshape(1, -1)

# Making a prediction using the trained model
prediction = model.predict(input_data_reshaped)

if prediction[0] == 0:
    print("The person is non-diabetic.")
else:
    print("The person is diabetic.")
```

## Results

The accuracy of the model is evaluated using the accuracy score on both training and test datasets. Below are the performance metrics:
- **Training Accuracy**: `X%` (78.66%)
- **Test Accuracy**: `Y%` (77.27%)

## Future Improvements

Potential improvements to increase model accuracy and generalizability include:

- **Testing Additional Algorithms**: Exploring other classification models, such as Decision Trees or Random Forests.
- **Feature Engineering**: Adding or transforming features to enhance model insights.
- **Hyperparameter Tuning**: Adjusting parameters of the SVM model for optimized performance.
- **Ensemble Learning**: Combining multiple models to enhance predictive power.

### Key Highlights of the Project:
This Project implements a machine learning model to predict diabetes based on patient data from the PIMA Diabetes Dataset. Here’s a summary of the key components:

### Data Collection:

- The dataset is loaded using Pandas.
- The data consists of features such as Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age, with the target variable being Outcome (0 = Non-Diabetic, 1 = Diabetic).

### Data Preprocessing:
-  ``StandardScaler`` is used to scale the input features, ensuring that all features are on a similar scale for better performance of the model.
-  The data is split into features (X) and the target label (Y).

### Model Building:
- The **Support Vector Machine (SVM)** algorithm is used to train the model.
- The dataset is split into training and test sets using ``train_test_split``.

### Model Evaluation:
- The performance of the model is evaluated using the  **accuracy** score on both the training and testing data.
- The accuracy of training data is **78%** and for test data is **77%**.


## Code Structure

### Code Snippet 1
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

### Code Snippet 2
```python
# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('/content/diabetes.csv')
```

### Code Snippet 3
```python
# printing the first 5 rows of the dataset
diabetes_dataset.head()
```
![1 github](https://github.com/user-attachments/assets/8e7250a1-2af7-42e7-a4d7-419c22de8b80)

### Code Snippet 4
```python
# number of rows and Columns in this dataset
diabetes_dataset.shape
```
![2 0](https://github.com/user-attachments/assets/48dcd9af-2112-4825-831f-c3f14406b016)

### Code Snippet 5
```python
# getting the statistical measures of the data
diabetes_dataset.describe()
```
![k](https://github.com/user-attachments/assets/caec38fc-ea66-415c-818b-1b654854a009)

### Code Snippet 6
```python
diabetes_dataset['Outcome'].value_counts()
```
![3](https://github.com/user-attachments/assets/dd69db4e-63c7-4b67-a72f-b69f689d2d56)

### Code Snippet 7
```python
diabetes_dataset.groupby('Outcome').mean()
```
![4](https://github.com/user-attachments/assets/3b1e2c9a-ab4e-476d-8dfb-662d5e8c833a)

### Code Snippet 8
```python
# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)   # axis = 0 for rows & axis = 1 for column
Y = diabetes_dataset['Outcome']
```

### Code Snippet 9 
```python
print(X)
```
![5](https://github.com/user-attachments/assets/e6bec1e7-2794-4f45-a1f9-0c75b46af821)

### Code Snippet 10
```python
print(Y)
```
![6](https://github.com/user-attachments/assets/391a6b82-61a9-4457-a19f-bad9f9ee47db)

### Code Snippet 11
```python
scaler = StandardScaler()
```

### Code Snippet 12
```python
scaler.fit(X)
```

### Code Snippet 13
```python
standardized_data = scaler.transform(X)
```

### Code Snippet 14
```python
print(standardized_data)
```
![7](https://github.com/user-attachments/assets/5e913544-e29e-43cb-8089-a0a3b0181a84)

### Code Snippet 15
```python
X = standardized_data
Y = diabetes_dataset['Outcome']
```

### Code Snippet 16
```python
print(X)
print(Y)
```
![8](https://github.com/user-attachments/assets/9a34fb86-f29c-4f56-985b-f5f0dea1c9b3)

### Code Snippet 17
```python
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
```

### Code Snippet 18
```python
print(X.shape, X_train.shape, X_test.shape)
```
![9](https://github.com/user-attachments/assets/40684831-502b-4e28-b59e-f3862086f1af)

### Code Snippet 19
```python
# Training the Model
```

### Code Snippet 20
```python
classifier = svm.SVC(kernel='linear')
```

### Code Snippet 21
```python
#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)
```

### Code Snippet 22
```
# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
```

### Code Snippet 23
```python
print('Accuracy score of the training data : ', training_data_accuracy)
```
![10](https://github.com/user-attachments/assets/e64a758f-0411-45ef-bcce-eb1d6d8c2695)

### Code Snippet 24
```python
# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
```

### Code Snippet 25
```python
print('Accuracy score of the test data : ', test_data_accuracy)
```
![11](https://github.com/user-attachments/assets/517e70dd-11bf-4051-b364-e52e3a342ade)

### Code Snippet 26
```python
# Making Predictive System
```

### Code Snippet 27
```python
input_data = [7,100,0,0,0,30,0.484,32]

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
```
![12](https://github.com/user-attachments/assets/2c7d635f-dcb1-487a-9a03-af6d639ad35e)








