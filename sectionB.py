# Section - B
# Source Code


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
Train=pd.read_csv(r'D:\train.csv')
Test=pd.read_csv(r'D:\test.csv')
Test.head()
Test.columns
print("shape: Test dataset  ", Test.shape)
Train.head()
print(" shape: Train dataset ", Train.shape)
print("Null values in Train dataset")
Train.isnull().sum()
Train["Gender"].fillna(Train["Gender"].mode()[0],inplace=True)
Train["Married"].fillna(Train["Married"].mode()[0],inplace=True)
Train['Dependents'].fillna(Train["Dependents"].mode()[0],inplace=True)
Train["Self_Employed"].fillna(Train["Self_Employed"].mode()[0],inplace=True)
Train["Credit_History"].fillna(Train["Credit_History"].mode()[0],inplace=True)
Train["Loan_Amount_Term"].value_counts()
Train["Loan_Amount_Term"].fillna(Train["Loan_Amount_Term"].mode()[0],inplace=True)
Train["Loan_Amount_Term"].value_counts()
Train["LoanAmount"].fillna(Train["LoanAmount"].median(),inplace=True)
Train.isnull().sum()
print("Null values in Train data set")
Train.isnull().sum()
Test["Gender"].fillna(Test["Gender"].mode()[0],inplace=True)
Test['Dependents'].fillna(Test["Dependents"].mode()[0],inplace=True)
Test["Self_Employed"].fillna(Test["Self_Employed"].mode()[0],inplace=True)
Test["Loan_Amount_Term"].fillna(Test["Loan_Amount_Term"].mode()[0],inplace=True)
Test["Credit_History"].fillna(Test["Credit_History"].mode()[0],inplace=True)
Test["LoanAmount"].fillna(Test["LoanAmount"].median(),inplace=True)
print("Null values in Test data set")
Test.isnull().sum()
Train.info()
Test.info()
print("Encoding categrical variable")  
Train_encoded = pd.get_dummies(Train,drop_first=True)
Train_encoded.head()
print("Split data Features and Target Varible")
X = Train_encoded.drop(columns='Loan_Status_Y')
y = Train_encoded['Loan_Status_Y']
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,f1_score
from sklearn import tree
import numpy as np
from sklearn.metrics import confusion_matrix
print("Splitting into train and test Data")
X_Train,X_Test,y_Train,y_Test = train_test_split(X,y,test_size=0.2,stratify =y,random_state =42)
print("handling Missing values")
impute = SimpleImputer(strategy='mean')
impute_Train = impute.fit(X_Train)
X_Train = impute_Train.transform(X_Train)
X_Test_imp = impute_Train.transform(X_Test)
treeclassifier = DecisionTreeClassifier()
treeclassifier.fit(X_Train,y_Train)
y_predict = treeclassifier.predict(X_Train)
print("Training Data Set Accuracy: ", accuracy_score(y_Train,y_predict))
print("Training Data F1 Score ", f1_score(y_Train,y_predict))
print("Validation Mean F1 Score: ",cross_val_score(treeclassifier,X_Train,y_Train,cv=5,scoring='f1_macro').mean())
print("Validation Mean Accuracy: ",cross_val_score(treeclassifier,X_Train,y_Train,cv=5,scoring='accuracy').mean())
validation_accuracy = []
training_accuracy = []
training_f1 = []
validation_f1 = []
tree_depths = []
for depth in range(1,20):
    treeclassifier = DecisionTreeClassifier(max_depth=depth)
    treeclassifier.fit(X_Train,y_Train)
    y_Training_pred = treeclassifier.predict(X_Train)
    Training_acc = accuracy_score(y_Train,y_Training_pred)
    Train_f1 = f1_score(y_Train,y_Training_pred)
    val_mean_f1 = cross_val_score(treeclassifier,X_Train,y_Train,cv=5,scoring='f1_macro').mean()
    val_mean_accuracy = cross_val_score(treeclassifier,X_Train,y_Train,cv=5,scoring='accuracy').mean()  
    training_accuracy.append(Training_acc)
    validation_accuracy.append(val_mean_accuracy)
    training_f1.append(Train_f1)
    validation_f1.append(val_mean_f1)
    tree_depths.append(depth)
Tuning_Max_depth = {"Training Accuracy": training_accuracy, "Validation Accuracy": validation_accuracy, "Training F1": training_f1, "Validation F1":validation_f1, "Max_Depth": tree_depths }
Tuning_Max_depth_df = pd.DataFrame.from_dict(Tuning_Max_depth)
plot_df = Tuning_Max_depth_df.melt('Max_Depth',var_name='Metrics',value_name="Values")
fig,ax = plt.subplots(figsize=(15,5))
sns.pointplot(x="Max_Depth", y="Values",hue="Metrics", data=plot_df,ax=ax)
training_accuracy = []
validation_accuracy = []
training_f1 = []
validation_f1 = []
min_samples_leaf = []
for samples_leaf in range(1,80,3): 
    treeclassifier = DecisionTreeClassifier(max_depth=3,min_samples_leaf = samples_leaf)
    treeclassifier.fit(X_Train,y_Train)
    y_Training_pred = treeclassifier.predict(X_Train)
    Training_acc = accuracy_score(y_Train,y_Training_pred)
    Train_f1 = f1_score(y_Train,y_Training_pred)
    val_mean_f1 = cross_val_score(treeclassifier,X_Train,y_Train,cv=5,scoring='f1_macro').mean()
    val_mean_accuracy = cross_val_score(treeclassifier,X_Train,y_Train,cv=5,scoring='accuracy').mean()
    training_accuracy.append(Training_acc)
    validation_accuracy.append(val_mean_accuracy)
    training_f1.append(Train_f1)
    validation_f1.append(val_mean_f1)
    min_samples_leaf.append(samples_leaf)
Tuning_min_samples_leaf = {"Training Accuracy": training_accuracy, "Validation Accuracy": validation_accuracy, "Training F1": training_f1, "Validation F1":validation_f1, "Min_Samples_leaf": min_samples_leaf }
Tuning_min_samples_leaf_df = pd.DataFrame.from_dict(Tuning_min_samples_leaf)
plot_df = Tuning_min_samples_leaf_df.melt('Min_Samples_leaf',var_name='Metrics',value_name="Values")
fig,ax = plt.subplots(figsize=(15,5))
sns.pointplot(x="Min_Samples_leaf", y="Values",hue="Metrics", data=plot_df,ax=ax)
treeclassifier = DecisionTreeClassifier(max_depth=3,min_samples_leaf = 35)
treeclassifier.fit(X_Train,y_Train)
y_predict = treeclassifier.predict(X_Test_imp)
print("Test Accuracy: ",accuracy_score(y_Test,y_predict))
print("Test F1 Score: ",f1_score(y_Test,y_predict))
print("Confusion Matrix on Test Data")
pd.crosstab(y_Test, y_predict, rownames=['True'], colnames=['Predicted'], margins=True)