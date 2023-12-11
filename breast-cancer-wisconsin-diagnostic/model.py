import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer





# IMPORTANT: getting the data and exploring it

cancer = load_breast_cancer()
#print(cancer)

# check what kinf of keys we have
#print(cancer.keys())

# print(cancer['DESCR'])
# print(cancer['target'])
# print(cancer['target_names'])
# print(cancer['feature_names'])
# print(cancer['data'].shape)


# Creating Dataframe
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target']))
# print(df.head())
# print(df.info())
# print(df.isnull().sum())
# print(df.dtypes)
# print(df['target'].value_counts())
# print(357 / len(df), 211/ len(df))

#sns.countplot(data=df, x='target')
#plt.savefig('countplot.png')  # Save as PNG image
# plt.show()

print(df.columns)
# sns.pairplot(data=df, vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness'], hue='target')
# plt.show()

#plt.figure(figsize=(20,15))
# sns.heatmap(df.corr(), annot=True, cmap='viridis_r')
# plt.show()

print(df.corr()['target'].sort_values(ascending=False))
# df.corr()['target'][:-1].sort_values().plot(kind='bar')
# plt.show()




# IMPORTANT: Modeling

# Train, Test and Split
# Separate Features and Labels
X = df.drop('target', axis=1)
y = df['target']

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Model Training
print("\n\nModel Training")
svc_model = SVC()
svc_model.fit(X_train, y_train)

# Model Evaluation
y_preds = svc_model.predict(X_test)
print(classification_report(y_test, y_preds))

cm = confusion_matrix(y_test, y_preds)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.show()




# IMPORTANT: Improving the model
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# scaling the data to bring all the features to the same level of magnitude (between 0 and 1)
scaled_X_train = scaler.fit_transform(X_train) 
scaled_X_test = scaler.transform(X_test)

# after scaling, as column names are removed. so we need to re-attach them
scaled_X_train = pd.DataFrame(scaled_X_train, columns=X_train.columns)
scaled_X_test = pd.DataFrame(scaled_X_test, columns=X_test.columns)
# print(scaled_X_train.head())
# print(scaled_X_test.head())

# # Before Scaling
# sns.scatterplot(data=X_train, x='mean area', y='mean smoothness', hue=y_train)
# plt.savefig('scatterplot_Before_Scaling.png')
# # After Scaling
# sns.scatterplot(x=scaled_X_train['mean area'], y=scaled_X_train['mean smoothness'], hue=y_train)
# plt.savefig('scatterplot_After_Scaling.png')




# IMPORTANT: Re-Train the model on the scaled data
print("\n\nRe-Training the model on the scaled data")
svc_model.fit(scaled_X_train, y_train)
y_preds = svc_model.predict(scaled_X_test)

print(classification_report(y_test, y_preds))

cm = confusion_matrix(y_test, y_preds)
print(cm)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.show()




# IMPORTANT: Improving the model even more with GridSearchCV
print("\n\nImproving the model even more with GridSearchCV")
from sklearn.model_selection import GridSearchCV

#help(SVC)
# help(GridSearchCV)

parameters = {
    'C': [0.1, 1, 10, 50, 100, 150],
    'kernel': ['poly', 'rbf', 'sigmoid'],
    'degree': [3, 4, 5, 6],
    'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]
}
# refit the model based on based parameters found
grid_model = GridSearchCV(SVC(), parameters, cv=5, refit=True, verbose=4) 
grid_model.fit(scaled_X_train, y_train)

# prints the best parameters 
print(grid_model.best_params_)




# IMPORTANT: Evaluating the model with the best parameters
print("\n\nEvaluating the model with the best parameters")

grid_predictions = grid_model.predict(scaled_X_test)

print(classification_report(y_test, grid_predictions))

cm = confusion_matrix(y_test, grid_predictions)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
