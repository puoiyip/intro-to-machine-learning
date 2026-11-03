import pandas as pd
import GWCutilities as util

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

print("\n-----\n")

#Create a variable to read the dataset
df = pd.read_csv("heartDisease_2020_sampling.csv")
print(
    "We will be performing data analysis on this Indicators of Heart Disease Dataset. Here is a sample of it: \n"
)

#Print the dataset's first five rows
print(df.head())
input("\n Press Enter to continue.\n")

#Data Cleaning
#Label encode the dataset
df = util.labelEncoder(df, ["HeartDisease", "GenHealth", "Smoking", "AlcoholDrinking", "AgeCategory", "PhysicalActivity"]) #changing value of df
print("\nHere is a preview of the dataset after label encoding. \n")
print(df.head())
input("\nPress Enter to continue.\n")

#One hot encode the dataset
df = util.oneHotEncoder(df, ["Sex", "Race"]) #returns a new dataframe with the columns one hot encoded
print(
    "\nHere is a preview of the dataset after one hot encoding. This will be the dataset used for data analysis: \n"
)
print(df.head())
input("\nPress Enter to continue.\n")

#Creates and trains Decision Tree Model
from sklearn.model_selection import train_test_split
#features
X = df.drop("HeartDisease", axis = 1) #creates dataframe without heartdisease bc it is the class, not a feature; axis = 1 tells model that heart disease is a column not a row
y = df["HeartDisease"] #target class
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 1) #train_test_split() randomly selects which data will be in each dataset

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 10, class_weight = "balanced") #clf = classifier; model gives more priority to minority class
clf = clf.fit(X_train, y_train)

#Test the model with the testing data set and prints accuracy score
test_predictions = clf.predict(X_test) #outputs list of predictions

from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test, test_predictions) #compares predictions with true outputs
print("The accuracy with the testing data set of the Decision Tree is: " + str(test_acc))

#Prints the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, test_predictions, labels= [1,0])
print("The confusion matrix of the tree is: ")
print(cm)

#Test the model with the training data set and prints accuracy score
train_predictions = clf.predict(X_train)
from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train, train_predictions)

print("The accuracy with the training data set of the Decision Tree is: " + str(train_acc))

input("\nPress Enter to continue.\n")

#Prints another application of Decision Trees and considerations
print("\nBelow is another application of decision trees and considerations for using them:\n")
print("Decision trees could be used in social media algorithms to cater posts to each user based on the past content that they've viewed and liked. Features of the decision tree could include the account, the hashtags, or the category of the content of each post. The decision tree would then classify whether or not the user will like the content, which would be confirmed by how they interact with the post (like, share, save, or watching the entire video). However, when creating this decision tree, it's important to consider that the user's interests may change over time, so outdated data should not be used. Moreover, the model should avoid recommending potentially sensitive content, even if its other features align with the user's interests in order to protect the mental health of users.")

#Prints a text representation of the Decision Tree
print("\nBelow is a text representation of how the Decision Tree makes choices:\n")
input("\nPress Enter to continue.\n")

util.printTree(clf, X.columns)

#Examine and explore feature importance
feature_importance = pd.DataFrame(clf.feature_importances_, index = X.columns)
print(feature_importance) #higher number = more impact on decision
import matplotlib.pyplot as plt 
feature_importance.plot(kind='bar', title='Feature Importance')
plt.show()
