# Import pandas, numpy, matplotlib,and seaborn. Then set %matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Read in the advertising.csv file and set it to a data frame called ad_data.
ad_data = pd.read_csv('advertising.csv')

# Check the head of ad_data, and check out its info() and describe() methods.
ad_data.head()
ad_data.info()
ad_data.describe()

# Exploratory Data Analysis

#  Create a histogram of the Age
plt.hist(ad_data['Age'],bins=40)

#Create a jointplot showing Area Income versus Age.
sns.jointplot('Age','Area Income' ,data=ad_data)

# Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.
sns.jointplot('Age','Daily Time Spent on Site' ,data=ad_data,kind='kde',color='red')

# Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage
sns.jointplot('Daily Time Spent on Site','Daily Internet Usage' ,data=ad_data,color='green')

#  Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.
sns.pairplot(ad_data,hue='Clicked on Ad')

#  Split the data into training set and testing set using train_test_split
from sklearn.model_selection import train_test_split
X= ad_data.drop(['Ad Topic Line','City','Timestamp','Clicked on Ad','Country'],axis=1)
y = ad_data['Clicked on Ad']
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=42)

#  Train and fit a logistic regression model on the training set.
from sklearn.linear_model import LogisticRegression
logrig = LogisticRegression()
logrig.fit(X_train,y_train)

# Predictions and Evaluations

#  Now predict values for the testing data.
prediction = logrig.predict(X_test)

#Create a classification report for the model
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))

