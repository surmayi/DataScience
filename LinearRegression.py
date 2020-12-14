# Import pandas, numpy, matplotlib,and seaborn. Then set %matplotlib inline 
import pandas as pd
import numpy as np
import matplotlib.pyplot as py
import seaborn as sns


# Read in the Ecommerce Customers csv file as a DataFrame called customers.
customers = pd.read_csv("Ecommerce Customers")

# Check the head of customers, and check out its info() and describe() methods.
customers.head()
customers.info()
customers.describe()

# Exploratory Data Analysis

# Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?
sns.jointplot(x='Time on Website',y='Yearly Amount Spent', data=customers)

#Do the same but with the Time on App column instead.
sns.jointplot(x='Time on App',y='Yearly Amount Spent', data=customers)

#Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.
sns.jointplot(x='Time on App',y='Length of Membership', data=customers,kind='hex')

# Let's explore these types of relationships across the entire data set. Use pairplot to recreate the plot below.
sns.pairplot(data=customers)

# Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?
# Length of Membership

# Create a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership.
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)

# Training and Testing Data

# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets. Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column
X =customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
X.head()
Y=customers['Yearly Amount Spent']
Y.head()

# Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3,random_state=101)

#Training the Model

# Now its time to train our model on our training data! Import LinearRegression from sklearn.linear_model
from sklearn.linear_model import LinearRegression

# Create an instance of a LinearRegression() model named lm.
lm = LinearRegression()

# Train/fit lm on the training data.
lm.fit(X_train,Y_train)

#Print out the coefficients of the model
print(lm.coef_)

# Predicting Test Data

# Now that we have fit our model, let's evaluate its performance by predicting off the test values! Use lm.predict() to predict off the X_test set of the data.
prediction = lm.predict(X_test)

# Create a scatterplot of the real test values versus the predicted values.
py.scatter(Y_test,prediction)

# Evaluating the Model

# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.
from sklearn import metrics
print('MAE= ', metrics.mean_absolute_error(Y_test,prediction) )
print('MSE= ', metrics.mean_squared_error(Y_test,prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, prediction)))

# Residuals

# Let's quickly explore the residuals to make sure everything was okay with our data. Plot a histogram of the residuals and make sure it looks normally distributed. Using plt.hist()
py.hist(prediction-Y_test,bins=50)

# Conclusion

# do we focus our efforst on mobile app or website development
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients

# How can you interpret these coefficients?

# Interpreting the coefficients:

# Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.
# Do you think the company should focus more on their mobile app or on their website?

# There are two ways to think about this: Develop the Website to catch up to the performance of the mobile app, or develop the app more since that is what is working better!!
