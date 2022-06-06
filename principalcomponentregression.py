#!/usr/bin/env python
# coding: utf-8

# # Assignment 2
# The following dataset belongs to 20 persons with Parkinson’s (6 female, 14 male) and 20 healthy individuals (10 female, 10 male). From all subjects, multiple types of sound recordings (26 voice samples including sustained vowels, numbers, words and short sentences) are taken. A group of 26 linear and time-frequency based features are extracted from each voice sample. Unified Parkinson’s Disease Rating Scale (UPDRS) score of each patient, which is determined by an expert physician is also available in this dataset.
# 
# The followings are the attribute information in the dataset:
# 
# - column 1: Subject id
# 
# - column 2-27: features
# 
# - features 1-5: Jitter (local), Jitter (local, absolute), Jitter (rap), Jitter (ppq5), Jitter (ddp), 
# 
# - features 6-11: Shimmer (local), Shimmer (local, dB), Shimmer (apq3), Shimmer (apq5), Shimmer (apq11), Shimmer (dda),
# 
# - features 12-14: AC, NTH, HTN,
# 
# - features 15-19: Median pitch, Mean pitch, Standard deviation, Minimum pitch, Maximum pitch,
# 
# - features 20-23: Number of pulses, Number of periods, Mean period, Standard deviation of period, 
# 
# - features 24-26: Fraction of locally unvoiced frames, Number of voice breaks, Degree of voice breaks
# 
# - column 28: UPDRS score
# 
# - column 29: class information (0: healthy, 1: person with Parkinson's)
# 
# Build a PCR model to predict the UPDRS scale based on the dataset.

# ## Libraries Loading

# In[2]:


# Handle table-like data and matrices
import pandas as pd
import numpy as np
import seaborn as sns

# Modelling helpers
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error

# Visualisation
import matplotlib.pyplot as plt

# Configure visualizations
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Preparation

# ### Import Dataset

# In[2]:


df = pd.read_table("Assignment2.txt", delimiter=",",
                   names=["subject_id",
                          "jitter_local", "jitter_localabs", "jitter_rap", 
                          "jitter_ppq5", "jitter_ddp", "shimmer_local",
                          "shimmer_localdB","shimmer_apq3","shimmer_apq5",
                          "shimmer_apq11","shimmer_dda","AC", "NTH", "HTN",
                          "median_pitch","mean_pitch","sd","min_pitch","max_pitch",
                          "no_pulses", "no_periods", "mean_period", "sd_period",
                          "fraction_unvoiced", "no_voicebreaks", "degree_voicebreaks",
                          "UPDRS_score","class_info"])
df.head()


# ## Exploratory Data Analysis

# In[3]:


df.info()


# In[4]:


df.shape


# ### Setting feature vector and target variable

# In[5]:


y = df.UPDRS_score

# Drop the column with subject ID and independent variable
X = df.drop(['subject_id', 'UPDRS_score', 'class_info'], axis=1)


# ### Correlation between the features:

# In[6]:


corr_matr = X.corr(method='pearson')
plt.figure(figsize=(10,10))
sns.heatmap(corr_matr, cmap='coolwarm', square=True)
plt.title("Pearson's correlation heatmap on X")
plt.show()


# We know that there are correlated features in this dataset, so non-robust to multicollinearity models might suffer

# ## Principal Component Analysis (PCA)

# Unfortunately sklearn does not have an implementation of PCA and regression combined, so we'll have to do it ourselves.
# 
# We'll start by performing Principal Components Analysis (PCA), remember to scale the data:

# In[7]:


plt.figure(figsize=(16,5))
X.boxplot()
plt.title("Distribution of X values")
plt.show()


# There is a large scale difference between variables. We need to standardize them to avoid those with large scales wrongly have too much weight in the calculations. Regarding abberant values, their effect should be reduced by methods that are not very sensitive to them.

# In[8]:


pca = PCA()
X_reduced = pca.fit_transform(scale(X)) 


# In[9]:


plt.figure(figsize=(16,5))
sns.boxplot(data=X_reduced)
plt.title("Distribution of X standardized predictors")
plt.grid()
plt.show()


# Looks better with standardization

# ### Print out the principal components:

# In[10]:


pd.DataFrame(pca.components_.T)


# It has 26 rows, 26 columns based on principal component.
# Each row represent variables.

# Now we'll perform 10-fold cross-validation to see how it influences the MSE:

# In[13]:


# 10-fold CV, with shuffle
n = len(X_reduced)
kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

regr = LinearRegression()
mse = []

# Calculate MSE with only the intercept (no principal components in regression)
score = -1*model_selection.cross_val_score(regr, np.ones((n,1)), y.ravel(), cv=kf_10, 
                                           scoring='neg_mean_squared_error').mean()    
mse.append(score)

# Calculate MSE using CV for 26 principle components, adding one component at the time
for i in np.arange(1, 27):
    score = -1*model_selection.cross_val_score(regr, X_reduced[:,:i],y.ravel(),cv=kf_10, 
                                               scoring='neg_mean_squared_error').mean()
    mse.append(score)
    
# Plot results    
plt.figure(figsize=(10,8))  
plt.plot(mse, '-v')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.title('UPDRS Score')
plt.xlim(xmin=-1);


# + We see that the smallest cross-validation error occurs when $M = 15$ components are used. 
# + This is barely fewer than $M = 13$, which amounts to simply performing least squares, because when all of the components are used in PCR no dimension reduction occurs. 
# + However, from the plot we also see that the cross-validation error is roughly the same when only one component is included in the model. 
# + This suggests that a model that uses just a small number of components might suffice. 

# In[12]:


## Amount of variance explained by adding each consecutive principal component:
np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)


# + We can think of this as the amount of information about the predictors or the response that is captured using $M$ principal components. 
# + For example, setting $M = 1$ only captures 40.55% of all the variance, or information, in the predictors. 
# + In contrast, using $M = 6$ increases the value to 87.45%. If we were to use all $M = p = 26$ components, this would increase to 100%.
# 
# 

# Now, perform PCA on the training data and evaluate its test set performance:

# In[50]:


pca2 = PCA()

# Split into training and test sets
X_train, X_test , y_train, y_test = model_selection.train_test_split(X, y, 
                                                                     test_size=0.3, 
                                                                     random_state=1)
# Scale the data
X_reduced_train = pca2.fit_transform(scale(X_train))
n = len(X_reduced_train)

# 10-fold CV, with shuffle
kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)
mse = []

# Calculate MSE with only the intercept (no principal components in regression)
score = -1*model_selection.cross_val_score(regr, np.ones((n,1)), y_train.ravel(), 
                                           cv=kf_10, 
                                           scoring='neg_mean_squared_error').mean()    
mse.append(score)

# Calculate MSE using CV for 26 principle components, adding one component at the time
for i in np.arange(1, 27):
    score = -1*model_selection.cross_val_score(regr, X_reduced_train[:,:i], 
                                               y_train.ravel(), cv=kf_10, 
                                               scoring='neg_mean_squared_error').mean()
    mse.append(score)

plt.figure(figsize=(8,6))    
plt.plot(np.array(mse), '-v')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.title('UPDRS Score')
plt.xlim(xmin=-1);


# Lowest cross-validation error occurs when  M=9  components are used.
# 
# Performance on the test data:

# In[51]:


X_reduced_test = pca2.transform(scale(X_test))[:,:9]

# Train regression model on training data 
regr = LinearRegression()
regr.fit(X_reduced_train[:,:9], y_train)

# Prediction with test data
pred = regr.predict(X_reduced_test)
mean_squared_error(y_test, pred)


# In[38]:


np.sqrt(202.71)


# Compare the RMSE value with training data:

# In[55]:


X_reduced_train = pca2.transform(scale(X_train))[:,:9]

# Train regression model on training data 
regr = LinearRegression()
regr.fit(X_reduced_train[:,:9], y_train)

# MSE with training data
pred = regr.predict(X_reduced_train)
mean_squared_error(y_train, pred)


# In[56]:


np.sqrt(mean_squared_error(y_train, pred))

