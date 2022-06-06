#!/usr/bin/env python
# coding: utf-8

# # Assignment 1
# Given a dataset containing values for 1024 binary attributes (molecular fingerprints) used to classify 8992 chemicals into 2 classes which are very toxic (positive) or not very toxic (negative). Details about the dataset can be found in the quoted reference: D. Ballabio, F. Grisoni, V. Consonni, R. Todeschini (2019), Integrated QSAR models to predict acute oral systemic toxicity, Molecular Informatics, 38.
# 
# Variables Information:
# - Variables in columns 1-1024: binary molecular fingerprint
# - Variable in column 1025: experimental class: positive (very toxic) and negative (not very toxic)
# 
# Develop a classification model using logistic regression to classify the experimental class based on the binary molecular fingerprints. Use 80:20 rules for validation purposes and find the test error rate.
# 
# Explain why Linear/Quadratic Discriminant Analysis is inappropriate for this dataset?

# ## Introduction
# World Health Organization has estimated 12 million deaths occur worldwide, every year due to Heart diseases. Half the deaths in the United States and other developed countries are due to cardio vascular diseases. The early prognosis of cardiovascular diseases can aid in making decisions on lifestyle changes in high risk patients and in turn reduce the complications. This research intends to pinpoint the most relevant/risk factors of heart disease as well as predict the overall risk using logistic regression.

# In[1]:


# Handle table-like data and matrices
import numpy as np
import pandas as pd
from scipy import stats
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Modelling Algorithms
from sklearn.linear_model import LogisticRegression

# Modelling Helpers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visualisations
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Preparation
# ### Source:
# The dataset is publically available on the Kaggle website, and it is from an ongoing ongoing cardiovascular study on residents of the town of Framingham, Massachusetts. The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD).The dataset provides the patients’ information. It includes over 4,000 records and 15 attributes.

# In[16]:


# Import dataset
df = pd.read_csv("data/qsar_oral_toxicity.csv", sep=';', header=None)

# Rename the columns
t = iter(range(1, 1025))
df.columns = ['x' + format(next(t)) if 0 <= i <= 1024 else x for i, 
              x in enumerate(df.columns, 1)]
df.columns = [*df.columns[:-1], 'Experimental']
df.head()


# ### Variables:
# Each attribute is a potential risk factor. There are both demographic, behavioural and medical risk factors.

# ### Missing Values

# In[17]:


# Check for missing values
df.isnull().sum()


# In[18]:


count=0
for i in df.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total number of rows with missing values is ', count)


# ## Exploratory Analysis

# In[19]:


def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribution",color='DarkRed')
        
    fig.tight_layout()  
    plt.show()
draw_histograms(df,df.columns,10,5)


# In[21]:


# Change categorical variable to dummy variable
df['Exp_bool'] = (df['Experimental'] == 'positive').astype(int)
df.drop(['Experimental'],axis=1,inplace=True)


# In[24]:


sns.countplot(x='Exp_bool',data=df)


# In[23]:


df.Exp_bool.value_counts()


# In[25]:


df.describe()


# ## Logistic Regression
# Logistic regression is a type of regression analysis in statistics used for prediction of outcome of a categorical dependent variable from a set of predictor or independent variables. In logistic regression the dependent variable is always binary. Logistic regression is mainly used to for prediction and also calculating the probability of success.

# In[26]:


from statsmodels.tools import add_constant as add_constant
df_cons = add_constant(df)
df_cons.head()


# In[31]:


stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
cols = df_cons.columns[1:-1]
model = sm.Logit(df.Exp_bool,df_cons[cols])
results = model.fit()
results.summary()


# In[28]:


results.pvalues


# The results above show some of the attributes with P value higher than the preferred alpha(5%) and thereby showing low statistically significant relationship with the probability of heart disease. Backward elemination approach is used here to remove those attributes with highest Pvalue one at a time follwed by running the regression repeatedly until all attributes have P Values less than 0.05.

# ### Feature Selection (P-value approach)

# In[60]:


sum(results.pvalues > 0.05)


# In[34]:


X = df.iloc[:-1]
a = X.columns[:-1]

b = results.pvalues > 0.05

from itertools import compress
list_a = a
fil = b
c = list(compress(list_a, fil))

df2 = df.drop(c, axis=1)

from statsmodels.tools import add_constant as add_constant
df_cons2 = add_constant(df2)

stats.chisqprob = lambda chisq, df2: stats.chi2.sf(chisq, df2)
cols2 = df_cons2.columns[1:-1]
model2 = sm.Logit(df2.Exp_bool,df_cons2[cols2])
results2 = model2.fit()
results2.summary()


# In[35]:


sum(results2.pvalues > 0.05)


# In[37]:


X2 = df2.iloc[:-1]
a2 = X2.columns[:-1]

b2 = results2.pvalues > 0.05

from itertools import compress
list_a2 = a2
fil2 = b2
c2 = list(compress(list_a2, fil2))

df3 = df2.drop(c2, axis=1)

from statsmodels.tools import add_constant as add_constant
df_cons3 = add_constant(df3)

stats.chisqprob = lambda chisq, df3: stats.chi2.sf(chisq, df3)
cols3 = df_cons3.columns[1:-1]
model3 = sm.Logit(df3.Exp_bool,df_cons3[cols3])
results3 = model3.fit()
results3.summary()


# In[38]:


sum(results3.pvalues < 0.05)


# In[39]:


X3 = df3.iloc[:-1]
a3 = X3.columns[:-1]

b3 = results3.pvalues > 0.05

from itertools import compress
list_a3 = a3
fil3 = b3
c3 = list(compress(list_a3, fil3))

df4 = df3.drop(c3, axis=1)

from statsmodels.tools import add_constant as add_constant
df_cons4 = add_constant(df4)

stats.chisqprob = lambda chisq, df4: stats.chi2.sf(chisq, df4)
cols4 = df_cons4.columns[1:-1]
model4 = sm.Logit(df4.Exp_bool,df_cons4[cols4])
results4 = model4.fit()
results4.summary()


# In[41]:


sum(results4.pvalues < 0.05)


# In[42]:


X4 = df4.iloc[:-1]
a4 = X4.columns[:-1]

b4 = results4.pvalues > 0.05

from itertools import compress
list_a4 = a4
fil4 = b4
c4 = list(compress(list_a4, fil4))

df5 = df4.drop(c4, axis=1)

from statsmodels.tools import add_constant as add_constant
df_cons5 = add_constant(df5)

stats.chisqprob = lambda chisq, df5: stats.chi2.sf(chisq, df5)
cols5 = df_cons5.columns[1:-1]
model5 = sm.Logit(df5.Exp_bool,df_cons5[cols5])
results5 = model5.fit()
results5.summary()


# In[43]:


sum(results5.pvalues < 0.05)


# In[44]:


X5 = df5.iloc[:-1]
a5 = X5.columns[:-1]

b5 = results5.pvalues > 0.05

from itertools import compress
list_a5 = a5
fil5 = b5
c5 = list(compress(list_a5, fil5))

df6 = df5.drop(c5, axis=1)

from statsmodels.tools import add_constant as add_constant
df_cons6 = add_constant(df6)

stats.chisqprob = lambda chisq, df6: stats.chi2.sf(chisq, df6)
cols6 = df_cons6.columns[1:-1]
model6 = sm.Logit(df6.Exp_bool,df_cons6[cols6])
results6 = model6.fit()
results6.summary()


# In[45]:


sum(results6.pvalues < 0.05)


# In[46]:


X6 = df6.iloc[:-1]
a6 = X6.columns[:-1]

b6 = results6.pvalues > 0.05

from itertools import compress
list_a6 = a6
fil6 = b6
c6 = list(compress(list_a6, fil6))

df7 = df6.drop(c6, axis=1)

from statsmodels.tools import add_constant as add_constant
df_cons7 = add_constant(df7)

stats.chisqprob = lambda chisq, df7: stats.chi2.sf(chisq, df7)
cols7 = df_cons7.columns[1:-1]
model7 = sm.Logit(df7.Exp_bool,df_cons7[cols7])
results7 = model7.fit()
results7.summary()


# In[47]:


sum(results7.pvalues < 0.05)


# In[48]:


X7 = df7.iloc[:-1]
a7 = X7.columns[:-1]

b7 = results7.pvalues > 0.05

from itertools import compress
list_a7 = a7
fil7 = b7
c7 = list(compress(list_a7, fil7))

df8 = df7.drop(c7, axis=1)

from statsmodels.tools import add_constant as add_constant
df_cons8 = add_constant(df8)

stats.chisqprob = lambda chisq, df8: stats.chi2.sf(chisq, df8)
cols8 = df_cons8.columns[1:-1]
model8 = sm.Logit(df8.Exp_bool,df_cons8[cols8])
results8 = model8.fit()
results8.summary()


# In[58]:


sum(results8.pvalues < 0.05)


# **Logistic regression equation**
# 
# $$y = \frac{e^{\beta_0 + \beta_1X}}{1 + e^{\beta_0 + \beta_1X}}$$
#  
# 
# When all features plugged in:
# 
# $$logit(p) = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n$$

# ### Splitting data to train and test split

# In[50]:


import sklearn
new_features = df8
X = new_features.iloc[:,:-1]
y = new_features.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 1)


# In[51]:


X_train.shape


# In[52]:


X_test.shape


# In[53]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)


# ## Model Evaluation
# ### Model Accuracy

# In[54]:


sklearn.metrics.accuracy_score(y_test,y_pred)


# In[64]:


sklearn.metrics.precision_score(y_test,y_pred)


# Accuracy of the model is 0.94

# ### Confusion Metric

# In[55]:


confusion_matrix(y_test, y_pred)


# In[56]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
conf_matrix = pd.DataFrame(data = cm, columns = ['Predicted:0','Predicted:1'],index = ['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# The confusion matrix shows 1639+45 = 1684 correct predictions and 28+87 = 115 incorrect ones.
# 
# True Positives: 45
# 
# True Negatives: 1639
# 
# False Positives: 28 (Type I error)
# 
# False Negatives: 87 (Type II error)

# In[57]:


TN = cm[0,0]
TP = cm[1,1]
FN = cm[1,0]
FP = cm[0,1]
sensitivity = TP/float(TP+FN)
specificity = TN/float(TN+FP)


# In[77]:


sensitivity


# In[78]:


specificity


# ### Model Evaluation - Statistics

# In[79]:


print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',

'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',

'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',

'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',

'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',

'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',

'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)


# From the above statistics it is clear that the model is highly specific than sensitive. The negative values are predicted more accurately than the positives.

# ### Predicted probabilities of 0 (not very toxic) and 1 (very toxic) for the test data with a default classification threshold of 0.5

# In[62]:


y_pred_prob = lr.predict_proba(X_test)[:,:]
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['not very toxic (0)','very toxic (1)'])
y_pred_prob_df.head()


# **Lower the threshold**
# 
# Since the model is predicting Heart disease too many type II errors is not advisable. A False Negative ( ignoring the probability of disease when there actualy is one) is more dangerous than a False Positive in this case. Hence inorder to increase the sensitivity, threshold can be lowered

# In[63]:


from sklearn.preprocessing import binarize
for i in range(1,5):
    cm2 = 0
    y_pred_prob_yes = lr.predict_proba(X_test)
    y_pred2 = binarize(y_pred_prob_yes,i/10)[:,1]
    cm2 = confusion_matrix(y_test,y_pred2)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors (False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')
    


# ### ROC Curve

# In[83]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for toxicity classifier')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)


# A common way to visualize the trade-offs of different thresholds is by using an ROC curve, a plot of the true positive rate (# true positives/ total # positives) versus the false positive rate (# false positives / total # negatives) for all possible choices of thresholds. A model with good classification accuracy should have significantly more true positives than false positives at all thresholds.
# 
# The optimum position for roc curve is towards the top left corner where the specificity and sensitivity are at optimum levels

# ### Area Under The Curve (AUC)
# 
# The area under the ROC curve quantifies model classification accuracy; the higher the area, the greater the disparity between true and false positives, and the stronger the model in classifying members of the training dataset. An area of 0.5 corresponds to a model that performs no better than random classification and a good classifier stays as far away from that as possible. An area of 1 is ideal. The closer the AUC to 1 the better.

# In[81]:


sklearn.metrics.roc_auc_score(y_test,y_pred_prob_yes[:,1])


# ## Conclusions
# 
# - All attributes selected after the elimination process show Pvalues lower than 5% and thereby suggesting significant role in the Heart disease prediction.
# 
# - Men seem to be more susceptible to heart disease than women.Increase in Age,number of cigarettes smoked per day and systolic Blood Pressure also show increasing odds of having heart disease.
# 
# - Total cholesterol shows no significant change in the odds of CHD. This could be due to the presence of 'good cholesterol(HDL) in the total cholesterol reading.Glucose too causes a very negligible change in odds (0.2%).
# 
# - The model predicted with 0.88 accuracy. The model is more specific than sensitive.
# 
# - The Area under the ROC curve is 73.5 which is somewhat satisfactory.
# 
# - Overall model could be improved with more data.
