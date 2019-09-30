#!/usr/bin/env python
# coding: utf-8

# In[592]:


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# In[593]:


df = pd.read_csv('hw5_treasury yield curve data.csv')


# In[594]:


df=df.dropna()


# In[595]:


df.head()


# In[596]:


#df.describe()


# In[597]:


datX = df[df.columns[1:31]]


# In[598]:


#datX


# In[599]:


datX = pd.concat([datX], axis=1)
from sklearn.preprocessing import StandardScaler

#sc = StandardScaler()
#sc.fit(datX)
#datX_std = sc.transform(datX)


# In[600]:


from pandas import DataFrame
#dataX = DataFrame.from_records(datX_std)
#not scaler at this point 
dataX = datX


# In[601]:


datanumy = df[df.columns[31]]


# In[602]:


#datanumy


# In[603]:


dataXy= pd.concat([dataX, datanumy],axis=1)
#dataXy


# In[604]:


cols = ['SVENF01', 'SVENF02', 'SVENF03', 'SVENF04', 'SVENF05','SVENF06','SVENF07','SVENF08','SVENF09','SVENF10','SVENF11','SVENF12','SVENF13','SVENF14','SVENF15','SVENF16','SVENF17','SVENF18','SVENF19','SVENF20','SVENF21','SVENF22','SVENF23','SVENF24','SVENF25','SVENF26','SVENF27','SVENF28','SVENF29','SVENF30','Adj_Close']


# In[605]:


col30 = ['SVENF01', 'SVENF02', 'SVENF03', 'SVENF04', 'SVENF05','SVENF06','SVENF07','SVENF08','SVENF09','SVENF10','SVENF11','SVENF12','SVENF13','SVENF14','SVENF15','SVENF16','SVENF17','SVENF18','SVENF19','SVENF20','SVENF21','SVENF22','SVENF23','SVENF24','SVENF25','SVENF26','SVENF27','SVENF28','SVENF29','SVENF30']


# In[606]:


import numpy as np

cm = np.corrcoef(df[cols].values.T)

cm30 = np.corrcoef(df[col30].values.T)
 


# In[607]:


heat_map = sns.heatmap(cm)
plt.show()


# In[608]:


heat_map = sns.heatmap(cm30)
#plt.axis('equal')

plt.show()


# In[609]:


import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# In[610]:


X = dataXy.iloc[:, 0:30].values


# In[611]:


X.shape


# In[612]:


y = dataXy['Adj_Close'].values


# In[613]:


#Split data into training and test sets.  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42)


# In[614]:


#2.pca
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sc.fit(X_train)
sc.fit(X_test)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# In[615]:


# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# bond
bond0 = X_train[:,0]
bond1 = X_train[:,1]
bond2 = X_train[:,2]

# yield
ye = y_train

# Scatter plot from each treasury, here we only see first 3 correlation
plt.scatter(bond0, bond1)
plt.scatter(bond1, bond2)
#plt.scatter(bond2, ye)
#plt.axis('')
plt.show()


# Calculate the Pearson correlation
correlation, pvalue = pearsonr(bond0, bond1)

# Display the correlation
print(correlation)


# In[627]:


# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
#del range
# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,pca)

# Fit the pipeline to 'X_train_std'
pipeline.fit(X_train_std)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()


# In[629]:


plt.bar(features, pca.explained_variance_ratio_)
plt.xlabel('PCA feature')
plt.ylabel('variance ratio')
plt.xticks(features)
plt.show()


# In[630]:


# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 3 components: pca
pca = PCA(n_components=3)

# Fit the PCA instance to the scaled samples
sc.fit(X_train)
X_train_std = sc.transform(X_train_std)
X_test_std = sc.transform(X_test_std)

pca.fit(X_train_std)

# Transform the scaled samples: pca_features
pca_features = pca.transform(X_train_std)
#pca_features_ = pca.transform(X_test_std)
# Print the shape of pca_features
print(pca_features.shape)


# In[631]:


print(pca.explained_variance_)


# In[632]:


pca.explained_variance_ratio_


# In[633]:


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[634]:


#3.
from sklearn.metrics import mean_squared_error
from math import sqrt
#rms = sqrt(mean_squared_error(y_actual, y_predicted))


# In[635]:


from sklearn.linear_model import LinearRegression
slr = LinearRegression()
#using original data to fit linear model
slr.fit(X, y)
y_pred = slr.predict(X)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)


# In[636]:


##R² score of our model,this is the percentage of explained variance of the predictions. 
slr.score(X,y)


# In[637]:


#evaluate the performance of the linear regression model using predictions and test set
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)


# In[638]:


print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y_train, y_train_pred)),
       sqrt(mean_squared_error(y_test, y_test_pred))))


# In[641]:


#using pca features to fit linear regression model
slr = LinearRegression()
slr.fit(X_train_std, y_train)
y_train_predpca = slr.predict(X_train_std)
y_test_predpca = slr.predict(X_test_std)


# In[642]:


print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_predpca),
        mean_squared_error(y_test, y_test_predpca)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_predpca),
        r2_score(y_test, y_test_predpca)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y_train, y_train_predpca)),
       sqrt(mean_squared_error(y_test, y_test_predpca))))


# In[644]:


#using SVR for original data 
from sklearn.svm import SVR


# In[646]:


#SV regressor model with original dataset
SVR = svm.SVR(kernel="linear").fit(X_train, y_train)
SVR.predict(X_test)

y_train_pred = SVR.predict(X_train)
y_test_pred = SVR.predict(X_test)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y_train, y_train_pred)),
       sqrt(mean_squared_error(y_test, y_test_pred))))
round(SVR.score(X_test, y_test), 4)


# In[649]:


#SVR regressor model with pca dataset
SVR = svm.SVR(kernel="linear").fit(X_train_pca, y_train)
SVR.predict(X_test_pca)

y_train_pred = SVR.predict(X_train_pca)
y_test_pred = SVR.predict(X_test_pca)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
print('RMSE train: %.3f, test: %.3f' % (
        sqrt(mean_squared_error(y_train, y_train_pred)),
       sqrt(mean_squared_error(y_test, y_test_pred))))
round(SVR.score(X_test_pca, y_test), 4)


# In[ ]:


print("My name is {Xuehui Chao}")
print("My NetID is: {xuehuic2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

