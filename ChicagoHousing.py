#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv('./dataset/chicagoHouse.csv')


# In[7]:


df.head(5)


# In[9]:


df.tail(10)


# In[11]:


df.shape


# In[13]:


df[['Space','Lot','Room']].info()


# In[15]:


df[df['Space'].isna()]


# In[17]:


df[df['Lot'].isna()]


# In[19]:


df[df['Room'].isna()]


# In[21]:


df = df.dropna()


# In[23]:


df.shape


# In[25]:


plt.scatter(df['Space'],df['Price'])


# In[27]:


plt.scatter(df['Room'],df['Price'])


# In[29]:


plt.scatter(df['Lot'],df['Price'])


# In[31]:


plt.scatter(df['Tax'],df['Price'])


# In[33]:


x = df[['Lot','Room','Space']]
y = df[['Price']]


# In[35]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.25,random_state=1)


# In[36]:


from sklearn.linear_model import LinearRegression


# In[37]:


regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[41]:


print("Multiple linear Regression - R squared score: ", regressor.score(x_val,y_val))


# In[43]:


results_df = pd.DataFrame({
    'Actual Values': np.squeeze(y_test),
    'Predicted Values': np.squeeze(regressor.predict(x_test))
})
results_df


# In[45]:


regressor.coef_[0][2]
regressor.intercept_[0]
ya = regressor.intercept_[0]+(regressor.coef_[0][2]*df['Space'])


# In[47]:


plt.scatter(df['Space'],df['Price'])
plt.plot(df['Space'],ya)


# In[49]:


from sklearn.preprocessing import PolynomialFeatures
#degree 2 Polynomial function model test  -Hyper parameters tuning
polyReg = PolynomialFeatures(degree=2)
x_poly2 = polyReg.fit_transform(x_train) #Feature extraction to multiple linear reg function
linReg2 = LinearRegression()
linReg2.fit(x_poly2,y_train)


# In[51]:


x_poly_val2 = polyReg.fit_transform(x_val)
print("Degree 2 - Training Error: ", linReg2.score(x_poly2,y_train))
print("Degree 2 - Validation Error: ", linReg2.score(x_poly_val2,y_val))


# In[53]:


#degree 3 Polynomial function model test  -Hyper parameters tuning
polyReg = PolynomialFeatures(degree=3)
x_poly3 = polyReg.fit_transform(x_train) #Feature extraction to multiple linear reg function
linReg3 = LinearRegression()
linReg3.fit(x_poly3,y_train)


# In[55]:


x_poly_val3 = polyReg.fit_transform(x_val)
print("Degree 3 - Training Error: ", linReg3.score(x_poly3,y_train))
print("Degree 3 - Validation Error: ", linReg3.score(x_poly_val3,y_val))


# In[57]:


results_df = pd.DataFrame({
    'Actual Values': np.squeeze(y_test),
    'Predicted Values': np.squeeze(linReg3.predict(polyReg.fit_transform(x_test)))
})
results_df


# In[ ]:




