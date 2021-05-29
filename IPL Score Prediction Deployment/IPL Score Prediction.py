#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


data=pd.read_csv("data/ipl.csv")


# In[7]:


data.head(10)


# In[8]:


data.columns


# In[9]:


data.dtypes


# # Data Cleaning
# 
# * Removing unwanted Columns
# * Keeping consistent teams only
# * Removing the first 5 overs data in every match
# * converting the date object from string to datetime object

# In[10]:


#columns to be removed
columns_to_remove=['mid','batsman','bowler','striker','non-striker','venue']

print('Before removing unwanted columns: {}'.format(data.shape))
data.drop(labels=columns_to_remove,axis=1,inplace=True)
print('After removing unwanted columns: {}'.format(data.shape))


# In[11]:


data.columns


# In[12]:


data.head()


# In[13]:


data.index


# In[14]:


data['bat_team'].unique()


# In[15]:


#keeping only consistent teams
consistent_teams=['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals','Kings XI Punjab',
       'Royal Challengers Bangalore', 'Delhi Daredevils','Sunrisers Hyderabad','Mumbai Indians']
print("Before Removing inconsistent teams: {}".format(data.shape))
data=data[(data['bat_team'].isin(consistent_teams))& (data['bowl_team'].isin(consistent_teams))]
print('After removing consistent teams: {}'.format(data.shape))


# In[16]:


data['bat_team'].unique()


# In[17]:


#removing first five overs from each match
print("Before Removing first 5 overs: {}".format(data.shape))
data=data[data['overs']>=5.0]
print("After removing first 5 overs: {}".format(data.shape))


# In[18]:


#converting the column 'date' from string to datetime object
from datetime import datetime
print("Before converting: {}".format(type(data.iloc[0,0])))
data['date']=data['date'].apply(lambda x:datetime.strptime(x,"%Y-%m-%d"))
print("After Removing: {}".format(type(data.iloc[0,0])))


# In[19]:


#selecting correlated feautures using heatmap

corr_matrix=data.corr()
top_corr_features=corr_matrix.index

#plotting the heatmap
plt.figure(figsize=(13,10))
g = sns.heatmap(data=data[top_corr_features].corr(),annot=True) #cmap="RdYlGn"


# # Data Preprocessing
# 
# * Handling categorical features
# * Splitting dataset into train and test set on the basis of date

# In[20]:


#converting the categorical features using onehotencoding method
encoded_data=pd.get_dummies(data=data,columns=['bat_team','bowl_team'])
encoded_data.columns


# In[21]:


encoded_data.head()


# In[22]:


# Rearranging the columns
encoded_data = encoded_data[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]


# In[23]:


#splitting the data into train and test set
X_train=encoded_data.drop(labels='total',axis=1)[encoded_data['date'].dt.year<=2016]
X_test=encoded_data.drop(labels='total',axis=1)[encoded_data['date'].dt.year<=2016]

y_train=encoded_data[encoded_data['date'].dt.year<=2016]['total'].values
y_test=encoded_data[encoded_data['date'].dt.year<=2016]['total'].values

#removing the date columns
X_train.drop(labels='date',axis=True,inplace=True)
X_test.drop(labels='date',axis=True,inplace=True)

print("Training set: {} and Test set: {}".format(X_train.shape,X_test.shape))


# # Model Building

# **1. Linear Regression**

# In[24]:


from sklearn.linear_model import LinearRegression
linear_regressor=LinearRegression()
linear_regressor.fit(X_train,y_train)


# In[25]:


#predicting the results
y_pred_lr=linear_regressor.predict(X_test)


# In[26]:


#Model Evaluation
from sklearn.metrics import mean_absolute_error as mae,mean_squared_error as mse,accuracy_score
print("-----------Linear Regression - Model Evaluation-----------")
print("Mean Absolute Error (MAE): {}".format(mae(y_test,y_pred_lr)))
print("Mean Sqaured Error (MSE): {}".format(mse(y_test,y_pred_lr)))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test,y_pred_lr))))


# **2. Decision Tree**

# In[27]:


from sklearn.tree import DecisionTreeRegressor
decision_regressor=DecisionTreeRegressor()
decision_regressor.fit(X_train,y_train)


# In[28]:


y_pred_dt=decision_regressor.predict(X_test)


# In[29]:


print("-----------Decision Tree - Model Evaluation-----------")
print("Mean Absolute Error (MAE): {}".format(mae(y_test,y_pred_dt)))
print("Mean Sqaured Error (MSE): {}".format(mse(y_test,y_pred_dt)))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test,y_pred_dt))))


# **3. Random Forest**

# In[30]:


from sklearn.ensemble import RandomForestRegressor
random_regressor = RandomForestRegressor()
random_regressor.fit(X_train,y_train)


# In[31]:


# Predicting results
y_pred_rf = random_regressor.predict(X_test)


# In[32]:


print("---- Random Forest Regression - Model Evaluation ----")
print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_rf)))
print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_rf)))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred_rf))))


# Note: Since Linear Regression model performs best as compared to other two, we use this model and boost it's performance using AdaBoost Algorithm
# 
# **4. AdaBoost Algorithm**

# In[33]:


from sklearn.ensemble import AdaBoostRegressor
ada_regressor=AdaBoostRegressor(base_estimator=linear_regressor,n_estimators=100)
ada_regressor.fit(X_train,y_train)


# In[34]:


y_pred_ada=ada_regressor.predict(X_test)


# In[35]:


print("---- AdaBoost Regression - Model Evaluation ----")
print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_ada)))
print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_ada)))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred_ada))))


# # Make Predictions
# 
# * Model trained on the data from IPL Seasons 1 to 9 ie: (2008 to 2016)
# * Model tested on data from IPL Season 10 ie: (2017)
# * Model predicts on data from IPL Seasons 11 to 12 ie: (2018 to 2019)

# In[36]:


def predict_score(batting_team='Chennai Super Kings', bowling_team='Mumbai Indians', overs=5.1, runs=50, wickets=0, runs_in_prev_5=50, wickets_in_prev_5=0):
    temp_array=list()
    
    #batting team
    if batting_team == 'Chennai Super Kings':
        temp_array = temp_array + [1,0,0,0,0,0,0,0]
    elif batting_team == 'Delhi Daredevils':
        temp_array = temp_array + [0,1,0,0,0,0,0,0]
    elif batting_team == 'Kings XI Punjab':
        temp_array = temp_array + [0,0,1,0,0,0,0,0]
    elif batting_team == 'Kolkata Knight Riders':
        temp_array = temp_array + [0,0,0,1,0,0,0,0]
    elif batting_team == 'Mumbai Indians':
        temp_array = temp_array + [0,0,0,0,1,0,0,0]
    elif batting_team == 'Rajasthan Royals':
        temp_array = temp_array + [0,0,0,0,0,1,0,0]
    elif batting_team == 'Royal Challengers Bangalore':
        temp_array = temp_array + [0,0,0,0,0,0,1,0]
    elif batting_team == 'Sunrisers Hyderabad':
        temp_array = temp_array + [0,0,0,0,0,0,0,1]
        
    if bowling_team == 'Chennai Super Kings':
        temp_array = temp_array + [1,0,0,0,0,0,0,0]
    elif bowling_team == 'Delhi Daredevils':
        temp_array = temp_array + [0,1,0,0,0,0,0,0]
    elif bowling_team == 'Kings XI Punjab':
        temp_array = temp_array + [0,0,1,0,0,0,0,0]
    elif bowling_team == 'Kolkata Knight Riders':
        temp_array = temp_array + [0,0,0,1,0,0,0,0]
    elif bowling_team == 'Mumbai Indians':
        temp_array = temp_array + [0,0,0,0,1,0,0,0]
    elif bowling_team == 'Rajasthan Royals':
        temp_array = temp_array + [0,0,0,0,0,1,0,0]
    elif bowling_team == 'Royal Challengers Bangalore':
        temp_array = temp_array + [0,0,0,0,0,0,1,0]
    elif bowling_team == 'Sunrisers Hyderabad':
        temp_array = temp_array + [0,0,0,0,0,0,0,1]

    
    #Overs, Runs, Wickets, Runs_in_prev_5, Wickets_in_prev_5
    temp_array = temp_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]

    #Converting into numpy array
    temp_array = np.array([temp_array])
    
    #Prediction
    return int(linear_regressor.predict(temp_array)[0])


# # Prediction 1
# 
# * Date: 16th April 2018
# * IPL : SEason 11
# * Match No : 13
# * Teams : KKR Vs DD
# * First Innings Final Score : 200/9

# In[37]:


final_score = predict_score(batting_team='Kolkata Knight Riders',bowling_team='Delhi Daredevils',overs=9.2,runs=72,wickets=2,runs_in_prev_5=60,wickets_in_prev_5=1)
print("The final predicted Score (rnage): {} to {}".format(final_score-10, final_score+5))


# # Prediction 2
# 
# * Date: 7th May 2018
# * IPL : Season 11
# * Match No : 39
# * Teams : SRH Vs RCB
# * First Innings Final Score : 146/10

# In[38]:


final_score = predict_score(batting_team='Sunrisers Hyderabad',bowling_team='Royal Challengers Bangalore',overs=10.2,runs=67,wickets=3,runs_in_prev_5=29,wickets_in_prev_5=1)
print("The final predicted Score (rnage): {} to {}".format(final_score-10, final_score+5))


# # Prediction 3
# 
# * Date: 17th May 2018
# * IPL : Season 11
# * Match No : 50
# * Teams : MI Vs KXIP
# * First Innings Final Score : 186/8

# In[39]:


final_score = predict_score(batting_team='Mumbai Indians',bowling_team='Kings XI Punjab',overs=14.4,runs=136,wickets=4,runs_in_prev_5=50,wickets_in_prev_5=0)
print("The final predicted Score (rnage): {} to {}".format(final_score-10, final_score+5))


# # Prediction 4
# * Date: 11th April 2019
# * IPL : Season 12
# * Match number: 25
# * Teams: RR Vs CSK
# * First Innings final score: 151/7

# In[40]:


final_score = predict_score(batting_team='Rajasthan Royals',bowling_team='Chennai Super Kings',overs=13.3,runs=92,wickets=5,runs_in_prev_5=27,wickets_in_prev_5=2)
print("The final predicted Score (rnage): {} to {}".format(final_score-10, final_score+5))


# # Prediction 5
# * Date: 14th April 2019
# * IPL : Season 12
# * Match number: 30
# * Teams: SRH Vs DD
# * First Innings final score: 155/7

# In[42]:


final_score = predict_score(batting_team='Sunrisers Hyderabad',bowling_team='Delhi Daredevils',overs=11.5,runs=98,wickets=3,runs_in_prev_5=41,wickets_in_prev_5=1)
print("The final predicted Score (rnage): {} to {}".format(final_score-10, final_score+5))


# In[44]:


import pickle

pickle.dump(ada_regressor,open('ipl_score_prediction.pkl','wb'))


# In[ ]:


ada_regressor=pickle.load(open('ipl_score_prediction.pkl','rb'))
print(model.predict([[37330,21]]))

