#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("Assignment - Junior Data Analyst (1).csv")


# In[3]:


numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns


# In[4]:


df.head(10)


# In[5]:


for col in numerical_cols:
    print(f"Univariate analysis of numerical column: {col}")
    print(df[col].describe())


# In[6]:


plt.figure(figsize=(10, 4))
sns.histplot(df[col], kde=True, bins=30)
plt.title(f'Distribution of {col}')
plt.show()


# In[8]:


print(df[col].dtype)
print(df[col].head())


# In[9]:


df['battery'] = df['battery'].str.extract('(\d+)').astype(float)


# In[10]:


correlation_matrix = df.select_dtypes(include=[float, int]).corr()


# In[11]:


correlation_matrix = df.apply(pd.to_numeric, errors='coerce').corr()


# In[12]:


correlation_matrix = df.apply(pd.to_numeric, errors='coerce').corr()


# In[ ]:


df_numeric = df.select_dtypes(include=[float, int])

correlation_matrix = df_numeric.corr()

print(correlation_matrix)


# In[ ]:


print(df.dtypes)
df['column_name'] = pd.to_numeric(df['column_name'], errors='coerce')


# In[30]:


df_numeric = df_numeric.dropna() 


# In[17]:


correlation_matrix = df.apply(pd.to_numeric, errors='coerce').corr()

price_corr = correlation_matrix['price'].sort_values(ascending=False)
rating_corr = correlation_matrix['rating'].sort_values(ascending=False)

print("Correlation with Price:")
print(price_corr)

print("\nCorrelation with Rating:")
print(rating_corr)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[18]:


# other insights that can we drawn are
sns.pairplot(df[['price', 'rating', 'battery', 'memory', 'camera']].apply(pd.to_numeric, errors='coerce'))
plt.show()


# In[19]:


# Distribution of Features
df[['price', 'rating', 'battery']].hist(bins=20, figsize=(10, 6))
plt.show()



# In[20]:


# Relationship between Features and Ratings
sns.boxplot(x='rating', y='battery', data=df)
plt.show()


# In[22]:


# Average Price and Rating by Feature
avg_price_rating = df.groupby('processor')[['price', 'rating']].mean().sort_values(by='price', ascending=False)
print(avg_price_rating)



# In[27]:


df['battery'] = df['battery'].astype(str).str.extract('(\d+)').astype(float)  # Extract numbers from battery string
df['camera'] = df['camera'].astype(str).str.extract('(\d+)').astype(float)    # Extract first number from camera string
df['warranty'] = df['warranty'].astype(str).str.extract('(\d+)').astype(float)  # Extract warranty period
df['price'] = pd.to_numeric(df['price'], errors='coerce')  # Convert price to numeric

df = df.fillna(df.mean())

print(f"Shape of dataset after preprocessing: {df.shape}")

X = df.drop(columns=['rating'])
y = df['rating']

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

if X.shape[0] > 0 and y.shape[0] > 0:
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
 
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
 
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Actual vs Predicted Ratings")
    plt.show()
else:
    print("Error: No data available for model training.")


# In[26]:


df['battery'] = df['battery'].astype(str).str.extract('(\d+)').astype(float)  
df['camera'] = df['camera'].astype(str).str.extract('(\d+)').astype(float)    
df['warranty'] = df['warranty'].astype(str).str.extract('(\d+)').astype(float)  
df['price'] = pd.to_numeric(df['price'], errors='coerce')  

df_numeric = df.drop(columns=['name', 'reviews', 'rating'])  

print("Before filling missing values:")
print(df_numeric.isna().sum())

df_numeric = df_numeric.fillna(df_numeric.mean())  

print("After filling missing values:")
print(df_numeric.isna().sum())  


print("Shape of the dataset after preprocessing:", df_numeric.shape)

if df_numeric.shape[0] == 0:
    print("Error: No valid samples left for clustering.")
else:
  
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(df_scaled)

    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['cluster'], cmap='viridis', s=50)
    plt.title('K-Means Clustering of Phones (PCA-reduced)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(label='Cluster')
    plt.show()


# In[ ]:




