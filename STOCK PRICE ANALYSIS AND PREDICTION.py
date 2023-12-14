#!/usr/bin/env python
# coding: utf-8

# # Data Analysis and Manipulation:

# In[1]:


import pandas as pd #data manipulation
import numpy as np


# # Data Visualization :

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# # Machine Learning:¶

# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[4]:


df = pd.read_csv('world-stock-prices-dataset.csv')


# In[5]:


df.head(3)


# In[6]:


df.tail(3)


# In[7]:


df.shape


# In[8]:


df.describe()


# In[9]:


df.info


# In[10]:


df.dtypes


# # DATA PREPROCESSING
# 

# In[11]:


df.isnull().sum()


# In[12]:


df.dropna(inplace=True)


# In[13]:


df


# In[14]:


df.nunique().sort_values()


# In[15]:


import warnings
warnings.filterwarnings('ignore')


# In[16]:


df['Date'] = pd.to_datetime(df['Date'], utc=True)
df['Month'] = df['Date'].dt.month
df


# In[17]:


monthly_avg = df.groupby('Month')['Close'].mean()
monthly_avg


# # UNIVARIATE ANALYSIS

# In[18]:


plt.figure(figsize=(10, 5))
plt.plot(monthly_avg.index, monthly_avg.values, marker='o', linestyle='-')
plt.title('Month-wise Average Closing Prices')
plt.xlabel('Month')
plt.ylabel('Average Closing Price')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.show()


# In[19]:


df['Date'] = pd.to_datetime(df['Date'], utc=True)
df['year'] = df['Date'].dt.year
df


# ## Histogram of Closing Prices

# In[20]:


plt.figure(figsize=(6, 6))
plt.hist(df['Close'], bins=20, color='skyblue')
plt.title('Histogram of Closing Prices')
plt.xlabel('Closing Price')
plt.ylabel('Frequency')
plt.show()


# ## Box Plot of Closing Prices
# 

# In[21]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='Brand_Name', y='Close', data=df, palette='Set3')
plt.title('Box Plot of Closing Prices by Brand')
plt.xticks(rotation=45)
plt.show()


# # Bivariate Analysis

# ## Scatter Plot of High vs. Low Prices

# In[22]:


plt.figure(figsize=(8, 6))
plt.scatter(df['High'], df['Low'], alpha=0.5, color='green')
plt.title('Scatter Plot of High vs. Low Prices')
plt.xlabel('High Price')
plt.ylabel('Low Price')
plt.show()


# ## Correlation Heatmap

# In[23]:


correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[24]:


df


# In[25]:


df.shape


# ## 3D Scatter Plot 

# In[26]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Open'], df['High'], df['Low'], c='skyblue', marker='o')
ax.set_xlabel('Open')
ax.set_ylabel('High')
ax.set_zlabel('Low')
plt.title('3D Scatter Plot')
plt.show()


# # OVERVIEW OF EDA
# 

# In[27]:


import pandas as pd
from pandas_profiling import ProfileReport

profile = ProfileReport(pd.read_csv('world-stock-prices-dataset.csv'), explorative=True)


profile.to_file("output.html")


# In[28]:


from IPython.display import IFrame

# Provide the path to your HTML report
report_path = "output.html"

# Display the HTML report using IFrame
IFrame(report_path, width="100%", height=800)


# # <span style="color:blue">MACHINE LEARNING</span>
# 

# #  <span style="color:#ff5733">FINDING MSE</span>
# 

# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd


features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

user_input_brand = input("Enter the brand name: ")

# Filter data for a specific brand
brand_data = df[df['Brand_Name'] == user_input_brand]

# Check if the brand exists in the dataset
if brand_data.empty:
    print(f"No data found for the brand: {user_input_brand}")
else:
    X = brand_data[features]
    y = brand_data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error for {user_input_brand}: {mse}')
    
    # Visualization of actual vs. predicted closing prices 
    plt.figure(figsize=(10, 6))

    # Scatter plot for actual values in blue
    sns.regplot(x=y_test, y=y_test, scatter_kws={'s': 20, 'alpha': 0.7, 'color': 'blue'}, label='Actual')

    # Scatter plot for predicted values in red
    sns.regplot(x=y_test, y=predictions, scatter_kws={'s': 20, 'alpha': 0.7, 'color': 'red'}, label='Predicted')

    plt.xlabel('Actual Closing Price')
    plt.ylabel('Predicted Closing Price')
    plt.title(f"Actual vs. Predicted Closing Prices for {user_input_brand}")
    plt.legend()
    plt.show()


# #  <span style="color:#ff5733">RESIDUAL PLOT</span>
# 

# In[30]:


import matplotlib.pyplot as plt

## Residual plot
residuals = y_test - predictions
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for Linear Regression')
plt.show()


# # <span style="color:#ff5733">BRAND WITH HIGHEST INFLUENCE</span>
# 

# In[31]:


from sklearn.linear_model import LinearRegression
import pandas as pd


features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

# Get user input for the year
user_input_year = int(input("Enter the year: "))

# Filter data for the specified year
yearly_data = df[df['year'] == user_input_year]

# Check if data exists for the specified year
if yearly_data.empty:
    print(f"No data found for the year (2000-2023) : {user_input_year}")
else:
    brand_with_highest_influence = None
    max_influence = float('-inf')

    # Iterate through unique brands
    for brand in yearly_data['Brand_Name'].unique():
        brand_data = yearly_data[yearly_data['Brand_Name'] == brand]

        # Calculate average closing price for the brand
        avg_closing_price = brand_data['Close'].mean()

        # Create a new DataFrame with average closing price
        brand_avg_data = pd.DataFrame({'Average_Close': [avg_closing_price] * len(brand_data)})

        X = brand_avg_data[['Average_Close']]
        y = brand_data[target]

        # Train the model
        model = LinearRegression()
        model.fit(X, y)

        # Get the slope of the line (influence)
        influence = model.coef_[0]

        # Check if the current brand has higher influence
        if influence > max_influence:
            max_influence = influence
            brand_with_highest_influence = brand

    print(f"The brand with the highest influence in the year {user_input_year} based on average closing prices is: {brand_with_highest_influence}")


# In[32]:


from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt


features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

# Create empty lists to store results
years = []
highest_influence_brands = []

# Iterate through unique years in the dataset
for year in df['year'].unique():
    yearly_data = df[df['year'] == year]

    brand_with_highest_influence = None
    max_influence = float('-inf')

    # Iterate through unique brands
    for brand in yearly_data['Brand_Name'].unique():
        brand_data = yearly_data[yearly_data['Brand_Name'] == brand]

        # Calculate average closing price for the brand
        avg_closing_price = brand_data['Close'].mean()

        # Create a new DataFrame with average closing price
        brand_avg_data = pd.DataFrame({'Average_Close': [avg_closing_price] * len(brand_data)})

        X = brand_avg_data[['Average_Close']]
        y = brand_data[target]

        # Train the model
        model = LinearRegression()
        model.fit(X, y)

        # Get the slope of the line (influence)
        influence = model.coef_[0]

        # Check if the current brand has higher influence
        if influence > max_influence:
            max_influence = influence
            brand_with_highest_influence = brand

    years.append(year)
    highest_influence_brands.append(brand_with_highest_influence)

result_df = pd.DataFrame({'Year': years, 'Brand_with_Highest_Influence': highest_influence_brands})

# Create a bar chart
plt.figure(figsize=(12, 6))
plt.bar(result_df['Year'], result_df['Brand_with_Highest_Influence'])
plt.title('Brand with Highest Influence Each Year')
plt.xlabel('Year')
plt.ylabel('Brand with Highest Influence')
plt.xticks(result_df['Year'])
plt.xticks(rotation=45)

plt.show()


# # <span style="color:#ff5733">Actual vs. Predicted Closing Prices for a Brand</span>
# 

# In[33]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt

# Assuming df is your DataFrame containing the stock data
features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

# Get user input for the brand name
user_input_brand = input("Enter the Brand Name: ")

# Filter data for a specific brand
brand_data = df[df['Brand_Name'] == user_input_brand]

# Check if there is sufficient data for the selected brand
if brand_data.empty:
    print(f"No data found for the brand: {user_input_brand}")
else:
    # Convert 'Date' to datetime
    brand_data['Date'] = pd.to_datetime(brand_data['Date'], utc=True)

    # Sort the data by date
    brand_data.sort_values(by='Date', inplace=True)

    # Features and target variable
    X = brand_data[features]
    y = brand_data[target]

    # Check if there is sufficient data for training
    if X.shape[0] <= 1:
        print("Insufficient data for training. At least 2 samples are required.")
    else:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Check if there is any data for training
        if X_train.shape[0] == 0:
            print("No data available for training.")
        else:
            # Train the model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Get user input for the month and year
            user_input_month = int(input("Enter the month (1-12): "))
            user_input_year = int(input("Enter the year (2000-2023): "))

            # Filter data for the specified month and year
            user_input_data = brand_data[(brand_data['Date'].dt.month == user_input_month) & (brand_data['Date'].dt.year == user_input_year)]

            # Check if there is data for prediction
            if not user_input_data.empty:
                # Predict closing price for the user-input month and year
                predictions = model.predict(user_input_data[features])

                print(f"Predicted closing prices for {user_input_brand} in {user_input_month}/{user_input_year}:\n{predictions}")

                # Visualization of actual vs. predicted closing prices
                plt.figure(figsize=(10, 6))
                plt.plot(user_input_data['Date'], user_input_data['Close'], label='Actual Closing Prices')
                plt.scatter(user_input_data['Date'], predictions, color='red', marker='o', label='Predicted Closing Prices')
                plt.xlabel('Date')
                plt.ylabel('Closing Price')
                plt.title(f"Actual vs. Predicted Closing Prices for {user_input_brand} in {user_input_month}/{user_input_year}")
                plt.legend()
                plt.xticks(rotation=30)
                
                plt.show()
            else:
                print(f"No data available for the specified month and year: {user_input_month}/{user_input_year}")

