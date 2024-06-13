import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

user_reviews_df = pd.read_csv('/Users/ilaydaiyikesici/Desktop/googleplaystore_user_reviews.csv')
app_data_df = pd.read_csv('/Users/ilaydaiyikesici/Desktop/googleplaystore.csv')
child_app_data_df = pd.read_csv('/Users/ilaydaiyikesici/Desktop/Apps.csv')
child_review_data_df = pd.read_csv('/Users/ilaydaiyikesici/Desktop/Reviews.csv')

import pandas as pd

dfParentApps = pd.read_csv("/Users/ilaydaiyikesici/Desktop/googleplaystore.csv")
dfParentReviews = pd.read_csv("/Users/ilaydaiyikesici/Desktop/googleplaystore_user_reviews.csv")
dfChildApps = pd.read_csv("/Users/ilaydaiyikesici/Desktop/Apps.csv")
dfChildReviews = pd.read_csv("/Users/ilaydaiyikesici/Desktop/Reviews.csv")

dfParentApps.rename(columns={"App": "App Name"}, inplace=True)
dfParentReviews.rename(columns={"App": "App Name"}, inplace=True)
dfChildApps.rename(columns={"title": "App Name", "appId": "app_Id"}, inplace=True)
dfChildReviews.rename(columns={"repliedAt": "Review Date"}, inplace=True)

mergedMain = pd.merge(dfParentApps, dfParentReviews, on='App Name', how='left')
mergedChild = pd.merge(dfChildApps, dfChildReviews, on='app_Id', how='left')



# Merging datasets m
mergedAll = pd.merge(mergedMain, mergedChild[['App Name', 'Review Date']], on='App Name', how='left')


mergedAll.drop_duplicates(inplace=True)

print(mergedAll)

# Function to detect and handle outliers using IQR method

def handle_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), np.nan, df[col])
    return df

# Convert Size to numeric values (in MB), handling non-numeric values
mergedAll['Size'] = mergedAll['Size'].replace('Varies with device', np.nan)
mergedAll['Size'] = mergedAll['Size'].str.replace('M', '').str.replace('k', 'e-3')
mergedAll['Size'] = pd.to_numeric(mergedAll['Size'], errors='coerce')

# Review Date nan drop
mergedAll.dropna(subset=['Review Date'], inplace=True)

# Drop any remaining rows with missing or non-numeric values in 'Size'
mergedAll.dropna(subset=['Size'], inplace=True)

# Installs to numeric values
mergedAll['Installs'] = mergedAll['Installs'].astype(str).str.replace('+', '').str.replace(',', '').astype(int)

# Encode the Price column
mergedAll['Price'] = mergedAll['Price'].astype(str).str.replace('$', '', regex=False).astype(float)

mergedAll['Size'] = mergedAll['Size'].astype(str).str.replace('M', '', regex=False).astype(float)

mergedAll['Rating'] = mergedAll['Rating'].astype(float)

# Convert 'Review Date' column to datetime format
mergedAll['Review Date'] = pd.to_datetime(mergedAll['Review Date'])

# Change the format of 'Review Date' to dd-mm-yyyy
mergedAll['Review Date'] = mergedAll['Review Date'].dt.strftime('%d-%m-%Y')

print(mergedAll)

#Standart Scaler
numeric_cols = ['Rating', 'Reviews', 'Size', 'Installs', 'Sentiment_Polarity', 'Sentiment_Subjectivity']

# Step 5: Do Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(mergedAll[numeric_cols])

# Convert scaled features back to DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=numeric_cols)

# Include categorical variables
categorical_vars = mergedAll[['Category', 'Type', 'Content Rating', 'Genres']]

# Encode categorical variables
label_encoder = LabelEncoder()
for col in categorical_vars.columns:
    categorical_vars[col] = label_encoder.fit_transform(categorical_vars[col])

# Combine scaled features and encoded categorical variables
final_df = pd.concat([scaled_df, categorical_vars.reset_index(drop=True)], axis=1)

# Step 6: Split the Data
X = final_df.drop('Rating', axis=1)
y = final_df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display training and test sets
print("Training Set:")
print(X_train.head())

print("\nTest Set:")
print(X_test.head())

#EDA PART
print("Basic Statistics:")
print(mergedAll.describe())

numeric_data = mergedAll[numeric_cols]

# Correlation matrix for numeric data
corr_matrix = numeric_data.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Between Numeric Features')
plt.show()

# Bar plot of mean Rating by Category
plt.figure(figsize=(12, 8))
sns.barplot(x='Category', y='Rating', data=mergedAll, ci=None)  # ci=None to remove error bars
plt.title('Mean Rating by Category')
plt.xticks(rotation=45)
plt.ylabel('Mean Rating')
plt.xlabel('Category')
plt.show()

# Relationship Between Features (Pair Plot)
sns.pairplot(mergedAll[['Rating', 'Reviews', 'Size', 'Installs', 'Sentiment_Polarity', 'Sentiment_Subjectivity']])
plt.suptitle('Pair Plot of Numerical Features', y=1.02)
plt.show()

# App Ratings Over Time
mergedAll['Review Date'] = pd.to_datetime(mergedAll['Review Date'])
ratings_over_time = mergedAll.groupby('Review Date')['Rating'].mean()
plt.figure(figsize=(12, 6))
ratings_over_time.plot()
plt.title('Average App Ratings Over Time')
plt.xlabel('Date')
plt.ylabel('Average Rating')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor  # Add this import statement
from sklearn.metrics import mean_squared_error, r2_score



# Model Building - RandomForestRegressor

# Split the data into training and testing sets
X = final_df.drop('Rating', axis=1)
y = final_df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions on training and test sets
y_train_pred = rf_regressor.predict(X_train)
y_test_pred = rf_regressor.predict(X_test)

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing RMSE: {test_rmse:.4f}")
print(f"Training R^2: {train_r2:.4f}")
print(f"Testing R^2: {test_r2:.4f}")

# Feature Importance
feature_importances = rf_regressor.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the grid search
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',  # Use RMSE as scoring metric
                           cv=5,  # 5-fold cross-validation
                           verbose=1,
                           n_jobs=-1)

# Perform grid search
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Lowest RMSE found: ", np.sqrt(-grid_search.best_score_))

# Get the best model
best_rf_model = grid_search.best_estimator_

# Make predictions and evaluate the best model
y_train_pred = best_rf_model.predict(X_train)
y_test_pred = best_rf_model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing RMSE: {test_rmse:.4f}")
print(f"Training R^2: {train_r2:.4f}")
print(f"Testing R^2: {test_r2:.4f}")

# Impact of a single feature (example: 'Installs') on the target variable
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train['Installs'], y=y_train, alpha=0.5)
plt.xlabel('Installs')
plt.ylabel('Rating')
plt.title('Impact of Installs on Rating')
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Reload the dataset and perform necessary preprocessing
df = mergedAll.copy()

# Select numeric columns for clustering
numeric_cols = ['Rating', 'Reviews', 'Size', 'Installs', 'Sentiment_Polarity', 'Sentiment_Subjectivity']
X = df[numeric_cols].copy()

# Impute missing values with the mean of each column
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_imputed)

# Scale the data to [0, 1] range using Min-Max scaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_pca)

# Apply K-Means clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Get cluster labels
cluster_labels = kmeans.labels_

# Add cluster labels to the dataframe
df['Cluster'] = cluster_labels

# Visualize the clustering results using PCA and t-SNE for dimensionality reduction
plt.figure(figsize=(15, 5))

# PCA visualization
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='viridis', alpha=0.6)
plt.title('PCA - K-Means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()

# Cluster analysis
cluster_stats = df.groupby('Cluster')[numeric_cols].mean()

plt.figure(figsize=(12, 6))

for i, col in enumerate(numeric_cols):
    plt.subplot(2, 3, i + 1)
    sns.barplot(x=cluster_stats.index, y=cluster_stats[col], palette='viridis')
    plt.title(f'Mean {col} by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(f'Mean {col}')

plt.tight_layout()
plt.show()
