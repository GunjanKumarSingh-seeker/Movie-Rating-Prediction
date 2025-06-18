import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data similar to Netflix dataset structure
np.random.seed(42)
n_samples = 10000

# data = {
#     'User_ID': np.random.randint(1, 1000, n_samples),
#     'Movie_ID': np.random.randint(1, 500, n_samples),
#     'Rating': np.random.randint(1, 5, n_samples)
# }
# df = pd.DataFrame(data)
df = pd.read_csv("C:/Users/win 10/OneDrive/Documents/Vs Code/Movie Rating Prediction/Netflix_Dataset_Rating.csv")


# Feature engineering
def create_features(df):
    # User features
    user_features = df.groupby('User_ID').agg({
        'Rating': ['mean', 'std', 'count']
    }).fillna(0)
    user_features.columns = ['user_avg_rating', 'user_std_rating', 'user_rating_count']
    
    # Movie features
    movie_features = df.groupby('Movie_ID').agg({
        'Rating': ['mean', 'std', 'count']
    }).fillna(0)
    movie_features.columns = ['movie_avg_rating', 'movie_std_rating', 'movie_rating_count']
    
    # Merge features
    df = df.merge(user_features, left_on='User_ID', right_index=True)
    df = df.merge(movie_features, left_on='Movie_ID', right_index=True)
    
    return df

# Prepare data
df_processed = create_features(df)
features = ['user_avg_rating', 'user_std_rating', 'user_rating_count',
            'movie_avg_rating', 'movie_std_rating', 'movie_rating_count']

X = df_processed[features]
y = df_processed['Rating']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, train_pred)):.3f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, test_pred)):.3f}")
print(f"Test MAE: {mean_absolute_error(y_test, test_pred):.3f}")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Prediction function
def predict_rating(user_id, movie_id, model, scaler, df):
    # Get user and movie features
    user_data = df[df['User_ID'] == user_id].iloc[0]
    movie_data = df[df['Movie_ID'] == movie_id].iloc[0]
    
    features = np.array([[
        user_data['user_avg_rating'],
        user_data['user_std_rating'],
        user_data['user_rating_count'],
        movie_data['movie_avg_rating'],
        movie_data['movie_std_rating'],
        movie_data['movie_rating_count']
    ]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    return round(prediction, 1)