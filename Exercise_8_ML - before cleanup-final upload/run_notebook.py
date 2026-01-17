
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

sheet_names = ['Sheet2', 'Sheet3']
all_metrics = []

for sheet in sheet_names:
    df = pd.read_excel("curse-of-dimensionality.xlsx", sheet_name=sheet)
    df_cleaned = df.dropna()
    
    X = df_cleaned.drop(columns=['y'])
    y = df_cleaned['y']

    # --- Original Features ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    train_r2 = model.score(X_train_scaled, y_train)
    test_r2 = model.score(X_test_scaled, y_test)
    train_rmse = root_mean_squared_error(y_train, model.predict(X_train_scaled))
    test_rmse = root_mean_squared_error(y_test, model.predict(X_test_scaled))
    
    all_metrics.append({'Sheet': sheet, 'Features': 'Original', 'Train R2': train_r2, 'Test R2': test_r2, 'Train RMSE': train_rmse, 'Test RMSE': test_rmse})

    # --- PCA Features ---
    X_scaled = scaler.fit_transform(X)
    
    for n_components in [2, 4]:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, train_size=0.8, random_state=42)
        
        model_pca = LinearRegression()
        model_pca.fit(X_train_pca, y_train_pca)
        
        train_r2_pca = model_pca.score(X_train_pca, y_train_pca)
        test_r2_pca = model_pca.score(X_test_pca, y_test_pca)
        train_rmse_pca = root_mean_squared_error(y_train_pca, model_pca.predict(X_train_pca))
        test_rmse_pca = root_mean_squared_error(y_test_pca, model_pca.predict(X_test_pca))
        
        all_metrics.append({'Sheet': sheet, 'Features': f'PCA {n_components}', 'Train R2': train_r2_pca, 'Test R2': test_r2_pca, 'Train RMSE': train_rmse_pca, 'Test RMSE': test_rmse_pca})

metrics_df = pd.DataFrame(all_metrics)
print(metrics_df)
