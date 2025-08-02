import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

class CropPricePredictionModel:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.model_metrics = {}
        
    def prepare_features(self, data):
        """Prepare features for training"""
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Encode crop names
        df['crop_encoded'] = self.label_encoder.fit_transform(df['crop'])
        
        # Create additional time-based features
        df['date'] = pd.to_datetime(df['date'])
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Create lag features (previous day's price)
        df = df.sort_values(['crop', 'date'])
        df['price_lag_1'] = df.groupby('crop')['price'].shift(1)
        df['price_lag_7'] = df.groupby('crop')['price'].shift(7)
        
        # Create moving averages
        df['price_ma_7'] = df.groupby('crop')['price'].rolling(window=7).mean().reset_index(0, drop=True)
        df['price_ma_30'] = df.groupby('crop')['price'].rolling(window=30).mean().reset_index(0, drop=True)
        
        # Create volatility features
        df['price_volatility_7'] = df.groupby('crop')['price'].rolling(window=7).std().reset_index(0, drop=True)
        
        # Weather interaction features
        df['temp_rainfall_interaction'] = df['temperature'] * df['rainfall']
        df['humidity_sunshine_interaction'] = df['humidity'] * df['sunshine_hours']
        
        # Market interaction features
        df['demand_supply_ratio'] = df['demand_index'] / (df['supply_index'] + 1)
        df['export_import_ratio'] = df['export_volume'] / (df['import_volume'] + 1)
        
        # Drop rows with NaN values (from lag features)
        df = df.dropna()
        
        # Select features for training
        feature_columns = [
            'crop_encoded', 'temperature', 'rainfall', 'humidity', 'sunshine_hours',
            'demand_index', 'supply_index', 'export_volume', 'import_volume', 'fuel_price',
            'month', 'quarter', 'day_of_year', 'week_of_year',
            'price_lag_1', 'price_lag_7', 'price_ma_7', 'price_ma_30', 'price_volatility_7',
            'temp_rainfall_interaction', 'humidity_sunshine_interaction',
            'demand_supply_ratio', 'export_import_ratio'
        ]
        
        X = df[feature_columns]
        y = df['price']
        
        self.feature_names = feature_columns
        
        return X, y, df
    
    def train_models(self, data):
        """Train all models and select the best one"""
        print("Preparing features...")
        X, y, processed_data = self.prepare_features(data)
        
        print(f"Training on {len(X)} samples with {len(X.columns)} features")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=X['crop_encoded']
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train and evaluate each model
        best_score = float('-inf')
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            if name == 'Linear Regression':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            self.model_metrics[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'CV_R2_mean': cv_scores.mean(),
                'CV_R2_std': cv_scores.std()
            }
            
            print(f"{name} - R2: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
            
            # Select best model based on R2 score
            if r2 > best_score:
                best_score = r2
                self.best_model = model
                self.best_model_name = name
        
        print(f"\nBest model: {self.best_model_name} with R2 score: {best_score:.4f}")
        
        return processed_data
    
    def predict(self, input_data):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_models() first.")
        
        # Prepare features
        if self.best_model_name == 'Linear Regression':
            input_scaled = self.scaler.transform(input_data)
            predictions = self.best_model.predict(input_scaled)
        else:
            predictions = self.best_model.predict(input_data)
        
        return predictions
    
    def get_feature_importance(self):
        """Get feature importance for tree-based models"""
        if self.best_model is None or self.best_model_name == 'Linear Regression':
            return None
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None
    
    def save_model(self, filepath='crop_price_model.pkl'):
        """Save the trained model"""
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_metrics': self.model_metrics
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='crop_price_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_metrics = model_data['model_metrics']
        print(f"Model loaded from {filepath}")
    
    def predict_future_prices(self, crop_name, days_ahead=30, weather_forecast=None, market_forecast=None):
        """Predict future prices for a specific crop"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_models() first.")
        
        # Get crop encoding
        try:
            crop_encoded = self.label_encoder.transform([crop_name])[0]
        except ValueError:
            raise ValueError(f"Crop '{crop_name}' not found in training data")
        
        # Create future data points
        future_predictions = []
        
        # Use default values if forecasts not provided
        if weather_forecast is None:
            weather_forecast = {
                'temperature': [25.0] * days_ahead,
                'rainfall': [50.0] * days_ahead,
                'humidity': [60.0] * days_ahead,
                'sunshine_hours': [8.0] * days_ahead
            }
        
        if market_forecast is None:
            market_forecast = {
                'demand_index': [100.0] * days_ahead,
                'supply_index': [100.0] * days_ahead,
                'export_volume': [1000.0] * days_ahead,
                'import_volume': [800.0] * days_ahead,
                'fuel_price': [80.0] * days_ahead
            }
        
        # Create prediction DataFrame
        from datetime import datetime, timedelta
        start_date = datetime.now()
        
        future_data = []
        for i in range(days_ahead):
            future_date = start_date + timedelta(days=i)
            
            # Create feature vector
            features = {
                'crop_encoded': crop_encoded,
                'temperature': weather_forecast['temperature'][i],
                'rainfall': weather_forecast['rainfall'][i],
                'humidity': weather_forecast['humidity'][i],
                'sunshine_hours': weather_forecast['sunshine_hours'][i],
                'demand_index': market_forecast['demand_index'][i],
                'supply_index': market_forecast['supply_index'][i],
                'export_volume': market_forecast['export_volume'][i],
                'import_volume': market_forecast['import_volume'][i],
                'fuel_price': market_forecast['fuel_price'][i],
                'month': future_date.month,
                'quarter': (future_date.month - 1) // 3 + 1,
                'day_of_year': future_date.timetuple().tm_yday,
                'week_of_year': future_date.isocalendar()[1],
                'price_lag_1': 250.0,  # Default values - in practice, use last known prices
                'price_lag_7': 250.0,
                'price_ma_7': 250.0,
                'price_ma_30': 250.0,
                'price_volatility_7': 10.0,
                'temp_rainfall_interaction': weather_forecast['temperature'][i] * weather_forecast['rainfall'][i],
                'humidity_sunshine_interaction': weather_forecast['humidity'][i] * weather_forecast['sunshine_hours'][i],
                'demand_supply_ratio': market_forecast['demand_index'][i] / (market_forecast['supply_index'][i] + 1),
                'export_import_ratio': market_forecast['export_volume'][i] / (market_forecast['import_volume'][i] + 1)
            }
            
            future_data.append(features)
        
        future_df = pd.DataFrame(future_data)
        
        # Make predictions
        predictions = self.predict(future_df[self.feature_names])
        
        # Create results DataFrame
        results = pd.DataFrame({
            'date': [start_date + timedelta(days=i) for i in range(days_ahead)],
            'crop': [crop_name] * days_ahead,
            'predicted_price': predictions
        })
        
        return results

# Example usage
if __name__ == "__main__":
    from data_generator import CropDataGenerator
    
    # Generate sample data
    generator = CropDataGenerator()
    data = generator.generate_all_crops_data()
    
    # Train model
    model = CropPricePredictionModel()
    model.train_models(data)
    
    # Get feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        print("\nTop 10 Most Important Features:")
        print(importance.head(10))
    
    # Save model
    model.save_model()
    
    print("\nModel training completed!")