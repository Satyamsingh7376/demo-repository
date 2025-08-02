import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class CropDataGenerator:
    def __init__(self, start_date='2020-01-01', end_date='2024-01-01'):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.crops = [
            'Wheat', 'Rice', 'Corn', 'Soybeans', 'Cotton', 
            'Barley', 'Oats', 'Sugarcane', 'Coffee', 'Cocoa'
        ]
        
    def generate_weather_data(self, num_days):
        """Generate synthetic weather data"""
        weather_data = {
            'temperature': np.random.normal(25, 8, num_days),  # Temperature in Celsius
            'rainfall': np.abs(np.random.normal(50, 30, num_days)),  # Rainfall in mm
            'humidity': np.clip(np.random.normal(60, 15, num_days), 0, 100),  # Humidity %
            'sunshine_hours': np.clip(np.random.normal(8, 3, num_days), 0, 12)  # Sunshine hours
        }
        return weather_data
    
    def generate_market_factors(self, num_days):
        """Generate synthetic market factors"""
        market_data = {
            'demand_index': np.random.normal(100, 15, num_days),
            'supply_index': np.random.normal(100, 20, num_days),
            'export_volume': np.abs(np.random.normal(1000, 300, num_days)),
            'import_volume': np.abs(np.random.normal(800, 250, num_days)),
            'fuel_price': np.abs(np.random.normal(80, 15, num_days))
        }
        return market_data
    
    def generate_seasonal_effect(self, dates, crop_type):
        """Generate seasonal price effects for different crops"""
        seasonal_multipliers = {
            'Wheat': [1.1, 1.0, 0.9, 0.85, 0.8, 0.85, 0.9, 1.0, 1.1, 1.2, 1.15, 1.1],
            'Rice': [0.9, 0.85, 0.8, 0.9, 1.0, 1.1, 1.2, 1.15, 1.0, 0.95, 0.9, 0.9],
            'Corn': [1.0, 0.95, 0.9, 0.85, 0.8, 0.9, 1.0, 1.1, 1.2, 1.15, 1.05, 1.0],
            'Soybeans': [1.05, 1.0, 0.95, 0.9, 0.85, 0.9, 1.0, 1.1, 1.15, 1.1, 1.05, 1.0],
            'Cotton': [1.0, 1.05, 1.1, 1.0, 0.9, 0.85, 0.8, 0.9, 1.0, 1.1, 1.05, 1.0]
        }
        
        default_multiplier = [1.0, 0.95, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 0.95, 0.9, 0.95]
        multipliers = seasonal_multipliers.get(crop_type, default_multiplier)
        
        seasonal_effects = []
        for date in dates:
            month = date.month - 1  # 0-indexed
            seasonal_effects.append(multipliers[month])
        
        return np.array(seasonal_effects)
    
    def generate_price_trend(self, num_days, base_price, volatility=0.1):
        """Generate realistic price trends with volatility"""
        # Create a trend component
        trend = np.cumsum(np.random.normal(0, 0.001, num_days))
        
        # Add volatility
        volatility_component = np.random.normal(0, volatility, num_days)
        
        # Combine components
        price_changes = trend + volatility_component
        prices = base_price * np.exp(np.cumsum(price_changes))
        
        return prices
    
    def generate_crop_data(self, crop_name, num_days=None):
        """Generate comprehensive data for a specific crop"""
        if num_days is None:
            num_days = (self.end_date - self.start_date).days
        
        # Generate date range
        dates = [self.start_date + timedelta(days=i) for i in range(num_days)]
        
        # Base prices for different crops (USD per ton)
        base_prices = {
            'Wheat': 250, 'Rice': 400, 'Corn': 200, 'Soybeans': 450,
            'Cotton': 1500, 'Barley': 180, 'Oats': 220, 'Sugarcane': 50,
            'Coffee': 3000, 'Cocoa': 2500
        }
        
        base_price = base_prices.get(crop_name, 300)
        
        # Generate weather data
        weather_data = self.generate_weather_data(num_days)
        
        # Generate market factors
        market_data = self.generate_market_factors(num_days)
        
        # Generate seasonal effects
        seasonal_effects = self.generate_seasonal_effect(dates, crop_name)
        
        # Generate base price trend
        base_prices_trend = self.generate_price_trend(num_days, base_price)
        
        # Calculate final prices with all factors
        weather_impact = (
            (weather_data['temperature'] - 25) * 0.002 +
            (weather_data['rainfall'] - 50) * 0.001 +
            (weather_data['humidity'] - 60) * 0.001
        )
        
        market_impact = (
            (market_data['demand_index'] - 100) * 0.01 +
            (100 - market_data['supply_index']) * 0.01 +
            (market_data['fuel_price'] - 80) * 0.005
        )
        
        final_prices = base_prices_trend * seasonal_effects * (1 + weather_impact + market_impact)
        
        # Create DataFrame
        data = {
            'date': dates,
            'crop': [crop_name] * num_days,
            'price': final_prices,
            'temperature': weather_data['temperature'],
            'rainfall': weather_data['rainfall'],
            'humidity': weather_data['humidity'],
            'sunshine_hours': weather_data['sunshine_hours'],
            'demand_index': market_data['demand_index'],
            'supply_index': market_data['supply_index'],
            'export_volume': market_data['export_volume'],
            'import_volume': market_data['import_volume'],
            'fuel_price': market_data['fuel_price'],
            'month': [d.month for d in dates],
            'year': [d.year for d in dates],
            'day_of_year': [d.timetuple().tm_yday for d in dates]
        }
        
        return pd.DataFrame(data)
    
    def generate_all_crops_data(self):
        """Generate data for all crops"""
        all_data = []
        
        for crop in self.crops:
            crop_data = self.generate_crop_data(crop)
            all_data.append(crop_data)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    
    def save_data(self, filename='crop_price_data.csv'):
        """Generate and save all crop data to CSV"""
        data = self.generate_all_crops_data()
        data.to_csv(filename, index=False)
        return data

# Example usage
if __name__ == "__main__":
    generator = CropDataGenerator()
    data = generator.save_data()
    print(f"Generated {len(data)} rows of crop price data")
    print(data.head())