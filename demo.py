#!/usr/bin/env python3
"""
AI Crop Price Prediction System - Demo Script

This script demonstrates how to use the crop price prediction system programmatically.
It generates data, trains models, and makes predictions for multiple crops.
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_generator import CropDataGenerator
from ml_model import CropPricePredictionModel

def main():
    print("🌾 AI Crop Price Prediction System - Demo")
    print("=" * 50)
    
    # Step 1: Generate synthetic data
    print("📊 Step 1: Generating synthetic crop price data...")
    generator = CropDataGenerator(start_date='2020-01-01', end_date='2024-01-01')
    data = generator.generate_all_crops_data()
    print(f"✅ Generated {len(data):,} records for {len(data['crop'].unique())} crops")
    
    # Show sample data
    print("\n📋 Sample Data:")
    print(data[['date', 'crop', 'price', 'temperature', 'rainfall', 'demand_index']].head(10))
    
    # Step 2: Data Analysis
    print("\n📈 Step 2: Basic Market Analysis...")
    
    # Price statistics by crop
    price_stats = data.groupby('crop')['price'].agg(['mean', 'std', 'min', 'max']).round(2)
    print(f"\n💰 Price Statistics (USD/ton):")
    print(price_stats)
    
    # Most volatile crops
    volatility = data.groupby('crop')['price'].std().sort_values(ascending=False)
    print(f"\n📊 Most Volatile Crops:")
    for i, (crop, vol) in enumerate(volatility.head(3).items(), 1):
        print(f"{i}. {crop}: ${vol:.2f} volatility")
    
    # Step 3: Train Machine Learning Models
    print(f"\n🤖 Step 3: Training Machine Learning Models...")
    model = CropPricePredictionModel()
    model.train_models(data)
    
    print(f"✅ Best model selected: {model.best_model_name}")
    
    # Show model performance
    print(f"\n📊 Model Performance Metrics:")
    for name, metrics in model.model_metrics.items():
        print(f"{name}:")
        print(f"  - R² Score: {metrics['R2']:.4f}")
        print(f"  - RMSE: ${metrics['RMSE']:.2f}")
        print(f"  - MAE: ${metrics['MAE']:.2f}")
    
    # Feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        print(f"\n🎯 Top 5 Most Important Features:")
        for i, row in importance.head(5).iterrows():
            print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")
    
    # Step 4: Make Predictions
    print(f"\n🔮 Step 4: Making Price Predictions...")
    
    # Predict prices for different crops
    crops_to_predict = ['Wheat', 'Rice', 'Corn']
    prediction_days = 14
    
    all_predictions = {}
    
    for crop in crops_to_predict:
        print(f"\n📈 Predicting {crop} prices for next {prediction_days} days...")
        
        # Custom weather and market forecasts
        weather_forecast = {
            'temperature': [25.0 + i*0.1 for i in range(prediction_days)],  # Gradually warming
            'rainfall': [50.0 - i*0.5 for i in range(prediction_days)],     # Decreasing rain
            'humidity': [60.0] * prediction_days,                           # Stable humidity
            'sunshine_hours': [8.0 + i*0.1 for i in range(prediction_days)] # More sunshine
        }
        
        market_forecast = {
            'demand_index': [105.0] * prediction_days,     # Higher demand
            'supply_index': [95.0] * prediction_days,      # Lower supply
            'export_volume': [1100.0] * prediction_days,   # Increased exports
            'import_volume': [750.0] * prediction_days,    # Decreased imports
            'fuel_price': [85.0] * prediction_days         # Higher fuel costs
        }
        
        predictions = model.predict_future_prices(
            crop, 
            days_ahead=prediction_days,
            weather_forecast=weather_forecast,
            market_forecast=market_forecast
        )
        
        all_predictions[crop] = predictions
        
        # Show prediction summary
        avg_price = predictions['predicted_price'].mean()
        min_price = predictions['predicted_price'].min()
        max_price = predictions['predicted_price'].max()
        
        print(f"  Average predicted price: ${avg_price:.2f}/ton")
        print(f"  Price range: ${min_price:.2f} - ${max_price:.2f}/ton")
    
    # Step 5: Generate Insights and Recommendations
    print(f"\n💡 Step 5: AI-Powered Market Insights...")
    
    # Get current prices for comparison
    latest_prices = data.groupby('crop')['price'].last()
    
    recommendations = []
    for crop in crops_to_predict:
        current_price = latest_prices[crop]
        predicted_avg = all_predictions[crop]['predicted_price'].mean()
        change_percent = ((predicted_avg - current_price) / current_price) * 100
        
        if change_percent > 5:
            recommendation = "🔥 STRONG BUY"
        elif change_percent > 2:
            recommendation = "📈 BUY"
        elif change_percent > -2:
            recommendation = "⚖️ HOLD"
        elif change_percent > -5:
            recommendation = "📉 SELL"
        else:
            recommendation = "🚨 STRONG SELL"
        
        recommendations.append({
            'Crop': crop,
            'Current Price': f"${current_price:.2f}",
            'Predicted Avg': f"${predicted_avg:.2f}",
            'Change': f"{change_percent:+.1f}%",
            'Recommendation': recommendation
        })
    
    print(f"\n📋 Trading Recommendations:")
    recommendations_df = pd.DataFrame(recommendations)
    print(recommendations_df.to_string(index=False))
    
    # Step 6: Export Results
    print(f"\n💾 Step 6: Exporting Results...")
    
    # Save predictions to CSV
    for crop, predictions in all_predictions.items():
        filename = f"{crop.lower()}_predictions.csv"
        predictions.to_csv(filename, index=False)
        print(f"📁 Saved {crop} predictions to {filename}")
    
    # Save model
    model.save_model('trained_crop_model.pkl')
    print(f"🤖 Saved trained model to trained_crop_model.pkl")
    
    # Step 7: Success Summary
    print(f"\n🎉 Demo Complete!")
    print("=" * 50)
    print("✅ Data generated and analyzed")
    print("✅ ML models trained and evaluated")
    print("✅ Price predictions generated")
    print("✅ Market insights provided")
    print("✅ Results exported")
    
    print(f"\n🌾 Next Steps:")
    print("1. Run 'streamlit run app.py' to use the web interface")
    print("2. Customize weather/market forecasts for different scenarios")
    print("3. Integrate with real market data APIs")
    print("4. Set up automated prediction scheduling")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()