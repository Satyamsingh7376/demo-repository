#!/usr/bin/env python3
"""
Test script to verify the functionality of the AI Crop Price Prediction System
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_data_generation():
    """Test the data generation module"""
    print("🧪 Testing Data Generation...")
    try:
        from data_generator import CropDataGenerator
        
        generator = CropDataGenerator(start_date='2023-01-01', end_date='2023-12-31')
        data = generator.generate_all_crops_data()
        
        assert len(data) > 0, "No data generated"
        assert 'price' in data.columns, "Price column missing"
        assert 'crop' in data.columns, "Crop column missing"
        assert len(data['crop'].unique()) == 10, "Should have 10 different crops"
        
        print(f"✅ Data generation successful: {len(data)} records, {len(data['crop'].unique())} crops")
        return True
    except Exception as e:
        print(f"❌ Data generation failed: {str(e)}")
        return False

def test_model_training():
    """Test the machine learning model"""
    print("🧪 Testing ML Model Training...")
    try:
        from data_generator import CropDataGenerator
        from ml_model import CropPricePredictionModel
        
        # Generate small dataset for testing
        generator = CropDataGenerator(start_date='2023-01-01', end_date='2023-03-31')
        data = generator.generate_all_crops_data()
        
        # Train model
        model = CropPricePredictionModel()
        model.train_models(data)
        
        assert model.best_model is not None, "No best model selected"
        assert model.best_model_name is not None, "Best model name not set"
        assert len(model.model_metrics) > 0, "No model metrics generated"
        
        print(f"✅ Model training successful: Best model is {model.best_model_name}")
        return True
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        return False

def test_prediction():
    """Test price prediction functionality"""
    print("🧪 Testing Price Prediction...")
    try:
        from data_generator import CropDataGenerator
        from ml_model import CropPricePredictionModel
        
        # Generate small dataset for testing
        generator = CropDataGenerator(start_date='2023-01-01', end_date='2023-06-30')
        data = generator.generate_all_crops_data()
        
        # Train model
        model = CropPricePredictionModel()
        model.train_models(data)
        
        # Test prediction
        predictions = model.predict_future_prices('Wheat', days_ahead=7)
        
        assert len(predictions) == 7, "Should predict 7 days"
        assert 'predicted_price' in predictions.columns, "Prediction column missing"
        assert predictions['predicted_price'].min() > 0, "Prices should be positive"
        
        print(f"✅ Price prediction successful: {len(predictions)} days predicted")
        return True
    except Exception as e:
        print(f"❌ Price prediction failed: {str(e)}")
        return False

def test_feature_importance():
    """Test feature importance functionality"""
    print("🧪 Testing Feature Importance...")
    try:
        from data_generator import CropDataGenerator
        from ml_model import CropPricePredictionModel
        
        generator = CropDataGenerator(start_date='2023-01-01', end_date='2023-03-31')
        data = generator.generate_all_crops_data()
        
        model = CropPricePredictionModel()
        model.train_models(data)
        
        importance = model.get_feature_importance()
        
        if importance is not None:
            assert len(importance) > 0, "No feature importance data"
            assert 'feature' in importance.columns, "Feature column missing"
            assert 'importance' in importance.columns, "Importance column missing"
            print(f"✅ Feature importance successful: {len(importance)} features analyzed")
        else:
            print("✅ Feature importance skipped (Linear Regression model)")
        
        return True
    except Exception as e:
        print(f"❌ Feature importance failed: {str(e)}")
        return False

def test_data_analysis():
    """Test data analysis functions"""
    print("🧪 Testing Data Analysis...")
    try:
        from data_generator import CropDataGenerator
        import pandas as pd
        
        generator = CropDataGenerator(start_date='2023-01-01', end_date='2023-12-31')
        data = generator.generate_all_crops_data()
        
        # Test basic analysis operations
        price_stats = data.groupby('crop')['price'].agg(['mean', 'std', 'min', 'max'])
        assert len(price_stats) == 10, "Should have stats for 10 crops"
        
        # Test correlation analysis
        numeric_cols = ['price', 'temperature', 'rainfall', 'humidity']
        correlation = data[numeric_cols].corr()
        assert correlation.shape == (4, 4), "Correlation matrix should be 4x4"
        
        # Test seasonal analysis
        monthly_avg = data.groupby(['crop', 'month'])['price'].mean()
        assert len(monthly_avg) > 0, "No monthly averages calculated"
        
        print("✅ Data analysis successful: Statistics, correlations, and trends calculated")
        return True
    except Exception as e:
        print(f"❌ Data analysis failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🌾 AI Crop Price Prediction System - Functionality Test")
    print("=" * 60)
    
    tests = [
        test_data_generation,
        test_model_training,
        test_prediction,
        test_feature_importance,
        test_data_analysis
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {str(e)}\n")
    
    print("=" * 60)
    print(f"🧪 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())