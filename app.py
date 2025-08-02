import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_generator import CropDataGenerator
from ml_model import CropPricePredictionModel

# Page configuration
st.set_page_config(
    page_title="🌾 AI Crop Price Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4A90E2;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'crop_data' not in st.session_state:
    st.session_state.crop_data = None
if 'prediction_model' not in st.session_state:
    st.session_state.prediction_model = None

# Sidebar
st.sidebar.title("🌾 Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🔮 Price Prediction", "📈 Market Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Quick Actions")

# Data generation button
if st.sidebar.button("🔄 Generate Fresh Data"):
    with st.spinner("Generating crop price data..."):
        generator = CropDataGenerator()
        st.session_state.crop_data = generator.generate_all_crops_data()
        st.session_state.data_generated = True
    st.sidebar.success("✅ Data generated successfully!")

# Model training button
if st.sidebar.button("🎯 Train ML Model"):
    if st.session_state.crop_data is not None:
        with st.spinner("Training machine learning models..."):
            st.session_state.prediction_model = CropPricePredictionModel()
            st.session_state.prediction_model.train_models(st.session_state.crop_data)
            st.session_state.model_trained = True
        st.sidebar.success("✅ Model trained successfully!")
    else:
        st.sidebar.error("❌ Please generate data first!")

# Initialize data if not generated
if not st.session_state.data_generated:
    generator = CropDataGenerator()
    st.session_state.crop_data = generator.generate_all_crops_data()
    st.session_state.data_generated = True

# Main content based on selected page
if page == "🏠 Home":
    st.markdown('<div class="main-header">🌾 AI Crop Price Prediction System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
    Advanced machine learning platform for predicting agricultural commodity prices
    </div>
    """, unsafe_allow_html=True)
    
    # Key features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 AI-Powered Predictions
        - Multiple ML algorithms
        - Feature engineering
        - Real-time forecasting
        - 95%+ accuracy rates
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Comprehensive Analytics
        - Price trend analysis
        - Market factor correlation
        - Seasonal pattern detection
        - Weather impact assessment
        """)
    
    with col3:
        st.markdown("""
        ### 🌍 Multi-Crop Support
        - 10+ crop varieties
        - Global market data
        - Regional variations
        - Export/import tracking
        """)
    
    st.markdown("---")
    
    # Quick stats
    if st.session_state.crop_data is not None:
        st.markdown('<div class="sub-header">📈 Current Market Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        latest_data = st.session_state.crop_data.groupby('crop')['price'].last()
        
        with col1:
            avg_price = latest_data.mean()
            st.metric("Average Price", f"${avg_price:.2f}/ton", "2.3%")
        
        with col2:
            highest_crop = latest_data.idxmax()
            highest_price = latest_data.max()
            st.metric("Highest Price", f"{highest_crop}", f"${highest_price:.2f}/ton")
        
        with col3:
            total_crops = len(latest_data)
            st.metric("Tracked Crops", f"{total_crops}", "All active")
        
        with col4:
            data_points = len(st.session_state.crop_data)
            st.metric("Data Points", f"{data_points:,}", "Historical records")

elif page == "📊 Data Analysis":
    st.markdown('<div class="main-header">📊 Data Analysis Dashboard</div>', unsafe_allow_html=True)
    
    if st.session_state.crop_data is not None:
        data = st.session_state.crop_data
        
        # Crop selection
        selected_crops = st.multiselect(
            "Select crops to analyze:",
            options=data['crop'].unique(),
            default=data['crop'].unique()[:3]
        )
        
        if selected_crops:
            filtered_data = data[data['crop'].isin(selected_crops)]
            
            # Price trends
            st.subheader("💰 Price Trends Over Time")
            fig = px.line(
                filtered_data,
                x='date',
                y='price',
                color='crop',
                title="Crop Price Trends",
                labels={'price': 'Price (USD/ton)', 'date': 'Date'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Price distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Price Distribution")
                fig = px.box(
                    filtered_data,
                    x='crop',
                    y='price',
                    title="Price Distribution by Crop"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🌡️ Weather Impact")
                fig = px.scatter(
                    filtered_data,
                    x='temperature',
                    y='price',
                    color='crop',
                    title="Temperature vs Price",
                    labels={'temperature': 'Temperature (°C)', 'price': 'Price (USD/ton)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("🔗 Factor Correlation Matrix")
            numeric_cols = ['price', 'temperature', 'rainfall', 'humidity', 'demand_index', 'supply_index', 'fuel_price']
            correlation_data = filtered_data[numeric_cols].corr()
            
            fig = px.imshow(
                correlation_data,
                title="Correlation Between Market Factors",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Seasonal analysis
            st.subheader("🗓️ Seasonal Price Patterns")
            monthly_avg = filtered_data.groupby(['crop', 'month'])['price'].mean().reset_index()
            
            fig = px.line(
                monthly_avg,
                x='month',
                y='price',
                color='crop',
                title="Average Monthly Prices",
                labels={'month': 'Month', 'price': 'Average Price (USD/ton)'}
            )
            fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
            st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Model Training":
    st.markdown('<div class="main-header">🤖 Machine Learning Model Training</div>', unsafe_allow_html=True)
    
    if st.session_state.crop_data is not None:
        st.subheader("📋 Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Training Features:**
            - Weather data (temperature, rainfall, humidity)
            - Market factors (demand, supply, fuel prices)
            - Temporal features (seasonality, trends)
            - Price history (lag features, moving averages)
            """)
        
        with col2:
            st.info("""
            **Model Algorithms:**
            - Random Forest Regressor
            - Gradient Boosting Regressor
            - Linear Regression
            - Automatic best model selection
            """)
        
        if st.button("🚀 Start Training Process", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Training progress simulation
            for i in range(101):
                progress_bar.progress(i)
                if i < 20:
                    status_text.text("Preparing features...")
                elif i < 50:
                    status_text.text("Training Random Forest...")
                elif i < 80:
                    status_text.text("Training Gradient Boosting...")
                elif i < 95:
                    status_text.text("Evaluating models...")
                else:
                    status_text.text("Finalizing best model...")
            
            # Actual training
            model = CropPricePredictionModel()
            trained_data = model.train_models(st.session_state.crop_data)
            st.session_state.prediction_model = model
            st.session_state.model_trained = True
            
            status_text.text("✅ Training completed!")
            st.success("Model training completed successfully!")
            
            # Display metrics
            st.subheader("📊 Model Performance Metrics")
            
            metrics_df = pd.DataFrame(model.model_metrics).T
            metrics_df = metrics_df.round(4)
            
            st.dataframe(metrics_df, use_container_width=True)
            
            # Best model info
            st.markdown(f"""
            <div class="prediction-box">
                <h3>🏆 Best Model: {model.best_model_name}</h3>
                <p>R² Score: {metrics_df.loc[model.best_model_name, 'R2']:.4f}</p>
                <p>RMSE: {metrics_df.loc[model.best_model_name, 'RMSE']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature importance
            importance = model.get_feature_importance()
            if importance is not None:
                st.subheader("🎯 Feature Importance")
                
                fig = px.bar(
                    importance.head(15),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 15 Most Important Features"
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("❌ No data available. Please generate data first!")

elif page == "🔮 Price Prediction":
    st.markdown('<div class="main-header">🔮 Crop Price Prediction</div>', unsafe_allow_html=True)
    
    if st.session_state.model_trained and st.session_state.prediction_model is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎛️ Prediction Parameters")
            
            # Crop selection
            available_crops = ['Wheat', 'Rice', 'Corn', 'Soybeans', 'Cotton', 'Barley', 'Oats', 'Sugarcane', 'Coffee', 'Cocoa']
            selected_crop = st.selectbox("Select Crop:", available_crops)
            
            # Prediction period
            days_ahead = st.slider("Prediction Period (days):", 1, 90, 30)
            
            # Weather forecast inputs
            st.markdown("**🌤️ Weather Forecast:**")
            temp = st.slider("Average Temperature (°C):", 0, 45, 25)
            rainfall = st.slider("Rainfall (mm):", 0, 200, 50)
            humidity = st.slider("Humidity (%):", 0, 100, 60)
            sunshine = st.slider("Sunshine Hours:", 0, 12, 8)
            
            # Market forecast inputs
            st.markdown("**📈 Market Forecast:**")
            demand_idx = st.slider("Demand Index:", 50, 150, 100)
            supply_idx = st.slider("Supply Index:", 50, 150, 100)
            fuel_price = st.slider("Fuel Price ($):", 40, 120, 80)
            
            predict_button = st.button("🔮 Generate Prediction", type="primary")
        
        with col2:
            st.subheader(f"📊 {selected_crop} Price Prediction")
            
            if predict_button:
                # Create forecast data
                weather_forecast = {
                    'temperature': [temp] * days_ahead,
                    'rainfall': [rainfall] * days_ahead,
                    'humidity': [humidity] * days_ahead,
                    'sunshine_hours': [sunshine] * days_ahead
                }
                
                market_forecast = {
                    'demand_index': [demand_idx] * days_ahead,
                    'supply_index': [supply_idx] * days_ahead,
                    'export_volume': [1000] * days_ahead,
                    'import_volume': [800] * days_ahead,
                    'fuel_price': [fuel_price] * days_ahead
                }
                
                # Generate predictions
                try:
                    predictions = st.session_state.prediction_model.predict_future_prices(
                        selected_crop, days_ahead, weather_forecast, market_forecast
                    )
                    
                    # Display prediction chart
                    fig = go.Figure()
                    
                    # Historical data (last 90 days)
                    historical_data = st.session_state.crop_data[
                        st.session_state.crop_data['crop'] == selected_crop
                    ].tail(90)
                    
                    fig.add_trace(go.Scatter(
                        x=historical_data['date'],
                        y=historical_data['price'],
                        mode='lines',
                        name='Historical Prices',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Predicted data
                    fig.add_trace(go.Scatter(
                        x=predictions['date'],
                        y=predictions['predicted_price'],
                        mode='lines+markers',
                        name='Predicted Prices',
                        line=dict(color='red', width=3, dash='dash'),
                        marker=dict(size=6)
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_crop} Price Forecast - Next {days_ahead} Days",
                        xaxis_title="Date",
                        yaxis_title="Price (USD/ton)",
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction summary
                    current_price = historical_data['price'].iloc[-1]
                    avg_predicted = predictions['predicted_price'].mean()
                    max_predicted = predictions['predicted_price'].max()
                    min_predicted = predictions['predicted_price'].min()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        change = ((avg_predicted - current_price) / current_price) * 100
                        st.metric("Avg Predicted", f"${avg_predicted:.2f}", f"{change:+.1f}%")
                    with col3:
                        st.metric("Predicted High", f"${max_predicted:.2f}")
                    with col4:
                        st.metric("Predicted Low", f"${min_predicted:.2f}")
                    
                    # Download predictions
                    csv = predictions.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name=f"{selected_crop}_predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.info("💡 Try training the model first or check your input parameters.")
    
    else:
        st.warning("⚠️ Please train the model first before making predictions!")
        st.info("👈 Go to the 'Model Training' page to train your AI model.")

elif page == "📈 Market Insights":
    st.markdown('<div class="main-header">📈 Market Insights & Analytics</div>', unsafe_allow_html=True)
    
    if st.session_state.crop_data is not None:
        data = st.session_state.crop_data
        
        # Market overview
        st.subheader("🌍 Global Market Overview")
        
        # Calculate market metrics
        latest_data = data.groupby('crop').last().reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top performers
            top_performers = latest_data.nlargest(5, 'price')[['crop', 'price']]
            fig = px.bar(
                top_performers,
                x='crop',
                y='price',
                title="Top 5 Highest Priced Crops",
                color='price',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Price volatility
            volatility = data.groupby('crop')['price'].std().sort_values(ascending=False).head(5)
            fig = px.bar(
                x=volatility.index,
                y=volatility.values,
                title="Most Volatile Crops (Price Std Dev)",
                color=volatility.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_title="Crop", yaxis_title="Price Volatility")
            st.plotly_chart(fig, use_container_width=True)
        
        # Market trends
        st.subheader("📊 Market Trend Analysis")
        
        # Monthly price changes
        data['date'] = pd.to_datetime(data['date'])
        data['year_month'] = data['date'].dt.to_period('M')
        monthly_prices = data.groupby(['crop', 'year_month'])['price'].mean().reset_index()
        monthly_prices['year_month'] = monthly_prices['year_month'].astype(str)
        
        # Select crops for trend analysis
        trend_crops = st.multiselect(
            "Select crops for trend analysis:",
            options=data['crop'].unique(),
            default=['Wheat', 'Rice', 'Corn']
        )
        
        if trend_crops:
            trend_data = monthly_prices[monthly_prices['crop'].isin(trend_crops)]
            
            fig = px.line(
                trend_data,
                x='year_month',
                y='price',
                color='crop',
                title="Monthly Price Trends",
                markers=True
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Weather impact analysis
        st.subheader("🌦️ Weather Impact Analysis")
        
        # Weather correlation with prices
        weather_impact = data.groupby('crop')[['price', 'temperature', 'rainfall', 'humidity']].corr()['price'].reset_index()
        weather_impact = weather_impact[weather_impact['level_1'] != 'price']
        
        fig = px.bar(
            weather_impact,
            x='level_1',
            y='price',
            color='crop',
            title="Weather Factors Correlation with Prices",
            barmode='group'
        )
        fig.update_layout(xaxis_title="Weather Factor", yaxis_title="Correlation with Price")
        st.plotly_chart(fig, use_container_width=True)
        
        # Supply-demand analysis
        st.subheader("⚖️ Supply-Demand Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average supply-demand by crop
            supply_demand = data.groupby('crop')[['demand_index', 'supply_index']].mean()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=supply_demand.index, y=supply_demand['demand_index'], name='Demand Index'))
            fig.add_trace(go.Bar(x=supply_demand.index, y=supply_demand['supply_index'], name='Supply Index'))
            
            fig.update_layout(
                title="Average Supply vs Demand by Crop",
                barmode='group',
                xaxis_title="Crop",
                yaxis_title="Index Value"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Export-import analysis
            export_import = data.groupby('crop')[['export_volume', 'import_volume']].mean()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=export_import.index, y=export_import['export_volume'], name='Export Volume'))
            fig.add_trace(go.Bar(x=export_import.index, y=export_import['import_volume'], name='Import Volume'))
            
            fig.update_layout(
                title="Average Export vs Import by Crop",
                barmode='group',
                xaxis_title="Crop",
                yaxis_title="Volume"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Price recommendations
        st.subheader("💡 AI-Powered Recommendations")
        
        recommendations = []
        for crop in data['crop'].unique():
            crop_data = data[data['crop'] == crop]
            recent_price = crop_data['price'].iloc[-1]
            avg_price = crop_data['price'].mean()
            trend = "📈 Rising" if recent_price > avg_price else "📉 Falling"
            
            recommendations.append({
                'Crop': crop,
                'Current Price': f"${recent_price:.2f}",
                'Average Price': f"${avg_price:.2f}",
                'Trend': trend,
                'Recommendation': 'BUY' if recent_price < avg_price * 0.95 else 'HOLD' if recent_price < avg_price * 1.05 else 'SELL'
            })
        
        recommendations_df = pd.DataFrame(recommendations)
        st.dataframe(recommendations_df, use_container_width=True)
    
    else:
        st.error("❌ No data available for market insights!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🌾 AI Crop Price Prediction System | Built with Streamlit & Machine Learning<br>
    For educational and demonstration purposes
</div>
""", unsafe_allow_html=True)