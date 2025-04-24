from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import re
from dateutil import parser

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load and preprocess the data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('indirapuram,-ghaziabad-air-quality.csv')
        
        # Strip spaces from column names
        df.columns = df.columns.str.strip()
        
        # Strip spaces and convert pollutant columns to numeric
        pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        for col in pollutant_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
            else:
                df[col] = np.nan

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'].astype(str).str.strip(), errors='coerce')

        # Calculate AQI using proper EPA method
        def calculate_aqi(row):
            # AQI breakpoints for PM2.5
            pm25_breakpoints = [
                (0, 12, 0, 50),
                (12.1, 35.4, 51, 100),
                (35.5, 55.4, 101, 150),
                (55.5, 150.4, 151, 200),
                (150.5, 250.4, 201, 300),
                (250.5, 350.4, 301, 400),
                (350.5, 500.4, 401, 500)
            ]
            
            # Get PM2.5 value
            pm25 = row['pm25']
            
            # Calculate AQI for PM2.5
            for low_conc, high_conc, low_aqi, high_aqi in pm25_breakpoints:
                if low_conc <= pm25 <= high_conc:
                    aqi = ((high_aqi - low_aqi) / (high_conc - low_conc)) * (pm25 - low_conc) + low_aqi
                    return round(aqi)
            
            return None

        # Apply AQI calculation
        df['aqi'] = df.apply(calculate_aqi, axis=1)

        # Drop rows where all pollutant values are missing
        df = df.dropna(subset=['date', 'pm25'], how='all')

        # Sort by date
        df = df.sort_values('date')
        
        # Fill missing values with forward fill then backward fill
        df[pollutant_cols] = df[pollutant_cols].fillna(method='ffill').fillna(method='bfill')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def parse_date_from_query(query):
    try:
        # Try parsing with dateutil
        try:
            return parser.parse(query, fuzzy=True)
        except:
            pass
        
        # Try custom parsing if dateutil fails
        query = query.lower()
        
        # List of month names and their variations
        months = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }
        
        for month_name, month_num in months.items():
            if month_name in query:
                # Find year (assuming 4 digits)
                year_match = re.search(r'\b20\d{2}\b', query)
                year = int(year_match.group()) if year_match else datetime.now().year
                
                # Find day (1-31)
                day_match = re.search(r'\b(\d{1,2})(st|nd|rd|th)?\b', query)
                day = int(day_match.group(1)) if day_match else 1
                
                return datetime(year=year, month=month_num, day=day)
                
        return None
    except Exception as e:
        print(f"Error parsing date: {str(e)}")
        return None

def get_date_specific_data(df, query):
    try:
        target_date = parse_date_from_query(query)
        if target_date:
            # Find the closest date in the dataset
            df['date'] = pd.to_datetime(df['date'])
            closest_date = df.iloc[(df['date'] - target_date).abs().argsort()[:1]]
            
            if len(closest_date) > 0:
                result = {
                    'date': closest_date['date'].iloc[0],
                    'aqi': closest_date['aqi'].iloc[0],
                    'pm25': closest_date['pm25'].iloc[0],
                    'pm10': closest_date['pm10'].iloc[0],
                    'o3': closest_date['o3'].iloc[0],
                    'no2': closest_date['no2'].iloc[0],
                    'so2': closest_date['so2'].iloc[0],
                    'co': closest_date['co'].iloc[0]
                }
                return result
    except Exception as e:
        print(f"Error in get_date_specific_data: {str(e)}")
    return None

def prepare_time_series_data(df):
    try:
        # Ensure we have at least some data
        if len(df) < 2:
            raise ValueError("Not enough data points")
            
        # Create daily data
        df_daily = df.set_index('date').resample('D').mean()
        
        # More aggressive filling of missing values
        df_daily = df_daily.fillna(method='ffill').fillna(method='bfill')
        
        # If still have NaN, fill with mean
        df_daily = df_daily.fillna(df_daily.mean())
        
        # If any NaN still remain, fill with 0
        df_daily = df_daily.fillna(0)
        
        return df_daily
    except Exception as e:
        print(f"Error in prepare_time_series_data: {str(e)}")
        # Create minimal viable dataset
        dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
        return pd.DataFrame({'aqi': [df['aqi'].mean()] * len(dates)}, index=dates)

def train_prophet_model(df):
    try:
        # Prepare data for Prophet
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = df.index
        prophet_df['y'] = df['aqi']
        
        # Use more conservative Prophet parameters
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative',
            interval_width=0.95,
            uncertainty_samples=1000,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
        )
        
        model.fit(prophet_df)
        return model
    except Exception as e:
        print(f"Error in train_prophet_model: {str(e)}")
        return None

def make_forecast(df, days=30):
    try:
        # Prepare data
        df_daily = prepare_time_series_data(df)
        
        # Validate AQI values
        df_daily['aqi'] = df_daily['aqi'].clip(0, 500)
        
        # Train Prophet model
        prophet_model = train_prophet_model(df_daily)
        
        if prophet_model is None:
            raise ValueError("Failed to train Prophet model")
        
        # Make future dataframe for Prophet
        future_dates = prophet_model.make_future_dataframe(periods=days)
        
        # Forecast
        forecast = prophet_model.predict(future_dates)
        
        # Validate forecast values
        forecast['yhat'] = forecast['yhat'].clip(0, 500)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(0, 500)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(0, 500)
        
        return forecast
    except Exception as e:
        print(f"Error in make_forecast: {str(e)}")
        # Return simple trend-based forecast
        try:
            last_value = df['aqi'].iloc[-1]
            future_dates = pd.date_range(start=df['date'].iloc[-1], periods=days+1, freq='D')[1:]
            forecast_values = [last_value] * days
            
            return pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast_values,
                'yhat_lower': [v * 0.9 for v in forecast_values],
                'yhat_upper': [v * 1.1 for v in forecast_values]
            })
        except:
            # Absolute fallback
            future_dates = pd.date_range(start=datetime.now(), periods=days, freq='D')
            return pd.DataFrame({
                'ds': future_dates,
                'yhat': [100] * days,
                'yhat_lower': [90] * days,
                'yhat_upper': [110] * days
            })

def create_forecast_visualization(df, forecast):
    try:
        # Validate data ranges
        df['aqi'] = df['aqi'].clip(0, 500)
        forecast['yhat'] = forecast['yhat'].clip(0, 500)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(0, 500)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(0, 500)
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['aqi'],
            name='Historical AQI',
            line=dict(color='blue')
        ))
        
        # Forecasted data
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            name='Forecasted AQI',
            line=dict(color='red', dash='dash')
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(255,0,0,0.2)',
            name='Upper Bound'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(255,0,0,0.2)',
            name='Lower Bound'
        ))
        
        fig.update_layout(
            title='AQI Forecast',
            xaxis_title='Date',
            yaxis_title='AQI (0-500)',
            hovermode='x unified',
            yaxis_range=[0, 500]
        )
        
        return fig
    except Exception as e:
        print(f"Error in create_forecast_visualization: {str(e)}")
        return None

def create_visualization(df, query):
    try:
        if any(word in query.lower() for word in ['forecast', 'predict', 'future']):
            # Generate forecast
            forecast = make_forecast(df)
            if forecast is not None:
                return create_forecast_visualization(df, forecast)
            else:
                # Create a simple trend line instead
                fig = px.line(df, x='date', y='aqi',
                             title='AQI Trend (Forecast Unavailable)')
                return fig
        
        elif any(word in query.lower() for word in ['trend', 'history', 'pattern']):
            # Create multiple line plot for all pollutants
            fig = go.Figure()
            for column in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']:
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df[column],
                    name=column.upper(),
                    mode='lines'
                ))
            fig.update_layout(title='Pollutant Trends Over Time',
                             xaxis_title='Date',
                             yaxis_title='Concentration')
            return fig
        
        elif 'correlation' in query.lower():
            # Create correlation matrix heatmap
            corr = df[['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']].corr()
            fig = px.imshow(corr,
                           labels=dict(color="Correlation"),
                           title='Pollutant Correlations')
            return fig
        
        elif any(word in query.lower() for word in ['pm2.5', 'pm25']):
            fig = px.line(df, x='date', y='pm25',
                         title='PM2.5 Levels Over Time')
            return fig
        
        return None
    except Exception as e:
        print(f"Error in create_visualization: {str(e)}")
        return None

def get_openai_response(prompt, df):
    try:
        # Check if this is a date-specific query
        date_data = get_date_specific_data(df, prompt)
        if date_data:
            response_text = f"""On {date_data['date'].strftime('%B %d, %Y')}, the air quality measurements were:
            
            AQI: {date_data['aqi']:.1f}
            PM2.5: {date_data['pm25']:.1f} μg/m³
            PM10: {date_data['pm10']:.1f} μg/m³
            O3: {date_data['o3']:.1f} ppb
            NO2: {date_data['no2']:.1f} ppb
            SO2: {date_data['so2']:.1f} ppb
            CO: {date_data['co']:.1f} ppm
            
            Would you like me to analyze these values or compare them with current levels?"""
            return response_text

        # Generate forecast if query is about prediction
        forecast_info = ""
        if any(word in prompt.lower() for word in ['forecast', 'predict', 'future']):
            forecast = make_forecast(df)
            if forecast is not None:
                next_month_avg = forecast['yhat'].tail(30).mean()
                max_aqi = forecast['yhat'].tail(30).max()
                min_aqi = forecast['yhat'].tail(30).min()
                
                forecast_info = f"""
                Forecast for the next 30 days (simplified model):
                Average AQI: {next_month_avg:.1f}
                Expected Range: {min_aqi:.1f} - {max_aqi:.1f}
                Note: This is a forecast based on available data and trends.
                """
            else:
                forecast_info = "Basic forecast based on recent trends."

        # Get recent statistics for all pollutants
        recent_stats = {
            'pm25': {
                'current': df['pm25'].iloc[-1],
                'avg': df['pm25'].tail(7).mean(),
                'max': df['pm25'].tail(7).max()
            },
            'pm10': {
                'current': df['pm10'].iloc[-1],
                'avg': df['pm10'].tail(7).mean(),
                'max': df['pm10'].tail(7).max()
            },
            'o3': {
                'current': df['o3'].iloc[-1],
                'avg': df['o3'].tail(7).mean(),
                'max': df['o3'].tail(7).max()
            }
        }
        
        messages = [
            {
                "role": "system",
                "content": f"""You are an expert AI assistant specialized in air quality analysis.
                Current Readings:
                PM2.5: {recent_stats['pm25']['current']:.1f} μg/m³
                PM10: {recent_stats['pm10']['current']:.1f} μg/m³
                O3: {recent_stats['o3']['current']:.1f} ppb
                
                7-day Averages:
                PM2.5: {recent_stats['pm25']['avg']:.1f} μg/m³
                PM10: {recent_stats['pm10']['avg']:.1f} μg/m³
                O3: {recent_stats['o3']['avg']:.1f} ppb
                
                {forecast_info}
                
                Provide detailed analysis about:
                1. Current pollution levels and health implications
                2. Recent trends in pollutant levels
                3. Relationships between different pollutants
                4. Health recommendations based on current levels
                5. Future predictions and trends
                
                Include specific numbers and scientific explanations. When there is not enough data to predict accurately, say that you do better forecasting with better datasets and that the forecasting you are going to present might not be very accurate. Also Mention Yug Johri and Prashant Yadav for this making this amazing project. Be cool and use popular hinglish terms sometimes."""
            },
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    st.title("🌬️ Air Quality Analysis and Forecasting System")
    st.write("Analyze air quality trends, pollutant levels, and future predictions using a closed source api")

    # Load data
    df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check the data source and try again.")
        return

    # Sidebar with current pollutant levels
    with st.sidebar:
        st.header("📊 Current Pollutant Levels")
        
        try:
            # Get the most recent non-null values for each pollutant
            current_values = {}
            avg_values = {}
            
            for pollutant in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']:
                # Get last non-null value
                non_null_values = df[df[pollutant].notnull()][pollutant]
                if len(non_null_values) > 0:
                    current_values[pollutant] = non_null_values.iloc[-1]
                    # Get 7-day average excluding null values
                    recent_values = non_null_values.tail(7)
                    avg_values[pollutant] = recent_values.mean() if len(recent_values) > 0 else current_values[pollutant]
                else:
                    current_values[pollutant] = 0
                    avg_values[pollutant] = 0

            # Display metrics
            for pollutant in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']:
                current_val = current_values[pollutant]
                avg_val = avg_values[pollutant]
                delta = current_val - avg_val if current_val is not None and avg_val is not None else 0
                
                # Format the display values
                if pollutant in ['pm25', 'pm10']:
                    units = "μg/m³"
                elif pollutant in ['o3', 'no2', 'so2']:
                    units = "ppb"
                else:  # CO
                    units = "ppm"
                
                st.metric(
                    label=f"{pollutant.upper()}",
                    value=f"{current_val:.1f} {units}" if current_val is not None else "N/A",
                    delta=f"{delta:+.1f}" if delta != 0 else "0"
                )

        except Exception as e:
            st.error(f"Error displaying metrics: {str(e)}")
            # Fallback display
            for pollutant in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']:
                st.metric(f"{pollutant.upper()}", "N/A", "0")

        # Add information about pollutants with units
        with st.expander("About Pollutants"):
            st.write("""
            - **PM2.5**: Fine particulate matter (≤2.5 μm) [μg/m³]
            - **PM10**: Coarse particulate matter (≤10 μm) [μg/m³]
            - **O3**: Ozone [ppb]
            - **NO2**: Nitrogen dioxide [ppb]
            - **SO2**: Sulfur dioxide [ppb]
            - **CO**: Carbon monoxide [ppm]
            """)

    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "visualization" in message:
                st.plotly_chart(message["visualization"])

    # Chat input
    if prompt := st.chat_input("Ask about air quality..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display response
        with st.chat_message("assistant"):
            response = get_openai_response(prompt, df)
            st.markdown(response)
            
            # Create visualization if relevant
            fig = create_visualization(df, prompt)
            if fig:
                st.plotly_chart(fig)
                
            # Add to chat history
            message_data = {
                "role": "assistant",
                "content": response
            }
            if fig:
                message_data["visualization"] = fig
            st.session_state.messages.append(message_data)

if __name__ == "__main__":
    main()