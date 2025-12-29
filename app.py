from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import random
import warnings
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)

class PredictionSystem:
    def __init__(self, model_path='login_attack_model.pkl'):
        """Load trained model and components"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_cols = model_data['feature_cols']
            self.label_encoders = model_data['label_encoders']
            self.categorical_cols = model_data.get('categorical_cols', [])
            self.model_loaded = True
        except:
            print("Warning: Model file not found. Using mock predictions.")
            self.model_loaded = False
        
        self.all_login_data = pd.DataFrame()
    
    def generate_sample_data(self, config):
        """Generate sample login data based on configuration"""
        num_records = int(config.get('num_records', 5))
        user_id_mode = config.get('user_id_mode', 'single')
        ip_mode = config.get('ip_mode', 'multiple')
        country_mode = config.get('country_mode', 'multiple')
        browser_mode = config.get('browser_mode', 'multiple')
        device_mode = config.get('device_mode', 'multiple')
        success_rate = float(config.get('success_rate', 50))
        
        # User ID generation
        if user_id_mode == 'single':
            user_ids = [f"223{random.randint(10, 99)}"] * num_records
        else:
            user_ids = [f"223{random.randint(10, 99)}" for _ in range(num_records)]
        
        # IP generation
        if ip_mode == 'single':
            base_ip = f"209.236.117.{random.randint(1, 255)}"
            ips = [base_ip] * num_records
        else:
            ips = [f"209.236.117.{random.randint(1, 255)}" for _ in range(num_records)]
        
        # Country generation with full names
        countries_list = [
            'United States', 'Brazil', 'China', 'Russia', 'Norway', 
            'India', 'Pakistan', 'Netherlands', 'Germany', 'France',
            'United Kingdom', 'Canada', 'Australia', 'Japan', 'South Korea',
            'Mexico', 'Spain', 'Italy', 'Singapore', 'Thailand',
            'United Arab Emirates', 'Saudi Arabia', 'South Africa', 'Egypt', 'Nigeria'
        ]
        if country_mode == 'single':
            country = random.choice(countries_list)
            countries = [country] * num_records
        else:
            countries = [random.choice(countries_list) for _ in range(num_records)]
        
        # Browser generation
        browsers_list = ['Firefox 20.0', 'Chrome Mobile 81.0', 'IE Mobile 11.0', 
                        'Chrome 91.0', 'Safari 14.0', 'Edge 90.0']
        if browser_mode == 'single':
            browser = random.choice(browsers_list)
            browsers = [browser] * num_records
        else:
            browsers = [random.choice(browsers_list) for _ in range(num_records)]
        
        # Device generation
        devices_list = ['mobile', 'desktop', 'tablet']
        if device_mode == 'single':
            device = random.choice(devices_list)
            devices = [device] * num_records
        else:
            devices = [random.choice(devices_list) for _ in range(num_records)]
        
        # OS based on device
        os_mapping = {
            'mobile': ['iOS 11.2.6', 'iOS 13.1.3', 'Android 5.0', 'Android 10.0'],
            'desktop': ['Windows 10', 'macOS 10.15', 'Linux Ubuntu 20.04'],
            'tablet': ['iOS 13.1.3', 'Android 9.0']
        }
        
        data = []
        base_time = datetime.now()
        
        for i in range(num_records):
            device_type = devices[i]
            os_options = os_mapping.get(device_type, ['Windows 10'])
            
            record = {
                'Login Timestamp': (base_time + timedelta(minutes=random.randint(1, i+1))).strftime('%Y-%m-%d %H:%M:%S'),
                'User ID': user_ids[i],
                'IP Address': ips[i],
                'Country': countries[i],
                'Region': random.choice([
                    'Unknown', 'California', 'Texas', 'Sao Paulo', 'Beijing',
                    'Maharashtra', 'New York', 'London', 'Tokyo', 'Ontario',
                    'New South Wales', 'Bavaria', 'Ile-de-France', 'Seoul',
                    'Moscow', 'Dubai', 'Gauteng', 'Cairo', 'Lagos'
                ]),
                'City': random.choice([
                    'Unknown', 'San Francisco', 'Houston', 'Rio', 'Shanghai',
                    'Mumbai', 'New York City', 'London', 'Tokyo', 'Toronto',
                    'Sydney', 'Munich', 'Paris', 'Seoul', 'Moscow',
                    'Dubai', 'Johannesburg', 'Cairo', 'Lagos', 'Los Angeles',
                    'Chicago', 'Berlin', 'Madrid', 'Rome', 'Singapore'
                ]),
                'ASN': random.choice([393398, 500194, 61349, 25400, 15169]),
                'User Agent String': f"Mozilla/5.0 ({device_type.title()}) {random.randint(1000, 9999)}",
                'Browser Name and Version': browsers[i],
                'OS Name and Version': random.choice(os_options),
                'Device Type': device_type,
                'Login Successful': random.random() * 100 < success_rate,
                'Is Account Takeover': False
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'])
        return df.sort_values('Login Timestamp').reset_index(drop=True)
    
    def create_manual_data(self, records):
        """Create dataframe from manual input"""
        data = []
        for record in records:
            data.append({
                'Login Timestamp': pd.to_datetime(record['timestamp']),
                'User ID': record['user_id'],
                'IP Address': record['ip_address'],
                'Country': record['country'],
                'Region': record.get('region', 'Unknown'),
                'City': record.get('city', 'Unknown'),
                'ASN': int(record.get('asn', 393398)),
                'User Agent String': record.get('user_agent', 'Mozilla/5.0'),
                'Browser Name and Version': record['browser'],
                'OS Name and Version': record['os'],
                'Device Type': record['device_type'],
                'Login Successful': record['login_successful'] == 'true',
                'Is Account Takeover': False
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('Login Timestamp').reset_index(drop=True)
    
    def calculate_features_batch(self, df):
        """Calculate all features from complete dataset"""
        df = df.copy()
        df = df.sort_values(['User ID', 'Login Timestamp']).reset_index(drop=True)
        
        # Time-based features
        df['hour'] = df['Login Timestamp'].dt.hour
        df['minute'] = df['Login Timestamp'].dt.minute
        df['month'] = df['Login Timestamp'].dt.month
        df['day_of_week'] = df['Login Timestamp'].dt.dayofweek
        df['week_of_year'] = df['Login Timestamp'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # User-based features
        df['time_since_last_login'] = df.groupby('User ID')['Login Timestamp'].diff().dt.total_seconds().fillna(0)
        df['login_velocity'] = df.groupby('User ID').cumcount() + 1
        
        # Login count features
        df['login_count_last_hour'] = df.groupby('User ID')['User ID'].transform('count')
        df['login_count_recent'] = df.groupby('User ID').cumcount() + 1
        
        # Country risk score with full country names
        country_risk = {
            'United States': 0.25, 'Brazil': 0.01, 'China':  0.25, 'Russia': 0.25, 
            'Norway': 0.01, 'Netherlands': 0.02, 'India': 0.15, 'Pakistan': 0.25,
            'Germany': 0.05, 'France': 0.05, 'United Kingdom': 0.08, 'Canada': 0.03,
            'Australia': 0.03, 'Japan': 0.04, 'South Korea': 0.06, 'Mexico': 0.18,
            'Spain': 0.07, 'Italy': 0.08, 'Singapore': 0.02, 'Thailand': 0.12,
            'United Arab Emirates': 0.10, 'Saudi Arabia': 0.15, 'South Africa': 0.14,
            'Egypt': 0.20, 'Nigeria': 0.22
        }
        df['country_risk_score'] = df['Country'].map(country_risk).fillna(0.15)
        
        # Location tracking
        df['country_changed'] = (df.groupby('User ID')['Country'].shift() != df['Country']).astype(int)
        df['region_changed'] = (df.groupby('User ID')['Region'].shift() != df['Region']).astype(int)
        df['Country_count'] = df.groupby('User ID')['Country'].transform('nunique')
        df['Region_count'] = df.groupby('User ID')['Region'].transform('nunique')
        df['City_count'] = df.groupby('User ID')['City'].transform('nunique')
        
        # IP change detection
        df['ip_changed'] = (df.groupby('User ID')['IP Address'].shift() != df['IP Address']).astype(int)
        df['user_ip_diversity'] = df.groupby('User ID')['IP Address'].transform('nunique')
        
        # Browser and OS family
        df['Browser_Family'] = df['Browser Name and Version'].str.split().str[0]
        df['OS_Family'] = df['OS Name and Version'].str.split().str[0]
        
        # Device tracking
        df['device_changed'] = (df.groupby('User ID')['Device Type'].shift() != df['Device Type']).astype(int)
        df['browser_changed'] = (df.groupby('User ID')['Browser_Family'].shift() != df['Browser_Family']).astype(int)
        df['os_changed'] = (df.groupby('User ID')['OS_Family'].shift() != df['OS_Family']).astype(int)
        df['user_agent_diversity'] = df.groupby('User ID')['User Agent String'].transform('nunique')
        
        # Device risk score
        device_risk = {'mobile': 0.14, 'desktop': 0.028, 'tablet': 0.1}
        df['device_risk_score'] = df['Device Type'].map(device_risk).fillna(0.1)
        
        # User success metrics
        df['total_logins'] = df.groupby('User ID').cumcount() + 1
        df['success_rate'] = df.groupby('User ID')['Login Successful'].expanding().mean().reset_index(level=0, drop=True)
        
        # Recent failed logins
        df['recent_failed_logins'] = df.groupby('User ID')['Login Successful'].transform(
            lambda x: (~x.astype(bool)).sum()
        )
        
        # Network metrics
        df['avg_rtt'] = 0
        df['rtt_deviation'] = 0
        
        # Risk calculations
        df['geo_risk'] = (df['country_risk_score'] * 0.6 + 
                          df['country_changed'] * 0.2 + 
                          df['region_changed'] * 0.2)
        
        df['network_risk'] = (df['ip_changed'] * 0.6 + 
                              (df['user_ip_diversity'] > 1).astype(int) * 0.4)
        
        df['device_risk'] = (df['device_risk_score'] * 0.5 + 
                             df['device_changed'] * 0.2 + 
                             df['browser_changed'] * 0.15 + 
                             df['os_changed'] * 0.15)
        
        df['behavioral_risk'] = (df['recent_failed_logins'] / df['total_logins'].replace(0, 1) * 0.4 + 
                                 (1 - df['success_rate'].fillna(0.5)) * 0.3 + 
                                 (df['login_count_last_hour'] > 5).astype(int) * 0.3)
        
        df['overall_risk_score'] = (df['geo_risk'] * 0.25 + 
                                    df['network_risk'] * 0.35 + 
                                    df['device_risk'] * 0.2 + 
                                    df['behavioral_risk'] * 0.2)
        
        return df
    
    def prepare_features_for_prediction(self, df):
        """Prepare features for model prediction"""
        if not self.model_loaded:
            return df
        
        df_encoded = df.copy()
        
        for col in self.categorical_cols:
            if col in df_encoded.columns:
                le = self.label_encoders.get(col)
                if le:
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    df_encoded[col] = le.transform(df_encoded[col].astype(str))
                else:
                    df_encoded[col] = 0
        
        X = df_encoded[self.feature_cols].fillna(0)
        return X
    
    def categorize_risk(self, risk_score):
        """Categorize risk level"""
        if risk_score >= 0.35:
            return 'HIGH', 'danger'
        elif risk_score >= 0.3:
            return 'MEDIUM', 'warning'
        else:
            return 'LOW', 'success'
    
    def predict(self, df):
        """Make predictions on dataframe"""
        df_with_features = self.calculate_features_batch(df)
        
        if self.model_loaded:
            X = self.prepare_features_for_prediction(df_with_features)
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]
        else:
            # Mock predictions based on risk scores
            probabilities = df_with_features['overall_risk_score'].values
            predictions = (probabilities > 0.4).astype(int)
        
        df_with_features['Is_Attack_Predicted'] = predictions
        df_with_features['Attack_Probability'] = probabilities
        df_with_features['Risk_Level'], df_with_features['Risk_Badge'] = zip(
            *df_with_features['Attack_Probability'].apply(self.categorize_risk)
        )
        
        return df_with_features

# Initialize prediction system
predictor = PredictionSystem('login_attack_model.pkl')

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.json
        mode = data.get('mode', 'generate')
        
        if mode == 'generate':
            # Generate random data
            config = data.get('config', {})
            df = predictor.generate_sample_data(config)
        else:
            # Manual input
            records = data.get('records', [])
            if not records:
                return jsonify({'error': 'No records provided'}), 400
            df = predictor.create_manual_data(records)
        
        # Make predictions
        results = predictor.predict(df)
        
        # Prepare response
        response_data = {
            'total_records': len(results),
            'high_risk': int((results['Risk_Level'] == 'HIGH').sum()),
            'medium_risk': int((results['Risk_Level'] == 'MEDIUM').sum()),
            'low_risk': int((results['Risk_Level'] == 'LOW').sum()),
            'records': []
        }
        
        for idx, row in results.iterrows():
            response_data['records'].append({
                'id': idx + 1,
                'timestamp': row['Login Timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'user_id': row['User ID'],
                'ip_address': row['IP Address'],
                'country': row['Country'],
                'region': row['Region'],
                'city': row['City'],
                'browser': row['Browser Name and Version'],
                'os': row['OS Name and Version'],
                'device_type': row['Device Type'],
                'login_successful': bool(row['Login Successful']),
                'is_attack': bool(row['Is_Attack_Predicted']),
                'risk_level': row['Risk_Level'],
                'risk_badge': row['Risk_Badge'],
                'geo_risk': float(row['geo_risk']),
                'network_risk': float(row['network_risk']),
                'device_risk': float(row['device_risk']),
                'behavioral_risk': float(row['behavioral_risk']),
                'overall_risk': float(row['overall_risk_score'])
            })
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export', methods=['POST'])
def export_results():
    """Export results to CSV"""
    try:
        data = request.json
        records = data.get('records', [])
        
        if not records:
            return jsonify({'error': 'No records to export'}), 400
        
        df = pd.DataFrame(records)
        csv_filename = f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(csv_filename, index=False)
        
        return jsonify({'filename': csv_filename, 'message': 'Export successful'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)