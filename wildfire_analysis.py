#!/usr/bin/env python3
"""
Wildfire Prediction Analysis
Intern Hiring Assessment - Adgama Digital Private Limited

This script implements a complete data science workflow for wildfire prediction
using machine learning models including custom neural networks and ensemble methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, accuracy_score, 
                           precision_score, recall_score, f1_score)

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Utilities
import joblib
import json
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class WildfirePredictor:
    """
    Complete wildfire prediction analysis pipeline
    """
    
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.results = {}
        
    def create_synthetic_dataset(self, n_samples=10000):
        """Create synthetic wildfire dataset"""
        print("Creating synthetic wildfire dataset...")
        
        # Generate synthetic features
        data = {
            'temperature': np.random.normal(25, 10, n_samples),
            'humidity': np.random.uniform(10, 90, n_samples),
            'wind_speed': np.random.exponential(15, n_samples),
            'precipitation': np.random.exponential(2, n_samples),
            'drought_index': np.random.uniform(0, 100, n_samples),
            'vegetation_density': np.random.uniform(0, 1, n_samples),
            'elevation': np.random.uniform(0, 3000, n_samples),
            'slope': np.random.uniform(0, 45, n_samples),
            'distance_to_road': np.random.exponential(5, n_samples),
            'population_density': np.random.exponential(100, n_samples)
        }
        
        self.df = pd.DataFrame(data)
        
        # Create realistic target variable
        fire_probability = (
            (self.df['temperature'] - self.df['temperature'].min()) / 
            (self.df['temperature'].max() - self.df['temperature'].min()) * 0.3 +
            (1 - (self.df['humidity'] - self.df['humidity'].min()) / 
             (self.df['humidity'].max() - self.df['humidity'].min())) * 0.25 +
            (self.df['wind_speed'] - self.df['wind_speed'].min()) / 
            (self.df['wind_speed'].max() - self.df['wind_speed'].min()) * 0.2 +
            (1 - (self.df['precipitation'] - self.df['precipitation'].min()) / 
             (self.df['precipitation'].max() - self.df['precipitation'].min())) * 0.15 +
            (self.df['drought_index'] - self.df['drought_index'].min()) / 
            (self.df['drought_index'].max() - self.df['drought_index'].min()) * 0.1
        )
        
        fire_probability += np.random.normal(0, 0.1, n_samples)
        self.df['fire_occurrence'] = (fire_probability > np.percentile(fire_probability, 75)).astype(int)
        
        # Introduce missing values
        missing_indices = np.random.choice(self.df.index, size=int(0.05 * len(self.df)), replace=False)
        missing_columns = np.random.choice(self.df.columns[:-1], size=len(missing_indices))
        for idx, col in zip(missing_indices, missing_columns):
            self.df.loc[idx, col] = np.nan
            
        print(f"Dataset created: {len(self.df)} samples, {len(self.df.columns)} features")
        print(f"Fire occurrence rate: {self.df['fire_occurrence'].mean():.2%}")
        
    def preprocess_data(self):
        """Complete data preprocessing pipeline"""
        print("Preprocessing data...")
        
        # Handle missing values
        features = self.df.columns[:-1]
        imputer = SimpleImputer(strategy='median')
        self.df[features] = imputer.fit_transform(self.df[features])
        
        # Handle outliers using IQR capping
        for column in features:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df[column] = self.df[column].clip(lower=lower_bound, upper=upper_bound)
        
        # Feature engineering
        self.df['fire_weather_index'] = (
            self.df['temperature'] * 0.3 + 
            (100 - self.df['humidity']) * 0.3 + 
            self.df['wind_speed'] * 0.2 + 
            self.df['drought_index'] * 0.2
        )
        
        self.df['temp_humidity_ratio'] = self.df['temperature'] / (self.df['humidity'] + 1)
        self.df['wind_precip_ratio'] = self.df['wind_speed'] / (self.df['precipitation'] + 1)
        self.df['terrain_risk'] = self.df['slope'] * self.df['vegetation_density']
        
        # Prepare features and target
        X = self.df.drop('fire_occurrence', axis=1)
        y = self.df['fire_occurrence']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature scaling
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Preprocessing completed successfully")
        
    def create_custom_nn(self, input_dim):
        """Create custom neural network"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_models(self):
        """Train all models"""
        print("Training models...")
        
        # Custom Neural Network
        print("Training Custom Neural Network...")
        custom_model = self.create_custom_nn(self.X_train_scaled.shape[1])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        
        history_custom = custom_model.fit(
            self.X_train_scaled, self.y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        self.models['custom_nn'] = custom_model
        
        # Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        rf_model.fit(self.X_train_scaled, self.y_train)
        self.models['random_forest'] = rf_model
        
        # Ensemble Model
        print("Training Ensemble Model...")
        rf_train_pred = rf_model.predict_proba(self.X_train_scaled)[:, 1].reshape(-1, 1)
        rf_test_pred = rf_model.predict_proba(self.X_test_scaled)[:, 1].reshape(-1, 1)
        
        X_train_ensemble = np.hstack([self.X_train_scaled, rf_train_pred])
        X_test_ensemble = np.hstack([self.X_test_scaled, rf_test_pred])
        
        ensemble_model = Sequential([
            Dense(96, activation='relu', input_shape=(X_train_ensemble.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(48, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(24, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        ensemble_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        ensemble_model.fit(
            X_train_ensemble, self.y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        self.models['ensemble'] = ensemble_model
        self.X_test_ensemble = X_test_ensemble
        
        print("All models trained successfully")
        
    def evaluate_models(self):
        """Evaluate all models"""
        print("Evaluating models...")
        
        # Generate predictions
        y_pred_custom = (self.models['custom_nn'].predict(self.X_test_scaled) > 0.5).astype(int).flatten()
        y_pred_custom_proba = self.models['custom_nn'].predict(self.X_test_scaled).flatten()
        
        y_pred_rf = self.models['random_forest'].predict(self.X_test_scaled)
        y_pred_rf_proba = self.models['random_forest'].predict_proba(self.X_test_scaled)[:, 1]
        
        y_pred_ensemble = (self.models['ensemble'].predict(self.X_test_ensemble) > 0.5).astype(int).flatten()
        y_pred_ensemble_proba = self.models['ensemble'].predict(self.X_test_ensemble).flatten()
        
        # Calculate metrics
        def calculate_metrics(y_true, y_pred, y_pred_proba):
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred),
                'roc_auc': roc_auc_score(y_true, y_pred_proba)
            }
        
        self.results = {
            'Custom NN': calculate_metrics(self.y_test, y_pred_custom, y_pred_custom_proba),
            'Random Forest': calculate_metrics(self.y_test, y_pred_rf, y_pred_rf_proba),
            'Ensemble': calculate_metrics(self.y_test, y_pred_ensemble, y_pred_ensemble_proba)
        }
        
        # Display results
        results_df = pd.DataFrame(self.results)
        print("\nModel Performance Comparison:")
        print(results_df.round(4))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.models['random_forest'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return results_df, feature_importance
    
    def save_models(self):
        """Save trained models"""
        print("Saving models...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save models
        self.models['custom_nn'].save('models/custom_wildfire_model.h5')
        self.models['ensemble'].save('models/ensemble_wildfire_model.h5')
        joblib.dump(self.models['random_forest'], 'models/random_forest_model.pkl')
        joblib.dump(self.scaler, 'models/feature_scaler.pkl')
        
        # Save results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('models/model_comparison_results.csv')
        
        print("Models saved successfully")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("=== WILDFIRE PREDICTION ANALYSIS ===")
        print("Adgama Digital Private Limited - Intern Assessment")
        print("=" * 50)
        
        self.create_synthetic_dataset()
        self.preprocess_data()
        self.train_models()
        results_df, feature_importance = self.evaluate_models()
        self.save_models()
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("✅ Dataset created and preprocessed")
        print("✅ Three models trained and evaluated")
        print("✅ Results saved to models/ directory")
        print("✅ Ready for submission")
        
        # Best model summary
        best_model = results_df.loc['f1_score'].idxmax()
        best_f1 = results_df.loc['f1_score'].max()
        print(f"\n🏆 Best Model: {best_model} (F1-Score: {best_f1:.3f})")
        
        return results_df, feature_importance

def main():
    """Main execution function"""
    predictor = WildfirePredictor()
    results_df, feature_importance = predictor.run_complete_analysis()
    
    print("\n" + "="*50)
    print("Analysis completed successfully!")
    print("Check the models/ directory for saved artifacts.")
    print("="*50)

if __name__ == "__main__":
    main()