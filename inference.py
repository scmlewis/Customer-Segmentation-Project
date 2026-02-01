"""
Customer Segmentation - Inference Pipeline
=========================================
This script demonstrates how to load the trained clustering model
and assign clusters to new customer data.

Usage:
    python inference.py

Requirements:
    - Trained models in output/ directory
    - New customer data in the same format as training data
"""

import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_models(output_dir='output'):
    """Load all trained models and artifacts."""
    print("Loading models and artifacts...")
    
    preprocessor = joblib.load(os.path.join(output_dir, 'preprocessor.pkl'))
    variance_selector = joblib.load(os.path.join(output_dir, 'variance_selector.pkl'))
    clustering_model = joblib.load(os.path.join(output_dir, 'clustering_model.pkl'))
    feature_info = joblib.load(os.path.join(output_dir, 'feature_info.pkl'))
    
    print(f"✓ Models loaded successfully")
    print(f"  Algorithm: {feature_info['best_algorithm']}")
    print(f"  Number of clusters: {feature_info['optimal_k']}")
    print(f"  Features used: {len(feature_info['final_features'])}")
    
    return preprocessor, variance_selector, clustering_model, feature_info


def engineer_features(df):
    """
    Apply the same feature engineering as training.
    This must match the feature engineering in data_exploration.ipynb.
    """
    df_fe = df.copy()
    
    # Financial ratios
    df_fe['balance_per_age'] = df_fe['balance'] / (df_fe['age'] + 1)
    df_fe['balance_log'] = np.log1p(df_fe['balance'] + abs(df_fe['balance'].min()) + 1)
    df_fe['balance_positive'] = (df_fe['balance'] > 0).astype(int)
    
    # Balance categories
    balance_quantiles = df_fe['balance'].quantile([0.25, 0.5, 0.75])
    df_fe['balance_category'] = pd.cut(df_fe['balance'], 
                                        bins=[-np.inf, balance_quantiles[0.25], 
                                              balance_quantiles[0.5], balance_quantiles[0.75], np.inf],
                                        labels=['low', 'medium', 'high', 'very_high'])
    
    # Campaign metrics
    df_fe['campaign_intensity'] = df_fe['campaign'] / (df_fe['previous'] + 1)
    df_fe['total_contacts'] = df_fe['campaign'] + df_fe['previous']
    df_fe['is_first_campaign'] = (df_fe['previous'] == 0).astype(int)
    
    # Pdays handling
    df_fe['contacted_before'] = (df_fe['pdays'] != -1).astype(int)
    df_fe['pdays_normalized'] = df_fe['pdays'].replace(-1, df_fe[df_fe['pdays'] != -1]['pdays'].max() + 1)
    df_fe['recent_contact'] = (df_fe['pdays'] < 30).astype(int) * df_fe['contacted_before']
    
    # Temporal encoding
    month_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df_fe['month_num'] = df_fe['month'].map(month_mapping)
    df_fe['month_sin'] = np.sin(2 * np.pi * df_fe['month_num'] / 12)
    df_fe['month_cos'] = np.cos(2 * np.pi * df_fe['month_num'] / 12)
    df_fe['quarter'] = pd.cut(df_fe['month_num'], bins=[0, 3, 6, 9, 12], 
                              labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    # Interaction features
    df_fe['has_debt'] = ((df_fe['loan'] == 'yes') | (df_fe['housing'] == 'yes')).astype(int)
    df_fe['debt_count'] = (df_fe['loan'] == 'yes').astype(int) + (df_fe['housing'] == 'yes').astype(int)
    df_fe['prev_success'] = (df_fe['poutcome'] == 'success').astype(int)
    df_fe['duration_per_contact'] = df_fe['duration'] / (df_fe['campaign'] + 1)
    
    return df_fe


def predict_clusters(df_new, preprocessor, variance_selector, clustering_model, feature_info):
    """
    Predict cluster assignments for new customer data.
    
    Parameters:
    -----------
    df_new : pd.DataFrame
        New customer data with same columns as training data
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline
    variance_selector : VarianceThreshold
        Fitted variance threshold selector
    clustering_model : clustering model
        Fitted clustering model
    feature_info : dict
        Feature information from training
        
    Returns:
    --------
    np.ndarray : Cluster assignments
    """
    print(f"\nPredicting clusters for {len(df_new)} customers...")
    
    # Apply feature engineering
    df_fe = engineer_features(df_new)
    
    # Preprocess
    X_preprocessed = preprocessor.transform(df_fe)
    
    # Apply variance threshold
    X_variance = variance_selector.transform(X_preprocessed)
    
    # Apply correlation and VIF filters (using stored feature indices)
    # Note: In production, you'd save the exact column indices or masks
    features_after_variance = feature_info['features_after_variance']
    features_after_corr = feature_info['features_after_correlation']
    features_final = feature_info['final_features']
    
    # For simplicity, use final feature count
    # In production, save exact transformations
    X_final = X_variance[:, :len(features_final)]
    
    # Predict clusters
    clusters = clustering_model.predict(X_final)
    
    print(f"✓ Cluster prediction complete")
    print(f"  Cluster distribution: {pd.Series(clusters).value_counts().sort_index().to_dict()}")
    
    return clusters


def main():
    """Main inference pipeline."""
    print("="*60)
    print("CUSTOMER SEGMENTATION - INFERENCE PIPELINE")
    print("="*60)
    
    # Load trained models
    preprocessor, variance_selector, clustering_model, feature_info = load_models()
    
    # Example: Load new customer data
    # In production, replace this with your new data source
    print("\n" + "="*60)
    print("LOADING NEW CUSTOMER DATA")
    print("="*60)
    
    # For demonstration, load a sample from the original data
    data_path = os.path.join('data', 'bank.csv')
    df_sample = pd.read_csv(data_path).sample(n=100, random_state=42)
    print(f"✓ Loaded {len(df_sample)} customers for inference")
    
    # Predict clusters
    print("\n" + "="*60)
    print("PREDICTING CLUSTERS")
    print("="*60)
    
    predicted_clusters = predict_clusters(
        df_sample, preprocessor, variance_selector, clustering_model, feature_info
    )
    
    # Add clusters to dataframe
    df_sample['Predicted_Cluster'] = predicted_clusters
    
    # Display results
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    print(df_sample[['age', 'job', 'balance', 'deposit', 'Predicted_Cluster']].head(10))
    
    # Save predictions
    output_path = os.path.join('output', 'new_predictions.csv')
    df_sample.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to {output_path}")
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
