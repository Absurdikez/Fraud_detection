import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import json

from data_utils import prepare_data, apply_smote
from model_utils import train_model_with_timer, save_model, load_model
from evaluation_utils import evaluate_model, find_optimal_threshold
from feature_utils import engineer_all_features

def parse_args():
    parser = argparse.ArgumentParser(description="Fraud Detection System")
    parser.add_argument("--data_path", type=str, default="AA_UseCase_Data.zip", 
                      help="Path to the input data file")
    parser.add_argument("--model_type", type=str, default="xgboost", 
                      choices=["xgboost", "lightgbm", "random_forest"],
                      help="Type of model to train")
    parser.add_argument("--output_dir", type=str, default="./output", 
                      help="Directory to save models and results")
    parser.add_argument("--sample_size", type=int, default=None, 
                      help="Sample size to use for development (None for full dataset)")
    parser.add_argument("--fast_mode", action="store_true", 
                      help="Use reduced parameter grid for faster training")
    return parser.parse_args()

def create_model_and_params(model_type, fast_mode):
    """Create model object and parameter grid based on model type"""
    if model_type == "xgboost":
        model = XGBClassifier(
            tree_method='gpu_hist',
            eval_metric='aucpr',
            device='cuda',
        )
        
        if fast_mode:
            param_grid = {
                'max_depth': [5],
                'n_estimators': [100],
                'learning_rate': [0.1],
                'subsample': [0.8],
                'colsample_bytree': [0.8]
            }
        else:
            param_grid = {
                'max_depth': [3, 5, 7],
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [1, 5, 10]
            }
    
    elif model_type == "lightgbm":
        model = lgbm.LGBMClassifier(
            objective='binary',
            metric='auc',
            device='gpu',
            verbose=-1
        )
        
        if fast_mode:
            param_grid = {
                'num_leaves': [31],
                'n_estimators': [100],
                'learning_rate': [0.1]
            }
        else:
            param_grid = {
                'num_leaves': [31, 63, 127],
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.1, 0.5],
                'min_child_samples': [20, 50, 100]
            }
    
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        
        if fast_mode:
            param_grid = {
                'n_estimators': [100],
                'max_depth': [10]
            }
        else:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.3]
            }
    
    return model, param_grid

def exploratory_data_analysis(df, output_dir):
    """Perform basic EDA and save visualizations"""
    os.makedirs(f"{output_dir}/eda", exist_ok=True)
    
    # Transaction type distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='transaction_type', data=df)
    plt.title('Distribution of Transaction Types')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eda/transaction_types.png")
    plt.close()
    
    # Transaction amount distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df['transaction_amount'], kde=True, log_scale=True)
    plt.title('Transaction Amount Distribution (Log Scale)')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='is_fraud', y='transaction_amount', data=df)
    plt.title('Transaction Amount by Fraud Status')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eda/amount_distribution.png")
    plt.close()
    
    # Fraud by transaction type
    plt.figure(figsize=(10, 6))
    fraud_by_type = pd.crosstab(df['transaction_type'], df['is_fraud'], 
                               normalize='index') * 100
    fraud_by_type.plot(kind='bar', stacked=True)
    plt.title('Fraud Percentage by Transaction Type')
    plt.xlabel('Transaction Type')
    plt.ylabel('Percentage')
    plt.legend(['Legitimate', 'Fraud'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eda/fraud_by_type.png")
    plt.close()
    
    # Correlation matrix of numerical features
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 5:  # Only create if we have enough numeric columns
        plt.figure(figsize=(16, 12))
        corr_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                   linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/eda/correlation_matrix.png")
        plt.close()
    
    print("EDA visualizations saved to", f"{output_dir}/eda/")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("📊 Loading and preprocessing data...")
    # Read data
    df = pd.read_csv(args.data_path)
    
    # Sample data for development if specified
    if args.sample_size:
        # Ensure we get some fraud cases by stratified sampling
        df_fraud = df[df['isFraud'] == 1].sample(min(args.sample_size // 50, df[df['isFraud'] == 1].shape[0]))
        df_normal = df[df['isFraud'] == 0].sample(args.sample_size - df_fraud.shape[0])
        df = pd.concat([df_normal, df_fraud])
        print(f"Using sample of {df.shape[0]} transactions ({df_fraud.shape[0]} fraud cases)")
    
    # Perform feature engineering
    print("🔧 Engineering features...")
    df_processed = engineer_all_features(df)
    
    # Save engineered feature dataset
    df_processed.to_csv(f"{args.output_dir}/processed_features.csv", index=False)
    
    # Perform EDA
    print("📈 Performing exploratory analysis...")
    exploratory_data_analysis(df_processed, args.output_dir)
    
    # Prepare data
    print("🔪 Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = prepare_data(df_processed)
    
    # Save feature names for reference
    with open(f"{args.output_dir}/feature_names.json", 'w') as f:
        json.dump(list(X_train.columns), f)
    
    # Apply SMOTE for balance
    print("⚖️ Applying SMOTE to balance training data...")
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    
    # Create model and parameter grid
    print(f"🤖 Creating {args.model_type} model...")
    model, param_grid = create_model_and_params(args.model_type, args.fast_mode)
    
    # Train model
    print("🏋️‍♀️ Training model with hyperparameter tuning...")
    trained_model = train_model_with_timer(model, param_grid, args.model_type, X_train_res, y_train_res)
     
    # Save model
    model_file = f"{args.output_dir}/{args.model_type}_model.pkl"
    save_model(trained_model, model_file)
    
    # Make predictions
    print("🔮 Making predictions...")
    y_train_pred = trained_model.predict(X_train_res)
    
    if hasattr(trained_model, 'predict_proba'):
        y_test_proba = trained_model.predict_proba(X_test)[:, 1]
        # Find optimal threshold based on cost analysis
        print("💰 Finding optimal classification threshold...")
        opt_threshold = find_optimal_threshold(y_test, y_test_proba)
        y_test_pred = (y_test_proba >= opt_threshold).astype(int)
    else:
        y_test_pred = trained_model.predict(X_test)
    
    # Evaluate model
    print("📊 Evaluating model performance...")
    metrics = evaluate_model(
        y_train_res, y_train_pred,
        y_test, y_test_pred,
        args.model_type,
        transaction_amounts=X_test['transaction_amount'],
        model=trained_model.best_estimator_ if hasattr(trained_model, 'best_estimator_') else trained_model,
        X_test=X_test
    )
    
    # Save metrics
    with open(f"{args.output_dir}/{args.model_type}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\n✅ Project execution complete! Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()