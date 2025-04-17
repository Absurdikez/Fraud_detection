import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

def prepare_data(df, target_col='is_fraud', test_size=0.2, random_state=37):
    """
    Prepares data for modeling by splitting and encoding.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_col (str): Name of target column
        test_size (float): Proportion of test set
        random_state (int): Random seed
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    df.drop(columns=['origin_account', 'destination_account'], inplace=True, errors='ignore')

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, sampling_strategy=0.05, random_state=37):
    """
    Applies SMOTE for handling imbalanced data.
    
    Args:
        X_train: Training features
        y_train: Training target
        sampling_strategy (float): Desired ratio of minority class
        random_state (int): Random seed
    
    Returns:
        tuple: Resampled X_train, y_train
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled