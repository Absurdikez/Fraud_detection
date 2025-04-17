import pandas as pd
import numpy as np
from scipy import stats

def rename_columns(df):
    """
    Renames columns to make them more readable and intuitive.
    
    Args:
        df (pd.DataFrame): Input DataFrame with original column names
        
    Returns:
        pd.DataFrame: DataFrame with renamed columns
    """
    column_mapping = {
        'step': 'time_step',
        'action': 'transaction_type',
        'amount': 'transaction_amount',
        'nameOrig': 'origin_account',
        'oldBalanceOrig': 'origin_old_balance',
        'newBalanceOrig': 'origin_new_balance',
        'nameDest': 'destination_account',
        'oldBalanceDest': 'destination_old_balance',
        'newBalanceDest': 'destination_new_balance',
        'isFraud': 'is_fraud',
        'isFlaggedFraud': 'is_flagged_fraud'
    }
    return df.rename(columns=column_mapping)

def calculate_success_rates(df):
    """
    Calculates transaction success rates for each origin account.
    
    Args:
        df (pd.DataFrame): Input DataFrame with transaction data
        
    Returns:
        pd.DataFrame: DataFrame with added success_rate column
    """
    group_means = 1 - df.groupby('origin_account')['is_fraud'].mean()
    df['success_rate'] = df['origin_account'].map(group_means)
    
    # Also add success rate for destination account
    dest_group_means = 1 - df.groupby('destination_account')['is_fraud'].mean()
    df['dest_success_rate'] = df['destination_account'].map(dest_group_means)
    
    return df

def calculate_balance_differences(df):
    """
    Calculates balance differences for origin and destination accounts.
    
    Args:
        df (pd.DataFrame): Input DataFrame with balance information
        
    Returns:
        pd.DataFrame: DataFrame with added balance difference columns
    """
    df['balance_diff_origin'] = df['origin_old_balance'] - df['origin_new_balance']
    df['balance_diff_dest'] = df['destination_old_balance'] - df['destination_new_balance']
    
    # Calculate expected balance changes based on transaction amount
    df['expected_origin_balance_diff'] = df['transaction_amount']
    df['expected_dest_balance_diff'] = -df['transaction_amount']
    
    # Check if the actual balance changes match expected changes
    df['origin_balance_mismatch'] = abs(df['balance_diff_origin'] - df['expected_origin_balance_diff']) > 0.01
    df['dest_balance_mismatch'] = abs(df['balance_diff_dest'] - df['expected_dest_balance_diff']) > 0.01
    
    # Calculate transaction amount as percentage of old balance
    df['amount_to_balance_ratio'] = df['transaction_amount'] / df['origin_old_balance'].replace(0, 0.01)
    
    return df

def identify_fraud_patterns(df):
    """
    Identifies various fraud patterns and creates indicator columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with added fraud pattern indicators
    """
    # Get accounts involved in fraudulent transactions
    fraud_origin_accounts = set(df.loc[df['is_fraud'] == 1, 'origin_account'])
    fraud_dest_accounts = set(df.loc[df['is_fraud'] == 1, 'destination_account'])
    
    df['origin_fraud_history'] = df['origin_account'].isin(fraud_origin_accounts)
    df['dest_fraud_history'] = df['destination_account'].isin(fraud_dest_accounts)
    
    # Calculate amount anomalies using Z-score
    amount_mean = df['transaction_amount'].mean()
    amount_std = df['transaction_amount'].std()
    df['amount_Z'] = (df['transaction_amount'] - amount_mean) / amount_std
    df['amount_anomaly'] = (df['amount_Z'] > 3)
    
    # Transaction type + amount anomaly combinations
    for tx_type in df['transaction_type'].unique():
        tx_type_key = tx_type.lower().replace('_', '')
        type_mask = df['transaction_type'] == tx_type
        type_amount_mean = df.loc[type_mask, 'transaction_amount'].mean()
        type_amount_std = df.loc[type_mask, 'transaction_amount'].std()
        
        if type_amount_std > 0:  # Avoid division by zero
            df.loc[type_mask, f'{tx_type_key}_amount_Z'] = (
                (df.loc[type_mask, 'transaction_amount'] - type_amount_mean) / type_amount_std
            )
            df.loc[~type_mask, f'{tx_type_key}_amount_Z'] = 0
            df[f'{tx_type_key}_amount_anomaly'] = df[f'{tx_type_key}_amount_Z'] > 3
    
    return df

def calculate_time_features(df):
    """
    Calculates time-based features for transactions.
    
    Args:
        df (pd.DataFrame): Input DataFrame with time_step column
        
    Returns:
        pd.DataFrame: DataFrame with added time-based features
    """
    # Calculate time since last transaction for each origin account
    df['prev_step'] = df.groupby('origin_account')['time_step'].shift(1).astype('float32')
    df['is_first_tx'] = df['prev_step'].isna()
    
    df['time_since_last_tx'] = (
        df['time_step'].astype('float32') - df['prev_step'].fillna(df['time_step'].astype('float32')))
    
    # Calculate transaction velocity (number of transactions in last N steps)
    for window in [5, 10, 20]:
        df[f'tx_count_last_{window}_steps'] = df.groupby('origin_account')['time_step'].transform(
            lambda x: x.rolling(window=window, min_periods=1).count()
        )
    
    # Add time indicators (assuming steps represent time periods)
    df['time_hour'] = df['time_step'] % 24  # Assuming 24 hours in a day
    df['is_night'] = ((df['time_hour'] >= 22) | (df['time_hour'] <= 5)).astype(int)
    df['is_weekend'] = (df['time_step'] % 7 >= 5).astype(int)  # Assuming 7 days in a week
    
    # Clean up temporary column
    df.drop(columns=['prev_step'], inplace=True)
    return df

def identify_suspicious_patterns(df):
    """
    Identifies suspicious transaction patterns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with added suspicious pattern indicators
    """
    # Identify suspicious accounts based on success rate
    df['suspicious_account'] = (df['success_rate'] <= 0.95).astype(int)
    
    # Identify transfer-cashout patterns
    transfer_mask = df['transaction_type'] == 'TRANSFER'
    cashout_mask = df['transaction_type'] == 'CASH_OUT'
    
    transfer_accounts = df.loc[transfer_mask, 'destination_account'].to_numpy()
    cashout_accounts = df.loc[cashout_mask, 'origin_account'].to_numpy()
    
    suspicious_accounts = np.intersect1d(transfer_accounts, cashout_accounts)
    df['transfer_cashout_pair'] = df['origin_account'].isin(suspicious_accounts)
    
    # Zero balance after transaction
    df['zero_balance_after_tx'] = (df['origin_new_balance'] == 0).astype(int)
    
    # Account emptying pattern (balance reduces to less than 10%)
    df['account_emptying'] = (df['origin_new_balance'] < df['origin_old_balance'] * 0.1).astype(int)
    
    return df

def create_aggregate_features(df):
    """
    Creates statistical aggregations for each account.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with added aggregation features
    """
    # Aggregate features for origin account
    origin_aggs = df.groupby('origin_account').agg(
        origin_tx_count=('transaction_amount', 'count'),
        origin_mean_amount=('transaction_amount', 'mean'),
        origin_max_amount=('transaction_amount', 'max'),
        origin_std_amount=('transaction_amount', 'std')
    ).reset_index()
    
    # Aggregate features for destination account
    dest_aggs = df.groupby('destination_account').agg(
        dest_tx_count=('transaction_amount', 'count'),
        dest_mean_amount=('transaction_amount', 'mean'),
        dest_max_amount=('transaction_amount', 'max'),
        dest_std_amount=('transaction_amount', 'std')
    ).reset_index()
    
    # Merge aggregations back to the original dataframe
    df = df.merge(origin_aggs, on='origin_account', how='left')
    df = df.merge(dest_aggs, on='destination_account', how='left')
    
    # Calculate deviation from mean transaction amount for the account
    df['origin_amount_deviation'] = abs(df['transaction_amount'] - df['origin_mean_amount'])
    df['origin_amount_deviation_ratio'] = df['origin_amount_deviation'] / (df['origin_std_amount'] + 1e-10)
    
    # Fill NaN values
    df = df.fillna(0)
    
    return df

def identify_risky_transactions(df):
    """
    Identifies high-risk transactions based on type and amount.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with added risk indicators
    """
    # Mark high-risk transaction types
    df['high_risk_transaction'] = np.where(
        df['transaction_type'].isin(['TRANSFER', 'CASH_OUT']), 1, 0)
    
    # Mark high-amount transactions
    high_amount_threshold = np.percentile(df['transaction_amount'], 95)
    df['high_transaction'] = df['transaction_amount'].apply(
        lambda x: 1 if x > high_amount_threshold else 0)
    
    # Combine risk factors
    df['multiple_risk_factors'] = (
        (df['high_risk_transaction'] == 1) & 
        (df['high_transaction'] == 1) & 
        (df['amount_anomaly'] == 1)
    ).astype(int)
    
    return df

def add_network_features(df):
    """
    Creates network-based features to capture relationships between accounts.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with added network features
    """
    # Create pairs of accounts that have transacted
    df['account_pair'] = df['origin_account'] + '_' + df['destination_account']
    
    # Count transactions between each pair of accounts
    pair_counts = df.groupby('account_pair')['time_step'].count().reset_index()
    pair_counts.columns = ['account_pair', 'pair_tx_count']
    
    # Merge back to original dataframe
    df = df.merge(pair_counts, on='account_pair', how='left')
    
    # Create features for accounts that frequently transact with many different counterparties
    origin_partner_counts = df.groupby('origin_account')['destination_account'].nunique().reset_index()
    origin_partner_counts.columns = ['origin_account', 'origin_partner_count']
    
    dest_partner_counts = df.groupby('destination_account')['origin_account'].nunique().reset_index()
    dest_partner_counts.columns = ['destination_account', 'dest_partner_count']
    
    df = df.merge(origin_partner_counts, on='origin_account', how='left')
    df = df.merge(dest_partner_counts, on='destination_account', how='left')
    
    return df

def engineer_all_features(df):
    """
    Applies all feature engineering steps in sequence.
    
    Args:
        df (pd.DataFrame): Raw input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with all engineered features
    """
    df = rename_columns(df)
    df = calculate_success_rates(df)
    df = calculate_balance_differences(df)
    df = identify_fraud_patterns(df)
    df = calculate_time_features(df)
    df = identify_suspicious_patterns(df)
    df = create_aggregate_features(df)
    df = identify_risky_transactions(df)
    df = add_network_features(df)
    
    # Create a final fraud risk score combining multiple signals
    risk_features = [
        'amount_anomaly', 'high_risk_transaction', 'high_transaction',
        'suspicious_account', 'transfer_cashout_pair', 'zero_balance_after_tx',
        'account_emptying', 'origin_balance_mismatch', 'dest_balance_mismatch',
        'multiple_risk_factors'
    ]
    
    # Simple weighted sum of risk factors
    df['fraud_risk_score'] = df[risk_features].sum(axis=1) / len(risk_features)
    
    return df