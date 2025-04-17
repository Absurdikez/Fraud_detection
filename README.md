# Fraud Detection Challenge Task Definition

## Background
The detection of fraud is a critical problem across multiple sectors, especially in banking where new technologies have made fraudulent transactions easier to commit. This challenge addresses fraud detection in mobile financial transactions using machine learning techniques applied to some data.

## Challenge Objective
Build a machine learning model capable of predicting whether a mobile money transaction is fraudulent or not, based on transaction characteristics and patterns.

## Dataset Description

### Training Data
The training dataset contains a subset of transactions from the PaySim simulation, including the target variable `isFraud` (0/1). This dataset should be used to build your prediction model.

### Data Structure
Each record represents a single transaction with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| step | int | Time unit (1 step = 1 hour) |
| type | text | Transaction type: CASH-IN, CASH-OUT, DEBIT, PAYMENT, or TRANSFER |
| amount | double | Transaction amount in local currency |
| nameOrig | text | Customer who initiated the transaction |
| oldBalanceOrig | double | Initial balance before the transaction |
| newBalanceOrig | double | New balance after the transaction |
| nameDest | text | Recipient of the transaction |
| oldbalanceDest | double | Initial recipient balance before transaction (not available for Merchants) |
| newbalanceDest | double | New recipient balance after transaction (not available for Merchants) |
| isFraud | int | Target variable: fraudulent (1) or legitimate (0) transaction |

### Transaction Types Explained
- **CASH-IN**: Increasing account balance by paying cash to a Merchant
- **CASH-OUT**: Withdrawing cash from a merchant, decreasing account balance
- **DEBIT**: Sending money from mobile money service to a bank account
- **PAYMENT**: Paying for goods/services to merchants
- **TRANSFER**: Sending money to another user through the mobile money platform

## Evaluation Criteria
Models will be evaluated on their ability to correctly identify fraudulent transactions while minimizing false positives. Given the class imbalance typical in fraud detection, appropriate performance metrics will be used beyond simple accuracy.

## Challenge Tasks

1. **Exploratory Data Analysis**
   - Analyze the distribution of transaction types
   - Examine patterns in fraudulent vs. legitimate transactions
   - Identify relationships between features and fraud likelihood

2. **Feature Engineering**
   - Create relevant features from the transaction data
   - Implement features from categories described in the documentation:
     - Account reliability features
     - Balance discrepancy features
     - Amount anomaly features
     - Transaction history features
     - Network analysis features
     - And other feature categories as appropriate

3. **Model Development**
   - Build and train machine learning models for fraud detection
   - Address class imbalance issues
   - Perform hyperparameter tuning

4. **Model Evaluation**
   - Evaluate model performance using appropriate metrics
   - Compare different modeling approaches
   - Identify most important features for fraud detection

5. **Implementation Recommendations**
   - Provide recommendations for implementing the model in a production environment
   - Suggest strategies for model monitoring and updating

## Deliverables
1. A trained machine learning model for fraud detection
2. Documentation of the approach and methodology
3. Analysis of model performance
4. Code and implementation details

## Notes on Special Considerations
- The dataset exhibits significant class imbalance, with fraudulent transactions being rare
- Temporal patterns may be important for detection
- Account and transaction relationships provide valuable signals
- Different transaction types have different baseline fraud rates

# Execution
```
python3 main.py \
  --data_path "/path/to/your/Data.zip" \
  --model_type "xgboost" \ 
  --output_dir "/path/to/output/folder" \
  --fast_mode
```

# **NOTE**
- For more detailed documentation, please visit explore.ipynb

