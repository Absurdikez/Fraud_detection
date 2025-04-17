import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, precision_recall_curve, 
    average_precision_score, roc_auc_score
)

def evaluate_model(y_train, y_train_pred, y_test, y_test_pred, model_name, 
                  transaction_amounts=None, model=None, X_test=None, proba=True):
    """
    Comprehensive model evaluation with metrics and analysis.
    
    Args:
        y_train: Training target values
        y_train_pred: Training predictions
        y_test: Test target values
        y_test_pred: Test predictions
        model_name: Name of the model
        transaction_amounts: Transaction amounts for fraud impact analysis
        model: Trained model object (for feature importance)
        X_test: Test features (for feature importance)
        proba: Whether to calculate probability-based metrics
    """
    # Basic metrics calculation
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # Get probability predictions if available
    if proba and hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_test)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate probability-based metrics
        prob_metrics = calculate_probability_metrics(y_test, y_test_proba)
        test_metrics.update(prob_metrics)
    
    # Print results
    print_evaluation_results(
        train_metrics, 
        test_metrics, 
        model_name, 
        y_test, 
        y_test_pred, 
        transaction_amounts
    )
    
    # Plot curves
    if proba and hasattr(model, 'predict_proba'):
        plot_evaluation_curves(y_test, y_test_proba, model_name)
    
    # Plot feature importance if available
    if model is not None and X_test is not None and hasattr(model, 'feature_importances_'):
        plot_feature_importance(model, X_test.columns, model_name)
    
    # Analyze misclassifications if transaction amounts are provided
    if transaction_amounts is not None:
        analyze_misclassifications(y_test, y_test_pred, transaction_amounts)
    
    return test_metrics

def calculate_metrics(y_true, y_pred):
    """
    Calculates basic classification metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        dict: Dictionary of metrics
    """
    # Handle the case where there are no positive samples in y_true
    if sum(y_true) == 0:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

def calculate_probability_metrics(y_true, y_proba):
    """
    Calculates probability-based metrics.
    
    Args:
        y_true: True target values
        y_proba: Predicted probabilities for the positive class
        
    Returns:
        dict: Dictionary of metrics
    """
    # Handle the case where there are no positive samples in y_true
    if sum(y_true) == 0:
        return {
            'roc_auc': 0.5,
            'pr_auc': 0.0
        }
    
    return {
        'roc_auc': roc_auc_score(y_true, y_proba),
        'pr_auc': average_precision_score(y_true, y_proba)
    }

def print_evaluation_results(train_metrics, test_metrics, model_name, y_test, y_test_pred, transaction_amounts=None):
    """
    Prints detailed evaluation results.
    
    Args:
        train_metrics: Dictionary of training metrics
        test_metrics: Dictionary of test metrics
        model_name: Name of the model
        y_test: Test target values
        y_test_pred: Test predictions
        transaction_amounts: Transaction amounts for fraud impact analysis
    """
    print(f"\n{'='*50}")
    print(f"Model Evaluation: {model_name}")
    print(f"{'='*50}")
    
    print("\nTraining Metrics:")
    print(f"Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Precision: {train_metrics['precision']:.4f}")
    print(f"Recall:    {train_metrics['recall']:.4f}")
    print(f"F1 Score:  {train_metrics['f1']:.4f}")
    
    print("\nTest Metrics:")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    
    if 'roc_auc' in test_metrics:
        print(f"ROC AUC:   {test_metrics['roc_auc']:.4f}")
        print(f"PR AUC:    {test_metrics['pr_auc']:.4f}")
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Negative  Positive")
    print(f"Actual Negative   {cm[0, 0]:<8}  {cm[0, 1]:<8}")
    print(f"Actual Positive   {cm[1, 0]:<8}  {cm[1, 1]:<8}")
    
    # Calculate financial impact if transaction amounts are provided
    if transaction_amounts is not None:
        # Convert y_test and transaction_amounts to numpy arrays
        y_test_array = np.array(y_test)
        amounts_array = np.array(transaction_amounts)
        
        # Calculate fraud amounts
        actual_fraud_amount = np.sum(amounts_array[y_test_array == 1])
        detected_fraud_amount = np.sum(amounts_array[(y_test_array == 1) & (y_test_pred == 1)])
        missed_fraud_amount = np.sum(amounts_array[(y_test_array == 1) & (y_test_pred == 0)])
        false_positive_amount = np.sum(amounts_array[(y_test_array == 0) & (y_test_pred == 1)])
        
        print("\nFinancial Impact Analysis:")
        print(f"Total Fraud Amount:        ${actual_fraud_amount:.2f}")
        print(f"Detected Fraud Amount:     ${detected_fraud_amount:.2f} ({detected_fraud_amount / max(actual_fraud_amount, 1) * 100:.2f}%)")
        print(f"Missed Fraud Amount:       ${missed_fraud_amount:.2f} ({missed_fraud_amount / max(actual_fraud_amount, 1) * 100:.2f}%)")
        print(f"False Positive Amount:     ${false_positive_amount:.2f}")
        
        # Calculate estimated savings
        investigation_cost = 50  # Estimated cost per investigation
        fraud_recovery_rate = 0.7  # Estimated recovery rate for detected fraud
        
        true_positives = cm[1, 1]
        false_positives = cm[0, 1]
        
        investigation_costs = (true_positives + false_positives) * investigation_cost
        recovered_amount = detected_fraud_amount * fraud_recovery_rate
        net_benefit = recovered_amount - investigation_costs
        
        print(f"\nEstimated Impact:")
        print(f"Investigation Costs:       ${investigation_costs:.2f}")
        print(f"Potential Recovery:        ${recovered_amount:.2f}")
        print(f"Net Benefit:               ${net_benefit:.2f}")

def plot_evaluation_curves(y_true, y_proba, model_name):
    """
    Plots ROC and Precision-Recall curves.
    
    Args:
        y_true: True target values
        y_proba: Predicted probabilities for the positive class
        model_name: Name of the model
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    ax1.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve - {model_name}')
    ax1.legend(loc="lower right")
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    ax2.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'Precision-Recall Curve - {model_name}')
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_curves.png")
    plt.close()
    
    print(f"\nEvaluation curves saved as '{model_name.lower().replace(' ', '_')}_curves.png'")

def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """
    Plots feature importance for tree-based models.
    
    Args:
        model: Trained model object
        feature_names: Names of features
        model_name: Name of the model
        top_n: Number of top features to show
    """
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top N features
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_names)), top_importances, align='center')
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_feature_importance.png")
    plt.close()
    
    print(f"\nFeature importance plot saved as '{model_name.lower().replace(' ', '_')}_feature_importance.png'")

def analyze_misclassifications(y_true, y_pred, transaction_amounts):
    """
    Analyzes misclassified transactions.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        transaction_amounts: Transaction amounts
    """
    # Convert inputs to numpy arrays for easier slicing
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    transaction_amounts = np.array(transaction_amounts)
    
    # Identify different types of predictions
    true_positives = (y_true == 1) & (y_pred == 1)
    false_positives = (y_true == 0) & (y_pred == 1)
    true_negatives = (y_true == 0) & (y_pred == 0)
    false_negatives = (y_true == 1) & (y_pred == 0)
    
    # Calculate statistics for each group
    stats = {
        'True Positives': {
            'count': np.sum(true_positives),
            'amount_mean': np.mean(transaction_amounts[true_positives]) if np.sum(true_positives) > 0 else 0,
            'amount_std': np.std(transaction_amounts[true_positives]) if np.sum(true_positives) > 0 else 0,
            'amount_min': np.min(transaction_amounts[true_positives]) if np.sum(true_positives) > 0 else 0,
            'amount_max': np.max(transaction_amounts[true_positives]) if np.sum(true_positives) > 0 else 0,
        },
        'False Positives': {
            'count': np.sum(false_positives),
            'amount_mean': np.mean(transaction_amounts[false_positives]) if np.sum(false_positives) > 0 else 0,
            'amount_std': np.std(transaction_amounts[false_positives]) if np.sum(false_positives) > 0 else 0,
            'amount_min': np.min(transaction_amounts[false_positives]) if np.sum(false_positives) > 0 else 0,
            'amount_max': np.max(transaction_amounts[false_positives]) if np.sum(false_positives) > 0 else 0,
        },
        'True Negatives': {
            'count': np.sum(true_negatives),
            'amount_mean': np.mean(transaction_amounts[true_negatives]) if np.sum(true_negatives) > 0 else 0,
            'amount_std': np.std(transaction_amounts[true_negatives]) if np.sum(true_negatives) > 0 else 0,
            'amount_min': np.min(transaction_amounts[true_negatives]) if np.sum(true_negatives) > 0 else 0,
            'amount_max': np.max(transaction_amounts[true_negatives]) if np.sum(true_negatives) > 0 else 0,
        },
        'False Negatives': {
            'count': np.sum(false_negatives),
            'amount_mean': np.mean(transaction_amounts[false_negatives]) if np.sum(false_negatives) > 0 else 0,
            'amount_std': np.std(transaction_amounts[false_negatives]) if np.sum(false_negatives) > 0 else 0,
            'amount_min': np.min(transaction_amounts[false_negatives]) if np.sum(false_negatives) > 0 else 0,
            'amount_max': np.max(transaction_amounts[false_negatives]) if np.sum(false_negatives) > 0 else 0,
        }
    }
    
    print("\nAnalysis of Classification Results:")
    for group, group_stats in stats.items():
        print(f"\n{group}:")
        print(f"  Count:       {group_stats['count']}")
        print(f"  Mean Amount: ${group_stats['amount_mean']:.2f}")
        print(f"  Std Amount:  ${group_stats['amount_std']:.2f}")
        print(f"  Min Amount:  ${group_stats['amount_min']:.2f}")
        print(f"  Max Amount:  ${group_stats['amount_max']:.2f}")
    
    # Visualize misclassification amounts
    plt.figure(figsize=(14, 8))
    
    # Create amount bins for plotting
    amount_bins = [0, 100, 1000, 10000, 100000, float('inf')]
    bin_labels = ['0-100', '100-1K', '1K-10K', '10K-100K', '100K+']
    
    # Bin amounts for each prediction type
    tp_bins = np.histogram(transaction_amounts[true_positives], bins=amount_bins)[0]
    fp_bins = np.histogram(transaction_amounts[false_positives], bins=amount_bins)[0]
    fn_bins = np.histogram(transaction_amounts[false_negatives], bins=amount_bins)[0]
    
    # Create a bar chart
    x = np.arange(len(bin_labels))
    width = 0.25
    
    plt.bar(x - width, tp_bins, width, label='True Positives')
    plt.bar(x, fp_bins, width, label='False Positives')
    plt.bar(x + width, fn_bins, width, label='False Negatives')
    
    plt.xlabel('Transaction Amount')
    plt.ylabel('Count')
    plt.title('Distribution of Transaction Amounts by Prediction Type')
    plt.xticks(x, bin_labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig('misclassification_analysis.png')
    plt.close()
    
    print("\nMisclassification analysis plot saved as 'misclassification_analysis.png'")

def find_optimal_threshold(y_true, y_proba, cost_fn=1000, cost_fp=50):
    """
    Finds the optimal threshold for classification based on cost considerations.
    
    Args:
        y_true: True target values
        y_proba: Predicted probabilities
        cost_fn: Cost of a false negative (missed fraud)
        cost_fp: Cost of a false positive (false alarm)
        
    Returns:
        float: Optimal threshold value
    """
    # Initialize variables
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    
    # Calculate cost for each threshold
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        # Calculate confusion matrix values
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        
        # Calculate total cost
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        costs.append(total_cost)
    
    # Find threshold with minimum cost
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    
    # Plot cost versus threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs)
    plt.axvline(x=optimal_threshold, color='r', linestyle='--')
    plt.text(optimal_threshold+0.02, np.min(costs), f'Threshold = {optimal_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Total Cost')
    plt.title('Cost vs. Threshold')
    plt.grid(True)
    plt.savefig('optimal_threshold.png')
    plt.close()
    
    print(f"\nOptimal threshold analysis saved as 'optimal_threshold.png'")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    return optimal_threshold

def analyze_misclassifications(y_true, y_pred, transaction_amounts):
    """
    Analyzes misclassified transactions.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        transaction_amounts: Transaction amounts
    """
    # Convert inputs to numpy arrays for easier slicing
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    transaction_amounts = np.array(transaction_amounts)
    
    # Identify different types of predictions
    true_positives = (y_true == 1) & (y_pred == 1)
    false_positives = (y_true == 0) & (y_pred == 1)
    true_negatives = (y_true == 0) & (y_pred == 0)
    false_negatives = (y_true == 1) & (y_pred == 0)
    
    # Calculate statistics for each group
    stats = {
        'True Positives': {
            'count': np.sum(true_positives),
            'amount_mean': np.mean(transaction_amounts[true_positives]) if np.sum(true_positives) > 0 else 0,
            'amount_std': np.std(transaction_amounts[true_positives]) if np.sum(true_positives) > 0 else 0,
            'amount_min': np.min(transaction_amounts[true_positives]) if np.sum(true_positives) > 0 else 0,
            'amount_max': np.max(transaction_amounts[true_positives]) if np.sum(true_positives) > 0 else 0,
        },
        'False Positives': {
            'count': np.sum(false_positives),
            'amount_mean': np.mean(transaction_amounts[false_positives]) if np.sum(false_positives) > 0 else 0,
            'amount_std': np.std(transaction_amounts[false_positives]) if np.sum(false_positives) > 0 else 0,
            'amount_min': np.min(transaction_amounts[false_positives]) if np.sum(false_positives) > 0 else 0,
            'amount_max': np.max(transaction_amounts[false_positives]) if np.sum(false_positives) > 0 else 0,
        },
        'True Negatives': {
            'count': np.sum(true_negatives),
            'amount_mean': np.mean(transaction_amounts[true_negatives]) if np.sum(true_negatives) > 0 else 0,
            'amount_std': np.std(transaction_amounts[true_negatives]) if np.sum(true_negatives) > 0 else 0,
            'amount_min': np.min(transaction_amounts[true_negatives]) if np.sum(true_negatives) > 0 else 0,
            'amount_max': np.max(transaction_amounts[true_negatives]) if np.sum(true_negatives) > 0 else 0,
        },
        'False Negatives': {
            'count': np.sum(false_negatives),
            'amount_mean': np.mean(transaction_amounts[false_negatives]) if np.sum(false_negatives) > 0 else 0,
            'amount_std': np.std(transaction_amounts[false_negatives]) if np.sum(false_negatives) > 0 else 0,
            'amount_min': np.min(transaction_amounts[false_negatives]) if np.sum(false_negatives) > 0 else 0,
            'amount_max': np.max(transaction_amounts[false_negatives]) if np.sum(false_negatives) > 0 else 0,
        }
    }
    
    print("\nAnalysis of Classification Results:")
    for group, group_stats in stats.items():
        print(f"\n{group}:")
        print(f"  Count:       {group_stats['count']}")
        print(f"  Mean Amount: ${group_stats['amount_mean']:.2f}")
        print(f"  Std Amount:  ${group_stats['amount_std']:.2f}")
        print(f"  Min Amount:  ${group_stats['amount_min']:.2f}")
        print(f"  Max Amount:  ${group_stats['amount_max']:.2f}")
    
    # Visualize misclassification amounts
    plt.figure(figsize=(14, 8))
    
    # Create amount bins for plotting
    amount_bins = [0, 100, 1000, 10000, 100000, float('inf')]
    bin_labels = ['0-100', '100-1K', '1K-10K', '10K-100K', '100K+']
    
    # Bin amounts for each prediction type
    tp_bins = np.histogram(transaction_amounts[true_positives], bins=amount_bins)[0]
    fp_bins = np.histogram(transaction_amounts[false_positives], bins=amount_bins)[0]
    fn_bins = np.histogram(transaction_amounts[false_negatives], bins=amount_bins)[0]
    
    # Create a bar chart
    x = np.arange(len(bin_labels))
    width = 0.25
    
    plt.bar(x - width, tp_bins, width, label='True Positives')
    plt.bar(x, fp_bins, width, label='False Positives')
    plt.bar(x + width, fn_bins, width, label='False Negatives')
    
    plt.xlabel('Transaction Amount')
    plt.ylabel('Count')
    plt.title('Distribution of Transaction Amounts by Prediction Type')
    plt.xticks(x, bin_labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig('misclassification_analysis.png')
    plt.close()
    
    print("\nMisclassification analysis plot saved as 'misclassification_analysis.png'")
    
    # Analyze characteristics of misclassified transactions
    if np.sum(false_negatives) > 0:
        print("\nCharacteristics of Missed Fraud (False Negatives):")
        print(f"  - Total missed fraud amount: ${np.sum(transaction_amounts[false_negatives]):.2f}")
        print(f"  - Percentage of total fraud amount: {np.sum(transaction_amounts[false_negatives]) / (np.sum(transaction_amounts[y_true == 1]) + 1e-10) * 100:.2f}%")
        
        # Calculate recovery impact
        potential_recovery = np.sum(transaction_amounts[false_negatives]) * 0.7  # Assuming 70% recovery rate
        print(f"  - Potential lost recovery: ${potential_recovery:.2f}")
    
    if np.sum(false_positives) > 0:
        print("\nCharacteristics of False Alerts (False Positives):")
        print(f"  - Total false alert amount: ${np.sum(transaction_amounts[false_positives]):.2f}")
        print(f"  - Average transaction size: ${np.mean(transaction_amounts[false_positives]):.2f}")
        
        # Calculate operational cost
        investigation_cost_per_alert = 50  # Estimated cost per investigation
        total_investigation_cost = np.sum(false_positives) * investigation_cost_per_alert
        print(f"  - Estimated investigation cost: ${total_investigation_cost:.2f}")
        print(f"  - Investigation cost as percentage of transaction volume: {total_investigation_cost / (np.sum(transaction_amounts) + 1e-10) * 100:.4f}%")
    
    # Calculate financial impact metrics
    fraud_detected_amount = np.sum(transaction_amounts[true_positives])
    fraud_missed_amount = np.sum(transaction_amounts[false_negatives])
    false_alarm_amount = np.sum(transaction_amounts[false_positives])
    
    detection_rate_by_amount = fraud_detected_amount / (fraud_detected_amount + fraud_missed_amount + 1e-10)
    false_alarm_rate_by_amount = false_alarm_amount / (np.sum(transaction_amounts[y_true == 0]) + 1e-10)
    
    print("\nFinancial Impact Summary:")
    print(f"  - Fraud amount detected: ${fraud_detected_amount:.2f} ({detection_rate_by_amount * 100:.2f}%)")
    print(f"  - Fraud amount missed: ${fraud_missed_amount:.2f} ({(1 - detection_rate_by_amount) * 100:.2f}%)")
    print(f"  - False alarm amount: ${false_alarm_amount:.2f} ({false_alarm_rate_by_amount * 100:.4f}%)")
    
    # Create a threshold analysis to see how varying the threshold would impact these metrics
    if hasattr(y_pred, 'predict_proba'):
        thresholds = np.linspace(0.1, 0.9, 9)
        results = []
        
        for thresh in thresholds:
            y_pred_at_thresh = (y_pred.predict_proba(X_test)[:, 1] >= thresh).astype(int)
            tp = np.sum((y_true == 1) & (y_pred_at_thresh == 1))
            fp = np.sum((y_true == 0) & (y_pred_at_thresh == 1))
            fn = np.sum((y_true == 1) & (y_pred_at_thresh == 0))
            
            tp_amount = np.sum(transaction_amounts[(y_true == 1) & (y_pred_at_thresh == 1)])
            fn_amount = np.sum(transaction_amounts[(y_true == 1) & (y_pred_at_thresh == 0)])
            fp_amount = np.sum(transaction_amounts[(y_true == 0) & (y_pred_at_thresh == 1)])
            
            results.append({
                'threshold': thresh,
                'detected_fraud_pct': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'detected_amount_pct': tp_amount / (tp_amount + fn_amount) if (tp_amount + fn_amount) > 0 else 0,
                'false_positive_rate': fp / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0
            })
        
        # This information would typically be plotted, but we'll just print it for now
        print("\nThreshold Analysis would be performed here based on probability predictions")