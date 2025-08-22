import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
from collections import defaultdict

# Global variables for validation data
_validation_thresholds = {}
_roc_data = {}

def group_links_by_timestamp(predicted_links, actual_links):
    """Group links by timestamp for evaluation."""
    timestamp_data = defaultdict(lambda: {'predicted': [], 'actual': []})
    
    for src, dst, score, timestamp in predicted_links:
        timestamp_data[timestamp]['predicted'].append((src, dst, score))
    
    for src, dst, label, timestamp in actual_links:
        timestamp_data[timestamp]['actual'].append((src, dst, label))
    
    return dict(timestamp_data)

def calculate_timestamp_score(links, method='min'):
    """
    Calculate anomaly score for a timestamp (lower = more anomalous).
    
    Args:
        links: List of (src, dst, score) tuples
        method: Scoring method ('min' or 'bottom_1_percent')
    
    Returns:
        float: Timestamp anomaly score
    """
    if not links:
        return 1.0
    
    scores = [score for _, _, score in links]
    
    if method == 'min':
        return min(scores)
    elif method == 'bottom_1_percent':
        num_bottom = max(1, int(len(scores) * 0.01))
        return np.mean(sorted(scores)[:num_bottom])
    else:
        raise ValueError(f"Unknown method: {method}")

def is_timestamp_anomalous(links):
    """Check if timestamp contains any anomalous link."""
    return any(label == 1 for _, _, label in links)

def calculate_validation_thresholds(predicted_links, actual_links, methods):
    """Calculate optimal thresholds from validation data."""
    timestamp_data = group_links_by_timestamp(predicted_links, actual_links)
    thresholds = {}
    
    for method in methods:
        method_scores = [calculate_timestamp_score(data['predicted'], method) 
                        for data in timestamp_data.values()]
        
        if method_scores:
            if method == 'min':
                thresholds[f"{method}_min"] = min(method_scores)
                thresholds[f"{method}_mean"] = np.mean(method_scores)
            elif method == 'bottom_1_percent':
                thresholds[f"{method}_min"] = min(method_scores)
                thresholds[f"{method}_mean"] = np.mean(method_scores)
    
    return thresholds

def calculate_link_statistics(predicted_links, actual_links, non_exist_links):
    """Calculate score statistics by link type."""
    score_map = {(src, dst, ts): score for src, dst, score, ts in predicted_links}
    
    def get_stats(scores):
        return {
            'count': len(scores), 
            'mean': np.mean(scores) if scores else 0, 
            'std': np.std(scores) if scores else 0
        }
    
    benign_scores = []
    anomaly_scores = []
    
    for src, dst, label, ts in actual_links:
        if (src, dst, ts) in score_map:
            score = score_map[(src, dst, ts)]
            (benign_scores if label == 0 else anomaly_scores).append(score)
    
    non_exist_scores = [score for _, _, score, _ in non_exist_links]
    
    return {
        'benign': get_stats(benign_scores),
        'anomaly': get_stats(anomaly_scores),
        'non_exist': get_stats(non_exist_scores)
    }

def evaluate_timestamp_detection(predicted_links, actual_links, method, threshold):
    """Evaluate timestamp-level anomaly detection."""
    timestamp_data = group_links_by_timestamp(predicted_links, actual_links)
    
    scores, labels, predictions = [], [], []
    
    for data in timestamp_data.values():
        score = calculate_timestamp_score(data['predicted'], method)
        label = int(is_timestamp_anomalous(data['actual']))
        prediction = int(score < threshold)
        
        scores.append(score)
        labels.append(label)
        predictions.append(prediction)
    
    # Handle confusion matrix
    cm = confusion_matrix(labels, predictions)
    if cm.shape == (1, 1):
        tn = fp = fn = tp = 0
        if labels[0] == 0:
            tn = cm[0, 0]
        else:
            tp = cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()
    
    total = tp + tn + fp + fn
    return {
        'cm': {'TP': int(tp), 'FN': int(fn), 'FP': int(fp), 'TN': int(tn)},
        'metrics': {
            'accuracy': (tp + tn) / total if total > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        },
        'summary': {'total': len(labels), 'anomalous': sum(labels)},
        'method': method,
        'threshold': threshold
    }

def calculate_global_metrics(predicted_links, actual_links, method, validate, names):
    """Calculate ROC AUC and Average Precision globally (without threshold)."""
    global _roc_data
    
    # Suppress matplotlib warnings
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    timestamp_data = group_links_by_timestamp(predicted_links, actual_links)
    
    scores = [calculate_timestamp_score(data['predicted'], method) for data in timestamp_data.values()]
    labels = [int(is_timestamp_anomalous(data['actual'])) for data in timestamp_data.values()]
    
    if len(set(labels)) <= 1:
        print(f"\nGlobal metrics (method={method}): Cannot calculate (only one class)")
        print(f"  Timestamps: {len(labels)}")
        return None, None
    
    # Invert scores (lower original score = higher anomaly likelihood)
    inverted_scores = [1 - score for score in scores]
    
    roc_auc = roc_auc_score(labels, inverted_scores)
    avg_precision = average_precision_score(labels, inverted_scores)
    
    print(f"\nGlobal metrics (method={method}):")
    print(f"  ROC AUC: {roc_auc:.3f}")
    print(f"  Average Precision: {avg_precision:.3f}")
    print(f"  Timestamps: {len(labels)} (normal: {labels.count(0)}, anomalous: {labels.count(1)})")
    
    if not validate:
        fpr, tpr, _ = roc_curve(labels, inverted_scores)
        _roc_data[method] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
    
    return roc_auc, avg_precision

def create_threshold_distribution_plots(predicted_links, actual_links, names):
    """Create distribution plots with thresholds for both methods."""
    global _validation_thresholds
    
    os.makedirs('pdf_anom', exist_ok=True)
    
    timestamp_data = group_links_by_timestamp(predicted_links, actual_links)
    methods = ['min', 'bottom_1_percent']
    
    for method in methods:
        scores = []
        colors = []
        
        # Collect scores and their corresponding colors
        for data in timestamp_data.values():
            score = calculate_timestamp_score(data['predicted'], method)
            is_anomalous = is_timestamp_anomalous(data['actual'])
            
            scores.append(score)
            colors.append('red' if is_anomalous else 'blue')
        
        if not scores:
            continue
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Create fine histogram bins
        num_bins = min(200, max(50, int(len(scores) * 2)))
        bins = np.linspace(min(scores), max(scores), num_bins)
        
        # Separate anomalous and benign scores
        anomalous_scores = [score for score, color in zip(scores, colors) if color == 'red']
        benign_scores = [score for score, color in zip(scores, colors) if color == 'blue']
        
        # Plot histograms with fine bins
        plt.hist(benign_scores, bins=bins, alpha=0.7, color='blue', 
                label=f'Benign ({len(benign_scores)})', edgecolor='none', linewidth=0)
        plt.hist(anomalous_scores, bins=bins, alpha=0.7, color='red', 
                label=f'Anomalous ({len(anomalous_scores)})', edgecolor='none', linewidth=0)
        
        # Add threshold lines
        threshold_min_key = f"{method}_min"
        threshold_mean_key = f"{method}_mean"
        
        if threshold_min_key in _validation_thresholds:
            plt.axvline(_validation_thresholds[threshold_min_key], color='darkgreen', 
                       linestyle='-', linewidth=2, 
                       label=f'Threshold Min: {_validation_thresholds[threshold_min_key]:.6f}')
        
        if threshold_mean_key in _validation_thresholds:
            plt.axvline(_validation_thresholds[threshold_mean_key], color='lightgreen', 
                       linestyle='--', linewidth=2, 
                       label=f'Threshold Mean: {_validation_thresholds[threshold_mean_key]:.6f}')
        
        plt.xlabel('Score Values', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(f'Score Distribution - {method.replace("_", " ").title()} Method\nDataset: {names}', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        filename = f'pdf_anom/distribution_{method}_{names}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Distribution plot saved: {filename}")

def save_combined_roc_curve(names):
    """Save combined ROC curves for all methods."""
    global _roc_data
    
    if not _roc_data:
        return
    
    os.makedirs('courbes_auc', exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    colors = ['darkorange', 'darkgreen', 'darkblue', 'darkred', 'purple']
    
    for i, (method, data) in enumerate(_roc_data.items()):
        plt.plot(data['fpr'], data['tpr'], color=colors[i % len(colors)], lw=2, 
                label=f'{method.replace("_", " ").title()} (AUC = {data["roc_auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', label='Random (AUC = 0.500)')
    
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison - Timestamp Anomaly Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'courbes_auc/roc_timestamp_comparison_{names}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Combined ROC curve saved: {filename}")
    _roc_data.clear()

def save_timestamp_details(predicted_links, actual_links, method, names):
    """Save detailed timestamp information to JSON."""
    timestamp_data = group_links_by_timestamp(predicted_links, actual_links)
    
    details = []
    for timestamp, data in timestamp_data.items():
        benign_count = sum(1 for _, _, label in data['actual'] if label == 0)
        anomaly_count = sum(1 for _, _, label in data['actual'] if label == 1)
        
        scores = [score for _, _, score in data['predicted']]
        if scores:
            confidence_mean = np.mean(scores)
            confidence_min = min(scores)
            num_bottom = max(1, int(len(scores) * 0.01))
            bottom_1_percent_mean = np.mean(sorted(scores)[:num_bottom])
        else:
            confidence_mean = confidence_min = bottom_1_percent_mean = None
        
        details.append({
            'timestamp': timestamp,
            'nb_links_benign': benign_count,
            'nb_links_anomaly': anomaly_count,
            'is_suspicious': anomaly_count > 0,
            'total_predicted_links': len(data['predicted']),
            'confidence_mean': confidence_mean,
            'confidence_min': confidence_min,
            'confidence_bottom_1_percent': bottom_1_percent_mean,
            'method_score': calculate_timestamp_score(data['predicted'], method) if data['predicted'] else None
        })
    
    os.makedirs('json', exist_ok=True)
    filename = f'json/timestamp_details_{method}_{names}.json'
    
    with open(filename, 'w') as f:
        json.dump({
            'method': method,
            'dataset': names,
            'total_timestamps': len(details),
            'suspicious_timestamps': sum(1 for d in details if d['is_suspicious']),
            'details': details
        }, f, indent=2)
    
    print(f"  Timestamp details saved: {filename}")

def print_results(results, threshold_source="validation"):
    """Print evaluation results."""
    cm = results['cm']
    metrics = results['metrics']
    
    print(f"\n--- RESULTS (Threshold: {results['threshold']:.4f} [{threshold_source}], Method: {results['method']}) ---")
    print(f"Confusion Matrix: TP={cm['TP']}, FN={cm['FN']}, FP={cm['FP']}, TN={cm['TN']}")
    print(f"Metrics: Acc={metrics['accuracy']:.3f}, Prec={metrics['precision']:.3f}, Rec={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    print(f"Summary: {results['summary']['total']} timestamps ({results['summary']['anomalous']} anomalous)")

def print_link_stats(stats):
    """Print link score statistics."""
    print(f"\n--- LINK SCORE STATISTICS ---")
    for link_type, data in stats.items():
        print(f"{link_type.capitalize()} links: Count={data['count']}, Mean={data['mean']:.4f}, Std={data['std']:.4f}")

def timestamp_anomaly_result(predicted_links, actual_links, non_exist_links, validate, 
                           names=None, methods=['min', 'bottom_1_percent']):
    """
    Main function for timestamp-based anomaly detection evaluation.
    
    Args:
        predicted_links: List of (src, dst, score, timestamp) tuples
        actual_links: List of (src, dst, label, timestamp) tuples
        non_exist_links: List of non-existing links with scores
        validate: Boolean flag for validation vs test mode
        names: Dataset name for file naming
        methods: List of scoring methods to evaluate
    """
    global _validation_thresholds
    
    print(f"\n{'='*80}")
    print(f"TIMESTAMP ANOMALY DETECTION - {'VALIDATE' if validate else 'TEST'} {names}")
    print(f"{'='*80}")
    
    # Calculate and print link statistics
    link_stats = calculate_link_statistics(predicted_links, actual_links, non_exist_links)
    print_link_stats(link_stats)
    
    if validate:
        print("Logic: Timestamps with >=1 anomalous link are anomalous")
        _validation_thresholds = calculate_validation_thresholds(predicted_links, actual_links, methods)
        
        print(f"\n--- VALIDATION THRESHOLDS ---")
        for method, threshold in _validation_thresholds.items():
            print(f"  {method}: {threshold:.8f}")
        print(f"Validation complete. Thresholds saved for test phase.")
        
    else:
        print("Logic: Timestamps with >=1 anomalous link are anomalous")
        
        # Create distribution plots
        create_threshold_distribution_plots(predicted_links, actual_links, names)
        
        # Evaluate each method
        for method in methods:
            print(f"\n{'='*50}")
            print(f"METHOD: {method.upper()}")
            print(f"{'='*50}")
            
            calculate_global_metrics(predicted_links, actual_links, method, validate, names)
            save_timestamp_details(predicted_links, actual_links, method, names)
            
            print(f"\n--- THRESHOLD EVALUATION ---")
            for threshold_name, threshold in _validation_thresholds.items():
                if threshold_name.startswith(method):
                    results = evaluate_timestamp_detection(predicted_links, actual_links, method, threshold)
                    threshold_source = "min" if threshold_name.endswith("_min") else "mean"
                    print_results(results, threshold_source)
        
        # Save combined ROC curves
        save_combined_roc_curve(names)