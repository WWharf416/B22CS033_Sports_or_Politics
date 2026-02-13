import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, roc_auc_score, precision_recall_curve
)
from wordcloud import WordCloud
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def plot_confusion_matrices(all_results, output_dir='results/plots'):
    """
    Plot confusion matrices for all models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for feature_name, models in all_results.items():
        n_models = len(models)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Confusion Matrices - {feature_name}', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, (model_name, result) in enumerate(models.items()):
            cm = result['metrics']['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Sport', 'Politics'],
                       yticklabels=['Sport', 'Politics'],
                       ax=axes[idx])
            
            axes[idx].set_title(f'{model_name}\nAccuracy: {result["metrics"]["accuracy"]:.3f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide unused subplots
        for idx in range(len(models), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        filename = f'confusion_matrices_{feature_name.replace(" ", "_").lower()}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

def plot_model_comparison(df_comparison, output_dir='results/plots'):
    """
    Plot comprehensive model comparison
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(14, 8))
    
    # Group by feature method
    feature_methods = df_comparison['Feature Method'].unique()
    
    for feature_method in feature_methods:
        df_subset = df_comparison[df_comparison['Feature Method'] == feature_method]
        plt.plot(df_subset['Model'], df_subset['Accuracy'], 
                marker='o', label=feature_method, linewidth=2, markersize=8)
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Accuracy Comparison Across Feature Extraction Methods', 
             fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    print("Saved: accuracy_comparison.png")
    plt.close()
    
    # 2. F1-Score Comparison
    plt.figure(figsize=(14, 8))
    
    for feature_method in feature_methods:
        df_subset = df_comparison[df_comparison['Feature Method'] == feature_method]
        plt.plot(df_subset['Model'], df_subset['F1-Score'], 
                marker='s', label=feature_method, linewidth=2, markersize=8)
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
    plt.title('Model F1-Score Comparison Across Feature Extraction Methods', 
             fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1score_comparison.png'), dpi=300, bbox_inches='tight')
    print("Saved: f1score_comparison.png")
    plt.close()
    
    # 3. Heatmap of all metrics
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Create pivot tables for heatmaps
    for idx, metric in enumerate(['Accuracy', 'F1-Score']):
        pivot_data = df_comparison.pivot(index='Model', 
                                         columns='Feature Method', 
                                         values=metric)
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlGnBu', 
                   ax=axes[idx], cbar_kws={'label': metric})
        axes[idx].set_title(f'{metric} Heatmap', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Feature Method', fontsize=12)
        axes[idx].set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'), dpi=300, bbox_inches='tight')
    print("Saved: metrics_heatmap.png")
    plt.close()
    
    # 4. Bar chart for best models
    plt.figure(figsize=(12, 6))
    
    best_models = df_comparison.groupby('Model')['Accuracy'].max().sort_values(ascending=False)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(best_models)))
    bars = plt.bar(range(len(best_models)), best_models.values, color=colors)
    plt.xticks(range(len(best_models)), best_models.index, rotation=45, ha='right')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Best Accuracy', fontsize=12, fontweight='bold')
    plt.title('Best Accuracy Achieved by Each Model', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_accuracy_by_model.png'), dpi=300, bbox_inches='tight')
    print("Saved: best_accuracy_by_model.png")
    plt.close()
    
    # 5. Training time comparison
    plt.figure(figsize=(12, 6))
    
    df_time = df_comparison.groupby('Model')['Training Time (s)'].mean().sort_values()
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(df_time)))
    bars = plt.barh(range(len(df_time)), df_time.values, color=colors)
    plt.yticks(range(len(df_time)), df_time.index)
    plt.xlabel('Average Training Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.title('Average Training Time by Model', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for idx, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.2f}s',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'), dpi=300, bbox_inches='tight')
    print("Saved: training_time_comparison.png")
    plt.close()

def plot_detailed_classification_reports(all_results, y_test, output_dir='results/plots'):
    """
    Generate detailed classification reports for each model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    class_names = ['Sport', 'Politics']
    
    for feature_name, models in all_results.items():
        reports_text = f"Classification Reports - {feature_name}\n"
        reports_text += "=" * 80 + "\n\n"
        
        for model_name, result in models.items():
            predictions = result['predictions']
            
            reports_text += f"{model_name}\n"
            reports_text += "-" * 80 + "\n"
            reports_text += classification_report(y_test, predictions, 
                                                 target_names=class_names)
            reports_text += "\n\n"
        
        # Save to file
        filename = f'classification_reports_{feature_name.replace(" ", "_").lower()}.txt'
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.write(reports_text)
        
        print(f"Saved: {filename}")

def create_wordclouds(df, output_dir='results/plots'):
    """
    Create word clouds for Sport and Politics categories
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate texts by category
    sport_texts = ' '.join(df[df['category'] == 'Sport']['text'].values)
    politics_texts = ' '.join(df[df['category'] == 'Politics']['text'].values)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Sport WordCloud
    wordcloud_sport = WordCloud(width=800, height=400, 
                               background_color='white',
                               colormap='Greens').generate(sport_texts)
    
    axes[0].imshow(wordcloud_sport, interpolation='bilinear')
    axes[0].set_title('Sport Category - Word Cloud', fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # Politics WordCloud
    wordcloud_politics = WordCloud(width=800, height=400, 
                                   background_color='white',
                                   colormap='Blues').generate(politics_texts)
    
    axes[1].imshow(wordcloud_politics, interpolation='bilinear')
    axes[1].set_title('Politics Category - Word Cloud', fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wordclouds.png'), dpi=300, bbox_inches='tight')
    print("Saved: wordclouds.png")
    plt.close()

def generate_summary_report(df_comparison, all_results, output_dir='results'):
    """
    Generate a comprehensive summary report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report = "=" * 80 + "\n"
    report += "SPORT vs POLITICS TEXT CLASSIFICATION - SUMMARY REPORT\n"
    report += "=" * 80 + "\n\n"
    
    # Overall best model
    best_acc_idx = df_comparison['Accuracy'].idxmax()
    best_model = df_comparison.loc[best_acc_idx]
    
    report += "BEST OVERALL MODEL\n"
    report += "-" * 80 + "\n"
    report += f"Feature Method: {best_model['Feature Method']}\n"
    report += f"Model: {best_model['Model']}\n"
    report += f"Accuracy: {best_model['Accuracy']:.4f}\n"
    report += f"Precision: {best_model['Precision']:.4f}\n"
    report += f"Recall: {best_model['Recall']:.4f}\n"
    report += f"F1-Score: {best_model['F1-Score']:.4f}\n"
    report += f"Training Time: {best_model['Training Time (s)']:.2f} seconds\n\n"
    
    # Best model for each feature method
    report += "BEST MODEL FOR EACH FEATURE METHOD\n"
    report += "-" * 80 + "\n"
    
    for feature_method in df_comparison['Feature Method'].unique():
        df_subset = df_comparison[df_comparison['Feature Method'] == feature_method]
        best_idx = df_subset['Accuracy'].idxmax()
        best = df_subset.loc[best_idx]
        
        report += f"\n{feature_method}:\n"
        report += f"  Model: {best['Model']}\n"
        report += f"  Accuracy: {best['Accuracy']:.4f}\n"
        report += f"  F1-Score: {best['F1-Score']:.4f}\n"
    
    # Top 5 models overall
    report += "\n\nTOP 5 MODELS (by Accuracy)\n"
    report += "-" * 80 + "\n"
    
    top_5 = df_comparison.nlargest(5, 'Accuracy')
    for idx, row in top_5.iterrows():
        report += f"\n{idx + 1}. {row['Model']} ({row['Feature Method']})\n"
        report += f"   Accuracy: {row['Accuracy']:.4f}, F1-Score: {row['F1-Score']:.4f}\n"
    
    # Statistics
    report += "\n\nSTATISTICS\n"
    report += "-" * 80 + "\n"
    report += f"Average Accuracy: {df_comparison['Accuracy'].mean():.4f}\n"
    report += f"Std Dev Accuracy: {df_comparison['Accuracy'].std():.4f}\n"
    report += f"Average F1-Score: {df_comparison['F1-Score'].mean():.4f}\n"
    report += f"Std Dev F1-Score: {df_comparison['F1-Score'].std():.4f}\n"
    
    # Save report
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write(report)
    
    print("\nSaved: summary_report.txt")
    print("\n" + report)

def main():
    """
    Main evaluation and visualization pipeline
    """
    print("=" * 60)
    print("MODEL EVALUATION AND VISUALIZATION")
    print("=" * 60)
    
    # Load comparison results
    print("\nLoading results...")
    df_comparison = pd.read_csv('results/model_comparison.csv')
    
    # Load dataset for word clouds
    df = pd.read_csv('data/dataset.csv')
    
    # Load y_test
    y_test = np.load('models/features/y_test.npy')
    
    # Note: For full visualization, you would need to load all_results
    # For this simplified version, we'll create what we can from the CSV
    
    print("\nGenerating visualizations...")
    
    # Create word clouds
    create_wordclouds(df)
    
    # Plot comparisons
    plot_model_comparison(df_comparison)
    
    # Generate summary report
    generate_summary_report(df_comparison, None)
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("All plots saved to 'results/plots/'")
    print("Summary report saved to 'results/summary_report.txt'")
    print("=" * 60)

if __name__ == "__main__":
    main()
