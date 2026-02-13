import sys
from datetime import datetime

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")

def run_pipeline():
    """
    Run the complete classification pipeline
    """
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + " " * 20 + "SPORT vs POLITICS TEXT CLASSIFIER" + " " * 24 + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Step 1: Data Collection
        print_section("STEP 1: DATA COLLECTION")
        import data_collection
        df = data_collection.main()
        print("✓ Data collection completed successfully")
        
        # Step 2: Feature Extraction
        print_section("STEP 2: FEATURE EXTRACTION AND PREPROCESSING")
        import feature_extraction
        features, y_train, y_test = feature_extraction.main()
        print("✓ Feature extraction completed successfully")
        
        # Step 3: Model Training
        print_section("STEP 3: MODEL TRAINING")
        import model_training
        all_results, df_comparison = model_training.main()
        print("✓ Model training completed successfully")
        
        # Step 4: Evaluation and Visualization
        print_section("STEP 4: EVALUATION AND VISUALIZATION")
        import evaluation
        
        # Generate visualizations
        evaluation.create_wordclouds(df)
        evaluation.plot_model_comparison(df_comparison)
        
        # Generate confusion matrices if we have all_results
        if all_results:
            evaluation.plot_confusion_matrices(all_results)
            evaluation.plot_detailed_classification_reports(all_results, y_test)
        
        # Generate summary report
        evaluation.generate_summary_report(df_comparison, all_results)
        
        print("✓ Evaluation and visualization completed successfully")
        
        # Final Summary
        print_section("PIPELINE COMPLETED SUCCESSFULLY!")
        print("Results saved in the following directories:")
        print("  • data/              - Dataset")
        print("  • models/            - Trained models and features")
        print("  • results/           - Comparison results and plots")
        print("\nKey files:")
        print("  • results/model_comparison.csv    - Detailed comparison of all models")
        print("  • results/summary_report.txt      - Summary of best models")
        print("  • results/plots/                  - All visualization plots")
        
        # Display best model
        best_idx = df_comparison['Accuracy'].idxmax()
        best_model = df_comparison.loc[best_idx]
        
        print("\n" + "-" * 80)
        print("BEST MODEL:")
        print("-" * 80)
        print(f"Feature Method: {best_model['Feature Method']}")
        print(f"Model:          {best_model['Model']}")
        print(f"Accuracy:       {best_model['Accuracy']:.4f}")
        print(f"F1-Score:       {best_model['F1-Score']:.4f}")
        print("-" * 80)
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + "*" * 80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
