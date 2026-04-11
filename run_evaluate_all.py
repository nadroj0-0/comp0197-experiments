import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from models_config import MODEL_REGISTRY

def plot_individual_model(item_id, actual_series, name, preds, output_dir):
    """Generates a detailed visualization for a single model's performance."""
    plt.figure(figsize=(12, 6))
    
    item_preds = preds[preds['id'] == item_id].sort_values('day_ahead')
    if item_preds.empty:
        return

    # 1. Plot Actuals
    plt.plot(actual_series.values, label='Actual Sales', color='black', linewidth=2)
    
    # 2. Plot Quantile Ranges (The 'Fan')
    # Inner 50% CI
    plt.fill_between(range(28), item_preds['q0.25'].values, item_preds['q0.75'].values, 
                     color='blue', alpha=0.3, label='50% Confidence Interval')
    # Outer 95% CI
    plt.fill_between(range(28), item_preds['q0.025'].values, item_preds['q0.975'].values, 
                     color='blue', alpha=0.1, label='95% Confidence Interval')
    
    # 3. Plot Median
    plt.plot(item_preds['q0.5'].values, label='Predicted Median', color='blue', linewidth=2)

    plt.title(f"{name} Forecast: {item_id}", fontsize=14)
    plt.xlabel("Days Ahead", fontsize=12)
    plt.ylabel("Sales", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, f"forecast_plot_{item_id}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Individual plot for {name} saved to {plot_path}")

def plot_model_comparison(item_id, actual_series, all_model_preds, main_output_dir):
    """Generates a master visual comparison of all models in one plot."""
    plt.figure(figsize=(14, 7))
    plt.plot(actual_series.values, label='Actual Sales', color='black', linewidth=2, zorder=3)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, preds) in enumerate(all_model_preds.items()):
        item_preds = preds[preds['id'] == item_id].sort_values('day_ahead')
        if item_preds.empty: continue
        color = colors[i % len(colors)]
        
        plt.plot(item_preds['q0.5'].values, label=f'{name} Median', color=color, 
                 linestyle='--' if 'linear' in name.lower() else '-')
        plt.fill_between(range(28), item_preds['q0.025'].values, item_preds['q0.975'].values, 
                         color=color, alpha=0.1)

    plt.title(f"Master Model Comparison: {item_id}", fontsize=14)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(main_output_dir, f"master_comparison_{item_id}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"📈 Master comparison saved to {plot_path}")

def main():
    summary_stats = []
    all_predictions = {}
    model_dirs = {} # Store model directories for later plotting
    main_output_dir = "outputs"
    os.makedirs(main_output_dir, exist_ok=True)
    
    actual_data = None
    top_items = None

    for name, model_class in MODEL_REGISTRY.items():
        print(f"\n--- Evaluating Model: {name} ---")
        m = model_class()
        m.load_and_split_data() 
        model_dirs[name] = m.output_dir # Capture the subfolder path
        
        pred_file = f"{m.model_name}_predictions.csv"
        pred_path = os.path.join(m.output_dir, pred_file)

        if os.path.exists(pred_path):
            print(f"📂 Loading predictions from {pred_path}")
            preds_df = pd.read_csv(pred_path)
        else:
            print(f"🚀 Running inference for {name}...")
            m.preprocess()
            preds_df = m.predict() 
        
        all_predictions[name] = preds_df
        summary_stats.append(m.evaluate(preds_df)) 
        
        if actual_data is None:
            actual_data = m.test_raw[m.test_raw['d_num'] >= m.TARGET_START]
            top_items = m.item_weights.sort_values(ascending=False).head(3).index.tolist()

    # --- NEW: Generate plots for each model and the master comparison ---
    print("\n--- Generating All Visualizations ---")
    for item_id in top_items:
        item_actuals = actual_data[actual_data['id'] == item_id].sort_values('d_num')['sales']
        
        # 1. Individual plots (one per model folder)
        for name, preds in all_predictions.items():
            plot_individual_model(item_id, item_actuals, name, preds, model_dirs[name])
            
        # 2. Master comparison plot (in main outputs folder)
        plot_model_comparison(item_id, item_actuals, all_predictions, main_output_dir)

    df_compare = pd.DataFrame(summary_stats)
    print("\n" + "="*50 + "\n FINAL MODEL COMPARISON\n" + "="*50)
    print(df_compare.to_string(index=False))
    df_compare.to_csv(os.path.join(main_output_dir, "model_comparison.csv"), index=False)

if __name__ == "__main__":
    main()