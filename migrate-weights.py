import os
import shutil
from models_config import MODEL_REGISTRY

def migrate():
    base_output = "outputs"
    
    for name, model_class in MODEL_REGISTRY.items():
        # Initialize model to get the correct subfolder path
        m = model_class()
        target_dir = m.output_dir # e.g., 'outputs/LightGBM_NN_Hybrid'
        
        # Ensure the subfolder exists
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"📂 Organizing files for: {name} -> {target_dir}")

        # Define which files belong to which model
        # 1. Standard .pth weights and predictions
        files_to_move = [
            f"{m.model_name}.pth",
            f"{m.model_name}_predictions.csv",
            f"{m.model_name}_results.json"
        ]

        # 2. Add Hybrid-specific files if applicable
        if "LightGBM" in m.model_name:
            hybrid_files = [
                "nn_embedding.pth", "scaler.pkl", "features.pkl", 
                "cat_maps.pkl", "n_items.pkl"
            ]
            # Add quantile model files (lgb_q_0.025.txt, etc.)
            for q in [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]:
                hybrid_files.append(f"lgb_q_{q}.txt")
            files_to_move.extend(hybrid_files)

        # Move the files
        for filename in files_to_move:
            source = os.path.join(base_output, filename)
            destination = os.path.join(target_dir, filename)
            
            if os.path.exists(source):
                print(f"  Moving: {filename}")
                shutil.move(source, destination)
            else:
                # Silently skip if the file doesn't exist (e.g. model wasn't run yet)
                pass

    print("\n✅ Migration complete! Your /outputs folder is now organized.")

if __name__ == "__main__":
    migrate()