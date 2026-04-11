import os
from models_config import MODEL_REGISTRY

def main():
    for name, model_class in MODEL_REGISTRY.items():
        # Initialize model - this now creates outputs/model_name/ via BaseModel
        m = model_class()
        
        # --- FIX: Handle different weight naming conventions ---
        # Standard models use model_name.pth, Hybrid uses nn_embedding.pth
        if "LightGBM" in m.model_name:
            check_file = "nn_embedding.pth"
        else:
            check_file = f"{m.model_name}.pth"

        weights_path = os.path.join(m.output_dir, check_file)

        print(f"\n--- Checking Model: {name} ---")
        print(f"Target Directory: {m.output_dir}")

        # Skip logic: now checks the correct subfolder
        if os.path.exists(weights_path):
            print(f"⏭️  Weights found at {weights_path}. Skipping training phase.")
            continue
        
        print(f"🚀 No weights found. Starting training for {name}...")
        
        # 1. Load the 815MB raw data cache
        m.load_and_split_data() 
        
        # 2. Build the model-specific tensors/datasets
        m.preprocess() 
        
        # 3. Execute training
        # We can dynamically set epochs based on the model name
        epochs = 5 if ("lstm" in name.lower() or "lightgbm" in name.lower()) else 50
        m.train(epochs=epochs)
        
        print(f"✅ {name} training complete. Weights saved in {m.output_dir}")

if __name__ == "__main__":
    main()