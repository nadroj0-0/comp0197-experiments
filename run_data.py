import os
# Import the model from the models folder (package)
from models.h_lstm import HierarchicalLSTMModel
# New version
from models.base_model import BaseModel

def main():
    print("Step 1: Starting Data Pre-processing...")
    
    # 1. Instantiate the model
    # This automatically creates 'data' and 'outputs' folders via BaseModel.__init__
    m = HierarchicalLSTMModel()
    
    # 2. Run the shared data pipeline
    # This checks for 'data/raw_split.pkl'. If not found, it downloads 
    # the M5 CSVs, melts them into 59M rows, and saves the cache.
    m.load_and_split_data()
    
    print("✅ Data cached successfully in data/raw_split.pkl")

if __name__ == "__main__":
    main()