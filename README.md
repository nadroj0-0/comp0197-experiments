# M5 Forecasting Pipeline

This project is a modularized version of the M5 Uncertainty competition. It is designed to let you compare different forecasting models (like LSTMs vs. Linear Baselines) in a clean, organized way.

---

### 📦 Project Structure
* **`data/`**: Stores the raw data and the 815MB processed cache (`raw_split.pkl`).
* **`models/`**: **This is where you put your models.** Every model must be a file here (e.g., `h_lstm.py`) and inherit from the `BaseModel` template.
* **`outputs/`**: Holds trained weights (`.pth` files), prediction CSVs, and comparison charts.
* **`run_data.py`**: Run this first to download and prepare the data.
* **`run_train_all.py`**: Iterates through the registry to train all models.
* **`run_evaluate_all.py`**: Generates the final scores and "Fan Chart" visualizations.

---

### 🚀 How to Use It

1.  **Prepare Data**:
    Run `python run_data.py`. It will create a `raw_split.pkl` file. This is the 815MB cache that stores the pre-split training and testing sets.
2.  **Add a Model**:
    Drop your model class into the `models/` folder. Ensure it has `preprocess`, `train`, and `predict` methods as required by the `BaseModel`.
3.  **Register the Model**:
    Open `models_config.py` and add your new model class to the `MODEL_REGISTRY` dictionary.
4.  **Train and Evaluate**:
    Run `python run_train_all.py` to train, then `python run_evaluate_all.py` to see the results.

---

### 📈 Results & Visuals
The system automatically generates a comparison table in the console and saves it to `outputs/model_comparison.csv`. In our tests, the **Hierarchical LSTM** significantly outperformed the **Linear Baseline** because it understands complex patterns and uncertainty.

| Model | WSPL (Lower is better) | CRPS | 95% Coverage |
| :--- | :--- | :--- | :--- |
| **H-LSTM** | **0.398** | **1.001** | **97% (Accurate)** |
| **Linear** | 1.604 | 5.188 | 13% (Way off) |

#### The "Fan Chart" Visualization
The pipeline produces "Fan Charts" for the top-weighted items to show how the models handle the 28-day forecast horizon. 

![M5 Forecast Comparison](https://raw.githubusercontent.com/nadroj0-0/Applied-Deep-Learning-COMP019-GROUP/luke-modules/outputs/comparison_FOODS_3_120_CA_3_evaluation.png)

* **Black Line**: Actual sales (Ground Truth).
* **Blue Cloud**: H-LSTM 95% Confidence Interval (q0.025 to q0.975).
* **Red Dashed Line**: Linear Baseline (Deterministic).

If the black line stays inside the blue cloud, the model is successfully capturing the true uncertainty of the market.
---
