import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from task import TabularNet

# ==========================================
# ⚙️ PREDICTION CONTROL PANEL
# ==========================================
MODEL_FILE_PATH = "./output/best_model.pth"
EVAL_FILE_PATH = "./data/global_test.pt"
PLOT_FILE_PATH = "./output/fedavg_confusion_matrix.png"
CLASS_NAMES = ["Low Yield (0)", "Med Yield (1)", "High Yield (2)"]
NUM_DEBUG_SAMPLES = 10   # Number of individual tabular samples to inspect visually (Ground Truth Class v/s Predicted Class)
# ==========================================

def run_prediction():
    if not os.path.exists(MODEL_FILE_PATH):
        print(f"❌ Error: Model weights not found at '{MODEL_FILE_PATH}'. Did you finish rounds?")
        return

    if not os.path.exists(EVAL_FILE_PATH):
        print(f"❌ Error: Global test file not found at '{EVAL_FILE_PATH}'. Run prepare_tabular_data.py first!")
        return

    print("==================================================")
    print("🎯 STARTING GLOBAL MODEL EVALUATION (UNSEEN TEST SET)")
    print("==================================================")

    # 1. Load Trained TabularNet Model
    net = TabularNet()
    net.load_state_dict(torch.load(MODEL_FILE_PATH, weights_only=True))
    net.eval()
    print("✅ Successfully loaded trained global model weights.")

    # 2. Load Unseen Global Test Data
    test_data = torch.load(EVAL_FILE_PATH, weights_only=True)
    X_test, y_test = test_data["x"].float(), test_data["y"].long()
    print(f"📦 Loaded {len(X_test)} unseen test samples.")

    # 3. Perform Inference
    with torch.no_grad():
        outputs = net(X_test)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)

    y_true = y_test.numpy()
    y_pred = predictions.numpy()

    # 4. Compute Accuracy Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"\n🌐 GLOBAL UNSEEN TEST ACCURACY: {acc * 100:.2f}%\n")

    # ==========================================
    # 🔍 TABULAR SAMPLE PREDICTION VISUAL INSPECTION
    # ==========================================
    print(f"--- 🔍 Visual Inspection of {NUM_DEBUG_SAMPLES} Random Test Samples ---")
    
    # Pick N random indices from test set
    random_indices = np.random.choice(len(X_test), size=min(NUM_DEBUG_SAMPLES, len(X_test)), replace=False)
    
    sample_records = []
    for idx in random_indices:
        true_cls = y_true[idx]
        pred_cls = y_pred[idx]
        confidence = probabilities[idx][pred_cls].item() * 100
        
        status = "✅ MATCH" if true_cls == pred_cls else "❌ MISMATCH"
        
        sample_records.append({
            "Sample ID": f"Row #{idx:03d}",
            "True Label": f"{CLASS_NAMES[true_cls]}",
            "Predicted Label": f"{CLASS_NAMES[pred_cls]}",
            "Confidence": f"{confidence:.1f}%",
            "Status": status
        })
    
    debug_df = pd.DataFrame(sample_records)
    print(debug_df.to_string(index=False))
    print("-" * 65 + "\n")

    # 5. Print Detailed Classification Report for Paper
    print("--- Detailed Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)

    # 6. Generate and Display Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"FedAvg Baseline - Global Test Confusion Matrix\nAccuracy: {acc*100:.2f}%")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    
    plt.savefig(PLOT_FILE_PATH)
    print(f"📊 Confusion Matrix saved to '{PLOT_FILE_PATH}'.")
    plt.show()
    print("📊 Inference complete. Check the pop-up window for results.")

if __name__ == "__main__":
    run_prediction()
