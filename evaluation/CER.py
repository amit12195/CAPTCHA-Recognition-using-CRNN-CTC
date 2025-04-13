import pandas as pd
import editdistance
import os

def extract_ground_truth(filename):
    """
    Extract ground truth from filename like 'image_12876.jpg' -> '12876'
    """
    if not isinstance(filename, str):
        return ""
    base = os.path.basename(filename)
    if "_" in base and "." in base:
        return base.split('_', 1)[1].rsplit('.', 1)[0]
    return ""

def evaluate_from_csv(csv_path):
    df = pd.read_csv(csv_path, header=None)

    total_cer = 0
    total_chars = 0
    correct = 0
    total = 0

    for _, row in df.iterrows():
        img_name = str(row[0]) if pd.notnull(row[0]) else ""
        pred = str(row[1]) if pd.notnull(row[1]) else ""

        gt = extract_ground_truth(img_name)
        if not gt:
            
            continue

        cer = editdistance.eval(gt, pred)
        total_cer += cer
        total_chars += len(gt)
        if gt == pred:
            correct += 1
        total += 1
        print(f"[{img_name}] GT: {gt} | Pred: {pred} | CER: {cer / len(gt):.2f}")

    if total_chars == 0:
        print("No valid data to evaluate.")
        return

    avg_cer = total_cer / total_chars
    accuracy = correct / total

    print("\n=== Evaluation ===")
    print(f"Character Error Rate (CER): {avg_cer:.4f}")
    print(f"Exact Match Accuracy     : {accuracy:.4f}")

    return avg_cer, accuracy



csv_path = 'ground_predicted_values.csv'
evaluate_from_csv(csv_path)
