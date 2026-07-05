import os
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm

from amc.features import FeatureExtractor

def worker_extract(row_data) -> dict:
    """
    Worker function to process a single signal row and extract its features.
    """
    idx, rel_path, modulation, snr_db, sample_rate = row_data
    
    # Resolve absolute path to the dataset sample
    dataset_dir = "dataset"
    file_path = os.path.join(dataset_dir, rel_path)
    
    if not os.path.exists(file_path):
        return {}

    try:
        # Load the complex IQ signal
        signal = np.load(file_path)
        
        # Instantiate extractor and calculate features
        extractor = FeatureExtractor()
        extracted = extractor.extract(signal)
        
        # Inject metadata labels
        extracted["file_path"] = rel_path
        extracted["modulation"] = modulation
        extracted["snr_db"] = float(snr_db)
        
        return extracted
    except Exception as e:
        print(f"Error processing {rel_path}: {e}")
        return {}

def main():
    metadata_path = os.path.join("dataset", "metadata.csv")
    output_path = os.path.join("dataset", "features.csv")

    if not os.path.exists(metadata_path):
        print(f"Error: metadata.csv not found at {metadata_path}. Please run generate_dataset.py first.")
        return

    print("Loading metadata...")
    df_meta = pd.read_csv(metadata_path)

    # Prepare inputs for multiprocessing pool
    # Zip: index, file_path, modulation, snr_db, sample_rate
    tasks = list(zip(
        df_meta.index,
        df_meta["file_path"],
        df_meta["modulation"],
        df_meta["snr_db"],
        df_meta["sample_rate"]
    ))

    print(f"Starting feature extraction for {len(tasks)} samples...")
    num_workers = multiprocessing.cpu_count()
    print(f"Utilizing {num_workers} parallel workers.")

    extracted_rows = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(worker_extract, tasks), total=len(tasks), desc="Extracting features"):
            if result:  # Check if non-empty
                extracted_rows.append(result)

    # Convert to DataFrame
    df_features = pd.DataFrame(extracted_rows)

    # Reorder columns to place key metadata fields at the front
    meta_cols = ["file_path", "modulation", "snr_db"]
    feature_cols = [col for col in df_features.columns if col not in meta_cols]
    
    # Sort columns alphabetically for consistency
    feature_cols.sort()
    df_features = df_features[meta_cols + feature_cols]

    # Save to CSV
    df_features.to_csv(output_path, index=False)
    print(f"\nFeature extraction completed successfully.")
    print(f"Features saved to: {os.path.abspath(output_path)}")
    print(f"Total features extracted per sample: {len(feature_cols)}")

if __name__ == "__main__":
    main()
