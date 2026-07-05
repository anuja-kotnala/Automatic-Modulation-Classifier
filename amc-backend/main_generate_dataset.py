import os
import csv
import argparse
import multiprocessing
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm

from amc.constants import ModulationType
from amc.generator import (
    AMModulator,
    FMModulator,
    BPSKModulator,
    QPSKModulator,
    QAM16Modulator,
    QAM64Modulator,
    OFDMModulator
)
from amc.config import AppConfig
from amc.channel.channel_pipeline import ChannelPipeline

def get_modulator_instance(mod_type: ModulationType, sample_rate: float, seed: int):
    """
    Helper factory to instantiate the correct modulator with default config values.
    """
    if mod_type == ModulationType.AM:
        return AMModulator(sample_rate=sample_rate, random_seed=seed)
    elif mod_type == ModulationType.FM:
        return FMModulator(sample_rate=sample_rate, random_seed=seed)
    elif mod_type == ModulationType.BPSK:
        return BPSKModulator(sample_rate=sample_rate, random_seed=seed)
    elif mod_type == ModulationType.QPSK:
        return QPSKModulator(sample_rate=sample_rate, random_seed=seed)
    elif mod_type == ModulationType.QAM16:
        return QAM16Modulator(sample_rate=sample_rate, random_seed=seed)
    elif mod_type == ModulationType.QAM64:
        return QAM64Modulator(sample_rate=sample_rate, random_seed=seed)
    elif mod_type == ModulationType.OFDM:
        return OFDMModulator(sample_rate=sample_rate, random_seed=seed)
    else:
        raise ValueError(f"Unsupported modulation type: {mod_type}")

def worker_generate_sample(args) -> Dict[str, Any]:
    """
    Worker function to generate a single signal sample.
    Runs inside a multiprocessing pool.
    """
    mod_name, snr, sample_idx, out_dir, config_path, sample_rate, num_samples = args
    
    # Re-parse config inside the process to avoid pickle issues
    config = AppConfig.load_from_yaml(config_path)
    
    # Unique seed per sample/SNR/Modulation
    seed = hash((mod_name, snr, sample_idx)) % (2**32)
    np.random.seed(seed)

    mod_type = ModulationType(mod_name)
    modulator = get_modulator_instance(mod_type, sample_rate, seed)

    # Generate noise-free signal first (AWGN will be applied by channel pipeline or manually)
    raw_signal = modulator.generate(num_samples, snr_db=None)

    # Apply impairments via pipeline (override pipeline SNR with current loop SNR)
    ch_config = config.channel
    ch_config.awgn.snr_db = float(snr)
    
    # If the user disabled AWGN in config, we force it enabled to respect the loop SNR
    ch_config.awgn.enabled = True

    pipeline = ChannelPipeline(ch_config, seed=seed)
    impaired_signal = pipeline.apply(raw_signal, sample_rate)

    # Create destination subdirectory: dataset/raw/<Modulation>/snr_<SNR>/
    sub_dir = os.path.join(out_dir, "raw", mod_name, f"snr_{snr}")
    os.makedirs(sub_dir, exist_ok=True)

    filename = f"sample_{sample_idx:03d}.npy"
    file_path = os.path.join(sub_dir, filename)
    
    # Save as numpy array
    np.save(file_path, impaired_signal)

    # Relative path for metadata CSV
    rel_path = os.path.relpath(file_path, out_dir)

    return {
        "file_path": rel_path,
        "modulation": mod_name,
        "snr_db": snr,
        "sample_index": sample_idx,
        "sample_rate": sample_rate,
        "num_samples": num_samples
    }

def main():
    parser = argparse.ArgumentParser(description="AMC & Spectrum Analyzer Dataset Generation Pipeline")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Output root directory")
    parser.add_argument("--samples_per_snr", type=int, default=100, help="Number of samples to generate per SNR value")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel processes")
    args = parser.parse_args()

    # Load configuration
    config = AppConfig.load_from_yaml(args.config)
    sample_rate = config.generator.sample_rate
    num_samples = config.generator.samples_per_signal # 1024 points

    # Defined parameters
    snrs = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    modulations = [mod.value for mod in ModulationType]

    print(f"Starting dataset generation pipeline...")
    print(f"Modulations: {modulations}")
    print(f"SNR range: {snrs} dB")
    print(f"Samples per SNR: {args.samples_per_snr}")
    print(f"Total samples to generate: {len(modulations) * len(snrs) * args.samples_per_snr}")
    print(f"Utilizing {args.num_workers} parallel workers.")

    # Prepare directories
    os.makedirs(args.output_dir, exist_ok=True)

    # Compile task arguments list
    tasks = []
    for mod in modulations:
        for snr in snrs:
            for idx in range(args.samples_per_snr):
                tasks.append((
                    mod, 
                    snr, 
                    idx, 
                    args.output_dir, 
                    args.config, 
                    sample_rate, 
                    num_samples
                ))

    # Run multiprocessing pool with tqdm progress tracking
    metadata_list = []
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        # Use imap_unordered for fast processing and wrap in tqdm
        for result in tqdm(pool.imap_unordered(worker_generate_sample, tasks), total=len(tasks), desc="Generating signals"):
            metadata_list.append(result)

    # Save metadata.csv
    csv_path = os.path.join(args.output_dir, "metadata.csv")
    csv_columns = ["file_path", "modulation", "snr_db", "sample_index", "sample_rate", "num_samples"]
    
    # Sort metadata to keep the CSV readable
    metadata_list.sort(key=lambda x: (x["modulation"], x["snr_db"], x["sample_index"]))

    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()
        for row in metadata_list:
            writer.writerow(row)

    print(f"\nDataset generation completed successfully.")
    print(f"Metadata saved to: {os.path.abspath(csv_path)}")

if __name__ == "__main__":
    main()
