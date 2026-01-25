import argparse
import time
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

# Internal imports
from glem.detector import GLEMPipeline
from glem.utils import (
    setup_logger,
    logger,
    deterministic_setup,
    read_image_safe
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="GLEM: Global-Local Evidence Mining Benchmark Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing datasets")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save CSV results")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per class per dataset")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument("--fp32", action="store_true", help="Force float32 precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # VAE Config
    parser.add_argument("--repo_id", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--offline", action="store_true", help="Do not try to connect to HF Hub")

    # Detection Hyperparameters
    parser.add_argument("--gate_threshold", type=float, default=0.801,
                        help="Global consistency threshold for early exit")
    parser.add_argument("--hf_cutoff", type=float, default=0.3, help="Spectral high-frequency filtering radius")

    # Mining Hyperparameters
    parser.add_argument("--scales", nargs="+", type=float, default=[0.9, 0.8, 0.7],
                        help="Crop ratios for sliding window")
    parser.add_argument("--top_k", type=int, default=3, help="Number of salient patches to verify")
    parser.add_argument("--rand_patches", type=int, default=2, help="Number of random patches to verify")
    parser.add_argument("--stride", type=int, default=16, help="Stride for sliding window search")
    parser.add_argument("--no_refine", action="store_true", help="Disable local search refinement")

    return parser.parse_args()


def scan_dataset(root: Path, limit: int = None) -> Tuple[List[Path], List[Path]]:
    """Recursively find images in 'nature' and 'ai' subdirectories."""
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif'}

    real_imgs = []
    fake_imgs = []

    # Heuristic: Look for '0_real'/'1_fake' or 'nature'/'ai' folders
    for r, d, f in os.walk(root):
        for file in f:
            if Path(file).suffix.lower() in exts:
                full_path = Path(r) / file
                parent = full_path.parent.name.lower()

                if parent in ['nature', '0_real', 'real']:
                    real_imgs.append(full_path)
                elif parent in ['ai', '1_fake', 'fake']:
                    fake_imgs.append(full_path)

    # Sort for determinism
    real_imgs.sort()
    fake_imgs.sort()

    if limit:
        real_imgs = real_imgs[:limit]
        fake_imgs = fake_imgs[:limit]

    return real_imgs, fake_imgs


import os


def main():
    args = parse_arguments()

    # 1. Setup
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_file=str(out_dir / "benchmark.log"))
    deterministic_setup(args.seed)

    logger.info("Configuration:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # 2. Initialize Pipeline
    config = {
        'repo_id': args.repo_id,
        'fp16': not args.fp32,
        'offline_mode': args.offline,
        'img_size': 512,  # Fixed for standard LDM
        'gate_threshold': args.gate_threshold,
        'hf_cutoff': args.hf_cutoff,
        'crop_ratios': args.scales,
        'top_k': args.top_k,
        'rand_patches': args.rand_patches,
        'stride': args.stride,
        'refine': not args.no_refine,
        'seed': args.seed
    }

    try:
        detector = GLEMPipeline(config, args.device)
    except Exception as e:
        logger.critical(f"Failed to initialize detector: {e}")
        return

    # 3. Dataset Discovery
    data_root = Path(args.data_root)
    datasets = [d for d in data_root.iterdir() if d.is_dir()]
    logger.info(f"Found {len(datasets)} potential datasets in {data_root}")

    # 4. Evaluation Loop
    overall_metrics = []

    for ds_path in datasets:
        ds_name = ds_path.name
        logger.info(f"--- Processing {ds_name} ---")

        reals, fakes = scan_dataset(ds_path, args.max_samples)
        if not reals and not fakes:
            logger.warning(f"Skipping {ds_name}: No images found.")
            continue

        logger.info(f"Loaded: Real={len(reals)}, Fake={len(fakes)}")

        # Create work queue
        queue = [(p, 0) for p in reals] + [(p, 1) for p in fakes]

        # Results container
        records = []

        t_start_ds = time.time()

        for img_path, label in tqdm(queue, desc=f"Eval {ds_name}"):
            # Read
            img = read_image_safe(img_path)
            if img is None:
                continue

            # Infer
            t0 = time.time()
            try:
                # Use path hash for consistent patch randomization
                path_seed = hash(str(img_path)) % 100000
                res = detector.process_single(img, seed_offset=path_seed)
                inference_time = time.time() - t0

                record = {
                    "path": str(img_path),
                    "label": label,
                    "score": res['score'],
                    "global_err": res['global'],
                    "local_err": res['local'],
                    "gate_passed": res['gate'],
                    "time": inference_time
                }
                records.append(record)

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

        # Save Dataset Results
        if not records:
            continue

        df = pd.DataFrame(records)
        csv_path = out_dir / f"{ds_name}_details.csv"
        df.to_csv(csv_path, index=False)

        # Calculate AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(df['label'], df['score'])
        except:
            auc = 0.0

        # Calc Efficiency
        avg_time = df['time'].mean()
        gate_pass_rate = df['gate_passed'].mean()

        logger.info(
            f"Dataset: {ds_name} | AUC: {auc:.4f} | Avg Time: {avg_time:.3f}s | Gate Pass: {gate_pass_rate:.1%}")

        overall_metrics.append({
            "dataset": ds_name,
            "auc": auc,
            "n_samples": len(df),
            "avg_time": avg_time,
            "gate_rate": gate_pass_rate
        })

    # 5. Final Summary
    if overall_metrics:
        summary_df = pd.DataFrame(overall_metrics)
        summary_path = out_dir / "final_summary_report.csv"
        summary_df.to_csv(summary_path, index=False)
        print("\n" + "=" * 40)
        print(summary_df)
        print("=" * 40)
        logger.info(f"Benchmarks completed. Results saved to {out_dir}")
    else:
        logger.warning("No valid results generated.")


if __name__ == "__main__":
    main()