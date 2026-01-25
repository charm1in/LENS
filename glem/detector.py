import torch
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from .probe import SpectralProbe
from .miner import EvidenceMiner, Proposal
from .utils import (
    MetricAggregator,
    ProcessingStats,
    logger,
    tensor_to_numpy
)


class GLEMPipeline:
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        self.cfg = config
        self.device = torch.device(device)
        self.stats = ProcessingStats()

        self._init_modules()

    def _init_modules(self):
        logger.info("Initializing GLEM Pipeline modules...")

        # 1. Probe
        self.probe = SpectralProbe(
            model_path=self.cfg['repo_id'],
            device=self.device,
            dtype=torch.float16 if self.cfg.get('fp16', True) else torch.float32,
            local_files_only=self.cfg.get('offline_mode', False)
        )

        # 2. Miner
        self.miner = EvidenceMiner(
            scales=self.cfg.get('crop_ratios', [0.9, 0.7]),
            top_k=self.cfg.get('top_k', 2),
            stride=self.cfg.get('stride', 16),
            refine_steps=1 if self.cfg.get('refine', False) else 0
        )

        # 3. Aggregators
        self.aggregator = MetricAggregator(
            mode="softmin",
            temp=self.cfg.get('softmin_temp', 0.05)
        )

    def _preprocess(self, image: np.ndarray, roi: Optional[Proposal] = None) -> torch.Tensor:
        target_size = self.cfg['img_size']

        if roi is not None:
            # Crop logic
            crop = image[roi.y: roi.y + roi.h, roi.x: roi.x + roi.w]
            # Resize patch
            img_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        else:
            # Full image resize
            img_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

        # To Tensor: HWC -> CHW, Normalize 0-1
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).contiguous()
        return tensor.float().div(255.0)

    def process_single(self, image: np.ndarray, seed_offset: int = 0) -> Dict[str, float]:
        """
        Main inference entry point for a single image.
        """
        # Determine unique seed for this sample
        sample_seed = self.cfg['seed'] + seed_offset

        # --- Stage 1: Global Gate ---
        full_tensor = self._preprocess(image).unsqueeze(0).to(self.device, self.probe.dtype)

        global_score = self.probe.process_batch(
            full_tensor,
            hf_cutoff=0.0,  # Global check is usually full-spectrum
            seed=sample_seed
        ).item()

        gate_passed = False
        final_score = -global_score  # Default: higher reconstruction error => lower score (more real)
        # We define: Higher Score = More Fake.


        threshold = self.cfg['gate_threshold']

        # Logic: If Error < Threshold, it is VERY Real. Stop.
        if global_score < threshold:
            # Early Exit
            return {
                "score": global_score,  # Direct Error
                "global": global_score,
                "local": 0.0,
                "stage": 1,
                "gate": False
            }

        # --- Stage 2: Local Evidence Mining ---
        gate_passed = True

        proposals = self.miner.mine_proposals(image)

        rand_k = self.cfg.get('rand_patches', 0)
        if rand_k > 0:
            rand_props = self.miner.get_random_proposals(image, rand_k, sample_seed)
            proposals.extend(rand_props)

        if not proposals:
            # Fallback if mining fails
            return {
                "score": global_score,
                "global": global_score,
                "local": 0.0,
                "stage": 1,
                "gate": True
            }

        patch_tensors = []
        for p in proposals:
            pt = self._preprocess(image, roi=p)
            patch_tensors.append(pt)

        batch_input = torch.stack(patch_tensors).to(self.device, self.probe.dtype)

        # Split into mini-batches to save VRAM
        bs = self.cfg.get('batch_size', 4)
        local_errors = []

        for i in range(0, len(batch_input), bs):
            chunk = batch_input[i: i + bs]
            chunk_errs = self.probe.process_batch(
                chunk,
                hf_cutoff=self.cfg.get('hf_cutoff', 0.3),
                seed=sample_seed
            )
            local_errors.extend(chunk_errs.tolist())


        peak_error = np.max(local_errors)  # Hard Max

        # Score is purely local evidence if triggered
        final_score = peak_error

        return {
            "score": final_score,
            "global": global_score,
            "local": peak_error,
            "stage": 2,
            "gate": True
        }