import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from typing import Optional, Tuple, Dict
import logging

from .utils import logger


class SpectralProbe(nn.Module):
    def __init__(
            self,
            model_path: str,
            device: torch.device,
            dtype: torch.dtype = torch.float16,
            cache_dir: Optional[str] = None,
            local_files_only: bool = False
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.vae = self._initialize_vae(cache_dir, local_files_only)
        self.vae.eval()
        self._freeze()

        # Precomputed spectral weights buffer
        self.register_buffer("spectral_filter", None)

    def _initialize_vae(self, cache_dir: Optional[str], local: bool) -> AutoencoderKL:
        logger.info(f"Initializing VAE Probe from {self.model_path}")
        try:
            # Attempt loading as subfolder first (standard diffusers format)
            return AutoencoderKL.from_pretrained(
                self.model_path,
                subfolder="vae",
                cache_dir=cache_dir,
                local_files_only=local
            ).to(self.device, self.dtype)
        except Exception as e_sub:
            logger.debug(f"Subfolder load failed: {e_sub}. Trying direct load.")
            try:
                # Fallback to direct repository load
                return AutoencoderKL.from_pretrained(
                    self.model_path,
                    cache_dir=cache_dir,
                    local_files_only=local
                ).to(self.device, self.dtype)
            except Exception as e_direct:
                logger.critical(f"Critical failure loading VAE: {e_direct}")
                raise RuntimeError("VAE Probe initialization failed.")

    def _freeze(self):
        for param in self.vae.parameters():
            param.requires_grad = False

    def _build_filter(self, size: int, r0: float) -> torch.Tensor:
        # Generates a high-pass frequency mask
        if self.spectral_filter is not None:
            if self.spectral_filter.shape[-1] == size:
                return self.spectral_filter

        coords = torch.linspace(-0.5, 0.5, size, device=self.device)
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        r = torch.sqrt(x ** 2 + y ** 2)

        mask = torch.ones_like(r)
        if r0 > 0:
            mask = (r - r0).clamp(min=0.0) / (0.5 - r0 + 1e-7)
            mask = mask.clamp(0.0, 1.0)

        self.spectral_filter = mask
        return mask

    @torch.no_grad()
    def forward_reconstruction(
            self,
            images: torch.Tensor,
            generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        # Input validation
        if images.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {images.shape}")

        x = images * 2.0 - 1.0
        posterior = self.vae.encode(x).latent_dist

        if generator is not None:
            latents = posterior.sample(generator=generator)
        else:
            latents = posterior.sample()

        recon = self.vae.decode(latents).sample
        recon = (recon / 2.0 + 0.5).clamp(0.0, 1.0)
        return recon

    def compute_spectral_consistency(
            self,
            original: torch.Tensor,
            reconstructed: torch.Tensor,
            hf_cutoff: float = 0.0
    ) -> torch.Tensor:
        """
        Computes Log-Spectrum Distance (LSD) between original and recon.
        Returns: tensor [B]
        """
        assert original.shape == reconstructed.shape
        b, c, h, w = original.shape

        # Color conversion weights (RGB -> Gray)
        rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=self.device).view(1, 3, 1, 1)

        gray_orig = (original * rgb_weights).sum(dim=1)
        gray_recon = (reconstructed * rgb_weights).sum(dim=1)

        # FFT Analysis
        fft_orig = torch.fft.fft2(gray_orig)
        fft_recon = torch.fft.fft2(gray_recon)

        # Log Magnitude with numerical stability epsilon
        log_mag_orig = torch.log(torch.abs(torch.fft.fftshift(fft_orig)) + 1e-6)
        log_mag_recon = torch.log(torch.abs(torch.fft.fftshift(fft_recon)) + 1e-6)

        # Weighted difference
        weight_map = self._build_filter(h, hf_cutoff)
        diff_sq = (log_mag_orig - log_mag_recon).pow(2)
        weighted_diff = diff_sq * weight_map

        # Mean over spatial dims
        scores = weighted_diff.mean(dim=(1, 2)) / (weight_map.mean() + 1e-8)
        return torch.sqrt(scores)

    def process_batch(
            self,
            images: torch.Tensor,
            hf_cutoff: float,
            seed: int
    ) -> torch.Tensor:
        gen = torch.Generator(device=self.device).manual_seed(seed)
        recon = self.forward_reconstruction(images, generator=gen)
        return self.compute_spectral_consistency(images, recon, hf_cutoff)