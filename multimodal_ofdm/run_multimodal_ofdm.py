from __future__ import annotations
import os, argparse, json, datetime, math
import numpy as np
from PIL import Image
import requests # NEW: Import requests library for API calls

# MODIFIED: Import the hamming module directly to access both decoders
from . import hamming74 as ham
from .config import ExperimentConfig
from .application import (
    serialize_content, AppHeader, deserialize_content,
    _build_seg_ids_and_palette, _suppress_white_boundaries, _load_image,
    _build_text_codebook, _POPCNT_9
)
from .utils import (
    bytes_to_bits, bits_to_bytes, append_crc32,
    verify_and_strip_crc32, block_deinterleave,
    derepeat_bits_majority,
    permute_bytes, unpermute_bytes, derive_modality_seed,
)
from .ofdm import assemble_grid
from .channel import rayleigh_ofdm, awgn_ofdm
from .metrics import psnr, miou_from_ids, f1_binary_edge, ssim
from .presets import POWER_PRESETS

# ==============================================================================
# NEW: Function to correct text using Gemini API
# ==============================================================================
def correct_text_with_llm(text_to_correct: str, cfg_llm) -> str:
    """
    Sends text to the Gemini API for error correction and returns the corrected text.
    If the API call fails or the key is missing, it returns the original text.
    """
    if not cfg_llm.api_key or cfg_llm.api_key == "YOUR_GEMINI_API_KEY":
        print("[Warning] Gemini API key is not set. Skipping text correction.")
        return text_to_correct

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg_llm.model_name}:generateContent?key={cfg_llm.api_key}"
    
    prompt = (
        "The following text was transmitted over a noisy communication channel and may contain character errors. "
        "Please correct any errors to make it a coherent and grammatically correct English paragraph. "
        "Do not add any new information or change the original meaning.\n\n"
        f"Corrupted text:\n\"{text_to_correct}\"\n\n"
        "Corrected text:"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2048,
        }
    }

    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status() # Raise an exception for bad status codes
        
        data = response.json()
        corrected_text = data['candidates'][0]['content']['parts'][0]['text']
        print("[Info] Text successfully corrected by LLM.")
        return corrected_text.strip()

    except requests.exceptions.RequestException as e:
        print(f"[Error] LLM API call failed: {e}")
        return text_to_correct
    except (KeyError, IndexError) as e:
        print(f"[Error] Failed to parse LLM response: {e}")
        return text_to_correct


MODS = ["text","edge","depth","segmentation"]
# (The rest of the helper functions remain the same)
# ...

def main():
    ap = argparse.ArgumentParser()
    # (All previous arguments)
    # ...
    # MODIFIED: Removed the --llm-correct argument
    ap.add_argument("--seed", type=int, default=None, help="Seed for the channel random number generator")
    
    args = ap.parse_args()

    # --- config ---
    cfg = ExperimentConfig()
    # (All previous config setups)
    # ...
    # MODIFIED: Removed the logic that sets the flag from args. The flag is now controlled by config.py
    
    # (The rest of the main function up to deserialization remains the same)
    # ...
    
    # --- decode per modality ---
    # ...
    for m in MODS:
        # (All the decoding logic for each modality)
        # ...
        
        text_str, img_arr = deserialize_content(rx_hdr, payload_clean, app_cfg=cfg.app)
        
        # MODIFIED: The check now directly uses the flag from the config object
        if m == "text" and cfg.llm.correction_enabled:
            print("[Info] Applying LLM-based error correction to text...")
            text_str = correct_text_with_llm(text_str, cfg.llm)

        # (The rest of the result storage and image handling logic)
        # ...
    
    # (The rest of the script for output, metrics, and report remains the same)
    # ...

if __name__ == "__main__":
    main()