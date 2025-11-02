from datasets import load_dataset, load_from_disk
from pathlib import Path
from transformers import AutoTokenizer
import torch
import os
import numpy as np
import json
from datetime import datetime
from similarity_util_rewrite import *
from tqdm import tqdm


def chunk_with_bos(tensor, chunk_size, tokenizer):
    """
    Chunk a tokenized tensor and prepend BOS token to all chunks except the first.
    All chunks will be exactly chunk_size.
    If tokenizer doesn't have BOS token, chunks are created without BOS prepending.
    
    Args:
        tensor: torch.Tensor of shape [1, seq_len] - tokenized tensor
        chunk_size: int - size of each chunk (all chunks will be this size)
        tokenizer: transformers tokenizer - to get BOS and PAD token IDs
    
    Returns:
        list of torch.Tensors - chunks with BOS prepended (except first) if BOS exists,
                                otherwise chunks without BOS, each of shape [chunk_size]
    """
    # Get token IDs
    bos_token_id = tokenizer.bos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    has_bos = bos_token_id is not None
    
    # Remove batch dimension to get [seq_len]
    tokens = tensor.squeeze(0)  # Shape: [seq_len]
    seq_len = tokens.shape[0]
    
    chunks = []
    i = 0
    
    if has_bos:
        # Original behavior: prepend BOS to subsequent chunks
        # First chunk: take exactly chunk_size tokens (already has BOS)
        if seq_len > 0:
            first_chunk = tokens[i:i + chunk_size]
            if len(first_chunk) < chunk_size:
                # Pad if first chunk is shorter (shouldn't happen but handle it)
                padding = torch.full((chunk_size - len(first_chunk),), pad_token_id, dtype=first_chunk.dtype)
                first_chunk = torch.cat([first_chunk, padding])
            chunks.append(first_chunk)
            i += chunk_size
        
        # Subsequent chunks: take (chunk_size - 1) tokens + 1 BOS = chunk_size total
        while i < seq_len:
            # Take chunk_size - 1 tokens from the sequence (since we'll prepend BOS)
            remaining = seq_len - i
            if remaining < chunk_size - 1:
                # Not enough tokens for full chunk, pad it
                chunk_tokens = tokens[i:]
                bos_tensor = torch.tensor([bos_token_id], dtype=tokens.dtype)
                chunk = torch.cat([bos_tensor, chunk_tokens])
                
                # Pad to chunk_size
                padding_needed = chunk_size - len(chunk)
                if padding_needed > 0:
                    padding = torch.full((padding_needed,), pad_token_id, dtype=tokens.dtype)
                    chunk = torch.cat([chunk, padding])
            else:
                # Normal case: take (chunk_size - 1) tokens + prepend BOS
                chunk_tokens = tokens[i:i + chunk_size - 1]
                bos_tensor = torch.tensor([bos_token_id], dtype=tokens.dtype)
                chunk = torch.cat([bos_tensor, chunk_tokens])
            
            chunks.append(chunk)
            i += chunk_size - 1  # Move forward by (chunk_size - 1) since we consumed that many
    else:
        # No BOS token: create chunks of exactly chunk_size without BOS prepending
        while i < seq_len:
            remaining = seq_len - i
            if remaining < chunk_size:
                # Not enough tokens for full chunk, pad it
                chunk_tokens = tokens[i:]
                padding_needed = chunk_size - len(chunk_tokens)
                if padding_needed > 0:
                    padding = torch.full((padding_needed,), pad_token_id, dtype=tokens.dtype)
                    chunk = torch.cat([chunk_tokens, padding])
                else:
                    chunk = chunk_tokens
            else:
                # Normal case: take exactly chunk_size tokens
                chunk = tokens[i:i + chunk_size]
            
            chunks.append(chunk)
            i += chunk_size
    
    return chunks

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    save_name = script_dir / "wikitext"
    model_name = "Qwen/Qwen3-4B"
    device = "mps"
    torch.mps.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if save_name.exists():
        print(f"Dataset already exists at {save_name}, loading from disk...")
        ds = load_from_disk(str(save_name))
    else:
        print(f"Downloading dataset and saving to {save_name}...")
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        ds.save_to_disk(str(save_name))
    combined  = " ".join([ds['train'][i]['text'] for i in range(3000)]) # 3000 for this experiment
    combined_tokenized = tokenizer(combined, return_tensors="pt").input_ids
    chunked_items = chunk_with_bos(combined_tokenized, 1000, tokenizer)
    foa = finalOutputAnalysis(model_name, device)

    cos_inds = []
    cos_vals = []
    logit_inds = []
    logit_vals = []
    intersection_counts = []
    for tensor in tqdm(chunked_items):
        foa.pass_input(tensor, topk=10)
        foa.find_intersect_numbers() 
        cos_inds.append(foa.max_cosine_indices.cpu())
        cos_vals.append(foa.max_cosine_values.cpu())
        logit_inds.append(foa.max_logit_indices.cpu())
        logit_vals.append(foa.max_logit_values.cpu())  
        intersection_counts.append(foa.intersection_count)
    
    # Save results
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Create timestamp for unique results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize model name for directory name (replace / with _)
    model_name_safe = model_name.replace("/", "_")
    output_dir = results_dir / f"wikitext_results_{model_name_safe}_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    
    # Convert lists of tensors to numpy arrays and save
    print(f"\nSaving results to {output_dir}...")
    
    # Stack tensors into single arrays: [num_chunks, batch_size, seq_len, topk]
    cos_inds_array = torch.stack(cos_inds).numpy()
    cos_vals_array = torch.stack(cos_vals).numpy()
    logit_inds_array = torch.stack(logit_inds).numpy()
    logit_vals_array = torch.stack(logit_vals).numpy()
    
    # Convert intersection_counts from list of lists to 2D array: [num_chunks, seq_len]
    intersection_counts_array = np.array(intersection_counts)
    
    # Save as numpy compressed format (.npz)
    np.savez_compressed(
        output_dir / "results.npz",
        cos_indices=cos_inds_array,
        cos_values=cos_vals_array,
        logit_indices=logit_inds_array,
        logit_values=logit_vals_array,
        intersection_counts=intersection_counts_array
    )
    
    # Save metadata as JSON
    metadata = {
        "model_name": model_name,
        "device": device,
        "num_chunks": len(chunked_items),
        "chunk_size": 1000,
        "topk": 10,
        "num_text_samples": 3000,
        "timestamp": timestamp,
        "shapes": {
            "cos_indices": list(cos_inds_array.shape),
            "cos_values": list(cos_vals_array.shape),
            "logit_indices": list(logit_inds_array.shape),
            "logit_values": list(logit_vals_array.shape),
            "intersection_counts": list(intersection_counts_array.shape),
        }
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Results saved successfully!")
    print(f"  - Data: {output_dir / 'results.npz'}")
    print(f"  - Metadata: {output_dir / 'metadata.json'}")


