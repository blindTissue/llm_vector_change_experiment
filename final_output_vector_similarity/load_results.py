"""
Helper script to load and explore saved experiment results.
"""
import numpy as np
import json
from pathlib import Path


def load_results(results_dir):
    """
    Load saved experiment results.
    
    Args:
        results_dir: Path to results directory (can be string or Path)
                     e.g., "results/wikitext_results_20251031_215803" (relative to script dir)
                     or Path("results/wikitext_results_20251031_215803")
    
    Returns:
        dict: Dictionary containing all data and metadata
    """
    results_dir = Path(results_dir)
    
    # If relative path, assume it's relative to script directory
    if not results_dir.is_absolute():
        script_dir = Path(__file__).parent
        results_dir = script_dir / results_dir
    
    # Load metadata
    with open(results_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load numpy data
    data = np.load(results_dir / "results.npz")
    
    # Extract arrays
    results = {
        "metadata": metadata,
        "cos_indices": data["cos_indices"],           # Shape: [num_chunks, 1, seq_len, topk]
        "cos_values": data["cos_values"],             # Shape: [num_chunks, 1, seq_len, topk]
        "logit_indices": data["logit_indices"],       # Shape: [num_chunks, 1, seq_len, topk]
        "logit_values": data["logit_values"],         # Shape: [num_chunks, 1, seq_len, topk]
        "intersection_counts": data["intersection_counts"],  # Shape: [num_chunks, seq_len]
    }
    
    return results


def print_summary(results):
    """Print a summary of the loaded results."""
    metadata = results["metadata"]
    
    print("=" * 60)
    print("Experiment Summary")
    print("=" * 60)
    print(f"Model: {metadata['model_name']}")
    print(f"Device: {metadata['device']}")
    print(f"Number of chunks: {metadata['num_chunks']}")
    print(f"Chunk size: {metadata['chunk_size']}")
    print(f"Top-k: {metadata['topk']}")
    print(f"Timestamp: {metadata['timestamp']}")
    print()
    
    print("Data Shapes:")
    print(f"  Cosine indices: {results['cos_indices'].shape}")
    print(f"  Cosine values: {results['cos_values'].shape}")
    print(f"  Logit indices: {results['logit_indices'].shape}")
    print(f"  Logit values: {results['logit_values'].shape}")
    print(f"  Intersection counts: {results['intersection_counts'].shape}")
    print("=" * 60)


if __name__ == "__main__":

    results_dir = "results/wikitext_results_20251031_215803"
    
    
    # Load results
    results = load_results(results_dir)
    
    # Print summary
    print_summary(results)
    
    # Example: Access specific data
    print("\nExample: Accessing data for chunk 0, position 0")
    print("-" * 60)
    chunk_idx = 0
    pos_idx = 0
    
    top_cos_indices = results["cos_indices"][chunk_idx, 0, pos_idx, :]
    top_cos_values = results["cos_values"][chunk_idx, 0, pos_idx, :]
    top_logit_indices = results["logit_indices"][chunk_idx, 0, pos_idx, :]
    top_logit_values = results["logit_values"][chunk_idx, 0, pos_idx, :]
    intersection_count = results["intersection_counts"][chunk_idx, pos_idx]
    
    print(f"Top 10 cosine indices: {top_cos_indices}")
    print(f"Top 10 cosine values: {top_cos_values}")
    print(f"Top 10 logit indices: {top_logit_indices}")
    print(f"Top 10 logit values: {top_logit_values}")
    print(f"Intersection count: {intersection_count}")
    
    print("\nExample: Aggregate statistics")
    print("-" * 60)
    avg_intersection_per_chunk = results["intersection_counts"].mean(axis=1)
    print(f"Average intersection count per chunk (first 5): {avg_intersection_per_chunk[:5]}")
    
    total_chunks = results["cos_indices"].shape[0]
    seq_len = results["cos_indices"].shape[2]
    print(f"\nYou can now analyze {total_chunks} chunks with {seq_len} positions each!")

