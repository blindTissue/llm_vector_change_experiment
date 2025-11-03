import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_graph_argmax_token(data_path, save_path=None):
    data = np.load(data_path)
    best_cosine_values = data['cos_values'].squeeze(1)[:,:,0]
    best_logit_values = data['logit_values'].squeeze(1)[:,:,0]

    mean_cosine_values = np.mean(best_cosine_values, axis=0)
    mean_logit_values = np.mean(best_logit_values, axis=0)

    variance_cosine_values = np.var(best_cosine_values, axis=0)
    variance_logit_values = np.var(best_logit_values, axis=0)

    timestamps = np.arange(best_logit_values.shape[1])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot mean values
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel("Mean Cosine Similarity", color='tab:blue')
    line1 = ax1.plot(timestamps, mean_cosine_values, color='tab:blue', linewidth=1.5, label="Mean Cosine Similarity")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    ax1_2 = ax1.twinx()
    ax1_2.set_ylabel("Mean Logit Values", color='tab:orange')
    line2 = ax1_2.plot(timestamps, mean_logit_values, color='tab:orange', linewidth=1.5, label="Mean Logit Values")
    ax1_2.tick_params(axis='y', labelcolor='tab:orange')

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.set_title('Mean Values Over Time')

    # Plot variance values
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel("Variance Cosine Similarity", color='tab:blue')
    line3 = ax2.plot(timestamps, variance_cosine_values, color='tab:blue', linewidth=1.5, label="Variance Cosine Similarity")
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.grid(True, alpha=0.3)

    ax2_2 = ax2.twinx()
    ax2_2.set_ylabel("Variance Logit Values", color='tab:orange')
    line4 = ax2_2.plot(timestamps, variance_logit_values, color='tab:orange', linewidth=1.5, label="Variance Logit Values")
    ax2_2.tick_params(axis='y', labelcolor='tab:orange')

    # Combine legends
    lines = line3 + line4
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    ax2.set_title('Variance Values Over Time')

    plt.tight_layout()
    
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output_graphs"
    output_dir.mkdir(exist_ok=True)
    # Save to designated subdirectory
    if save_path is None:
        data_path_obj = Path(data_path)
        save_path = output_dir / f"{data_path_obj.parent.name}_argmax_token.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to: {save_path}")
    #plt.show()

def compare_probability_vs_logit(data_path, save_path=None):
    """
    Compare the probability values vs logit values for the highest probability token.
    Creates a scatter plot showing the relationship between logits and probabilities.
    """
    data = np.load(data_path)

    # Extract highest probability values (index 0 of topk dimension)
    # Shape: [chunks, batch, timesteps, topk] -> [chunks, timesteps]
    best_prob_values = data['prob_values'].squeeze(1)[:,:,0]
    best_logit_values = data['logit_values'].squeeze(1)[:,:,0]

    # Flatten arrays for scatter plot
    prob_flat = best_prob_values.flatten()
    logit_flat = best_logit_values.flatten()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax1.scatter(logit_flat, prob_flat, alpha=0.3, s=1)
    ax1.set_xlabel('Logit Value')
    ax1.set_ylabel('Probability')
    ax1.set_title('Probability vs Logit (Highest Probability Token)')
    ax1.grid(True, alpha=0.3)

    # Add theoretical softmax curve for reference
    logit_range = np.linspace(logit_flat.min(), logit_flat.max(), 1000)
    # For visualization: assuming other logits are at mean of observed logits
    mean_logit = np.mean(logit_flat)
    # Simple approximation: prob ≈ exp(logit) / (exp(logit) + constant)
    # This is simplified since we don't have all logit values
    ax1.plot(logit_range, 1 / (1 + np.exp(mean_logit - logit_range)),
             'r-', linewidth=2, label='Theoretical (simplified)', alpha=0.7)
    ax1.legend()

    # 2D histogram/density plot
    hist = ax2.hist2d(logit_flat, prob_flat, bins=100, cmap='Blues', cmin=1)
    ax2.set_xlabel('Logit Value')
    ax2.set_ylabel('Probability')
    ax2.set_title('Density: Probability vs Logit')
    plt.colorbar(hist[3], ax=ax2, label='Count')

    plt.tight_layout()

    # Save figure
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output_graphs"
    output_dir.mkdir(exist_ok=True)

    if save_path is None:
        data_path_obj = Path(data_path)
        save_path = output_dir / f"{data_path_obj.parent.name}_prob_vs_logit.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Probability vs Logit graph saved to: {save_path}")

    # Print some statistics
    print(f"\nStatistics for {Path(data_path).parent.name}:")
    print(f"  Logit range: [{logit_flat.min():.3f}, {logit_flat.max():.3f}]")
    print(f"  Probability range: [{prob_flat.min():.3f}, {prob_flat.max():.3f}]")
    print(f"  Mean logit: {logit_flat.mean():.3f}")
    print(f"  Mean probability: {prob_flat.mean():.3f}")

    plt.close()

def plot_max_probability_over_time(data_path, save_path=None):
    """
    Plot how the maximum probability changes across positions/timesteps.
    Shows mean and variance of max probability over time.
    """
    data = np.load(data_path)

    # Extract highest probability values (index 0 of topk dimension)
    # Shape: [chunks, batch, timesteps, topk] -> [chunks, timesteps]
    max_prob_values = data['prob_values'].squeeze(1)[:,:,0]

    # Calculate statistics across chunks
    mean_prob = np.mean(max_prob_values, axis=0)
    std_prob = np.std(max_prob_values, axis=0)
    variance_prob = np.var(max_prob_values, axis=0)

    timestamps = np.arange(max_prob_values.shape[1])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot mean with standard deviation band
    ax1.plot(timestamps, mean_prob, color='tab:blue', linewidth=2, label='Mean Max Probability')
    ax1.fill_between(timestamps, mean_prob - std_prob, mean_prob + std_prob,
                      color='tab:blue', alpha=0.2, label='±1 Std Dev')
    ax1.set_ylabel('Max Probability', fontsize=11)
    ax1.set_title('Maximum Probability Over Time (Position)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_ylim([0, 1])  # Probabilities are between 0 and 1

    # Plot variance
    ax2.plot(timestamps, variance_prob, color='tab:orange', linewidth=2)
    ax2.set_xlabel('Position/Timestep', fontsize=11)
    ax2.set_ylabel('Variance of Max Probability', fontsize=11)
    ax2.set_title('Variance of Maximum Probability Over Time', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output_graphs"
    output_dir.mkdir(exist_ok=True)

    if save_path is None:
        data_path_obj = Path(data_path)
        save_path = output_dir / f"{data_path_obj.parent.name}_max_prob_over_time.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Max probability over time graph saved to: {save_path}")

    # Print some statistics
    print(f"\nMax Probability Statistics for {Path(data_path).parent.name}:")
    print(f"  Overall mean: {mean_prob.mean():.4f}")
    print(f"  Overall std: {std_prob.mean():.4f}")
    print(f"  Min mean probability: {mean_prob.min():.4f} at position {mean_prob.argmin()}")
    print(f"  Max mean probability: {mean_prob.max():.4f} at position {mean_prob.argmax()}")

    plt.close()

def plot_combined_metrics_over_time(data_path, save_path=None):
    """
    Plot cosine similarity, logit values, and probabilities together over time.
    Shows how all three metrics evolve across positions/timesteps.
    """
    data = np.load(data_path)

    # Extract highest values (index 0 of topk dimension)
    # Shape: [chunks, batch, timesteps, topk] -> [chunks, timesteps]
    max_cos_values = data['cos_values'].squeeze(1)[:,:,0]
    max_logit_values = data['logit_values'].squeeze(1)[:,:,0]
    max_prob_values = data['prob_values'].squeeze(1)[:,:,0]

    # Calculate mean across chunks
    mean_cos = np.mean(max_cos_values, axis=0)
    mean_logit = np.mean(max_logit_values, axis=0)
    mean_prob = np.mean(max_prob_values, axis=0)

    # Calculate std for shading
    std_cos = np.std(max_cos_values, axis=0)
    std_logit = np.std(max_logit_values, axis=0)
    std_prob = np.std(max_prob_values, axis=0)

    timestamps = np.arange(max_prob_values.shape[1])

    # Create figure with three subplots sharing x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Cosine Similarity
    ax1.plot(timestamps, mean_cos, color='tab:blue', linewidth=2, label='Mean Cosine Similarity')
    ax1.fill_between(timestamps, mean_cos - std_cos, mean_cos + std_cos,
                      color='tab:blue', alpha=0.2, label='±1 Std Dev')
    ax1.set_ylabel('Cosine Similarity', fontsize=11)
    ax1.set_title('Combined Metrics Over Time (Position)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    # Plot 2: Logit Values
    ax2.plot(timestamps, mean_logit, color='tab:orange', linewidth=2, label='Mean Logit Value')
    ax2.fill_between(timestamps, mean_logit - std_logit, mean_logit + std_logit,
                      color='tab:orange', alpha=0.2, label='±1 Std Dev')
    ax2.set_ylabel('Logit Value', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    # Plot 3: Probability
    ax3.plot(timestamps, mean_prob, color='tab:green', linewidth=2, label='Mean Probability')
    ax3.fill_between(timestamps, mean_prob - std_prob, mean_prob + std_prob,
                      color='tab:green', alpha=0.2, label='±1 Std Dev')
    ax3.set_ylabel('Probability', fontsize=11)
    ax3.set_xlabel('Position/Timestep', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    ax3.set_ylim([0, 1])

    plt.tight_layout()

    # Save figure
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output_graphs"
    output_dir.mkdir(exist_ok=True)

    if save_path is None:
        data_path_obj = Path(data_path)
        save_path = output_dir / f"{data_path_obj.parent.name}_combined_metrics.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Combined metrics graph saved to: {save_path}")

    # Print correlation statistics
    print(f"\nCombined Metrics Statistics for {Path(data_path).parent.name}:")
    print(f"  Cosine Similarity - Mean: {mean_cos.mean():.4f}, Std: {mean_cos.std():.4f}")
    print(f"  Logit Value - Mean: {mean_logit.mean():.4f}, Std: {mean_logit.std():.4f}")
    print(f"  Probability - Mean: {mean_prob.mean():.4f}, Std: {mean_prob.std():.4f}")

    # Calculate correlations between metrics
    from scipy.stats import pearsonr
    corr_cos_logit, _ = pearsonr(mean_cos, mean_logit)
    corr_cos_prob, _ = pearsonr(mean_cos, mean_prob)
    corr_logit_prob, _ = pearsonr(mean_logit, mean_prob)

    print(f"\n  Correlations:")
    print(f"    Cosine Similarity vs Logit: {corr_cos_logit:.4f}")
    print(f"    Cosine Similarity vs Probability: {corr_cos_prob:.4f}")
    print(f"    Logit vs Probability: {corr_logit_prob:.4f}")

    plt.close()

def plot_intersection_analysis(data_path, save_path=None):
    """
    Analyze how often top-k tokens by cosine similarity match top-k tokens by logit.
    Shows agreement between the two ranking methods.
    """
    data = np.load(data_path)

    # intersection_counts shape: [chunks, timesteps]
    intersection_counts = data['intersection_counts']
    topk = data['cos_indices'].shape[-1]  # Get k value

    # Calculate statistics
    mean_intersection = np.mean(intersection_counts, axis=0)
    std_intersection = np.std(intersection_counts, axis=0)

    # Calculate percentage
    mean_intersection_pct = (mean_intersection / topk) * 100
    std_intersection_pct = (std_intersection / topk) * 100

    timestamps = np.arange(intersection_counts.shape[1])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plot 1: Absolute intersection counts
    ax1.plot(timestamps, mean_intersection, color='tab:purple', linewidth=2, label='Mean Intersection')
    ax1.fill_between(timestamps, mean_intersection - std_intersection,
                      mean_intersection + std_intersection,
                      color='tab:purple', alpha=0.2, label='±1 Std Dev')
    ax1.axhline(y=topk, color='r', linestyle='--', linewidth=1, label=f'Max Possible (k={topk})')
    ax1.set_ylabel('Intersection Count', fontsize=11)
    ax1.set_title(f'Top-{topk} Cosine Similarity vs Logit Ranking Agreement', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_ylim([0, topk + 1])

    # Plot 2: Percentage agreement
    ax2.plot(timestamps, mean_intersection_pct, color='tab:purple', linewidth=2, label='Mean Agreement %')
    ax2.fill_between(timestamps, mean_intersection_pct - std_intersection_pct,
                      mean_intersection_pct + std_intersection_pct,
                      color='tab:purple', alpha=0.2, label='±1 Std Dev')
    ax2.axhline(y=100, color='r', linestyle='--', linewidth=1, label='100% Agreement')
    ax2.set_ylabel('Agreement Percentage (%)', fontsize=11)
    ax2.set_xlabel('Position/Timestep', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_ylim([0, 105])

    plt.tight_layout()

    # Save figure
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output_graphs"
    output_dir.mkdir(exist_ok=True)

    if save_path is None:
        data_path_obj = Path(data_path)
        save_path = output_dir / f"{data_path_obj.parent.name}_intersection_analysis.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Intersection analysis graph saved to: {save_path}")

    # Print statistics
    print(f"\nIntersection Analysis for {Path(data_path).parent.name}:")
    print(f"  Mean agreement: {mean_intersection.mean():.2f}/{topk} tokens ({mean_intersection_pct.mean():.1f}%)")
    print(f"  Min agreement: {mean_intersection.min():.2f} at position {mean_intersection.argmin()}")
    print(f"  Max agreement: {mean_intersection.max():.2f} at position {mean_intersection.argmax()}")

    plt.close()

def compare_cosine_vs_probability(data_path, save_path=None):
    """
    Scatter plot comparing cosine similarity values vs probability values.
    Shows if high cosine similarity correlates with high probability.
    """
    data = np.load(data_path)

    # Extract highest values (index 0 of topk dimension)
    max_cos_values = data['cos_values'].squeeze(1)[:,:,0]
    max_prob_values = data['prob_values'].squeeze(1)[:,:,0]

    # Flatten for scatter plot
    cos_flat = max_cos_values.flatten()
    prob_flat = max_prob_values.flatten()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax1.scatter(cos_flat, prob_flat, alpha=0.3, s=1)
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Probability')
    ax1.set_title('Probability vs Cosine Similarity (Highest Prob Token)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # 2D histogram/density plot
    hist = ax2.hist2d(cos_flat, prob_flat, bins=100, cmap='Greens', cmin=1)
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Probability')
    ax2.set_title('Density: Probability vs Cosine Similarity')
    plt.colorbar(hist[3], ax=ax2, label='Count')
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    # Save figure
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output_graphs"
    output_dir.mkdir(exist_ok=True)

    if save_path is None:
        data_path_obj = Path(data_path)
        save_path = output_dir / f"{data_path_obj.parent.name}_cosine_vs_prob.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Cosine vs Probability graph saved to: {save_path}")

    # Calculate correlation
    from scipy.stats import pearsonr, spearmanr
    pearson_corr, _ = pearsonr(cos_flat, prob_flat)
    spearman_corr, _ = spearmanr(cos_flat, prob_flat)

    print(f"\nCosine vs Probability for {Path(data_path).parent.name}:")
    print(f"  Pearson correlation: {pearson_corr:.4f}")
    print(f"  Spearman correlation: {spearman_corr:.4f}")

    plt.close()

def plot_topk_probability_spread(data_path, save_path=None):
    """
    Analyze the spread of probability across top-k tokens.
    Shows how concentrated vs distributed the probability mass is.
    """
    data = np.load(data_path)

    # Get all top-k probabilities
    # Shape: [chunks, batch, timesteps, topk] -> [chunks, timesteps, topk]
    topk_probs = data['prob_values'].squeeze(1)

    # Calculate metrics for each timestep
    # Mean of top-1, top-2, ..., top-k across chunks
    mean_probs_by_rank = np.mean(topk_probs, axis=0)  # [timesteps, topk]

    # Calculate cumulative probability (how much of total prob is in top-k)
    cumsum_probs = np.cumsum(mean_probs_by_rank, axis=1)  # [timesteps, topk]

    # Calculate entropy across top-k for each position
    # H = -sum(p * log(p))
    eps = 1e-10
    entropy = -np.sum(topk_probs * np.log(topk_probs + eps), axis=-1)  # [chunks, timesteps]
    mean_entropy = np.mean(entropy, axis=0)  # [timesteps]

    timestamps = np.arange(topk_probs.shape[1])
    topk = topk_probs.shape[-1]

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Probability by rank (showing top-5 only for clarity)
    ranks_to_show = min(5, topk)
    colors = plt.cm.viridis(np.linspace(0, 0.8, ranks_to_show))
    for i in range(ranks_to_show):
        ax1.plot(timestamps, mean_probs_by_rank[:, i],
                label=f'Rank {i+1}', linewidth=1.5, color=colors[i])
    ax1.set_ylabel('Probability', fontsize=11)
    ax1.set_title('Probability Distribution Across Top-K Tokens', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', ncol=2)
    ax1.set_ylim([0, 1])

    # Plot 2: Cumulative probability coverage
    for i in [0, 4, 9]:  # Show top-1, top-5, top-10
        if i < topk:
            ax2.plot(timestamps, cumsum_probs[:, i],
                    label=f'Top-{i+1}', linewidth=2)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_ylim([0, 1])

    # Plot 3: Entropy over time
    ax3.plot(timestamps, mean_entropy, color='tab:red', linewidth=2)
    ax3.set_ylabel('Entropy (nats)', fontsize=11)
    ax3.set_xlabel('Position/Timestep', fontsize=11)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output_graphs"
    output_dir.mkdir(exist_ok=True)

    if save_path is None:
        data_path_obj = Path(data_path)
        save_path = output_dir / f"{data_path_obj.parent.name}_topk_spread.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Top-K spread graph saved to: {save_path}")

    # Print statistics
    print(f"\nTop-K Probability Spread for {Path(data_path).parent.name}:")
    print(f"  Mean top-1 probability: {mean_probs_by_rank[:, 0].mean():.4f}")
    if topk >= 5:
        print(f"  Mean top-5 cumulative: {cumsum_probs[:, 4].mean():.4f}")
    if topk >= 10:
        print(f"  Mean top-10 cumulative: {cumsum_probs[:, 9].mean():.4f}")
    print(f"  Mean entropy: {mean_entropy.mean():.4f}")

    plt.close()

def plot_cosine_vs_random_distribution(data_path, save_path=None):
    """
    Compare the distribution of actual cosine similarities vs random cosine similarities.
    This helps determine if the observed similarities are meaningfully different from random chance.
    """
    data = np.load(data_path)

    # Extract actual cosine similarities (all top-k values)
    # Shape: [chunks, batch, timesteps, topk]
    actual_cos = data['cos_values'].squeeze(1)  # [chunks, timesteps, topk]
    random_cos = data['random_values'].squeeze(1)  # [chunks, timesteps, topk]

    # Flatten all values
    actual_flat = actual_cos.flatten()
    random_flat = random_cos.flatten()

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Subplot 1: Overlapping histograms
    ax1 = fig.add_subplot(gs[0, :])
    bins = np.linspace(min(actual_flat.min(), random_flat.min()),
                       max(actual_flat.max(), random_flat.max()), 100)
    ax1.hist(actual_flat, bins=bins, alpha=0.6, label='Actual Cosine Similarities',
             color='tab:blue', density=True, edgecolor='black', linewidth=0.5)
    ax1.hist(random_flat, bins=bins, alpha=0.6, label='Random Cosine Similarities',
             color='tab:red', density=True, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Cosine Similarity', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Distribution Comparison: Actual vs Random Cosine Similarities',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Cumulative Distribution Functions (CDF)
    ax2 = fig.add_subplot(gs[1, 0])
    actual_sorted = np.sort(actual_flat)
    random_sorted = np.sort(random_flat)
    actual_cdf = np.arange(1, len(actual_sorted) + 1) / len(actual_sorted)
    random_cdf = np.arange(1, len(random_sorted) + 1) / len(random_sorted)

    ax2.plot(actual_sorted, actual_cdf, label='Actual', color='tab:blue', linewidth=2)
    ax2.plot(random_sorted, random_cdf, label='Random', color='tab:red', linewidth=2)
    ax2.set_xlabel('Cosine Similarity', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title('Cumulative Distribution Function (CDF)', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Box plots
    ax3 = fig.add_subplot(gs[1, 1])
    box_data = [actual_flat, random_flat]
    bp = ax3.boxplot(box_data, labels=['Actual', 'Random'], patch_artist=True,
                     showfliers=False)  # Hide outliers for clarity
    bp['boxes'][0].set_facecolor('tab:blue')
    bp['boxes'][1].set_facecolor('tab:red')
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    ax3.set_ylabel('Cosine Similarity', fontsize=11)
    ax3.set_title('Distribution Summary (Box Plot)', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    # Subplot 4: Mean cosine similarity over time
    ax4 = fig.add_subplot(gs[2, 0])
    mean_actual_over_time = np.mean(actual_cos[:, :, 0], axis=0)  # Best match over time
    mean_random_over_time = np.mean(random_cos[:, :, 0], axis=0)
    timestamps = np.arange(len(mean_actual_over_time))

    ax4.plot(timestamps, mean_actual_over_time, label='Actual (Best Match)',
             color='tab:blue', linewidth=2)
    ax4.plot(timestamps, mean_random_over_time, label='Random (Best Match)',
             color='tab:red', linewidth=2)
    ax4.set_xlabel('Position/Timestep', fontsize=11)
    ax4.set_ylabel('Mean Cosine Similarity', fontsize=11)
    ax4.set_title('Mean Cosine Similarity Over Time', fontsize=12)
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    # Subplot 5: Statistics text box
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    # Calculate statistics
    from scipy import stats

    # Basic statistics
    actual_mean = np.mean(actual_flat)
    random_mean = np.mean(random_flat)
    actual_std = np.std(actual_flat)
    random_std = np.std(random_flat)
    actual_median = np.median(actual_flat)
    random_median = np.median(random_flat)

    # Statistical tests
    # T-test for difference in means
    t_stat, t_pval = stats.ttest_ind(actual_flat, random_flat)

    # Kolmogorov-Smirnov test for distribution difference
    ks_stat, ks_pval = stats.ks_2samp(actual_flat, random_flat)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((actual_std**2 + random_std**2) / 2)
    cohens_d = (actual_mean - random_mean) / pooled_std

    stats_text = f"""Statistical Comparison:

Actual Cosine Similarities:
  Mean: {actual_mean:.4f}
  Std: {actual_std:.4f}
  Median: {actual_median:.4f}

Random Cosine Similarities:
  Mean: {random_mean:.4f}
  Std: {random_std:.4f}
  Median: {random_median:.4f}

Difference:
  Mean Δ: {actual_mean - random_mean:.4f}

Statistical Tests:
  T-test p-value: {t_pval:.2e}
  KS-test p-value: {ks_pval:.2e}
  Cohen's d: {cohens_d:.4f}

Interpretation:
  {"Significantly different" if ks_pval < 0.001 else "Not significantly different"}
  Effect size: {"Large" if abs(cohens_d) > 0.8 else "Medium" if abs(cohens_d) > 0.5 else "Small"}
"""

    ax5.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round',
             facecolor='wheat', alpha=0.3))

    # Save figure
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output_graphs"
    output_dir.mkdir(exist_ok=True)

    if save_path is None:
        data_path_obj = Path(data_path)
        save_path = output_dir / f"{data_path_obj.parent.name}_cosine_vs_random.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Cosine vs Random distribution graph saved to: {save_path}")

    # Print statistics to console
    print(f"\nCosine vs Random Analysis for {Path(data_path).parent.name}:")
    print(f"  Actual mean: {actual_mean:.4f} ± {actual_std:.4f}")
    print(f"  Random mean: {random_mean:.4f} ± {random_std:.4f}")
    print(f"  Difference: {actual_mean - random_mean:.4f}")
    print(f"  T-test p-value: {t_pval:.2e}")
    print(f"  KS-test p-value: {ks_pval:.2e}")
    print(f"  Cohen's d: {cohens_d:.4f}")

    plt.close()

def plot_multi_model_comparison(data_paths, model_names, metric='probability', save_path=None):
    """
    Compare the same metric across multiple models.

    Args:
        data_paths: List of paths to .npz files
        model_names: List of model names for labels
        metric: 'probability', 'logit', or 'cosine'
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(data_paths)))

    for i, (data_path, model_name) in enumerate(zip(data_paths, model_names)):
        data = np.load(data_path)

        # Extract the appropriate metric
        if metric == 'probability':
            values = data['prob_values'].squeeze(1)[:,:,0]
            ylabel = 'Max Probability'
        elif metric == 'logit':
            values = data['logit_values'].squeeze(1)[:,:,0]
            ylabel = 'Max Logit Value'
        elif metric == 'cosine':
            values = data['cos_values'].squeeze(1)[:,:,0]
            ylabel = 'Max Cosine Similarity'
        else:
            raise ValueError(f"Unknown metric: {metric}")

        mean_values = np.mean(values, axis=0)
        std_values = np.std(values, axis=0)
        var_values = np.var(values, axis=0)

        timestamps = np.arange(values.shape[1])

        # Plot mean values
        ax1.plot(timestamps, mean_values, label=model_name,
                linewidth=2, color=colors[i])

        # Plot variance
        ax2.plot(timestamps, var_values, label=model_name,
                linewidth=2, color=colors[i])

    # Formatting
    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.set_title(f'Multi-Model Comparison: {metric.capitalize()}', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    if metric == 'probability':
        ax1.set_ylim([0, 1])

    ax2.set_ylabel(f'Variance of {ylabel}', fontsize=11)
    ax2.set_xlabel('Position/Timestep', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)

    plt.tight_layout()

    # Save figure
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output_graphs"
    output_dir.mkdir(exist_ok=True)

    if save_path is None:
        save_path = output_dir / f"multi_model_comparison_{metric}.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Multi-model comparison ({metric}) saved to: {save_path}")

    plt.close()

if __name__ == "__main__":
    # Generate argmax token graphs
    load_and_graph_argmax_token("final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-1B_20251103_144808/results.npz")
    load_and_graph_argmax_token("final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-3B_20251103_150939/results.npz")

    load_and_graph_argmax_token("final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-0.6B_20251103_142800/results.npz")
    load_and_graph_argmax_token("final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-4B_20251103_144144/results.npz")

    # Generate probability vs logit comparison graphs
    compare_probability_vs_logit("final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-1B_20251103_144808/results.npz")
    compare_probability_vs_logit("final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-3B_20251103_150939/results.npz")

    compare_probability_vs_logit("final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-0.6B_20251103_142800/results.npz")
    compare_probability_vs_logit("final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-4B_20251103_144144/results.npz")

    # Generate max probability over time graphs
    plot_max_probability_over_time("final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-1B_20251103_144808/results.npz")
    plot_max_probability_over_time("final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-3B_20251103_150939/results.npz")

    plot_max_probability_over_time("final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-0.6B_20251103_142800/results.npz")
    plot_max_probability_over_time("final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-4B_20251103_144144/results.npz")

    # Generate combined metrics graphs
    plot_combined_metrics_over_time("final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-1B_20251103_144808/results.npz")
    plot_combined_metrics_over_time("final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-3B_20251103_150939/results.npz")

    plot_combined_metrics_over_time("final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-0.6B_20251103_142800/results.npz")
    plot_combined_metrics_over_time("final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-4B_20251103_144144/results.npz")

    # Generate intersection analysis graphs
    plot_intersection_analysis("final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-1B_20251103_144808/results.npz")
    plot_intersection_analysis("final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-3B_20251103_150939/results.npz")

    plot_intersection_analysis("final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-0.6B_20251103_142800/results.npz")
    plot_intersection_analysis("final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-4B_20251103_144144/results.npz")

    # Generate cosine vs probability graphs
    compare_cosine_vs_probability("final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-1B_20251103_144808/results.npz")
    compare_cosine_vs_probability("final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-3B_20251103_150939/results.npz")

    compare_cosine_vs_probability("final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-0.6B_20251103_142800/results.npz")
    compare_cosine_vs_probability("final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-4B_20251103_144144/results.npz")

    # Generate top-k probability spread graphs
    plot_topk_probability_spread("final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-1B_20251103_144808/results.npz")
    plot_topk_probability_spread("final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-3B_20251103_150939/results.npz")

    plot_topk_probability_spread("final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-0.6B_20251103_142800/results.npz")
    plot_topk_probability_spread("final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-4B_20251103_144144/results.npz")

    # Generate cosine vs random distribution comparison graphs
    plot_cosine_vs_random_distribution("final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-1B_20251103_144808/results.npz")
    plot_cosine_vs_random_distribution("final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-3B_20251103_150939/results.npz")

    plot_cosine_vs_random_distribution("final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-0.6B_20251103_142800/results.npz")
    plot_cosine_vs_random_distribution("final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-4B_20251103_144144/results.npz")

    # Generate multi-model comparison graphs
    data_paths = [
        "final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-1B_20251103_144808/results.npz",
        "final_output_vector_similarity/results/wikitext_results_meta-llama_Llama-3.2-3B_20251103_150939/results.npz",
        "final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-0.6B_20251103_142800/results.npz",
        "final_output_vector_similarity/results/wikitext_results_Qwen_Qwen3-4B_20251103_144144/results.npz"
    ]
    model_names = ["Llama-3.2-1B", "Llama-3.2-3B", "Qwen3-0.6B", "Qwen3-4B"]

    plot_multi_model_comparison(data_paths, model_names, metric='probability')
    plot_multi_model_comparison(data_paths, model_names, metric='logit')
    plot_multi_model_comparison(data_paths, model_names, metric='cosine')
