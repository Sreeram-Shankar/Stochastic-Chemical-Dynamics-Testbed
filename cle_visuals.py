import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import correlate

#defines the global visual settings for light mode (orange theme)
light_settings = {
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "lines.linewidth": 2.2,
    "lines.markersize": 5,
    "legend.fontsize": 10,
    "figure.figsize": (6, 4),
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cc5500",
    "axes.labelcolor": "#cc5500",
    "text.color": "#cc5500",
    "xtick.color": "#cc5500",
    "ytick.color": "#cc5500",
    "grid.color": "#ffaa66",
    "figure.edgecolor": "#ffffff",
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#cc5500",
    "legend.labelcolor": "#cc5500",
}

#defines the global visual settings for dark mode (orange theme)
dark_settings = {
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "lines.linewidth": 2.2,
    "lines.markersize": 5,
    "legend.fontsize": 10,
    "figure.figsize": (6, 4),
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "figure.facecolor": "#0d0d0d",
    "axes.facecolor": "#0d0d0d",
    "axes.edgecolor": "#ffaa66",
    "axes.labelcolor": "#ffaa66",
    "text.color": "#ffaa66",
    "xtick.color": "#ffaa66",
    "ytick.color": "#ffaa66",
    "grid.color": "#cc5500",
    "figure.edgecolor": "#0d0d0d",
    "legend.facecolor": "#0d0d0d",
    "legend.edgecolor": "#ffaa66",
    "legend.labelcolor": "#ffaa66",
}

#function to apply theme settings
def apply_theme(theme="light"):
    if theme.lower() == "dark": plt.rcParams.update(dark_settings)
    else: plt.rcParams.update(light_settings)

#function to ensure the directory exists
def ensure_dir(path): return os.makedirs(path, exist_ok=True)

#plots the time series of the CLE simulation
def plot_cle_time_series(t_grid, Y_all, output_dir, theme="light", n_paths=10):
    apply_theme(theme)
    ensure_dir(output_dir)
    
    M, N_plus_1, _ = Y_all.shape
    n_paths = min(n_paths, M, 20)
    n_paths = max(n_paths, 5)
    
    #selects representative paths (evenly spaced)
    path_indices = np.linspace(0, M-1, n_paths, dtype=int)
    
    plt.figure()
    for idx in path_indices:
        plt.plot(t_grid, Y_all[idx, :, 0], alpha=0.6, linewidth=1.5)
    plt.xlabel("Time")
    plt.ylabel("X(t)")
    plt.title(f"CLE Trajectories ({n_paths} Representative Paths)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, "cle_time_series.png")
    plt.savefig(filepath)
    plt.close()
    return filepath

#plots the mean and variance of the CLE simulation
def plot_cle_mean_variance(t_grid, Y_all, output_dir, theme="light"):
    apply_theme(theme)
    ensure_dir(output_dir)
    
    #computes ensemble statistics
    mean_t = np.mean(Y_all[:, :, 0], axis=0)
    var_t = np.var(Y_all[:, :, 0], axis=0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    ax1.plot(t_grid, mean_t, color="#ff8c42", linewidth=2)
    ax1.set_ylabel("⟨X(t)⟩")
    ax1.set_title("CLE Ensemble Mean vs Time")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t_grid, var_t, color="#ff6347", linewidth=2)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Var[X(t)]")
    ax2.set_title("CLE Ensemble Variance vs Time")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "cle_mean_variance.png")
    plt.savefig(filepath)
    plt.close()
    return filepath

#plots the stationary distribution of the CLE simulation
def plot_cle_stationary_distribution(t_grid, Y_all, output_dir, theme="light", use_kde=True):
    apply_theme(theme)
    ensure_dir(output_dir)
    
    #extracts final states
    final_states = Y_all[:, -1, 0].astype(float)
    final_states = final_states[final_states >= 0] 
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    #applies a linear scale
    ax1.hist(final_states, bins=50, density=True, alpha=0.7, color="#ff8c42", edgecolor="#cc5500")
    if use_kde and len(final_states) > 10:
        try:
            kde = stats.gaussian_kde(final_states)
            x_kde = np.linspace(final_states.min(), final_states.max(), 200)
            ax1.plot(x_kde, kde(x_kde), color="#cc5500", linewidth=2, label="KDE")
            ax1.legend()
        except: pass
    ax1.set_xlabel("X(T)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("CLE Stationary Distribution (Linear Scale)")
    ax1.grid(True, alpha=0.3)
    
    #applies a log scale
    positive_states = final_states[final_states > 0]
    if len(positive_states) > 0:
        ax2.hist(positive_states, bins=50, density=True, alpha=0.7, color="#ff8c42", edgecolor="#cc5500")
        ax2.set_xscale('log')
        if use_kde and len(positive_states) > 10:
            try:
                kde = stats.gaussian_kde(np.log(positive_states))
                x_log = np.logspace(np.log10(positive_states.min()), np.log10(positive_states.max()), 200)
                ax2.plot(x_log, kde(np.log(x_log)) / x_log, color="#cc5500", linewidth=2, label="KDE")
                ax2.legend()
            except:
                pass
        ax2.set_xlabel("X(T) [log scale]")
        ax2.set_ylabel("Probability Density")
        ax2.set_title("CLE Stationary Distribution (Log Scale)")
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "cle_stationary_distribution.png")
    plt.savefig(filepath)
    plt.close()
    return filepath

#plots the quantiles of the CLE simulation
def plot_cle_quantiles(t_grid, Y_all, output_dir, theme="light"):
    apply_theme(theme)
    ensure_dir(output_dir)
    
    #extracts the final states
    final_states = Y_all[:, -1, 0].astype(float)
    final_states = final_states[final_states >= 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
     #creates a box plot
    ax1.boxplot([final_states], labels=["CLE"], patch_artist=True, boxprops=dict(facecolor="#ff8c42", alpha=0.7), medianprops=dict(color="#cc5500", linewidth=2))
    ax1.set_ylabel("X(T)")
    ax1.set_title("CLE Final State Distribution (Box Plot)")
    ax1.grid(True, alpha=0.3, axis='y')
    
    #violin plot
    parts = ax2.violinplot([final_states], positions=[0], showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor("#ff8c42")
        pc.set_alpha(0.7)
    ax2.set_xticks([0])
    ax2.set_xticklabels(["CLE"])
    ax2.set_ylabel("X(T)")
    ax2.set_title("CLE Final State Distribution (Violin Plot)")
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "cle_quantiles.png")
    plt.savefig(filepath)
    plt.close()
    return filepath

#plots the autocorrelation function of the CLE simulation
def plot_cle_autocorrelation(t_grid, Y_all, output_dir, theme="light", max_lag=None):
    apply_theme(theme)
    ensure_dir(output_dir)
    
    #computes the mean trajectory
    mean_traj = np.mean(Y_all[:, :, 0], axis=0)
    mean_val = np.mean(mean_traj)
    
    #computes the autocorrelation for the mean trajectory
    centered = mean_traj - mean_val
    if max_lag is None:
        max_lag = len(centered) // 4
    
    autocorr = correlate(centered, centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr[:max_lag] / autocorr[0]
    
    #computes the time lags
    dt = t_grid[1] - t_grid[0] if len(t_grid) > 1 else 1.0
    lags = np.arange(len(autocorr)) * dt
    
    plt.figure()
    plt.plot(lags, autocorr, color="#ff8c42", linewidth=2)
    plt.xlabel("Lag τ")
    plt.ylabel("Autocorrelation C(τ)")
    plt.title("CLE Autocorrelation Function")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, "cle_autocorrelation.png")
    plt.savefig(filepath)
    plt.close()
    return filepath

#plots the switching events of the CLE simulation
def plot_cle_switching_events(t_grid, Y_all, output_dir, theme="light", threshold_low=None, threshold_high=None):
    apply_theme(theme)
    ensure_dir(output_dir)
    
    #determines the thresholds if not provided (uses the median split)
    if threshold_low is None or threshold_high is None:
        all_values = Y_all[:, :, 0].flatten()
        threshold_low = np.percentile(all_values, 33)
        threshold_high = np.percentile(all_values, 67)
    
    #detects the switching events for each path
    switching_times = []
    residence_times_low = []
    residence_times_high = []
    
    for m in range(Y_all.shape[0]):
        traj = Y_all[m, :, 0]
        state = "low" if traj[0] < threshold_low else "high" if traj[0] > threshold_high else "mid"
        
        for n in range(1, len(traj)):
            if traj[n] < threshold_low:
                new_state = "low"
            elif traj[n] > threshold_high:
                new_state = "high"
            else:
                new_state = "mid"
            
            if new_state != state and state != "mid":
                switching_times.append(t_grid[n])
                if state == "low":
                    residence_times_low.append(t_grid[n] - (switching_times[-2] if len(switching_times) > 1 else t_grid[0]))
                state = new_state
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    #creates a histogram of the switching times
    if switching_times:
        ax1.hist(switching_times, bins=30, color="#ff8c42", alpha=0.7, edgecolor="#cc5500")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Number of Switches")
        ax1.set_title("Switching Event Times")
        ax1.grid(True, alpha=0.3)
    
    #creates a histogram of the residence times
    if residence_times_low: ax2.hist(residence_times_low, bins=30, color="#ff6347", alpha=0.7, edgecolor="#cc5500", label="Low State")
    if residence_times_high: ax2.hist(residence_times_high, bins=30, color="#ff8c42", alpha=0.7, edgecolor="#cc5500", label="High State")
    ax2.set_xlabel("Residence Time")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residence Time Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "cle_switching_events.png")
    plt.savefig(filepath)
    plt.close()
    return filepath

#plots the solver bias summary of the CLE simulation
def plot_cle_solver_bias_summary(t_grid, Y_all, solver_name, output_dir, theme="light"):
    apply_theme(theme)
    ensure_dir(output_dir)
    
    final_states = Y_all[:, -1, 0].astype(float)
    final_states = final_states[final_states >= 0]
    
    #computes the statistics
    mean_val = np.mean(final_states)
    median_val = np.median(final_states)
    std_val = np.std(final_states)
    q25, q75 = np.percentile(final_states, [25, 75])
    iqr = q75 - q25
    
    #computes the tail mass (probability of being in top 5% or bottom 5%)
    tail_high = np.percentile(final_states, 95)
    tail_low = np.percentile(final_states, 5)
    tail_mass = np.mean((final_states > tail_high) | (final_states < tail_low))
    
    #creates a bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    stats_names = ["Mean", "Median", "Std", "IQR", "Tail Mass"]
    stats_values = [mean_val, median_val, std_val, iqr, tail_mass * 100]
    
    bars = ax.bar(stats_names, stats_values, color="#ff8c42", alpha=0.7, edgecolor="#cc5500")
    ax.set_ylabel("Value")
    ax.set_title(f"Solver Bias Summary: {solver_name}")
    ax.grid(True, alpha=0.3, axis='y')
    
    #adds value labels on bars
    for bar, val in zip(bars, stats_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    #sanitizes the solver name for filename 
    safe_name = solver_name.replace(' ', '_').replace('–', '-').replace('/', '_').replace('\\', '_')
    safe_name = safe_name.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
    safe_name = safe_name.replace('{', '').replace('}', '').replace(':', '_').replace('*', '_')
    safe_name = safe_name.replace('?', '').replace('"', '').replace('<', '').replace('>', '')
    safe_name = safe_name.replace('|', '_')
    filepath = os.path.join(output_dir, f"cle_solver_bias_{safe_name}.png")
    plt.savefig(filepath)
    plt.close()
    return filepath

#plots the performance vs accuracy of the CLE simulation
def plot_cle_performance_accuracy(runtimes, variances, tail_masses, solver_names, output_dir, theme="light"):
    apply_theme(theme)
    ensure_dir(output_dir)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    #creates a scatter plot of the runtime vs variance
    ax1.scatter(runtimes, variances, s=100, alpha=0.7, color="#ff8c42", edgecolor="#cc5500")
    for i, name in enumerate(solver_names): ax1.annotate(name, (runtimes[i], variances[i]), fontsize=8)
    ax1.set_xlabel("Runtime (s)")
    ax1.set_ylabel("Variance")
    ax1.set_title("Performance vs Variance")
    ax1.grid(True, alpha=0.3)
    
    #creates a scatter plot of the runtime vs tail mass
    ax2.scatter(runtimes, tail_masses, s=100, alpha=0.7, color="#ff6347", edgecolor="#cc5500")
    for i, name in enumerate(solver_names): ax2.annotate(name, (runtimes[i], tail_masses[i]), fontsize=8)
    ax2.set_xlabel("Runtime (s)")
    ax2.set_ylabel("Tail Mass")
    ax2.set_title("Performance vs Tail Mass")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "cle_performance_accuracy.png")
    plt.savefig(filepath)
    plt.close()
    return filepath

#plots the stationary distributions of the SSA and CLE simulations
def plot_ssa_vs_cle_distributions(t_grid_cle, Y_all_cle, t_grid_ssa, Y_all_ssa, output_dir, theme="light"):
    apply_theme(theme)
    ensure_dir(output_dir)
    
    cle_final = Y_all_cle[:, -1, 0].astype(float)
    cle_final = cle_final[cle_final >= 0]
    ssa_final = Y_all_ssa[:, -1, 0].astype(float)
    ssa_final = ssa_final[ssa_final >= 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    #applies a linear scale
    ax1.hist(ssa_final, bins=50, density=True, alpha=0.6, label='SSA', color="#ffaa66", edgecolor="#cc5500")
    ax1.hist(cle_final, bins=50, density=True, alpha=0.6, label='CLE', color="#ff8c42", edgecolor="#cc5500")
    ax1.set_xlabel("X(T)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("SSA vs CLE Stationary Distribution (Linear Scale)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    #applies a log scale
    cle_positive = cle_final[cle_final > 0]
    ssa_positive = ssa_final[ssa_final > 0]
    if len(cle_positive) > 0 and len(ssa_positive) > 0:
        ax2.hist(ssa_positive, bins=50, density=True, alpha=0.6, label='SSA', color="#ffaa66", edgecolor="#cc5500")
        ax2.hist(cle_positive, bins=50, density=True, alpha=0.6, label='CLE', color="#ff8c42", edgecolor="#cc5500")
        ax2.set_xscale('log')
        ax2.set_xlabel("X(T) [log scale]")
        ax2.set_ylabel("Probability Density")
        ax2.set_title("SSA vs CLE Stationary Distribution (Log Scale)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "ssa_vs_cle_distributions.png")
    plt.savefig(filepath)
    plt.close()
    return filepath

#plots the quantiles of the SSA and CLE simulations
def plot_ssa_vs_cle_quantiles(t_grid_cle, Y_all_cle, t_grid_ssa, Y_all_ssa, output_dir, theme="light"):
    apply_theme(theme)
    ensure_dir(output_dir)
    
    cle_final = Y_all_cle[:, -1, 0].astype(float)
    cle_final = cle_final[cle_final >= 0]
    ssa_final = Y_all_ssa[:, -1, 0].astype(float)
    ssa_final = ssa_final[ssa_final >= 0]
    
    #computes the quantiles
    quantiles = [25, 50, 75, 95]
    cle_quantiles = [np.percentile(cle_final, q) for q in quantiles]
    ssa_quantiles = [np.percentile(ssa_final, q) for q in quantiles]
    
    x = np.arange(len(quantiles))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, ssa_quantiles, width, label='SSA', color="#ffaa66", alpha=0.7, edgecolor="#cc5500")
    ax.bar(x + width/2, cle_quantiles, width, label='CLE', color="#ff8c42", alpha=0.7, edgecolor="#cc5500")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Value")
    ax.set_title("SSA vs CLE Quantile Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f'Q{q}' for q in quantiles])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "ssa_vs_cle_quantiles.png")
    plt.savefig(filepath)
    plt.close()
    return filepath

#plots the mean / variance bias of the SSA and CLE simulations
def plot_ssa_vs_cle_bias(t_grid_cle, Y_all_cle, t_grid_ssa, Y_all_ssa, output_dir, theme="light"):
    apply_theme(theme)
    ensure_dir(output_dir)
    
    cle_final = Y_all_cle[:, -1, 0].astype(float)
    cle_final = cle_final[cle_final >= 0]
    ssa_final = Y_all_ssa[:, -1, 0].astype(float)
    ssa_final = ssa_final[ssa_final >= 0]
    
    #computes the statistics
    cle_mean = np.mean(cle_final)
    cle_var = np.var(cle_final)
    cle_median = np.median(cle_final)
    
    ssa_mean = np.mean(ssa_final)
    ssa_var = np.var(ssa_final)
    ssa_median = np.median(ssa_final)
    
    #computes the bias
    bias_mean = cle_mean - ssa_mean
    bias_var = cle_var - ssa_var
    bias_median = cle_median - ssa_median
    
    fig, ax = plt.subplots(figsize=(8, 5))
    stats_names = ["Mean", "Variance", "Median"]
    bias_values = [bias_mean, bias_var, bias_median]
    
    colors = ["#ff8c42" if v > 0 else "#ff6347" for v in bias_values]
    bars = ax.bar(stats_names, bias_values, color=colors, alpha=0.7, edgecolor="#cc5500")
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel("Bias (CLE - SSA)")
    ax.set_title("CLE Bias Relative to SSA")
    ax.grid(True, alpha=0.3, axis='y')
    
    #adds the value labels to the bars
    for bar, val in zip(bars, bias_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "ssa_vs_cle_bias.png")
    plt.savefig(filepath)
    plt.close()
    return filepath

#plots the probability mass in the physical regions of the SSA and CLE simulations
def plot_probability_mass_regions(t_grid_cle, Y_all_cle, t_grid_ssa, Y_all_ssa, output_dir, theme="light", low_threshold=None, high_threshold=None):
    apply_theme(theme)
    ensure_dir(output_dir)
    
    cle_final = Y_all_cle[:, -1, 0].astype(float)
    cle_final = cle_final[cle_final >= 0]
    ssa_final = Y_all_ssa[:, -1, 0].astype(float)
    ssa_final = ssa_final[ssa_final >= 0]
    
    #determines the thresholds if not provided
    if low_threshold is None or high_threshold is None:
        all_values = np.concatenate([cle_final, ssa_final])
        low_threshold = np.percentile(all_values, 33)
        high_threshold = np.percentile(all_values, 67)
    
    #computes the probabilities
    cle_low = np.mean(cle_final < low_threshold)
    cle_high = np.mean(cle_final > high_threshold)
    cle_extreme = np.mean((cle_final < np.percentile(cle_final, 5)) | (cle_final > np.percentile(cle_final, 95)))
    
    ssa_low = np.mean(ssa_final < low_threshold)
    ssa_high = np.mean(ssa_final > high_threshold)
    ssa_extreme = np.mean((ssa_final < np.percentile(ssa_final, 5)) | (ssa_final > np.percentile(ssa_final, 95)))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    regions = ["Low State", "High State", "Extreme"]
    x = np.arange(len(regions))
    width = 0.35
    
    ax.bar(x - width/2, [ssa_low, ssa_high, ssa_extreme], width, label='SSA', color="#ffaa66", alpha=0.7, edgecolor="#cc5500")
    ax.bar(x + width/2, [cle_low, cle_high, cle_extreme], width, label='CLE', color="#ff8c42", alpha=0.7, edgecolor="#cc5500")
    ax.set_xlabel("Region")
    ax.set_ylabel("Probability Mass")
    ax.set_title("Probability Mass in Physical Regions")
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "probability_mass_regions.png")
    plt.savefig(filepath)
    plt.close()
    return filepath

#plots the switching statistics of the SSA and CLE simulations
def plot_switching_statistics(t_grid_cle, Y_all_cle, t_grid_ssa, Y_all_ssa, output_dir, theme="light", threshold_low=None, threshold_high=None):
    apply_theme(theme)
    ensure_dir(output_dir)
    
    #helper function to compute the switching rate
    def compute_switching_rate(t_grid, Y, thresh_low, thresh_high):
        switches = 0
        for m in range(Y.shape[0]):
            traj = Y[m, :, 0]
            state = "low" if traj[0] < thresh_low else "high" if traj[0] > thresh_high else "mid"
            for n in range(1, len(traj)):
                if traj[n] < thresh_low:
                    new_state = "low"
                elif traj[n] > thresh_high:
                    new_state = "high"
                else:
                    new_state = "mid"
                if new_state != state and state != "mid":
                    switches += 1
                state = new_state
        T_total = t_grid[-1] * Y.shape[0]
        return switches / T_total if T_total > 0 else 0.0
    
    #determines the thresholds
    if threshold_low is None or threshold_high is None:
        all_values = np.concatenate([Y_all_cle[:, :, 0].flatten(), Y_all_ssa[:, :, 0].flatten()])
        threshold_low = np.percentile(all_values, 33)
        threshold_high = np.percentile(all_values, 67)
    
    cle_rate = compute_switching_rate(t_grid_cle, Y_all_cle, threshold_low, threshold_high)
    ssa_rate = compute_switching_rate(t_grid_ssa, Y_all_ssa, threshold_low, threshold_high)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ["SSA", "CLE"]
    rates = [ssa_rate, cle_rate]
    
    bars = ax.bar(methods, rates, color=["#ffaa66", "#ff8c42"], alpha=0.7, edgecolor="#cc5500")
    ax.set_ylabel("Switching Rate (events/time)")
    ax.set_title("Switching Statistics: SSA vs CLE")
    ax.grid(True, alpha=0.3, axis='y')
    
    #adds the value labels to the bars
    for bar, val in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "switching_statistics.png")
    plt.savefig(filepath)
    plt.close()
    return filepath

#plots the time-resolved comparison of the SSA and CLE simulations
def plot_time_resolved_comparison(t_grid_cle, Y_all_cle, t_grid_ssa, Y_all_ssa, output_dir, theme="light"):
    apply_theme(theme)
    ensure_dir(output_dir)
    
    #computes the ensemble statistics
    cle_mean_t = np.mean(Y_all_cle[:, :, 0], axis=0)
    cle_var_t = np.var(Y_all_cle[:, :, 0], axis=0)
    ssa_mean_t = np.mean(Y_all_ssa[:, :, 0], axis=0)
    ssa_var_t = np.var(Y_all_ssa[:, :, 0], axis=0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    ax1.plot(t_grid_ssa, ssa_mean_t, label='SSA', color="#ffaa66", linewidth=2)
    ax1.plot(t_grid_cle, cle_mean_t, label='CLE', color="#ff8c42", linewidth=2)
    ax1.set_ylabel("⟨X(t)⟩")
    ax1.set_title("Time-Resolved Mean: SSA vs CLE")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t_grid_ssa, ssa_var_t, label='SSA', color="#ffaa66", linewidth=2)
    ax2.plot(t_grid_cle, cle_var_t, label='CLE', color="#ff8c42", linewidth=2)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Var[X(t)]")
    ax2.set_title("Time-Resolved Variance: SSA vs CLE")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "time_resolved_comparison.png")
    plt.savefig(filepath)
    plt.close()
    return filepath