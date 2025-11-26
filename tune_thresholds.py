#!/usr/bin/env python3
"""Interactive tool to tune classification thresholds."""
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pandas as pd

import config_loader
import validate_classification


def grid_search_thresholds(
    events_file: Path,
    config_path: Optional[Path],
    param_ranges: Dict[str, List[float]]
) -> Dict:
    """Perform grid search to find optimal thresholds."""
    print("Performing grid search...")
    print(f"Testing {len(param_ranges)} parameters")
    
    best_config = None
    best_f1 = 0.0
    best_accuracy = 0.0
    
    # Load base config
    base_config = config_loader.load_config(config_path)
    
    # Generate all combinations (simplified - only test key parameters)
    similarity_range = param_ranges.get("similarity_threshold", [0.7, 0.75, 0.8, 0.85, 0.9])
    low_freq_range = param_ranges.get("low_freq_energy_threshold", [0.2, 0.25, 0.3, 0.35, 0.4])
    max_duration_range = param_ranges.get("max_duration_sec", [1.5, 2.0, 2.5])
    energy_conc_range = param_ranges.get("energy_concentration_threshold", [0.4, 0.5, 0.6])
    
    total_combinations = (
        len(similarity_range) * len(low_freq_range) * 
        len(max_duration_range) * len(energy_conc_range)
    )
    print(f"Testing {total_combinations} combinations...")
    print()
    
    results = []
    
    for sim_thresh in similarity_range:
        for low_freq_thresh in low_freq_range:
            for max_dur in max_duration_range:
                for energy_conc in energy_conc_range:
                    # Create modified config
                    test_config = json.loads(json.dumps(base_config))  # Deep copy
                    test_config["chirp_classification"]["similarity_threshold"] = sim_thresh
                    test_config["chirp_classification"]["frequency_filtering"]["low_freq_energy_threshold"] = low_freq_thresh
                    test_config["chirp_classification"]["temporal_filtering"]["max_duration_sec"] = max_dur
                    test_config["chirp_classification"]["temporal_filtering"]["energy_concentration_threshold"] = energy_conc
                    
                    # Save temporary config
                    temp_config_path = Path("config.temp.json")
                    with temp_config_path.open("w") as f:
                        json.dump(test_config, f, indent=2)
                    
                    # Validate with this config
                    try:
                        metrics = validate_with_config(events_file, temp_config_path)
                        
                        if metrics:
                            results.append({
                                "similarity_threshold": sim_thresh,
                                "low_freq_energy_threshold": low_freq_thresh,
                                "max_duration_sec": max_dur,
                                "energy_concentration_threshold": energy_conc,
                                "accuracy": metrics["accuracy"],
                                "precision": metrics["precision"],
                                "recall": metrics["recall"],
                                "f1": metrics["f1"],
                                "true_positives": metrics["true_positives"],
                                "false_positives": metrics["false_positives"],
                                "false_negatives": metrics["false_negatives"],
                            })
                            
                            if metrics["f1"] > best_f1:
                                best_f1 = metrics["f1"]
                                best_accuracy = metrics["accuracy"]
                                best_config = test_config
                            
                            print(f"  sim={sim_thresh:.2f}, low_freq={low_freq_thresh:.2f}, "
                                  f"max_dur={max_dur:.1f}, energy_conc={energy_conc:.2f}: "
                                  f"F1={metrics['f1']:.3f}, Acc={metrics['accuracy']:.3f}")
                    except Exception as e:
                        print(f"  Error with config: {e}")
                    
                    # Clean up
                    if temp_config_path.exists():
                        temp_config_path.unlink()
    
    if not results:
        print("No valid results found")
        return {}
    
    # Find best result
    results_df = pd.DataFrame(results)
    best_idx = results_df["f1"].idxmax()
    best_result = results_df.loc[best_idx]
    
    print()
    print("=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    print(f"Similarity Threshold: {best_result['similarity_threshold']:.3f}")
    print(f"Low-Freq Energy Threshold: {best_result['low_freq_energy_threshold']:.3f}")
    print(f"Max Duration: {best_result['max_duration_sec']:.1f}s")
    print(f"Energy Concentration: {best_result['energy_concentration_threshold']:.3f}")
    print()
    print(f"Metrics:")
    print(f"  Accuracy: {best_result['accuracy']:.1%}")
    print(f"  Precision: {best_result['precision']:.1%}")
    print(f"  Recall: {best_result['recall']:.1%}")
    print(f"  F1 Score: {best_result['f1']:.1%}")
    print()
    print(f"  True Positives: {best_result['true_positives']}")
    print(f"  False Positives: {best_result['false_positives']}")
    print(f"  False Negatives: {best_result['false_negatives']}")
    print()
    
    # Save results
    results_df.to_csv("tuning_results.csv", index=False)
    print("All results saved to tuning_results.csv")
    
    # Generate recommended config
    recommended_config = json.loads(json.dumps(base_config))
    recommended_config["chirp_classification"]["similarity_threshold"] = float(best_result['similarity_threshold'])
    recommended_config["chirp_classification"]["frequency_filtering"]["low_freq_energy_threshold"] = float(best_result['low_freq_energy_threshold'])
    recommended_config["chirp_classification"]["temporal_filtering"]["max_duration_sec"] = float(best_result['max_duration_sec'])
    recommended_config["chirp_classification"]["temporal_filtering"]["energy_concentration_threshold"] = float(best_result['energy_concentration_threshold'])
    
    with Path("config.recommended.json").open("w") as f:
        json.dump(recommended_config, f, indent=2)
    
    print("Recommended config saved to config.recommended.json")
    print("Review and rename to config.json if you want to use it")
    
    return recommended_config


def validate_with_config(events_file: Path, config_path: Path) -> Optional[Dict]:
    """Validate classification with a specific config and return metrics."""
    import monitor
    import config_loader
    
    config = config_loader.load_config(config_path)
    fingerprint_info = monitor.load_chirp_fingerprint(config)
    
    df = pd.read_csv(events_file) if events_file.exists() else pd.DataFrame()
    if df.empty:
        return None
    
    # Filter to reviewed events
    if "reviewed" not in df.columns:
        return None
    
    reviewed_df = df[df["reviewed"].notna() & (df["reviewed"] != "")]
    if reviewed_df.empty:
        return None
    
    reviewed_df = reviewed_df.copy()
    reviewed_df["ground_truth"] = reviewed_df["reviewed"].str.lower().str.contains("chirp|true|yes|positive")
    
    # Re-classify
    correct = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    
    clips_dir = Path(config["event_clips"]["clips_dir"])
    
    for idx, row in reviewed_df.iterrows():
        clip_file = row.get("clip_file", "")
        if not clip_file:
            continue
        
        clip_path = clips_dir / Path(clip_file).name
        if not clip_path.exists():
            clip_path = Path(clip_file)
        
        is_chirp, _, _, _ = validate_classification.classify_clip(clip_path, config, fingerprint_info)
        predicted = is_chirp
        actual = row["ground_truth"]
        
        if predicted == actual:
            correct += 1
            if predicted:
                true_positives += 1
            else:
                true_negatives += 1
        else:
            if predicted:
                false_positives += 1
            else:
                false_negatives += 1
    
    total = len(reviewed_df)
    if total == 0:
        return None
    
    accuracy = correct / total
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune classification thresholds")
    parser.add_argument("--config", type=Path, help="Path to base config.json")
    parser.add_argument("--events", type=Path, default=Path("events.csv"), help="Path to events.csv")
    parser.add_argument("--similarity-range", nargs="+", type=float, help="Similarity threshold range (e.g., 0.7 0.75 0.8 0.85 0.9)")
    parser.add_argument("--low-freq-range", nargs="+", type=float, help="Low-freq threshold range")
    parser.add_argument("--max-duration-range", nargs="+", type=float, help="Max duration range")
    parser.add_argument("--energy-conc-range", nargs="+", type=float, help="Energy concentration range")
    
    args = parser.parse_args()
    
    param_ranges = {}
    if args.similarity_range:
        param_ranges["similarity_threshold"] = args.similarity_range
    if args.low_freq_range:
        param_ranges["low_freq_energy_threshold"] = args.low_freq_range
    if args.max_duration_range:
        param_ranges["max_duration_sec"] = args.max_duration_range
    if args.energy_conc_range:
        param_ranges["energy_concentration_threshold"] = args.energy_conc_range
    
    grid_search_thresholds(args.events, args.config, param_ranges)

