# Tuning Guide

Step-by-step guide to tuning classification thresholds for optimal accuracy.

## Overview

The classification system uses multiple filters:
1. **Spectral Similarity**: Cosine similarity to trained chirp fingerprint
2. **Frequency Filtering**: Rejects fan noise (low-freq) and requires high-freq content
3. **Temporal Filtering**: Rejects sustained sounds by checking duration and energy distribution
4. **Confidence Scoring**: Weighted combination of all features

## Step 1: Collect and Review Events

1. Run the monitor and collect events over a period (e.g., 24-48 hours)
2. Review clips in `clips/` directory
3. Mark events in `events.csv` `reviewed` column:
   - Chirps: `chirp`, `true`, `yes`, or `positive`
   - Non-chirps: `not_chirp`, `false`, `no`, or `negative`

## Step 2: Validate Current Performance

```bash
python3 scripts/validate_classification.py
```

This shows:
- Accuracy, precision, recall, F1 score
- Confusion matrix
- List of false positives and false negatives
- Tuning recommendations

## Step 3: Analyze Clip Characteristics

```bash
python3 scripts/analyze_clips.py
```

Generates `clip_analysis.csv` with features for all clips, helping identify patterns in chirps vs non-chirps.

## Step 4: Diagnose Specific Events

```bash
python3 scripts/diagnose_event.py clips/clip_2025-11-24_20-02-05.wav
```

Shows detailed analysis of why an event was/wasn't classified as chirp:
- All filter results (frequency, temporal, similarity)
- Which filters passed/failed
- Confidence score breakdown

## Step 5: Tune Thresholds

### Automated Grid Search

```bash
python3 scripts/tune_thresholds.py --events data/events.csv
```

This will:
- Test all combinations of threshold values
- Find the configuration with best F1 score
- Save results to `tuning_results.csv`
- Generate `config.recommended.json` with optimal settings

### Custom Ranges

```bash
python3 scripts/tune_thresholds.py \
  --similarity-range 0.7 0.75 0.8 0.85 0.9 \
  --low-freq-range 0.2 0.25 0.3 0.35 \
  --max-duration-range 1.5 2.0 2.5 \
  --energy-conc-range 0.4 0.5 0.6
```

Review `config.recommended.json` and update your `config.json` if desired.

## Step 6: Iterate

Repeat steps 2-5 until satisfied with accuracy.

## Common Tuning Scenarios

### Too Many False Positives

**Symptoms**: Non-chirps (fan noise, doors, etc.) classified as chirps

**Solutions**:
- Increase `similarity_threshold` (e.g., 0.8 → 0.85)
- Decrease `low_freq_energy_threshold` (e.g., 0.30 → 0.25) - stricter fan noise rejection
- Decrease `max_duration_sec` (e.g., 2.0 → 1.5) - reject longer sounds
- Increase `energy_concentration_threshold` (e.g., 0.50 → 0.60) - require more energy in first half

### Too Many False Negatives

**Symptoms**: Actual chirps not being detected

**Solutions**:
- Decrease `similarity_threshold` (e.g., 0.8 → 0.75)
- Increase `high_freq_energy_min_ratio` (e.g., 0.10 → 0.08) - allow less high-freq content
- Increase `max_duration_sec` (e.g., 2.0 → 2.5) - allow longer events
- Decrease `energy_concentration_threshold` (e.g., 0.50 → 0.40) - allow more spread-out energy

### Chirps Too Quiet

**Symptoms**: Chirps below baseline threshold not detected

**Solutions**:
- Decrease `threshold_above_baseline_db` (e.g., 10 → 8)
- Check baseline calibration - may need to recalibrate in quieter conditions

### Too Many Short Events

**Symptoms**: Very brief sounds triggering events

**Solutions**:
- Increase `min_event_duration_sec` (e.g., 0.5 → 1.0)

## Threshold Reference

| Threshold | Default | Range | Effect |
|-----------|---------|-------|--------|
| `similarity_threshold` | 0.8 | 0.0-1.0 | Higher = stricter (fewer false positives) |
| `low_freq_energy_threshold` | 0.30 | 0.0-1.0 | Lower = stricter fan noise rejection |
| `high_freq_energy_min_ratio` | 0.10 | 0.0-1.0 | Higher = requires more high-freq content |
| `max_duration_sec` | 2.0 | 0.5-5.0 | Lower = rejects longer sounds |
| `energy_concentration_threshold` | 0.50 | 0.0-1.0 | Higher = requires energy in first half |

## Retraining Fingerprint

If tuning thresholds doesn't improve accuracy, consider retraining the fingerprint:

1. Collect better training samples:
   - More diverse chirp examples
   - More non-chirp examples (fan, doors, other sounds you want to reject)
2. Retrain: `make train`
3. Deploy: `make deploy`
4. Re-validate: `python3 scripts/validate_classification.py`

