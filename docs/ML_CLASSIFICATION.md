# ML-Based Chirp Classification

## Overview

This document describes the improved ML-based classification system that replaces the simple spectral fingerprinting approach.

## Why ML?

The original system used **template matching** (averaging spectra, cosine similarity). This has limitations:

1. **No learning**: Can't adapt from mistakes
2. **Loses temporal info**: Averaging spectra removes time-varying characteristics
3. **Simple features**: Only uses raw spectral magnitude
4. **Static thresholds**: Manual tuning required

The new ML approach:
- ✅ **Learns patterns** from training data
- ✅ **MFCC features** (more robust, perceptually relevant)
- ✅ **Temporal features** (captures attack/decay, energy distribution)
- ✅ **Automatic optimization** (model finds best decision boundaries)
- ✅ **Confidence scores** (probability-based, not just similarity)

## Architecture

### Feature Extraction

**MFCC Features (52 dimensions)**:
- 13 MFCC coefficients
- Mean, std, min, max across time → 52 features
- Captures spectral envelope and timbre

**Additional Features (7 dimensions)**:
- RMS energy
- Zero-crossing rate
- Spectral centroid
- Spectral rolloff
- Energy in low/mid/high bands

**Total: 59 features per clip**

### Model Options

**Random Forest (Recommended)**:
- Fast inference (~1-2ms on Pi)
- Handles non-linear patterns
- Feature importance analysis
- Robust to overfitting

**SVM (Alternative)**:
- Good for small datasets
- Slightly slower inference
- Better generalization with limited data

### Training Pipeline

```bash
# Train model
python3 scripts/train_chirp_ml.py

# Or use SVM
python3 scripts/train_chirp_ml.py --model-type svm
```

**Output files**:
- `data/chirp_model.pkl` - Trained model
- `data/chirp_scaler.pkl` - Feature scaler (normalization)
- `data/chirp_model_metadata.json` - Model info and metrics

## Performance

**On Raspberry Pi 4**:
- Inference time: ~1-2ms per clip
- Memory: ~5-10MB for model
- CPU: Minimal (Random Forest is very efficient)

**Accuracy improvements**:
- Better handling of edge cases
- Learns from negative examples
- Reduces false positives

## Integration

The ML classifier can run alongside or replace the existing fingerprint system:

1. **Hybrid mode**: Use ML for primary classification, fingerprint as fallback
2. **ML-only mode**: Replace fingerprint entirely
3. **A/B testing**: Compare both systems

## Incremental Learning

When you mark clips as "not chirp", you can retrain:

```bash
# Add new examples to training/not_chirp/
# Then retrain
python3 scripts/train_chirp_ml.py
```

The model learns from all examples, improving over time.

## Deployment

**Local (Mac)**:
```bash
make train-ml
```

**Deploy to Pi**:
```bash
make deploy-ml
```

The model files are small enough to sync via rsync.

## Comparison: Fingerprint vs ML

| Aspect | Fingerprint | ML Model |
|--------|------------|----------|
| Training | Instant | ~5-10 seconds |
| Inference | ~0.5ms | ~1-2ms |
| Accuracy | Good | Better |
| Adaptability | None | Retrainable |
| Features | Spectral only | MFCC + temporal |
| False positives | Manual tuning | Learned rejection |

## Migration Path

1. **Phase 1**: Train ML model, test locally
2. **Phase 2**: Deploy to Pi, run in parallel with fingerprint
3. **Phase 3**: Compare results, tune thresholds
4. **Phase 4**: Switch to ML-only if better

## Troubleshooting

**Model too large for Pi?**
- Use Random Forest with fewer trees: `n_estimators=50`
- Or use SVM (smaller model)

**Inference too slow?**
- Random Forest is already optimized
- Consider quantization (future work)

**Accuracy not improving?**
- Add more training examples (especially edge cases)
- Check feature extraction (verify MFCCs look reasonable)
- Try different model parameters

