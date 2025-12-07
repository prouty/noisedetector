# Scripts Directory Refactoring Plan

## Overview
This plan refactors the `scripts/` directory to follow SOLID principles, eliminate duplication, and properly separate concerns between scripts (CLI entry points) and core modules (reusable business logic).

## Goals
1. **Extract reusable logic** from scripts into `core/` modules
2. **Eliminate duplication** between scripts
3. **Make scripts thin** - they should only handle CLI args and call core functions
4. **Improve testability** - core modules can be unit tested
5. **Maintain backward compatibility** - existing scripts continue to work

## Current Issues

### Duplication
- `classify_chirp_ml.py` duplicates classification logic (should use `core/classifier.py`)
- `capture_ml.py` duplicates capture decision logic (should use `core/classifier.py`)
- Feature extraction duplicated across `train_chirp_ml.py`, `classify_chirp_ml.py`, `analyze_clips.py`
- Email/report logic duplicated in `email_report.py` and `generate_chirp_report.py`

### Mixed Concerns
- Scripts contain business logic that should be in core modules
- Feature extraction, model training, and file I/O all mixed together
- Email sending, report generation, and data loading in same files

### Missing Abstractions
- No shared feature extraction module
- No shared email/reporting module
- No shared training utilities module

## Proposed Structure

### New Core Modules

#### `core/features.py`
**Purpose**: Feature extraction for audio analysis
**Functions to extract**:
- `load_mono_wav()` - from `train_chirp_ml.py`, `train_chirp_fingerprint.py`, `analyze_clips.py`
- `extract_mfcc_features()` - from `train_chirp_ml.py`, `classify_chirp_ml.py`
- `extract_additional_features()` - from `train_chirp_ml.py`
- `compute_spectral_features()` - from `analyze_clips.py`
- `compute_temporal_features()` - from `analyze_clips.py`
- `compute_avg_spectrum()` - from `train_chirp_fingerprint.py`
- `create_mel_filterbank()` - from `train_chirp_ml.py`
- `dct()` - from `train_chirp_ml.py`

**Dependencies**: numpy, wave

#### `core/email.py`
**Purpose**: Email functionality
**Classes/Functions to extract**:
- `EmailConfig` - dataclass for email configuration
- `get_email_config()` - from `email_report.py`
- `send_email()` - from `email_report.py`
- `EmailSender` class (optional) - wrapper for email operations

**Dependencies**: smtplib, email.mime

#### `core/reporting.py`
**Purpose**: Report generation
**Functions to extract**:
- `load_events()` - from `email_report.py`, `generate_chirp_report.py`, `validate_classification.py`, etc.
- `filter_recent_events()` - from `email_report.py`
- `generate_email_report()` - from `email_report.py`
- `generate_chirp_report()` - from `generate_chirp_report.py`
- `add_date_column()` - from `generate_chirp_report.py`
- `choose_latest_date()` - from `generate_chirp_report.py`

**Dependencies**: pandas, datetime

#### `core/training.py`
**Purpose**: Training utilities
**Functions to extract**:
- `load_training_data()` - from `train_chirp_ml.py`
- `prepare_training_data()` - common pattern across training scripts
- `save_model()` - common model saving logic
- `load_model()` - common model loading logic

**Dependencies**: joblib, numpy, pathlib

### Scripts to Refactor

#### `scripts/email_report.py`
**Current**: ~290 lines, mixes data loading, report generation, email sending
**After**: ~50 lines, thin wrapper that:
- Parses CLI args
- Calls `core.reporting.load_events()`
- Calls `core.reporting.generate_email_report()`
- Calls `core.email.send_email()`

#### `scripts/generate_chirp_report.py`
**Current**: ~120 lines, duplicates event loading
**After**: ~30 lines, uses `core.reporting` functions

#### `scripts/train_chirp_ml.py`
**Current**: ~430 lines, mixes feature extraction, training, file I/O
**After**: ~150 lines, uses `core.features` and `core.training`

#### `scripts/classify_chirp_ml.py`
**Status**: **DELETE** - functionality already in `core/classifier.py`
**Action**: Verify all functionality is in `core/classifier.py`, then remove

#### `scripts/capture_ml.py`
**Status**: **REVIEW** - check if functionality is in `core/classifier.py` or `core/detector.py`
**Action**: If duplicate, remove; if unique, extract to `core/`

#### `scripts/test_chirp_ml.py`
**Status**: **REVIEW** - development/testing script
**Action**: Keep if useful for development, or move to `tests/` directory

### Scripts to Keep As-Is (Minor Changes)

These scripts are already well-structured or are simple utilities:
- `check_chirps.py` - Simple query script, could use `core.reporting.load_events()`
- `mark_clip.py` - Training data management, mostly fine
- `validate_classification.py` - Validation script, could use `core.reporting.load_events()`
- `tune_thresholds.py` - Threshold tuning, mostly fine
- `rediagnose_events.py` - Re-classification, mostly fine
- `compare_classifiers.py` - Comparison tool, mostly fine
- `diagnose_event.py` - Diagnosis tool, mostly fine
- `merge_events.py` - CSV merging utility, fine as-is
- `pull_*.py` - Data sync utilities, fine as-is
- `health_check.py` - System health check, fine as-is
- `debug_state.py` - Debug utility, fine as-is
- `analyze_clips.py` - Analysis tool, should use `core.features`

## Implementation Steps

### Phase 1: Create Core Modules (No Breaking Changes)
1. **Create `core/features.py`**
   - Extract feature extraction functions
   - Add unit tests
   - Update imports in scripts (backward compatible)

2. **Create `core/email.py`**
   - Extract email functions from `email_report.py`
   - Add unit tests
   - Update `email_report.py` to use it

3. **Create `core/reporting.py`**
   - Extract report generation functions
   - Extract `load_events()` helper
   - Add unit tests
   - Update scripts to use it

4. **Create `core/training.py`**
   - Extract training utilities
   - Add unit tests
   - Update training scripts to use it

### Phase 2: Refactor Scripts (Backward Compatible)
5. **Refactor `email_report.py`**
   - Use `core.reporting` and `core.email`
   - Verify functionality unchanged

6. **Refactor `generate_chirp_report.py`**
   - Use `core.reporting`
   - Verify functionality unchanged

7. **Refactor `train_chirp_ml.py`**
   - Use `core.features` and `core.training`
   - Verify functionality unchanged

8. **Refactor `analyze_clips.py`**
   - Use `core.features`
   - Verify functionality unchanged

### Phase 3: Remove Duplicates
9. **Remove `classify_chirp_ml.py`**
   - Verify all functionality in `core/classifier.py`
   - Check for any imports/references
   - Remove file

10. **Review and remove `capture_ml.py`**
    - Check if functionality exists in `core/`
    - Remove if duplicate

11. **Review `test_chirp_ml.py`**
    - Move to `tests/` if it's a test
    - Remove if no longer needed

### Phase 4: Update Remaining Scripts
12. **Update scripts to use `core.reporting.load_events()`**
    - `check_chirps.py`
    - `validate_classification.py`
    - `compare_classifiers.py`
    - `rediagnose_events.py`
    - Any others that load events.csv

13. **Update scripts to use `core.features`**
    - Any scripts doing feature extraction

### Phase 5: Documentation & Testing
14. **Update documentation**
    - Update `docs/USAGE.md` if script interfaces change
    - Document new core modules in `docs/ARCHITECTURE.md`

15. **Add unit tests**
    - Test `core/features.py`
    - Test `core/email.py`
    - Test `core/reporting.py`
    - Test `core/training.py`

## File Structure After Refactoring

```
core/
  ├── __init__.py
  ├── audio.py          (existing)
  ├── baseline.py        (existing)
  ├── classifier.py     (existing)
  ├── detector.py       (existing)
  ├── repository.py     (existing)
  ├── features.py        (NEW - feature extraction)
  ├── email.py           (NEW - email functionality)
  ├── reporting.py       (NEW - report generation)
  └── training.py        (NEW - training utilities)

scripts/
  ├── email_report.py           (refactored - thin wrapper)
  ├── generate_chirp_report.py  (refactored - uses core.reporting)
  ├── train_chirp_ml.py         (refactored - uses core.features, core.training)
  ├── train_chirp_fingerprint.py (refactored - uses core.features)
  ├── analyze_clips.py          (refactored - uses core.features)
  ├── check_chirps.py           (updated - uses core.reporting)
  ├── validate_classification.py (updated - uses core.reporting)
  ├── [other scripts - minor updates or unchanged]
  └── [removed: classify_chirp_ml.py, capture_ml.py?]

tests/  (optional - for unit tests)
  ├── test_features.py
  ├── test_email.py
  ├── test_reporting.py
  └── test_training.py
```

## Benefits

1. **Reduced duplication**: Feature extraction, event loading, email sending in one place
2. **Better testability**: Core modules can be unit tested independently
3. **Easier maintenance**: Changes to feature extraction affect all scripts automatically
4. **Clearer separation**: Scripts are thin CLI wrappers, core contains business logic
5. **Reusability**: Core modules can be used by other parts of the application

## Risks & Mitigation

1. **Breaking changes**: Mitigate by keeping script interfaces the same
2. **Import errors**: Test all scripts after each phase
3. **Missing functionality**: Verify all functionality is preserved before removing files
4. **Circular imports**: Careful module design to avoid circular dependencies

## Success Criteria

- [ ] All core modules created and tested
- [ ] All scripts refactored to use core modules
- [ ] No duplicate code between scripts
- [ ] All existing functionality preserved
- [ ] Unit tests for core modules
- [ ] Documentation updated
- [ ] All scripts still work as before

## Notes

- This refactoring maintains backward compatibility - existing scripts continue to work
- Each phase can be done independently and tested
- Can stop at any phase if issues arise
- Focus on extracting common patterns first, then refactoring scripts

