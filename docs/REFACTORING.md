# Code Refactoring - SOLID Principles

## Overview

The codebase has been refactored to follow SOLID principles and establish a consistent architectural pattern. The new architecture separates concerns, uses dependency injection, and makes the codebase more maintainable and testable.

## Architecture

### Core Module (`core/`)

The `core/` module contains the domain model following SOLID principles:

#### Single Responsibility Principle (SRP)

Each class has one clear responsibility:

- **`AudioCapture`**: Handles audio I/O from ALSA arecord
- **`BaselineTracker`**: Manages baseline noise level calculation
- **`EventDetector`**: Detects noise events above threshold
- **`EventRepository`**: Handles event CSV file operations
- **`SegmentRepository`**: Handles WAV segment file operations

#### Open/Closed Principle (OCP)

The `Classifier` interface allows adding new classifier types without modifying existing code:

- **`Classifier`** (ABC): Abstract base class defining the interface
- **`FingerprintClassifier`**: Spectral fingerprint-based classification
- **`MLClassifier`**: Machine learning-based classification

New classifiers can be added by implementing the `Classifier` interface.

#### Liskov Substitution Principle (LSP)

All classifier implementations are interchangeable - any `Classifier` can be used wherever a classifier is expected.

#### Interface Segregation Principle (ISP)

Interfaces are small and focused:
- `Classifier.classify()` - Single method for classification
- `Classifier.is_available()` - Check if classifier is ready

#### Dependency Inversion Principle (DIP)

High-level modules depend on abstractions:
- `EventDetector` depends on `BaselineTracker` (abstraction)
- `run_monitor()` depends on `Classifier` interface, not concrete implementations
- Components are injected via constructor (dependency injection)

## File Structure

```
core/
├── __init__.py          # Module exports
├── audio.py             # AudioCapture, AudioChunk
├── baseline.py          # BaselineTracker
├── classifier.py        # Classifier interface and implementations
├── detector.py          # EventDetector, Event
└── repository.py        # EventRepository, SegmentRepository
```

## Usage

### Refactored Monitor

The refactored `monitor.py` demonstrates the clean architecture:

```python
from core import (
    AudioCapture,
    BaselineTracker,
    EventDetector,
    EventRepository,
    SegmentRepository,
    create_classifier
)

# Initialize components (dependency injection)
audio_capture = AudioCapture(config)
baseline_tracker = BaselineTracker(config)
event_detector = EventDetector(config, baseline_tracker)
event_repo = EventRepository(config)
segment_repo = SegmentRepository(config)
classifier = create_classifier(config)

# Use components
audio_capture.start()
while audio_capture.is_running():
    chunk = audio_capture.read_chunk()
    event = event_detector.process_chunk(chunk, chunk.raw_bytes)
    if event:
        # Process event...
```

## Benefits

1. **Testability**: Each component can be tested in isolation
2. **Maintainability**: Clear separation of concerns makes code easier to understand
3. **Extensibility**: New features can be added without modifying existing code
4. **Reusability**: Components can be reused in different contexts
5. **Flexibility**: Easy to swap implementations (e.g., different classifiers)

## Migration Status

The refactoring is complete. The old `monitor.py` has been replaced with the refactored version that uses the `core/` module architecture. The old code is preserved in `monitor_old_backup.py` for reference.

## Design Patterns Used

- **Repository Pattern**: `EventRepository`, `SegmentRepository` abstract data persistence
- **Strategy Pattern**: `Classifier` interface with multiple implementations
- **Dependency Injection**: Components injected via constructors
- **Factory Pattern**: `create_classifier()` factory function

## Comparison: Before vs After

### Before (monitor.py)
- 1400+ lines in single file
- Mixed responsibilities
- Tight coupling
- Hard to test
- Global state

### After (core/ + monitor.py)
- Separated into focused classes
- Clear responsibilities
- Loose coupling via interfaces
- Easy to test (mock dependencies)
- No global state

