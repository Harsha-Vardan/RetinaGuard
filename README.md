# RetinaGuard

**Lesion-Guided, Causally Informed Diabetic Retinopathy Diagnosis**

A research-oriented machine learning framework for robust diabetic retinopathy (DR) grading
from retinal fundus photographs. RetinaGuard investigates whether lesion-guided learning and
domain-robust representation learning can improve generalization across datasets, hospitals,
and imaging devices.

> **⚠️ Research Prototype** — This is NOT a clinically validated diagnostic tool.

## Research Questions

1. **Cross-domain generalization**: How much does DR grading degrade on unseen datasets?
2. **Lesion guidance**: Does lesion supervision improve classification?
3. **Domain robustness**: Can we reduce reliance on dataset-specific features?
4. **Representation separation**: Can disease and acquisition information be disentangled?
5. **Trustworthiness**: Do model explanations target relevant retinal regions?

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Discover and download datasets
python -m src.data.download_datasets

# Audit datasets
python -m src.data.audit_dataset --dataset IDRiD

# Build unified metadata
python -m src.data.build_metadata

# Run preprocessing
python -m src.data.preprocessing --config configs/preprocessing.yaml

# Train baseline
python -m src.training.train_baseline --config configs/baseline.yaml

# Evaluate
python -m src.evaluation.evaluate_cross_domain --config configs/evaluation.yaml
```

## Project Structure

```
retinaguard/
├── configs/          # YAML configuration files
├── data/             # Raw, interim, processed data and metadata
├── checkpoints/      # Model checkpoints
├── experiments/      # Experiment logs and results
├── src/              # Source code modules
├── tests/            # Automated tests
├── docs/             # Documentation
└── outputs/          # Figures, reports, predictions
```

## Datasets

| Dataset | Purpose | Priority |
|---------|---------|----------|
| IDRiD | DR grading + lesion annotations | P0 |
| APTOS 2019 | Five-class DR classification | P1 |
| DDR | DR grading + lesion annotations | P2 |
| EyePACS | Large-scale DR grading | P2 |

## License

Research use only. Individual datasets are subject to their own licenses.
