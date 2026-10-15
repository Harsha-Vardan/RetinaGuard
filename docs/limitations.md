# Limitations and Clinical Caveats

To ensure scientific honesty and medical transparency, the following technical and clinical limitations of the current framework are explicitly documented:

---

## 1. Ground Truth Lesion Annotations
- Pixel-level segmentation masks for all four cardinal DR lesions (Microaneurysms, Haemorrhages, Hard Exudates, Soft Exudates) are currently provided in the IDRiD cohort (81 annotated training images).
- Larger registries such as APTOS and EyePACS provide image-level grades without dense lesion annotations. In unannotated cohorts, attention supervision relies on weak self-guidance or pseudo-lesion maps derived from the trained spatial attention head. Readers and practitioners must recognize that pseudo-lesion maps are model-derived hypotheses and not clinician-certified segmentations.

---

## 2. Device Information and Proxy Clustering
- True camera hardware parameters (e.g., optical lens sensor dimensions, exact illumination flash wavelength, sensor pixel pitch) are often stripped in anonymized public competition archives.
- While the framework incorporates a multi-feature device-proxy clustering module (resolution, FOV diameter, color temperature, vignetting profile), this remains an empirical proxy rather than a hardware ground truth.

---

## 3. Prevalence and Spectrum Bias
- Disease prevalence differs significantly across screening cohorts (e.g., higher severity in tertiary referral centers like IDRiD compared to primary screening programs like EyePACS).
- While causal disentanglement penalizes spurious statistical correlation, severe class imbalance (especially in Grade 3 and Grade 4) requires class-balanced sampling to maintain sensitivity.

---

## 4. Regulatory and Clinical Status
- **Research Software Only**: This codebase, including trained checkpoints, inference modules, and web demonstration tools, is distributed strictly for scientific and academic research.
- It has not received clearance or approval from the U.S. FDA, European CE-MDR, or national regulatory bodies as a medical diagnostic device.
