# Clinical AI Ethics, Patient Privacy, and Governance

## 1. Ethical Mission
The objective of the RetinaGuard project is to address clinical disparities and diagnostic decay in AI-assisted diabetic retinopathy screening caused by demographic, scanner, and institutional distribution shifts. By explicitly isolating disease-causal lesion features from site-specific shortcuts, the system aims to promote equitable diagnostic fidelity regardless of clinical setting or imaging hardware.

---

## 2. Patient Data Privacy and De-Identification
- **DICOM / EXIF Sanitization**: All medical images utilized in this framework undergo automatic de-identification. Headers containing Protected Health Information (PHI) such as patient names, dates of birth, social security numbers, or internal medical record numbers (MRNs) are permanently expunged.
- **Salted Hash Identifiers**: Patient and visit tracking is maintained exclusively through cryptographic salted SHA-256 hashes (`hash_patient_id`), preventing re-identification while preserving patient-level split integrity.
- **Zero Cloud Leakage**: Patient imaging data is maintained strictly within local encrypted environments. No raw patient records are transmitted to third-party APIs or unvetted cloud stores.

---

## 3. Institutional Review Board (IRB) Compliance
- Public cohorts (IDRiD, APTOS 2019, DDR, EyePACS) were obtained under open-access research licenses governed by their respective primary institutional ethics review boards.
- Private multi-site hospital datasets require explicit IRB protocol approval from the participating institutions. If utilized, ethics committee approvals are recorded under reference `IRB-RETINAGUARD-CLINICAL-2026-A`.

---

## 4. Human-in-the-Loop Safeguards
- RetinaGuard is engineered strictly as an assistive diagnostic decision-support system (AI-DSS).
- Outputs include explicit multi-panel explanation figures (lesion attention maps, predictive entropy, and referable triage recommendations) designed to augment clinical scrutiny rather than replace ophthalmologist evaluation.
- The system incorporates selective prediction (abstention) triggers when encountering extreme sensor blur, illumination failure, or ungradable image quality.
