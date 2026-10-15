# Datasheet for Diabetic Retinopathy Datasets (Gebru et al. Standard)

## 1. Motivation
- **For what purpose was the dataset created?**  
  To benchmark automated diabetic retinopathy (DR) grading algorithms and validate out-of-distribution generalizability across distinct clinical capture protocols, camera devices, and demographic populations.
- **Who created the dataset and on behalf of which entity?**  
  - *IDRiD*: Dr. Prasanna Porwal et al., Shri Guru Gobind Singhji Institute of Engineering and Technology, Nanded, India.
  - *APTOS 2019*: Aravind Eye Hospital, India, in collaboration with the Asia Pacific Tele-Ophthalmology Society.
  - *EyePACS / DDR*: EyePACS LLC and CAS respectively.

---

## 2. Composition
- **What do the instances represent?**  
  Digital color retinal fundus photographs captured via table-top fundus cameras (e.g., Kowa VX-10α, Topcon TRC-NW8, Centervue DRS).
- **How many instances are there in total?**  
  - IDRiD: 516 graded images (train: 413, test: 103), plus 81 images with multi-lesion pixel masks.
  - APTOS 2019: 3,662 labeled images.
  - DDR: 13,673 images.
  - EyePACS: 88,702 images.
- **Does the dataset contain all possible instances or is it a sample?**  
  Sample of patients screened at participating eye clinics and tertiary ophthalmology departments.
- **What data does each instance consist of?**  
  RGB color fundus photograph, international DR severity grade (ICDR 0–4), optional diabetic macular edema (DME) status, and optional binary lesion masks (Microaneurysms, Haemorrhages, Hard Exudates, Soft Exudates).
- **Are labels assigned to instances?**  
  Yes, clinical ground truth graded by expert ophthalmologists and retina specialists.
- **Is any information missing?**  
  Device identifiers are partially missing in competition subsets; addressed via empirical device-proxy feature clustering.

---

## 3. Collection Process
- **How was the data acquired?**  
  Standard mydriatic or non-mydriatic fundus cameras with fields of view between 45° and 50°, centered on the macula and optic disc.
- **Who was involved in the data collection?**  
  Certified clinical staff, optometrists, and retina fellows.

---

## 4. Preprocessing, Cleaning, and Labeling
- **Was any preprocessing applied?**  
  FOV segmentation and border cropping, Graham local illumination equalization / CLAHE, aspect ratio preservation, and standard 512×512 resolution resampling.
- **Were raw data kept?**  
  Yes, raw unaltered source images are archived under `data/raw/`.

---

## 5. Uses
- **Has the dataset been used for previous tasks?**  
  Widely used in Kaggle benchmarks, IEEE ISBI Grand Challenges, and international clinical AI trials.
- **Is there a repository that links to relevant publications?**  
  Full references are cataloged in `paper/refs.bib` and `docs/DATA_ACCESS.md`.

---

## 6. Distribution and Maintenance
- **How will the dataset be distributed?**  
  Under CC BY 4.0 (IDRiD) and Kaggle research competition terms (APTOS).
- **Who supports and hosts the dataset?**  
  IEEE DataPort, Grand Challenge, and Kaggle.
