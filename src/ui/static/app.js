/**
 * RetinaGuard — Clinical Diagnostic UI Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const dropzoneIdle = document.getElementById("dropzone-idle");
  const dropzonePreview = document.getElementById("dropzone-preview");
  const previewThumbnail = document.getElementById("preview-thumbnail");
  const previewName = document.getElementById("preview-name");
  const btnClearImage = document.getElementById("btn-clear-image");
  const btnDiagnose = document.getElementById("btn-diagnose");
  const btnSpinner = document.getElementById("btn-spinner");
  const sampleChipsContainer = document.getElementById("sample-chips-container");

  const labelRetinaguard = document.getElementById("label-retinaguard");
  const labelBaseline = document.getElementById("label-baseline");

  const stateEmpty = document.getElementById("state-empty");
  const stateResults = document.getElementById("state-results");
  const cardDiagnosis = document.getElementById("card-diagnosis");

  const resGradeNum = document.getElementById("res-grade-num");
  const resSeverityTag = document.getElementById("res-severity-tag");
  const resReferableBadge = document.getElementById("res-referable-badge");
  const resReferableText = document.getElementById("res-referable-text");
  const resIcdrDesc = document.getElementById("res-icdr-desc");
  const resClinicalRec = document.getElementById("res-clinical-rec");
  const resConfidencePct = document.getElementById("res-confidence-pct");
  const resConfidenceFill = document.getElementById("res-confidence-fill");
  const probBarsContainer = document.getElementById("prob-bars-container");
  const resXaiMode = document.getElementById("res-xai-mode");
  const captionHeatmap = document.getElementById("caption-heatmap");

  const resConformalBox = document.getElementById("res-conformal-box");
  const resConformalBadge = document.getElementById("res-conformal-badge");
  const resConformalPills = document.getElementById("res-conformal-pills");
  const resConformalNote = document.getElementById("res-conformal-note");

  const imgRaw = document.getElementById("img-raw");
  const imgPrep = document.getElementById("img-prep");
  const imgHeatmap = document.getElementById("img-heatmap");
  const imgOverlay = document.getElementById("img-overlay");

  // State
  let currentImageBase64 = null;
  let currentSampleId = null;
  let currentModelType = "retinaguard";

  // 1. Initial System Status & Samples Fetch
  fetchSystemStatus();
  fetchSampleImages();

  // 2. Model Selection Toggle
  document.querySelectorAll("input[name='model_type']").forEach((radio) => {
    radio.addEventListener("change", (e) => {
      currentModelType = e.target.value;
      if (currentModelType === "retinaguard") {
        labelRetinaguard.classList.add("active");
        labelBaseline.classList.remove("active");
      } else {
        labelBaseline.classList.add("active");
        labelRetinaguard.classList.remove("active");
      }
    });
  });

  // 3. Dropzone & File Input Handlers
  dropzone.addEventListener("click", () => {
    if (!currentImageBase64 && !currentSampleId) {
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleUploadedFile(e.target.files[0]);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUploadedFile(e.dataTransfer.files[0]);
    }
  });

  btnClearImage.addEventListener("click", (e) => {
    e.stopPropagation();
    resetImageSelection();
  });

  // 4. Diagnose Button Handler
  btnDiagnose.addEventListener("click", () => {
    executeDiagnosis();
  });

  // Functions
  function handleUploadedFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file (JPEG, PNG, TIFF).");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      currentImageBase64 = e.target.result;
      currentSampleId = null;

      // Unselect any active sample chip
      document.querySelectorAll(".sample-chip").forEach((chip) => chip.classList.remove("selected"));

      // Update dropzone UI
      previewThumbnail.src = currentImageBase64;
      previewName.textContent = file.name;
      dropzoneIdle.classList.add("hidden");
      dropzonePreview.classList.remove("hidden");

      btnDiagnose.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function resetImageSelection() {
    currentImageBase64 = null;
    currentSampleId = null;
    fileInput.value = "";
    previewThumbnail.src = "";
    dropzonePreview.classList.add("hidden");
    dropzoneIdle.classList.remove("hidden");
    btnDiagnose.disabled = true;

    document.querySelectorAll(".sample-chip").forEach((chip) => chip.classList.remove("selected"));
  }

  async function fetchSystemStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      const statusPill = document.getElementById("pill-status");
      if (data.status === "online") {
        statusPill.textContent = `● System Online (${data.device.toUpperCase()})`;
      }
    } catch (err) {
      console.warn("Could not check status:", err);
    }
  }

  async function fetchSampleImages() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      sampleChipsContainer.innerHTML = "";

      if (!data.samples || data.samples.length === 0) {
        sampleChipsContainer.innerHTML = "<div class='chip-loading'>No samples found.</div>";
        return;
      }

      data.samples.forEach((sample) => {
        const chip = document.createElement("div");
        chip.className = "sample-chip";
        chip.dataset.id = sample.id;

        chip.innerHTML = `
          <div class="chip-left">
            <span class="chip-dataset">${sample.dataset}</span>
            <span class="chip-desc">${sample.description}</span>
          </div>
          <span class="chip-arrow">→</span>
        `;

        chip.addEventListener("click", () => {
          selectSample(sample, chip);
        });

        sampleChipsContainer.appendChild(chip);
      });
    } catch (err) {
      sampleChipsContainer.innerHTML = "<div class='chip-loading'>Error loading samples.</div>";
    }
  }

  function selectSample(sample, chipElement) {
    document.querySelectorAll(".sample-chip").forEach((c) => c.classList.remove("selected"));
    chipElement.classList.add("selected");

    currentSampleId = sample.id;
    currentImageBase64 = null;

    // Load sample image for thumbnail
    previewThumbnail.src = `/api/sample-image?id=${encodeURIComponent(sample.id)}`;
    previewName.textContent = `${sample.filename} — ${sample.description}`;

    dropzoneIdle.classList.add("hidden");
    dropzonePreview.classList.remove("hidden");
    btnDiagnose.disabled = false;
  }

  async function executeDiagnosis() {
    btnDiagnose.disabled = true;
    btnSpinner.classList.remove("hidden");

    const payload = {
      model_type: currentModelType,
    };

    if (currentSampleId) {
      payload.image_id = currentSampleId;
    } else if (currentImageBase64) {
      payload.image_base64 = currentImageBase64;
    } else {
      alert("Please upload or select an image first.");
      btnDiagnose.disabled = false;
      btnSpinner.classList.add("hidden");
      return;
    }

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const result = await response.json();
      if (result.error) {
        throw new Error(result.error);
      }

      renderResults(result);

    } catch (err) {
      alert(`Diagnosis Error: ${err.message}`);
    } finally {
      btnDiagnose.disabled = false;
      btnSpinner.classList.add("hidden");
    }
  }

  function renderResults(result) {
    stateEmpty.classList.add("hidden");
    stateResults.classList.remove("hidden");

    // Smooth scroll to results
    stateResults.scrollIntoView({ behavior: "smooth", block: "start" });

    // 1. Grade Badge & Labels
    resGradeNum.textContent = `Grade ${result.predicted_grade}`;
    resSeverityTag.textContent = result.severity.toUpperCase();

    // Reset card classes
    cardDiagnosis.className = `card result-card severity-${result.severity}`;

    // 2. Referable DR Indicator
    if (result.referable_dr) {
      resReferableBadge.className = "referable-badge is-referable";
      resReferableText.textContent = "REFERABLE DR DETECTED";
    } else {
      resReferableBadge.className = "referable-badge non-referable";
      resReferableText.textContent = "NON-REFERABLE DR";
    }

    // 3. Clinical Info
    resIcdrDesc.textContent = result.icdr_description;
    resClinicalRec.textContent = result.clinical_recommendation;

    // 4. Confidence
    resConfidencePct.textContent = `${result.confidence}%`;
    resConfidenceFill.style.width = `${result.confidence}%`;

    // 4b. Conformal Risk-Controlled Prediction Sets
    if (result.conformal_prediction) {
      const cp = result.conformal_prediction;
      resConformalBox.classList.remove("hidden");
      resConformalPills.innerHTML = "";

      cp.prediction_set.forEach((pText) => {
        const pill = document.createElement("span");
        pill.className = "set-pill";
        pill.textContent = pText;
        resConformalPills.appendChild(pill);
      });

      if (cp.is_singleton) {
        resConformalBadge.className = "conformal-badge safe";
        resConformalBadge.textContent = "CERTIFIED SAFE";
        resConformalNote.textContent = `High-confidence single grade prediction (Finite-sample coverage bound P(Y ∈ C(X)) ≥ 90%). Safe for automated reporting.`;
      } else {
        resConformalBadge.className = "conformal-badge triage";
        resConformalBadge.textContent = "SPECIALIST TRIAGE";
        resConformalNote.textContent = `Borderline ambiguity detected across ${cp.set_size} grades. Model safely abstains from single-label diagnosis and triages to retinal specialist (Coverage guarantee P(Y ∈ C(X)) ≥ 90% satisfied).`;
      }
    } else {
      resConformalBox.classList.add("hidden");
    }

    // 5. Probabilities Bars
    probBarsContainer.innerHTML = "";
    const grades = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"];
    const gradeLabels = ["0 (No DR)", "1 (Mild)", "2 (Moderate)", "3 (Severe)", "4 (PDR)"];

    grades.forEach((gKey, idx) => {
      const pct = result.probabilities[gKey] || 0;
      const isActive = idx === result.predicted_grade;

      const row = document.createElement("div");
      row.className = `prob-row ${isActive ? "active-grade" : ""}`;

      row.innerHTML = `
        <span class="prob-name">${gradeLabels[idx]}</span>
        <div class="prob-track">
          <div class="prob-val-fill" style="width: ${pct}%;"></div>
        </div>
        <span class="prob-pct">${pct}%</span>
      `;
      probBarsContainer.appendChild(row);
    });

    // 6. Visual Attribution Gallery
    imgRaw.src = result.images.raw;
    imgPrep.src = result.images.preprocessed;
    imgHeatmap.src = result.images.heatmap;
    imgOverlay.src = result.images.overlay;

    if (result.model_key === "retinaguard") {
      resXaiMode.textContent = "Spatial Lesion Attention";
      captionHeatmap.textContent = "Learned auxiliary lesion attention maps";
    } else {
      resXaiMode.textContent = "Grad-CAM Activation";
      captionHeatmap.textContent = "Backbone layer4 convolutional gradients";
    }
  }
});
