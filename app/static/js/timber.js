/**
 * Frontend JavaScript for the Timber Grading tab (SmartTimber-LK Module 3).
 * Fully independent from app.js's tree detection/classification/maturity logic:
 * its own upload flow, its own /api/timber/* endpoints, its own in-memory crop
 * session on the backend. The only thing shared with app.js is the tab nav
 * highlighting/panel-visibility wiring that already lives there.
 *
 * Exposes window.timberGradedCrops (+ a 'timber:cropGraded' CustomEvent) purely
 * as a UI-layer bridge so the Auction Bid Price tabs (auction.js) can offer
 * "use this graded crop" prefills - no backend coupling either direction.
 */

window.timberGradedCrops = window.timberGradedCrops || [];

document.addEventListener('DOMContentLoaded', () => {
    const tabTimber = document.getElementById('tabTimber');
    if (!tabTimber) return; // Timber Grading panel not present on this page

    const dropzone = document.getElementById('timberDropzone');
    const fileInput = document.getElementById('timberImageInput');
    const browseBtn = document.getElementById('timberBrowseBtn');
    const dropzonePrompt = document.getElementById('timberDropzonePrompt');
    const previewContainer = document.getElementById('timberPreviewContainer');
    const fileListEl = document.getElementById('timberFileList');
    const detectBtn = document.getElementById('timberDetectBtn');
    const detectStepper = document.getElementById('timberDetectStepper');
    const detectStatus = document.getElementById('timberDetectStatus');
    const detectionResults = document.getElementById('timberDetectionResults');
    const gradeCard = document.getElementById('timberGradeCard');
    const gradeBtn = document.getElementById('timberGradeBtn');
    const gradeStepper = document.getElementById('timberGradeStepper');
    const gradeStatus = document.getElementById('timberGradeStatus');

    let selectedFiles = [];
    let sessionId = null;
    let cropRecords = {}; // crop_id -> { removed: bool, cardEl, lengthInput, girthInput, volumeInput, statusBadge, gradeBody }

    function escapeHtml(value) {
        return String(value ?? '')
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#039;');
    }

    function setStatus(el, message, isError = false) {
        if (!el) return;
        el.textContent = message;
        el.classList.toggle('error', isError);
    }

    function stepperStart(containerEl, prefix, count) {
        if (containerEl) containerEl.classList.remove('hidden');
        if (containerEl) updateProgress(containerEl, 10, 'Analysis in progress');
        for (let i = 1; i <= count; i += 1) {
            const el = document.getElementById(`${prefix}${i}`);
            if (el) el.className = 'stepper-step active';
        }
    }

    function stepperFinish(prefix, count, success) {
        for (let i = 1; i <= count; i += 1) {
            const el = document.getElementById(`${prefix}${i}`);
            if (el) el.className = success ? 'stepper-step complete' : 'stepper-step';
        }
        const first = document.getElementById(`${prefix}1`);
        const container = first && first.closest('.stepper');
        if (container) updateProgress(container, success ? 100 : 0, success ? 'Analysis complete' : 'Analysis stopped', !success);
    }

    function updateProgress(container, percent, label, isError = false) {
        let progress = container.querySelector('.embedded-progress');
        if (!progress) {
            progress = document.createElement('div');
            progress.className = 'embedded-progress';
            progress.innerHTML = '<div><span class="embedded-progress-label"></span><strong class="embedded-progress-percent"></strong></div><div class="compact-progress-track"><span></span></div>';
            container.prepend(progress);
        }
        progress.classList.toggle('error', isError);
        progress.classList.toggle('running', percent > 0 && percent < 100);
        progress.querySelector('.embedded-progress-label').textContent = label;
        progress.querySelector('.embedded-progress-percent').textContent = `${percent}%`;
        progress.querySelector('.compact-progress-track span').style.width = `${percent}%`;
    }

    // --- File selection (multi-file, own dropzone) ---

    function renderFileList() {
        if (!selectedFiles.length) {
            previewContainer.classList.add('hidden');
            dropzonePrompt.classList.remove('hidden');
            detectBtn.disabled = true;
            fileListEl.innerHTML = '';
            return;
        }
        dropzonePrompt.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        detectBtn.disabled = false;
        fileListEl.innerHTML = selectedFiles
            .map((file, idx) => `
                <div class="timber-file-chip" data-idx="${idx}">
                    <span class="file-name">${escapeHtml(file.name)}</span>
                    <span class="file-size">${(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                    <button type="button" class="btn-icon timber-remove-file" data-idx="${idx}" title="Remove">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                    </button>
                </div>
            `)
            .join('');
        fileListEl.querySelectorAll('.timber-remove-file').forEach((btn) => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const idx = parseInt(e.currentTarget.dataset.idx, 10);
                selectedFiles.splice(idx, 1);
                renderFileList();
            });
        });
    }

    function addFiles(fileList) {
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
        let rejected = 0;
        for (const file of Array.from(fileList)) {
            if (!validTypes.includes(file.type) || file.size > 20 * 1024 * 1024) { rejected += 1; continue; }
            if (selectedFiles.some((selected) => selected.name === file.name && selected.size === file.size)) continue;
            selectedFiles.push(file);
        }
        renderFileList();
        setStatus(detectStatus, rejected ? `${rejected} unsupported or oversized file(s) were skipped.` : `${selectedFiles.length} photo(s) ready for detection.`, rejected > 0);
    }

    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
    dropzone.addEventListener('click', () => {
        if (!selectedFiles.length) fileInput.click();
    });
    dropzone.addEventListener('keydown', (event) => {
        if ((event.key === 'Enter' || event.key === ' ') && !selectedFiles.length) {
            event.preventDefault();
            fileInput.click();
        }
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length) addFiles(e.target.files);
        fileInput.value = '';
    });
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
    });
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('drag-over');
    });
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
        if (e.dataTransfer.files && e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
    });

    // --- Detect crops ---

    function volumeFromInputs(lengthM, girthM) {
        if (!(lengthM > 0) || !(girthM > 0)) return null;
        return (lengthM * girthM * girthM) / (4 * Math.PI);
    }

    function renderCropCard(imageResult, crop, staggerIdx) {
        const card = document.createElement('div');
        card.className = 'card tree-card fade-in-up';
        card.style.animationDelay = `${(staggerIdx || 0) * 50}ms`;
        card.id = `timberCrop_${crop.crop_id}`;

        const confPct = crop.confidence !== null && crop.confidence !== undefined ? `${(crop.confidence * 100).toFixed(1)}%` : 'N/A';
        const visPct = crop.visible_ratio !== null && crop.visible_ratio !== undefined ? `${(crop.visible_ratio * 100).toFixed(1)}%` : 'N/A';
        const sharp = crop.sharpness_score !== null && crop.sharpness_score !== undefined ? crop.sharpness_score.toFixed(1) : 'N/A';

        card.innerHTML = `
            <div class="tree-card-header">
                <span class="tree-name">${escapeHtml(crop.label)} &mdash; ${escapeHtml(imageResult.filename)}</span>
                <span class="tree-badge badge-success" id="timberCropStatus_${crop.crop_id}">Pending Measurements</span>
            </div>
            <div class="tree-card-body">
                <div class="timber-crop-layout">
                    <img class="timber-crop-preview" src="${crop.preview}" alt="${escapeHtml(crop.label)} preview">
                    <div>
                        <div class="tree-metrics">
                            <div class="metric-item">
                                <span class="label">Detection Method</span>
                                <span class="val">${escapeHtml(crop.method || 'N/A')}</span>
                            </div>
                            <div class="metric-item">
                                <span class="label">Confidence</span>
                                <span class="val">${confPct}</span>
                            </div>
                            <div class="metric-item">
                                <span class="label">Visible Face</span>
                                <span class="val">${visPct}</span>
                            </div>
                            <div class="metric-item">
                                <span class="label">Sharpness</span>
                                <span class="val">${sharp}</span>
                            </div>
                        </div>
                        <p style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 12px;">
                            Enter the <strong>whole log's</strong> length and mid-girth (not just this cropped face) &mdash;
                            these drive this log's volume, which is the single biggest factor in the Bid Price estimate on the
                            "Grading + Bid Price" tab.
                        </p>
                        <div class="mat-form-grid" style="margin-top: 10px;">
                            <label>
                                <span>Full Log Length (m)</span>
                                <input type="number" min="0" step="0.01" class="timber-length-input">
                            </label>
                            <label>
                                <span>Full Log Mid Girth (m)</span>
                                <input type="number" min="0" step="0.01" class="timber-girth-input">
                            </label>
                            <label class="wide">
                                <span>Volume (m&sup3;, computed)</span>
                                <input type="text" readonly class="timber-volume-input" value="-">
                            </label>
                        </div>
                        <div class="action-bar">
                            <button type="button" class="btn btn-secondary timber-remove-crop">Remove This Crop</button>
                        </div>
                        <p class="timber-card-validation">Enter both measurements to include this log in grading.</p>
                        <div class="timber-grade-body"></div>
                    </div>
                </div>
            </div>
        `;

        const lengthInput = card.querySelector('.timber-length-input');
        const girthInput = card.querySelector('.timber-girth-input');
        const volumeInput = card.querySelector('.timber-volume-input');
        const statusBadge = card.querySelector(`#timberCropStatus_${crop.crop_id}`);
        const gradeBody = card.querySelector('.timber-grade-body');
        const removeBtn = card.querySelector('.timber-remove-crop');

        function refreshVolume() {
            const volume = volumeFromInputs(parseFloat(lengthInput.value), parseFloat(girthInput.value));
            volumeInput.value = volume !== null ? volume.toFixed(4) : '-';
            card.classList.toggle('measurement-ready', volume !== null);
            card.querySelector('.timber-card-validation').textContent = volume !== null ? 'Ready for grading.' : 'Enter both measurements to include this log in grading.';
            updateGradeReadiness();
        }
        lengthInput.addEventListener('input', refreshVolume);
        girthInput.addEventListener('input', refreshVolume);

        removeBtn.addEventListener('click', () => {
            const record = cropRecords[crop.crop_id];
            if (!record) return;
            record.removed = true;
            card.classList.add('hidden');
            updateGradeReadiness();
        });

        cropRecords[crop.crop_id] = {
            removed: false,
            cardEl: card,
            lengthInput,
            girthInput,
            statusBadge,
            gradeBody,
        };

        return card;
    }

    function updateGradeReadiness() {
        const ready = Object.values(cropRecords).filter((record) => !record.removed && parseFloat(record.lengthInput.value) > 0 && parseFloat(record.girthInput.value) > 0).length;
        gradeBtn.textContent = ready ? `Grade ${ready} ${ready === 1 ? 'log' : 'logs'}` : 'Grade crops';
        gradeBtn.disabled = ready === 0;
    }

    function renderDetectionResults(data) {
        detectionResults.innerHTML = '';
        cropRecords = {};
        let totalCrops = 0;

        data.images.forEach((imageResult) => {
            const group = document.createElement('div');
            group.className = 'trees-details-section';
            group.innerHTML = `
                <div class="section-title">
                    <h2>${escapeHtml(imageResult.filename)}</h2>
                    <p>Detected ${imageResult.trunks_detected ?? 0} &middot; Passed for grading ${imageResult.trunks_passed_for_grading ?? 0} &middot; Rejected ${imageResult.trunks_rejected ?? 0}</p>
                </div>
            `;
            const list = document.createElement('div');
            list.className = 'trees-list-container';
            (imageResult.crops || []).forEach((crop) => {
                list.appendChild(renderCropCard(imageResult, crop, totalCrops));
                totalCrops += 1;
            });
            if (!imageResult.crops || !imageResult.crops.length) {
                const empty = document.createElement('p');
                empty.style.color = 'var(--text-secondary)';
                empty.textContent = 'No cut faces passed the confidence, visibility, and sharpness gates for this image.';
                list.appendChild(empty);
            }
            group.appendChild(list);
            detectionResults.appendChild(group);
        });

        gradeCard.classList.toggle('hidden', totalCrops === 0);
        updateGradeReadiness();
        setStatus(detectStatus, totalCrops
            ? `Detection complete. ${totalCrops} crop(s) ready for measurements.`
            : 'Detection complete. No crops passed the quality gates.');
    }

    detectBtn.addEventListener('click', async () => {
        if (!selectedFiles.length) return;

        detectBtn.disabled = true;
        setStatus(detectStatus, 'Detecting cut faces. This can take a little while on CPU.');
        stepperStart(detectStepper, 'timberDetectStep', 3);
        detectionResults.innerHTML = '';
        gradeCard.classList.add('hidden');
        gradeStatus.classList.add('hidden');

        const formData = new FormData();
        selectedFiles.forEach((file) => formData.append('files', file, file.name));

        try {
            const resp = await fetch('/api/timber/detect', { method: 'POST', body: formData });
            const data = await resp.json();
            if (!resp.ok) {
                throw new Error((data && data.detail) || 'Detection failed.');
            }
            sessionId = data.session_id;
            renderDetectionResults(data);
            stepperFinish('timberDetectStep', 3, true);
        } catch (err) {
            setStatus(detectStatus, err.message || 'Detection failed.', true);
            stepperFinish('timberDetectStep', 3, false);
        } finally {
            detectBtn.disabled = false;
        }
    });

    // --- Grade crops ---

    function renderGradeResult(cropId, result) {
        const record = cropRecords[cropId];
        if (!record) return;
        const prediction = result.prediction || {};

        if (prediction.status === 'ok') {
            record.statusBadge.textContent = `${prediction.grade} (${prediction.bucket})`;
            record.statusBadge.className = 'tree-badge badge-success';
            const topRows = (prediction.top_predictions || [])
                .map((p) => `
                    <div class="prob-row">
                        <div class="prob-header">
                            <span class="prob-name">${escapeHtml(p.grade)} (${escapeHtml(p.bucket || '-')})</span>
                            <span class="prob-pct">${(p.probability * 100).toFixed(1)}%</span>
                        </div>
                        <div class="prob-bar-track">
                            <div class="prob-bar-fill" style="width: ${(p.probability * 100).toFixed(1)}%;"></div>
                        </div>
                    </div>
                `)
                .join('');
            record.gradeBody.innerHTML = `
                <div class="tree-metrics" style="margin-top: 10px;">
                    <div class="metric-item">
                        <span class="label">Grade</span>
                        <span class="val bold">${escapeHtml(prediction.grade)}</span>
                    </div>
                    <div class="metric-item">
                        <span class="label">Quality Bucket</span>
                        <span class="val">${escapeHtml(prediction.bucket || '-')}</span>
                    </div>
                    <div class="metric-item">
                        <span class="label">Confidence</span>
                        <span class="val">${(prediction.confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
                <div class="prob-bars-container" style="margin-top: 10px;">${topRows}</div>
            `;

            // Bridge to the Auction Bid Price tabs: expose graded crops for the
            // "Grading + Bid Price" combined tab to read/prefill from. This is a
            // UI-layer convenience only - no backend coupling with app/timber or
            // app/auction, and grading itself is completely unaffected.
            const measurements = result.measurements || {};
            const gradedCrop = {
                crop_id: cropId,
                label: result.label,
                source_filename: result.source_filename,
                grade: prediction.grade,
                bucket: prediction.bucket,
                confidence: prediction.confidence,
                Length_M: measurements.Length_M,
                Mid_Girth_M: measurements.Mid_Girth_M,
                Volume: measurements.Volume,
            };
            window.timberGradedCrops = window.timberGradedCrops || [];
            const existingIdx = window.timberGradedCrops.findIndex((c) => c.crop_id === cropId);
            if (existingIdx >= 0) {
                window.timberGradedCrops[existingIdx] = gradedCrop;
            } else {
                window.timberGradedCrops.push(gradedCrop);
            }
            document.dispatchEvent(new CustomEvent('timber:cropGraded', { detail: gradedCrop }));
        } else {
            record.statusBadge.textContent = prediction.status === 'skipped' ? 'Skipped' : 'Error';
            record.statusBadge.className = 'tree-badge badge-warning';
            record.gradeBody.innerHTML = `<p style="color: var(--text-secondary); margin-top: 10px;">${escapeHtml(prediction.reason || 'Grading was not completed for this crop.')}</p>`;
        }
    }

    gradeBtn.addEventListener('click', async () => {
        if (!sessionId) return;

        const measurements = [];
        for (const [cropId, record] of Object.entries(cropRecords)) {
            if (record.removed) continue;
            const lengthM = parseFloat(record.lengthInput.value);
            const girthM = parseFloat(record.girthInput.value);
            if (!(lengthM > 0) || !(girthM > 0)) continue;
            measurements.push({ crop_id: cropId, Length_M: lengthM, Mid_Girth_M: girthM });
        }

        if (!measurements.length) {
            setStatus(gradeStatus, 'Enter Length and Mid Girth for at least one crop before grading.', true);
            gradeStatus.classList.remove('hidden');
            return;
        }

        gradeBtn.disabled = true;
        gradeStatus.classList.remove('hidden');
        setStatus(gradeStatus, 'Grading crops. This can take a little while on CPU.');
        stepperStart(gradeStepper, 'timberGradeStep', 3);

        try {
            const resp = await fetch('/api/timber/grade', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, measurements }),
            });
            const data = await resp.json();
            if (!resp.ok) {
                throw new Error((data && data.detail) || 'Grading failed.');
            }
            (data.results || []).forEach((result) => renderGradeResult(result.crop_id, result));
            setStatus(gradeStatus, `Grading complete for ${data.results.length} crop(s).`);
            stepperFinish('timberGradeStep', 3, true);
        } catch (err) {
            setStatus(gradeStatus, err.message || 'Grading failed.', true);
            stepperFinish('timberGradeStep', 3, false);
        } finally {
            updateGradeReadiness();
        }
    });
});
