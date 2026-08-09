/**
 * Frontend JavaScript for Tree Species Identification System.
 * Manages tabbed analysis modes (Tree Detection, Tree Classification, Tree Detection +
 * Classification, Maturity Assessment, and the full end-to-end pipeline), drag and drop
 * image uploads with EXIF orientation correction, real-time job polling, multi-candidate
 * process modal inspection, and interactive result rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Mode Navigation Elements, in pipeline order: detect -> classify -> detect+classify ->
    // maturity assessment -> detect+classify+maturity assessment.
    const tabDetector = document.getElementById('tabDetector');
    const tabClassifier = document.getElementById('tabClassifier');
    const tabIntegrated = document.getElementById('tabIntegrated');
    const tabMaturity = document.getElementById('tabMaturity');
    const tabAutoFlow = document.getElementById('tabAutoFlow');
    const tabTimber = document.getElementById('tabTimber');
    const tabGradingBid = document.getElementById('tabGradingBid');
    const tabAuction = document.getElementById('tabAuction');
    const tabButtons = [tabDetector, tabClassifier, tabIntegrated, tabMaturity, tabAutoFlow, tabTimber, tabGradingBid, tabAuction];

    // Upload Section Elements
    const uploadTitle = document.getElementById('uploadTitle');
    const uploadDesc = document.getElementById('uploadDesc');
    const dropzone = document.getElementById('dropzone');
    const imageInput = document.getElementById('imageInput');
    const fileLimitsText = document.getElementById('fileLimitsText');
    const browseBtn = document.getElementById('browseBtn');
    const dropzonePrompt = document.getElementById('dropzonePrompt');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const removeImgBtn = document.getElementById('removeImgBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const analyzeBtnText = document.getElementById('analyzeBtnText');

    // Timeline Elements (Tree Detection + Classification tab)
    const timelineCard = document.getElementById('timelineCard');
    const timelineContainer = document.getElementById('timelineContainer');

    // Full Pipeline Stepper (Tree Detection + Classification + Maturity Assessment tab)
    const autoFlowStepperCard = document.getElementById('autoFlowStepperCard');

    // Results Wrappers for each Mode
    const resultsWrapperIntegrated = document.getElementById('resultsWrapperIntegrated');
    const resultsWrapperDetector = document.getElementById('resultsWrapperDetector');
    const resultsWrapperClassifier = document.getElementById('resultsWrapperClassifier');
    const resultsWrapperAutoFlow = document.getElementById('resultsWrapperAutoFlow');

    // Maturity Assessment Panel (merged Single Tree / Multiple Trees)
    const autoFlowOptionalCard = document.getElementById('autoFlowOptionalCard');
    const maturityPanel = document.getElementById('maturityPanel');
    const matModeSingleBtn = document.getElementById('matModeSingleBtn');
    const matModeMultipleBtn = document.getElementById('matModeMultipleBtn');
    const matSingleSection = document.getElementById('matSingleSection');
    const matMultipleSection = document.getElementById('matMultipleSection');
    const uploadCard = document.getElementById('uploadCard');

    // Timber Grading Panel (independent module - own upload UI, wired up in timber.js)
    const timberPanel = document.getElementById('timberPanel');

    // Auction Bid Price Panels (independent module - own forms, wired up in auction.js)
    const gradingBidPanel = document.getElementById('gradingBidPanel');
    const auctionPanel = document.getElementById('auctionPanel');

    // Integrated Mode Elements
    const summaryDetected = document.getElementById('summaryDetected');
    const summaryAssessable = document.getElementById('summaryAssessable');
    const summaryBarkReady = document.getElementById('summaryBarkReady');
    const summaryIdentified = document.getElementById('summaryIdentified');
    const summaryUnknown = document.getElementById('summaryUnknown');
    const summaryUncertain = document.getElementById('summaryUncertain');
    const summaryNotAssessable = document.getElementById('summaryNotAssessable');
    const summaryCommercial = document.getElementById('summaryCommercial');
    const annotatedImage = document.getElementById('annotatedImage');
    const treesListContainer = document.getElementById('treesListContainer');

    // Detector Mode Elements
    const detectorDetected = document.getElementById('detectorDetected');
    const detectorAvgConf = document.getElementById('detectorAvgConf');
    const detectorTotalArea = document.getElementById('detectorTotalArea');
    const detectorResolution = document.getElementById('detectorResolution');
    const detectorAnnotatedImage = document.getElementById('detectorAnnotatedImage');
    const detectorListContainer = document.getElementById('detectorListContainer');

    // Classifier Mode Elements
    const clfDecisionBadge = document.getElementById('clfDecisionBadge');
    const clfSpeciesName = document.getElementById('clfSpeciesName');
    const clfRawConf = document.getElementById('clfRawConf');
    const clfProtoDist = document.getElementById('clfProtoDist');
    const clfEnergy = document.getElementById('clfEnergy');
    const clfUnknownScore = document.getElementById('clfUnknownScore');
    const clfProbBars = document.getElementById('clfProbBars');
    const clfAttentionGrid = document.getElementById('clfAttentionGrid');
    const clfPatchesGrid = document.getElementById('clfPatchesGrid');

    // Modal Elements
    const processModal = document.getElementById('processModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalSubtitle = document.getElementById('modalSubtitle');
    const modalBody = document.getElementById('modalBody');
    const closeModalBtn = document.getElementById('closeModalBtn');

    let currentMode = 'integrated'; // 'detector' | 'classifier' | 'integrated' | 'maturity' | 'autoflow' | 'timber' | 'grading-bid' | 'auction'
    let selectedFile = null;
    let pollInterval = null;
    let currentResultsData = null;

    // Backend Pipeline Stages (Tree Detection + Classification tab)
    const STAGE_NAMES = [
        "Image uploaded and validated",
        "Detecting tree trunks",
        "Checking trunk assessability",
        "Extracting bark candidates",
        "Evaluating bark quality",
        "Preparing classification inputs",
        "Executing species classification",
        "Performing open-set validation",
        "Performing multi-ROI consensus",
        "Applying commercial flag",
        "Rendering final results"
    ];

    const STANDARD_ACCEPT = '.jpg,.jpeg,.png,.webp';
    const STANDARD_ACCEPT_TEXT = 'Supported formats: JPEG, PNG, WEBP (Max 20MB) — EXIF orientation auto-corrected';
    const HEIC_ACCEPT = '.jpg,.jpeg,.png,.webp,.heic,.heif';
    const HEIC_ACCEPT_TEXT = 'Supported formats: JPEG, PNG, WEBP, HEIC, HEIF (Max 20MB) — EXIF orientation auto-corrected';

    // Tab Switching Logic
    tabButtons.forEach(btn => {
        if (!btn) return;
        btn.addEventListener('click', () => {
            const mode = btn.dataset.tab;
            if (currentMode === mode) return;

            currentMode = mode;
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Hide all result wrappers
            timelineCard.classList.add('hidden');
            autoFlowStepperCard.classList.add('hidden');
            resultsWrapperIntegrated.classList.add('hidden');
            resultsWrapperDetector.classList.add('hidden');
            resultsWrapperClassifier.classList.add('hidden');
            resultsWrapperAutoFlow.classList.add('hidden');

            // Shared single-image dropzone is used by detector/classifier/integrated/autoflow;
            // the Maturity Assessment tab uses its own dedicated form+queue UI instead.
            const usesSharedUpload = ['detector', 'classifier', 'integrated', 'autoflow'].includes(currentMode);
            uploadCard.classList.toggle('hidden', !usesSharedUpload);
            autoFlowOptionalCard.classList.toggle('hidden', currentMode !== 'autoflow');
            maturityPanel.classList.toggle('hidden', currentMode !== 'maturity');
            timberPanel.classList.toggle('hidden', currentMode !== 'timber');
            gradingBidPanel.classList.toggle('hidden', currentMode !== 'grading-bid');
            auctionPanel.classList.toggle('hidden', currentMode !== 'auction');

            // Only the full pipeline tab accepts HEIC on the shared dropzone (the standalone
            // Maturity Assessment tab has its own HEIC-capable inputs already).
            const allowHeic = currentMode === 'autoflow';
            imageInput.accept = allowHeic ? HEIC_ACCEPT : STANDARD_ACCEPT;
            fileLimitsText.textContent = allowHeic ? HEIC_ACCEPT_TEXT : STANDARD_ACCEPT_TEXT;

            // Update Upload Card Prompt Text
            if (currentMode === 'detector') {
                uploadTitle.textContent = "Upload Image for Tree Trunk Detection";
                uploadDesc.textContent = "Detect and count tree trunks in a forest or plantation image.";
                analyzeBtnText.textContent = "Run Tree Detection";
            } else if (currentMode === 'classifier') {
                uploadTitle.textContent = "Upload Bark Crop for Species Classification";
                uploadDesc.textContent = "Identify the species from a cropped bark image region, with open-set unknown detection.";
                analyzeBtnText.textContent = "Classify Species";
            } else if (currentMode === 'integrated') {
                uploadTitle.textContent = "Upload Forest Image";
                uploadDesc.textContent = "Select a high-resolution forest or plantation image containing one or more tree trunks for combined detection and classification.";
                analyzeBtnText.textContent = "Run Detection + Classification";
            } else if (currentMode === 'autoflow') {
                uploadTitle.textContent = "Upload Tree Photo for Full Pipeline";
                uploadDesc.textContent = "One photo: detect the trunk, identify the species, then automatically run DBH and maturity estimation on the same photo.";
                analyzeBtnText.textContent = "Run Full Pipeline";
            }
        });
    });

    // Maturity Assessment mode toggle: Single Tree <-> Multiple Trees
    function setMatMode(mode) {
        matModeSingleBtn.classList.toggle('active', mode === 'single');
        matModeMultipleBtn.classList.toggle('active', mode === 'multiple');
        matSingleSection.classList.toggle('hidden', mode !== 'single');
        matMultipleSection.classList.toggle('hidden', mode !== 'multiple');
    }
    matModeSingleBtn.addEventListener('click', () => setMatMode('single'));
    matModeMultipleBtn.addEventListener('click', () => setMatMode('multiple'));

    // Lightweight pipeline stepper (synchronous tabs with no live backend polling):
    // shows every stage the request performs, marks them running while in flight
    // and complete once the response comes back.
    function stepperStart(containerId, prefix, count) {
        const container = document.getElementById(containerId);
        if (container) container.classList.remove('hidden');
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
    }

    // File Drag and Drop Handlers
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        imageInput.click();
    });

    dropzone.addEventListener('click', () => {
        if (!selectedFile) imageInput.click();
    });

    imageInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
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
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    removeImgBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function handleFileSelect(file) {
        const ext = (file.name.split('.').pop() || '').toLowerCase();
        const allowHeic = currentMode === 'autoflow';
        const isHeic = allowHeic && (ext === 'heic' || ext === 'heif');
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];

        if (!isHeic && !validTypes.includes(file.type)) {
            alert(allowHeic
                ? 'Invalid file format. Please select a JPEG, PNG, WEBP, or HEIC/HEIF image.'
                : 'Invalid file format. Please select a JPEG, PNG, or WEBP image.');
            return;
        }

        if (file.size > 20 * 1024 * 1024) {
            alert('File size exceeds maximum 20MB limit.');
            return;
        }

        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
        dropzonePrompt.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        analyzeBtn.disabled = false;

        if (isHeic) {
            // Most browsers can't render HEIC/HEIF in an <img>; skip the thumbnail rather
            // than show a broken image icon. The filename/size chip above still confirms selection.
            imagePreview.src = '';
            imagePreview.classList.add('hidden');
        } else {
            imagePreview.classList.remove('hidden');
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    }

    function resetUpload() {
        selectedFile = null;
        imageInput.value = '';
        imagePreview.src = '';
        imagePreview.classList.remove('hidden');
        dropzonePrompt.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        analyzeBtn.disabled = true;

        if (pollInterval) clearInterval(pollInterval);
        timelineCard.classList.add('hidden');
        autoFlowStepperCard.classList.add('hidden');
        resultsWrapperIntegrated.classList.add('hidden');
        resultsWrapperDetector.classList.add('hidden');
        resultsWrapperClassifier.classList.add('hidden');
        resultsWrapperAutoFlow.classList.add('hidden');
    }

    // Main Analyze Click Event Listener
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        if (currentMode === 'integrated') {
            await startIntegratedPipeline();
        } else if (currentMode === 'detector') {
            await startDetectorOnlyAnalysis();
        } else if (currentMode === 'classifier') {
            await startClassifierOnlyAnalysis();
        } else if (currentMode === 'autoflow') {
            await startAutoFlowPipeline();
        }
    });

    // 1. Integrated Pipeline Execution
    async function startIntegratedPipeline() {
        analyzeBtn.disabled = true;
        analyzeBtnText.textContent = "Processing...";
        resultsWrapperIntegrated.classList.add('hidden');
        timelineCard.classList.remove('hidden');

        initTimeline();

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const resp = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (!resp.ok) {
                const errData = await resp.json();
                throw new Error(errData.detail || 'Server error uploading image.');
            }

            const data = await resp.json();
            const jobId = data.job_id;

            pollInterval = setInterval(() => pollJobStatus(jobId), 750);
        } catch (err) {
            alert(`Analysis failed: ${err.message}`);
            analyzeBtn.disabled = false;
            analyzeBtnText.textContent = "Run Detection + Classification";
            timelineCard.classList.add('hidden');
        }
    }

    function initTimeline() {
        timelineContainer.innerHTML = '';
        STAGE_NAMES.forEach((name, idx) => {
            const stageNum = idx + 1;
            const item = document.createElement('div');
            item.className = 'timeline-item';
            item.id = `timelineStage_${stageNum}`;

            item.innerHTML = `
                <div class="stage-num">${stageNum}</div>
                <div class="stage-details">
                    <span class="stage-title">${name}</span>
                    <span class="stage-desc" id="stageDesc_${stageNum}">Waiting...</span>
                </div>
            `;
            timelineContainer.appendChild(item);
        });
    }

    async function pollJobStatus(jobId) {
        try {
            const resp = await fetch(`/api/jobs/${jobId}`);
            if (!resp.ok) return;

            const job = await resp.json();
            const currentStage = job.stage_index;

            STAGE_NAMES.forEach((_, idx) => {
                const sNum = idx + 1;
                const el = document.getElementById(`timelineStage_${sNum}`);
                const descEl = document.getElementById(`stageDesc_${sNum}`);
                if (!el || !descEl) return;

                if (sNum < currentStage) {
                    el.className = 'timeline-item completed';
                    descEl.textContent = 'Completed';
                } else if (sNum === currentStage) {
                    if (job.status === 'completed') {
                        el.className = 'timeline-item completed';
                        descEl.textContent = 'Completed';
                    } else if (job.status === 'error') {
                        el.className = 'timeline-item error';
                        descEl.textContent = job.message;
                    } else {
                        el.className = 'timeline-item active';
                        descEl.textContent = job.message || 'Processing...';
                    }
                } else {
                    el.className = 'timeline-item';
                    descEl.textContent = 'Waiting...';
                }
            });

            if (job.status === 'completed') {
                clearInterval(pollInterval);
                await fetchAndRenderIntegratedResults(jobId);
            } else if (job.status === 'error') {
                clearInterval(pollInterval);
                alert(`Pipeline Error: ${job.message}`);
                analyzeBtn.disabled = false;
                analyzeBtnText.textContent = "Run Detection + Classification";
            }
        } catch (err) {
            console.error('Polling error:', err);
        }
    }

    async function fetchAndRenderIntegratedResults(jobId) {
        try {
            const resp = await fetch(`/api/results/${jobId}`);
            if (!resp.ok) throw new Error('Failed to retrieve results payload.');

            const data = await resp.json();
            currentResultsData = data;

            summaryDetected.textContent = data.summary.trees_detected;
            summaryAssessable.textContent = data.summary.assessable;
            summaryBarkReady.textContent = data.summary.bark_ready;
            summaryIdentified.textContent = data.summary.identified;
            summaryUnknown.textContent = data.summary.unknown;
            summaryUncertain.textContent = data.summary.uncertain || 0;
            summaryNotAssessable.textContent = data.summary.not_assessable;
            summaryCommercial.textContent = data.summary.commercial_flagged || 0;

            annotatedImage.src = data.image.annotated + `?t=${Date.now()}`;

            renderIntegratedTreeCards(data.trees);

            resultsWrapperIntegrated.classList.remove('hidden');
            analyzeBtn.disabled = false;
            analyzeBtnText.textContent = "Run Detection + Classification";
        } catch (err) {
            alert(`Error rendering results: ${err.message}`);
        }
    }

    function renderIntegratedTreeCards(trees) {
        treesListContainer.innerHTML = '';
        if (!trees || trees.length === 0) {
            treesListContainer.innerHTML = '<p style="color: var(--text-secondary);">No tree trunks identified in the uploaded image.</p>';
            return;
        }

        trees.forEach(tree => {
            const card = document.createElement('div');
            card.className = 'card tree-card';

            let statusBadgeClass = 'badge-success';
            let statusText = tree.raglo.final_species ? formatSpeciesName(tree.raglo.final_species) : 'Identified';

            if (tree.status === 'not_assessable') {
                statusBadgeClass = 'badge-danger';
                statusText = 'Not Assessable';
            } else if (tree.status === 'quality_failed') {
                statusBadgeClass = 'badge-warning';
                statusText = 'Bark Quality Failed';
            } else if (tree.status === 'open_set_rejected') {
                statusBadgeClass = 'badge-warning';
                statusText = 'Unknown / Open-Set Rejected';
            } else if (tree.status === 'uncertain') {
                statusBadgeClass = 'badge-warning';
                statusText = 'Uncertain Consensus';
            }

            const consensus = tree.raglo.consensus || {};
            const consensusStr = consensus.known_votes !== undefined ? 
                `${consensus.known_votes} / ${consensus.candidate_count} known candidates agreed` : 'N/A';

            const comm = tree.commercial || {};
            const commStr = comm.evaluated ? (comm.commercial_flag ? 'Commercial species' : 'Non-commercial') : 'Not evaluated';

            card.innerHTML = `
                <div class="tree-card-header">
                    <span class="tree-name">${tree.display_name}</span>
                    <span class="tree-badge ${statusBadgeClass}">${statusText}</span>
                </div>
                <div class="tree-card-body">
                    <div class="tree-metrics">
                        <div class="metric-item">
                            <span class="label">Detection Confidence</span>
                            <span class="val">${(tree.detection.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div class="metric-item">
                            <span class="label">Assessable</span>
                            <span class="val">${tree.assessability.passed ? 'Passed' : 'Failed'}</span>
                        </div>
                        <div class="metric-item">
                            <span class="label">Bark Ready</span>
                            <span class="val">${tree.bark.bark_ready ? 'Yes' : 'No'}</span>
                        </div>
                        <div class="metric-item">
                            <span class="label">Final Species</span>
                            <span class="val bold">${tree.raglo.final_species || 'N/A'}</span>
                        </div>
                        <div class="metric-item">
                            <span class="label">Multi-ROI Consensus</span>
                            <span class="val">${consensusStr}</span>
                        </div>
                        <div class="metric-item">
                            <span class="label">Commercial Flag</span>
                            <span class="val">${commStr}</span>
                        </div>
                    </div>
                </div>
                <div class="tree-card-footer">
                    <button type="button" class="btn btn-secondary btn-sm inspect-btn" data-tree-id="${tree.tree_id}">
                        View Identification Process &rarr;
                    </button>
                </div>
            `;
            treesListContainer.appendChild(card);
        });

        document.querySelectorAll('.inspect-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tId = parseInt(e.currentTarget.dataset.treeId);
                openProcessModal(tId);
            });
        });
    }

    // Section 26 & 27: Expandable Multi-Candidate Process View
    function openProcessModal(treeId) {
        if (!currentResultsData) return;
        const tree = currentResultsData.trees.find(t => t.tree_id === treeId);
        if (!tree) return;

        modalTitle.textContent = `${tree.display_name} — View Identification Process`;
        modalSubtitle.textContent = `Bounding Box: [${tree.detection.bbox.join(', ')}] | Status: ${tree.status}`;

        modalBody.innerHTML = '';
        const flowContainer = document.createElement('div');
        flowContainer.className = 'pipeline-flow';

        let flowHtml = '';

        // Stage 2 & 3 Artifacts
        if (tree.artifacts.tree_crop) {
            flowHtml += `
                <div class="flow-step">
                    <div class="flow-step-header">Trunk Crop &amp; Segmentation Mask</div>
                    <div class="flow-images-grid">
                        <div class="flow-img-card">
                            <img src="${tree.artifacts.tree_crop}" alt="Tree Crop">
                            <span>Original Trunk Bounding Box</span>
                        </div>
                        ${tree.artifacts.masked_tree ? `
                        <div class="flow-img-card">
                            <img src="${tree.artifacts.masked_tree}" alt="Masked Tree">
                            <span>Neutral Masked Trunk</span>
                        </div>` : ''}
                    </div>
                </div>
            `;
        }

        // Section 27: Multi-Candidate Breakdown Display
        const candidates = tree.raglo.candidates || [];
        if (candidates.length > 0) {
            flowHtml += `<div class="flow-step"><div class="flow-step-header">Multi-Candidate Bark ROI Evaluation &amp; Open-Set Validation</div>`;

            candidates.forEach((cand, idx) => {
                const q = cand.quality || {};
                const clf = cand.classification || {};
                const os = cand.open_set || {};
                const art = cand.artifacts || {};
                const att = cand.attention?.attention_weights || [];

                const isRejected = os.open_set_rejected;
                const statusBadge = isRejected ?
                    '<span class="tree-badge badge-warning">REJECTED</span>' :
                    '<span class="tree-badge badge-success">KNOWN</span>';

                flowHtml += `
                    <div class="candidate-detail-card" style="background: #f8fafc; border: 1px solid var(--border-color); border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 10px; margin-bottom: 12px;">
                            <strong style="font-size: 1.05rem;">Candidate ROI ${cand.candidate_id || idx+1}</strong>
                            ${statusBadge}
                        </div>
                        <div style="display: grid; grid-template-columns: 140px 1fr; gap: 16px; align-items: start;">
                            <div class="flow-img-card">
                                <img src="${art.bark_roi || tree.artifacts.selected_bark_roi}" style="width: 130px; height: 130px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border-color);">
                                <span>Bark ROI ${cand.candidate_id || idx+1}</span>
                            </div>
                            <div class="tree-metrics">
                                <div class="metric-item">
                                    <span class="label">Quality Score</span>
                                    <span class="val">${q.quality_score !== undefined ? q.quality_score.toFixed(4) : 'N/A'}</span>
                                </div>
                                <div class="metric-item">
                                    <span class="label">Raw Prediction</span>
                                    <span class="val">${clf.raw_predicted_class ? formatSpeciesName(clf.raw_predicted_class) : 'N/A'}</span>
                                </div>
                                <div class="metric-item">
                                    <span class="label">Raw Confidence</span>
                                    <span class="val">${clf.raw_confidence !== undefined ? (clf.raw_confidence * 100).toFixed(1) + '%' : 'N/A'}</span>
                                </div>
                                <div class="metric-item">
                                    <span class="label">Prototype Distance (Z)</span>
                                    <span class="val">${os.prototype_distance?.toFixed(4)} (Z: ${os.prototype_z?.toFixed(4)})</span>
                                </div>
                                <div class="metric-item">
                                    <span class="label">Energy Score (Z)</span>
                                    <span class="val">${os.energy?.toFixed(4)} (Z: ${os.energy_z?.toFixed(4)})</span>
                                </div>
                                <div class="metric-item">
                                    <span class="label">Combined Unknownness</span>
                                    <span class="val bold">${os.combined_unknownness?.toFixed(4)}</span>
                                </div>
                                <div class="metric-item">
                                    <span class="label">Open-Set Threshold</span>
                                    <span class="val">${os.threshold?.toFixed(4)}</span>
                                </div>
                                <div class="metric-item">
                                    <span class="label">Decision Margin</span>
                                    <span class="val">${os.decision_margin?.toFixed(4)}</span>
                                </div>
                            </div>
                        </div>

                        <!-- Patch Attention Weights & Visualizer -->
                        ${art.global_384 ? `
                        <div style="margin-top: 14px;">
                            <span class="label" style="display: block; margin-bottom: 6px; font-weight: 700;">Preprocessed Global &amp; Local Patches</span>
                            <div class="flow-images-grid" style="grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 8px;">
                                <div class="flow-img-card">
                                    <img src="${art.global_384}">
                                    <span>Global (384)</span>
                                </div>
                                ${art.patch_1_224 ? `<div class="flow-img-card"><img src="${art.patch_1_224}"><span>P1 W: ${att[0] ? (att[0]*100).toFixed(1)+'%' : 'N/A'}</span></div>` : ''}
                                ${art.patch_2_224 ? `<div class="flow-img-card"><img src="${art.patch_2_224}"><span>P2 W: ${att[1] ? (att[1]*100).toFixed(1)+'%' : 'N/A'}</span></div>` : ''}
                                ${art.patch_3_224 ? `<div class="flow-img-card"><img src="${art.patch_3_224}"><span>P3 W: ${att[2] ? (att[2]*100).toFixed(1)+'%' : 'N/A'}</span></div>` : ''}
                                ${art.patch_4_224 ? `<div class="flow-img-card"><img src="${art.patch_4_224}"><span>P4 W: ${att[3] ? (att[3]*100).toFixed(1)+'%' : 'N/A'}</span></div>` : ''}
                            </div>
                        </div>` : ''}
                    </div>
                `;
            });

            flowHtml += `</div>`;
        }

        flowContainer.innerHTML = flowHtml || '<p>No intermediate process artifacts available for this tree.</p>';
        modalBody.appendChild(flowContainer);

        processModal.classList.remove('hidden');
    }

    closeModalBtn.addEventListener('click', () => {
        processModal.classList.add('hidden');
    });

    // 2. Tree Detector Only Execution Mode
    async function startDetectorOnlyAnalysis() {
        analyzeBtn.disabled = true;
        analyzeBtnText.textContent = "Detecting Tree Trunks...";
        resultsWrapperDetector.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const resp = await fetch('/api/analyze/detector', {
                method: 'POST',
                body: formData
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Detector execution failed.');
            }

            const data = await resp.json();

            detectorDetected.textContent = data.summary.trees_detected;
            detectorAvgConf.textContent = `${(data.summary.avg_confidence * 100).toFixed(1)}%`;
            detectorTotalArea.textContent = `${data.summary.total_mask_area_px.toLocaleString()} px²`;
            detectorResolution.textContent = data.summary.image_dimensions;

            detectorAnnotatedImage.src = data.image.annotated + `?t=${Date.now()}`;

            detectorListContainer.innerHTML = '';
            if (data.trees.length === 0) {
                detectorListContainer.innerHTML = '<p style="color: var(--text-secondary);">No tree trunks detected in image.</p>';
            } else {
                data.trees.forEach(tree => {
                    const card = document.createElement('div');
                    card.className = 'card tree-card';
                    card.innerHTML = `
                        <div class="tree-card-header">
                            <span class="tree-name">${tree.display_name}</span>
                            <span class="tree-badge badge-success">Confidence: ${(tree.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div class="tree-card-body">
                            <div style="display: flex; gap: 16px; align-items: center;">
                                <img src="${tree.crop_url}" style="width: 80px; height: 100px; object-fit: cover; border-radius: 6px; border: 1px solid #e2e8f0;">
                                <div class="tree-metrics" style="flex: 1;">
                                    <div class="metric-item">
                                        <span class="label">Segmentation Boundary</span>
                                        <span class="val">${tree.polygon_vertex_count ? `${tree.polygon_vertex_count} Polygon Vertices` : 'Masked Polygon'}</span>
                                    </div>
                                    <div class="metric-item">
                                        <span class="label">Centroid (X, Y)</span>
                                        <span class="val">(${tree.centroid[0]}, ${tree.centroid[1]})</span>
                                    </div>
                                    <div class="metric-item">
                                        <span class="label">Mask Coverage Area</span>
                                        <span class="val">${tree.area_px.toLocaleString()} px²</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                    detectorListContainer.appendChild(card);
                });
            }

            resultsWrapperDetector.classList.remove('hidden');
        } catch (err) {
            alert(`Detector error: ${err.message}`);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtnText.textContent = "Run Tree Detection";
        }
    }

    // 3. Bark Classifier Only Execution Mode
    async function startClassifierOnlyAnalysis() {
        analyzeBtn.disabled = true;
        analyzeBtnText.textContent = "Classifying Bark Species...";
        resultsWrapperClassifier.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const resp = await fetch('/api/analyze/classifier', {
                method: 'POST',
                body: formData
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Classifier execution failed.');
            }

            const data = await resp.json();
            const res = data.result;
            const art = data.artifacts;
            const os = res.open_set || {};

            // Hero Badge & Title
            if (os.open_set_rejected) {
                clfDecisionBadge.textContent = "OPEN-SET REJECTED";
                clfDecisionBadge.className = "status-badge open_set_rejected";
                clfSpeciesName.textContent = "Unknown / Unrecognized species";
            } else {
                clfDecisionBadge.textContent = "KNOWN SPECIES";
                clfDecisionBadge.className = "status-badge known";
                clfSpeciesName.textContent = formatSpeciesName(res.final_species);
            }

            clfRawConf.textContent = `${(res.classification.raw_confidence * 100).toFixed(1)}%`;
            clfProtoDist.textContent = os.prototype_distance !== undefined ? os.prototype_distance.toFixed(4) : 'N/A';
            clfEnergy.textContent = os.energy !== undefined ? os.energy.toFixed(4) : 'N/A';
            clfUnknownScore.textContent = os.combined_unknownness !== undefined ? os.combined_unknownness.toFixed(4) : 'N/A';

            // Render Softmax Probability Distribution Bar Chart
            clfProbBars.innerHTML = '';
            if (res.classification.class_probabilities) {
                Object.entries(res.classification.class_probabilities).forEach(([clsName, probVal]) => {
                    const row = document.createElement('div');
                    row.className = 'prob-row';
                    const pct = (probVal * 100).toFixed(1);
                    row.innerHTML = `
                        <div class="prob-header">
                            <span class="prob-name">${formatSpeciesName(clsName)}</span>
                            <span class="prob-pct">${pct}%</span>
                        </div>
                        <div class="prob-bar-track">
                            <div class="prob-bar-fill" style="width: ${pct}%;"></div>
                        </div>
                    `;
                    clfProbBars.appendChild(row);
                });
            }

            // Render Patch Attention Weights Grid
            clfAttentionGrid.innerHTML = '';
            if (res.attention && res.attention.attention_weights) {
                res.attention.attention_weights.forEach((w, idx) => {
                    const card = document.createElement('div');
                    card.className = 'attention-card';
                    card.innerHTML = `
                        <div class="att-title">Patch ${idx + 1} Weight</div>
                        <div class="att-value">${(w * 100).toFixed(1)}%</div>
                    `;
                    clfAttentionGrid.appendChild(card);
                });
            }

            // Render Patches Visualizer
            clfPatchesGrid.innerHTML = `
                <div class="patch-box">
                    <img src="${art.global_384}" alt="Global 384">
                    <span>Global (384x384)</span>
                </div>
            `;
            if (art.patches_224) {
                art.patches_224.forEach((pUrl, idx) => {
                    clfPatchesGrid.innerHTML += `
                        <div class="patch-box">
                            <img src="${pUrl}" alt="Patch ${idx + 1}">
                            <span>Patch ${idx + 1} (224x224)</span>
                        </div>
                    `;
                });
            }

            resultsWrapperClassifier.classList.remove('hidden');
        } catch (err) {
            alert(`Classifier error: ${err.message}`);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtnText.textContent = "Classify Species";
        }
    }

    function formatSpeciesName(name) {
        if (!name) return 'Unknown';
        const map = {
            'mahogany': 'Mahogany (Swietenia macrophylla)',
            'pine': 'Pine (Pinus spp.)',
            'rubber': 'Rubber (Hevea brasiliensis)',
            'teak': 'Teak (Tectona grandis)'
        };
        return map[name.toLowerCase()] || name;
    }

    // =========================================================================
    // Module 2 (DBH + Maturity Estimation) — Complete Pipeline, Single Estimation,
    // and the auto-integrated Full Pipeline tab. All identifiers below are
    // prefixed with "mat"/"autoFlow" to avoid collisions with Module 1's code above.
    // =========================================================================

    const MAT_NUMERIC_FIELDS = [
        ['AgeYears', 'age_years'],
        ['HeightM', 'height_m'],
        ['AnnualRainfallMm', 'annual_rainfall_mm'],
        ['ElevationM', 'elevation_m'],
        ['StemsPerHa', 'stems_per_ha'],
    ];
    const MAT_CATEGORICAL_FIELDS = [
        ['ClimaticZone', 'climatic_zone'],
        ['Spacing', 'spacing'],
        ['SiteClass', 'site_class'],
        ['ManagementIntensity', 'management_intensity'],
        ['IntendedProduct', 'intended_product'],
    ];

    function matEscapeHtml(value) {
        return String(value ?? '')
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#039;');
    }

    function matSetStatus(elId, message, isError = false) {
        const el = document.getElementById(elId);
        if (!el) return;
        el.textContent = message;
        el.classList.toggle('error', isError);
    }

    function matPopulateSelect(select, values) {
        if (!select) return;
        select.innerHTML = '';
        for (const value of values || []) {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = value;
            select.appendChild(option);
        }
    }

    function matCollectOptionalInputs(prefix) {
        const optional = {};
        for (const [suffix, key] of MAT_NUMERIC_FIELDS) {
            const el = document.getElementById(`${prefix}${suffix}`);
            if (!el || el.value === '') continue;
            const num = Number(el.value);
            if (Number.isFinite(num) && num > 0) optional[key] = num;
        }
        for (const [suffix, key] of MAT_CATEGORICAL_FIELDS) {
            const el = document.getElementById(`${prefix}${suffix}`);
            if (!el) continue;
            const value = el.value;
            if (value && value !== 'Blank' && value !== 'Unknown') optional[key] = value;
        }
        return optional;
    }

    async function loadMaturityConfig() {
        try {
            const resp = await fetch('/api/maturity/config');
            if (!resp.ok) return;
            const config = await resp.json();
            const categoryOptions = config.category_options || {};

            matPopulateSelect(document.getElementById('matSpecies'), config.species);
            matPopulateSelect(document.getElementById('matSingleSpecies'), config.species);

            for (const prefix of ['mat', 'matSingle', 'autoFlow']) {
                for (const [suffix, key] of MAT_CATEGORICAL_FIELDS) {
                    matPopulateSelect(document.getElementById(`${prefix}${suffix}`), categoryOptions[key]);
                }
            }
        } catch (error) {
            console.error('Failed to load maturity configuration', error);
        }
    }

    function matRenderGallery(id, items) {
        const container = document.getElementById(id);
        if (!container) return false;
        if (!items || !items.length) {
            container.innerHTML = '<div class="empty">No images</div>';
            return true;
        }
        container.innerHTML = items
            .map((item) => `
                <figure>
                    <a href="${matEscapeHtml(item.url)}" target="_blank" rel="noreferrer">
                        <img src="${matEscapeHtml(item.url)}" alt="${matEscapeHtml(item.caption)}">
                    </a>
                    <figcaption>${matEscapeHtml(item.caption)} - ${matEscapeHtml(item.name)}</figcaption>
                </figure>
            `)
            .join('');
        return true;
    }

    function matRenderTable(headEl, bodyEl, rows) {
        if (!headEl || !bodyEl) return;
        if (!rows || !rows.length) {
            headEl.innerHTML = '';
            bodyEl.innerHTML = '<tr><td class="empty">No rows</td></tr>';
            return;
        }
        const columns = Object.keys(rows[0]);
        headEl.innerHTML = `<tr>${columns.map((c) => `<th>${matEscapeHtml(c)}</th>`).join('')}</tr>`;
        bodyEl.innerHTML = rows
            .map((row) => `<tr>${columns.map((c) => `<td>${matEscapeHtml(row[c] ?? '')}</td>`).join('')}</tr>`)
            .join('');
    }

    function matRenderCards(containerId, results) {
        const container = document.getElementById(containerId);
        if (!container) return;
        if (!results || !results.length) {
            container.innerHTML = '';
            return;
        }
        container.innerHTML = results
            .map((result) => {
                if (!result.dbh_success) {
                    return `
                        <article class="result-card error">
                            <h3>${matEscapeHtml(result.tree_id)}</h3>
                            <div>${matEscapeHtml(result.species || '')}</div>
                            <div class="note">${matEscapeHtml(result.reason || 'DBH could not be measured reliably.')}</div>
                        </article>
                    `;
                }
                const optional = result.optional_fields_used && result.optional_fields_used.length
                    ? result.optional_fields_used.join(', ')
                    : 'None';
                return `
                    <article class="result-card">
                        <h3>${matEscapeHtml(result.tree_id)}</h3>
                        <div>${matEscapeHtml(result.species)}</div>
                        <div class="metrics">
                            <div class="metric">
                                <span>Final DBH</span>
                                <strong>${Number(result.final_dbh_cm).toFixed(2)} cm</strong>
                            </div>
                            <div class="metric">
                                <span>Maturity</span>
                                <strong>${Number(result.maturity_score).toFixed(2)}%</strong>
                            </div>
                            <div class="metric">
                                <span>Class</span>
                                <strong>${matEscapeHtml(result.maturity_class)}</strong>
                            </div>
                            <div class="metric">
                                <span>DBH Quality</span>
                                <strong>${matEscapeHtml(String(result.dbh_quality).toUpperCase())}</strong>
                            </div>
                        </div>
                        <div class="note">
                            Images: ${result.successful_images} successful of ${result.images_uploaded}.
                            Optional fields: ${matEscapeHtml(optional)}.
                            ${matEscapeHtml(result.dbh_quality_reasons || '')}
                        </div>
                    </article>
                `;
            })
            .join('');
    }

    // --- Maturity: Complete Pipeline (multi-tree queue) ---

    const matQueue = [];

    function matNextTreeName() {
        return `Tree_${String(matQueue.length + 1).padStart(2, '0')}`;
    }

    function matResetFields() {
        document.getElementById('matTreeName').value = matNextTreeName();
        document.getElementById('matImages').value = '';
        for (const [suffix] of MAT_NUMERIC_FIELDS) document.getElementById(`mat${suffix}`).value = '';
        for (const [suffix] of MAT_CATEGORICAL_FIELDS) document.getElementById(`mat${suffix}`).value = 'Blank';
    }

    function matRenderQueue() {
        document.getElementById('matQueueCount').textContent = `${matQueue.length} ${matQueue.length === 1 ? 'tree' : 'trees'}`;
        const body = document.getElementById('matQueueBody');
        if (!matQueue.length) {
            body.innerHTML = '<tr><td colspan="4" class="empty">No trees queued</td></tr>';
            return;
        }
        body.innerHTML = matQueue
            .map((tree) => {
                const optionalNames = Object.keys(tree.optional_inputs);
                return `
                    <tr>
                        <td>${matEscapeHtml(tree.tree_id)}</td>
                        <td>${matEscapeHtml(tree.species)}</td>
                        <td>${tree.files.length}</td>
                        <td>${optionalNames.length ? matEscapeHtml(optionalNames.join(', ')) : 'None'}</td>
                    </tr>
                `;
            })
            .join('');
    }

    function matAddTreeToQueue() {
        const files = Array.from(document.getElementById('matImages').files || []);
        if (!files.length) {
            matSetStatus('matStatus', 'Upload at least one image for this tree.', true);
            return;
        }

        let treeId = (document.getElementById('matTreeName').value || '').trim();
        if (!treeId) treeId = matNextTreeName();

        const existing = new Set(matQueue.map((t) => t.tree_id));
        const base = treeId;
        let suffix = 2;
        while (existing.has(treeId)) {
            treeId = `${base}_${suffix}`;
            suffix += 1;
        }

        matQueue.push({
            tree_id: treeId,
            species: document.getElementById('matSpecies').value,
            files,
            optional_inputs: matCollectOptionalInputs('mat'),
        });

        matRenderQueue();
        matResetFields();
        matSetStatus('matStatus', `Added ${treeId}.`);
    }

    function matRemoveLastTree() {
        if (!matQueue.length) {
            matSetStatus('matStatus', 'The queue is already empty.');
            return;
        }
        const removed = matQueue.pop();
        matRenderQueue();
        document.getElementById('matTreeName').value = matNextTreeName();
        matSetStatus('matStatus', `Removed ${removed.tree_id}.`);
    }

    function matClearTreeQueue() {
        matQueue.length = 0;
        matRenderQueue();
        matResetFields();
        matSetStatus('matStatus', 'Tree queue cleared.');
    }

    function matBuildFormData() {
        const formData = new FormData();
        const payloadTrees = [];
        let fileIndex = 0;

        for (const tree of matQueue) {
            const fileIndexes = [];
            for (const file of tree.files) {
                formData.append('files', file, file.name);
                fileIndexes.push(fileIndex);
                fileIndex += 1;
            }
            payloadTrees.push({
                tree_id: tree.tree_id,
                species: tree.species,
                optional_inputs: tree.optional_inputs,
                file_indexes: fileIndexes,
            });
        }

        formData.append('payload', JSON.stringify({ trees: payloadTrees }));
        return formData;
    }

    async function matRunAllTrees() {
        if (!matQueue.length) {
            matSetStatus('matStatus', 'Add at least one tree before running analysis.', true);
            return;
        }

        const runBtn = document.getElementById('matRunAll');
        runBtn.disabled = true;
        matSetStatus('matStatus', 'Running analysis. Local CPU execution can take a long time.');
        document.getElementById('matLogOutput').textContent = '';
        stepperStart('matMultipleStepper', 'matMultipleStep', 3);

        try {
            const response = await fetch('/api/maturity/analyze', {
                method: 'POST',
                body: matBuildFormData(),
            });
            const body = await response.json();

            if (!response.ok) {
                const detail = body.detail || body;
                const message = typeof detail === 'string' ? detail : detail.message;
                document.getElementById('matLogOutput').textContent = typeof detail === 'object' ? detail.log || '' : '';
                matSetStatus('matStatus', message || 'Analysis failed.', true);
                stepperFinish('matMultipleStep', 3, false);
                return;
            }

            matRenderTable(document.getElementById('matSummaryHead'), document.getElementById('matSummaryBody'), body.summary);
            matRenderCards('matResultCards', body.tree_results);
            matRenderGallery('matOverlayGallery', body.overlays);
            matRenderGallery('matMeasurementGallery', body.measurement_guides);
            matRenderGallery('matMaturityGallery', body.maturity_plots || body.plots);
            document.getElementById('matLogOutput').textContent = body.log || '';

            const zipLink = document.getElementById('matZipLink');
            zipLink.href = body.zip_url;
            zipLink.classList.remove('hidden');
            matSetStatus('matStatus', `Analysis complete. Run ID: ${body.run_id}`);
            stepperFinish('matMultipleStep', 3, true);
        } catch (error) {
            matSetStatus('matStatus', error.message || 'Analysis failed.', true);
            stepperFinish('matMultipleStep', 3, false);
        } finally {
            runBtn.disabled = false;
        }
    }

    // --- Maturity: Single Estimation (one tree, one photo, full vision pipeline) ---

    async function matRunSingleEstimation() {
        const files = Array.from(document.getElementById('matSingleImages').files || []);
        if (!files.length) {
            matSetStatus('matSingleStatus', 'Upload at least one photo of the tree.', true);
            return;
        }

        const runBtn = document.getElementById('matSingleRun');
        runBtn.disabled = true;
        matSetStatus('matSingleStatus', 'Running analysis. Local CPU execution can take a long time.');
        document.getElementById('matSingleLogOutput').textContent = '';
        stepperStart('matSingleStepper', 'matSingleStep', 3);

        const formData = new FormData();
        const fileIndexes = [];
        files.forEach((file, idx) => {
            formData.append('files', file, file.name);
            fileIndexes.push(idx);
        });
        formData.append('payload', JSON.stringify({
            trees: [{
                tree_id: 'Tree_01',
                species: document.getElementById('matSingleSpecies').value,
                optional_inputs: matCollectOptionalInputs('matSingle'),
                file_indexes: fileIndexes,
            }],
        }));

        try {
            const response = await fetch('/api/maturity/analyze', {
                method: 'POST',
                body: formData,
            });
            const body = await response.json();

            if (!response.ok) {
                const detail = body.detail || body;
                const message = typeof detail === 'string' ? detail : detail.message;
                document.getElementById('matSingleLogOutput').textContent = typeof detail === 'object' ? detail.log || '' : '';
                matSetStatus('matSingleStatus', message || 'Analysis failed.', true);
                stepperFinish('matSingleStep', 3, false);
                return;
            }

            matRenderCards('matSingleResultCards', body.tree_results);
            matRenderGallery('matSingleOverlayGallery', body.overlays);
            matRenderGallery('matSingleMeasurementGallery', body.measurement_guides);
            matRenderGallery('matSingleMaturityGallery', body.maturity_plots || body.plots);
            document.getElementById('matSingleLogOutput').textContent = body.log || '';

            const zipLink = document.getElementById('matSingleZipLink');
            zipLink.href = body.zip_url;
            zipLink.classList.remove('hidden');
            matSetStatus('matSingleStatus', `Analysis complete. Run ID: ${body.run_id}`);
            stepperFinish('matSingleStep', 3, true);
        } catch (error) {
            matSetStatus('matSingleStatus', error.message || 'Analysis failed.', true);
            stepperFinish('matSingleStep', 3, false);
        } finally {
            runBtn.disabled = false;
        }
    }

    // --- Full Pipeline (auto-integrated): Module 1 species ID -> Module 2 DBH/maturity ---

    async function startAutoFlowPipeline() {
        analyzeBtn.disabled = true;
        analyzeBtnText.textContent = 'Processing...';
        resultsWrapperAutoFlow.classList.add('hidden');
        stepperStart('autoFlowStepperCard', 'autoFlowStep', 3);

        const formData = new FormData();
        formData.append('file', selectedFile);
        const optional = matCollectOptionalInputs('autoFlow');
        for (const [key, value] of Object.entries(optional)) {
            formData.append(key, value);
        }

        try {
            const resp = await fetch('/api/maturity/analyze/auto', {
                method: 'POST',
                body: formData,
            });

            const data = await resp.json();
            if (!resp.ok) {
                const detail = data.detail || data;
                throw new Error(typeof detail === 'string' ? detail : (detail.message || 'Full pipeline failed.'));
            }

            renderAutoFlowResult(data);
            resultsWrapperAutoFlow.classList.remove('hidden');
            stepperFinish('autoFlowStep', 3, true);
        } catch (err) {
            alert(`Full pipeline failed: ${err.message}`);
            stepperFinish('autoFlowStep', 3, false);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtnText.textContent = 'Run Full Pipeline';
        }
    }

    function renderAutoFlowResult(data) {
        const identification = data.identification || {};
        const trees = identification.trees || [];
        const tree = trees.find((t) => t.status === 'known') || trees[0];

        if (identification.image && identification.image.annotated) {
            document.getElementById('autoFlowAnnotatedImage').src = identification.image.annotated + `?t=${Date.now()}`;
        }

        const heroBadge = document.getElementById('autoFlowDecisionBadge');
        const heroSpecies = document.getElementById('autoFlowSpeciesName');
        const heroConfidence = document.getElementById('autoFlowConfidence');
        const heroCommercial = document.getElementById('autoFlowCommercial');

        if (tree) {
            const known = tree.status === 'known';
            heroBadge.textContent = known ? 'KNOWN SPECIES' : String(tree.status || 'unknown').toUpperCase().replaceAll('_', ' ');
            heroBadge.className = `status-badge ${known ? 'known' : 'open_set_rejected'}`;
            heroSpecies.textContent = tree.raglo && tree.raglo.final_species ? formatSpeciesName(tree.raglo.final_species) : 'Unknown';
            heroConfidence.textContent = `${((tree.detection?.confidence || 0) * 100).toFixed(1)}%`;
            const comm = tree.commercial || {};
            heroCommercial.textContent = comm.evaluated ? (comm.commercial_flag ? 'Commercial species' : 'Non-commercial') : 'Not evaluated';
        } else {
            heroBadge.textContent = 'NO TREE DETECTED';
            heroBadge.className = 'status-badge open_set_rejected';
            heroSpecies.textContent = 'N/A';
            heroConfidence.textContent = '0.0%';
            heroCommercial.textContent = 'Not evaluated';
        }

        const unresolvedCard = document.getElementById('autoFlowUnresolvedCard');
        const maturitySection = document.getElementById('autoFlowMaturitySection');

        if (!data.species_resolved) {
            unresolvedCard.classList.remove('hidden');
            document.getElementById('autoFlowUnresolvedMessage').textContent =
                data.message || 'Species could not be resolved automatically.';
            maturitySection.classList.add('hidden');
            return;
        }

        unresolvedCard.classList.add('hidden');
        maturitySection.classList.remove('hidden');

        const maturity = data.maturity || {};
        matRenderCards('autoFlowResultCards', maturity.tree_results);
        matRenderGallery('autoFlowOverlayGallery', maturity.overlays);
        matRenderGallery('autoFlowMeasurementGallery', maturity.measurement_guides);
        matRenderGallery('autoFlowMaturityGallery', maturity.maturity_plots || maturity.plots);
        document.getElementById('autoFlowLogOutput').textContent = maturity.log || '';

        const zipLink = document.getElementById('autoFlowZipLink');
        if (maturity.zip_url) {
            zipLink.href = maturity.zip_url;
            zipLink.classList.remove('hidden');
        } else {
            zipLink.classList.add('hidden');
        }
    }

    // --- Wire up Module 2 UI events ---

    document.getElementById('matAddTree').addEventListener('click', matAddTreeToQueue);
    document.getElementById('matResetForm').addEventListener('click', matResetFields);
    document.getElementById('matRemoveLast').addEventListener('click', matRemoveLastTree);
    document.getElementById('matClearQueue').addEventListener('click', matClearTreeQueue);
    document.getElementById('matRunAll').addEventListener('click', matRunAllTrees);
    document.getElementById('matSingleRun').addEventListener('click', matRunSingleEstimation);

    matRenderQueue();
    loadMaturityConfig();
});
