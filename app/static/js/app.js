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

    // App Shell Elements (sidebar nav + mobile off-canvas drawer)
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebarBackdrop = document.getElementById('sidebarBackdrop');
    const topbarTitle = document.getElementById('topbarTitle');
    const workspaceKicker = document.getElementById('workspaceKicker');
    const workspaceTitle = document.getElementById('workspaceTitle');
    const workspaceDescription = document.getElementById('workspaceDescription');

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
    const quickAnalysisProgress = document.getElementById('quickAnalysisProgress');
    const quickProgressLabel = document.getElementById('quickProgressLabel');
    const quickProgressPercent = document.getElementById('quickProgressPercent');
    const quickProgressFill = document.getElementById('quickProgressFill');
    const quickProgressMessage = document.getElementById('quickProgressMessage');

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
    const optionalFieldsToggle = document.getElementById('optionalFieldsToggle');
    const optionalFieldsContent = document.getElementById('optionalFieldsContent');
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

    optionalFieldsToggle.addEventListener('click', () => {
        const isExpanded = optionalFieldsToggle.getAttribute('aria-expanded') === 'true';
        optionalFieldsToggle.setAttribute('aria-expanded', String(!isExpanded));
        optionalFieldsContent.classList.toggle('expanded', !isExpanded);
        optionalFieldsToggle.querySelector('.toggle-label').textContent = isExpanded ? 'Expand' : 'Collapse';
    });

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
    const clfSpeciesProfile = document.getElementById('clfSpeciesProfile');
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

    const MODE_CONTEXT = {
        detector: ['Computer vision', 'Map the trees in your scene.', 'Locate, segment, and count visible trunks with confidence and spatial detail.'],
        classifier: ['Species intelligence', 'Know the species behind the bark.', 'Compare visual evidence against known species while screening unfamiliar samples.'],
        integrated: ['Inventory intelligence', 'Understand every tree in the frame.', 'Detect trunks, identify species, and turn field imagery into a clear, reviewable forest inventory.'],
        maturity: ['Growth intelligence', 'Measure readiness with confidence.', 'Estimate DBH and maturity from field photos with optional site context for stronger results.'],
        autoflow: ['End-to-end assessment', 'One image. The complete tree story.', 'Run detection, identification, and maturity assessment in one guided workflow.'],
        timber: ['Quality intelligence', 'Turn cut faces into quality grades.', 'Detect timber sections and combine visual evidence with measurements for consistent grading.'],
        'grading-bid': ['Commercial intelligence', 'Move from grade to market value.', 'Use graded timber evidence to prepare a transparent, editable bid estimate.'],
        auction: ['Market intelligence', 'Price timber with better context.', 'Model an expected auction value from log quality, dimensions, and market conditions.'],
    };

    // Mobile Sidebar Drawer (below ~900px the sidebar becomes an off-canvas drawer
    // toggled via the topbar hamburger button; above that it's an always-visible rail).
    function openSidebar() {
        sidebar.classList.add('open');
        sidebarBackdrop.classList.add('open');
        sidebarToggle.setAttribute('aria-expanded', 'true');
    }
    function closeSidebar() {
        sidebar.classList.remove('open');
        sidebarBackdrop.classList.remove('open');
        sidebarToggle.setAttribute('aria-expanded', 'false');
    }
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.contains('open') ? closeSidebar() : openSidebar();
        });
    }
    if (sidebarBackdrop) {
        sidebarBackdrop.addEventListener('click', closeSidebar);
    }

    // Tab Switching Logic
    tabButtons.forEach(btn => {
        if (!btn) return;
        btn.addEventListener('click', () => {
            const mode = btn.dataset.tab;
            closeSidebar();
            if (currentMode === mode) return;
            quickAnalysisProgress.classList.add('hidden');

            currentMode = mode;
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (topbarTitle) {
                const titleEl = btn.querySelector('.tab-title');
                topbarTitle.textContent = titleEl ? titleEl.textContent : '';
            }
            const context = MODE_CONTEXT[mode];
            if (context) {
                workspaceKicker.textContent = context[0];
                workspaceTitle.textContent = context[1];
                workspaceDescription.textContent = context[2];
            }

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
        const progressHost = container && (container.matches('.stepper') ? container : container.querySelector('.stepper'));
        if (progressHost) setCompactProgress(progressHost, 10, 'Analysis in progress');
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
        const firstStep = document.getElementById(`${prefix}1`);
        const container = firstStep && firstStep.closest('.stepper');
        if (container) setCompactProgress(container, success ? 100 : 0, success ? 'Analysis complete' : 'Analysis stopped', !success);
    }

    function setCompactProgress(container, percent, label, isError = false) {
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

    function quickProgressStart(label, message) {
        quickAnalysisProgress.classList.remove('hidden', 'complete', 'error');
        quickProgressLabel.textContent = label;
        quickProgressMessage.textContent = message;
        quickProgressPercent.textContent = '10%';
        quickProgressFill.style.width = '10%';
        quickProgressFill.classList.add('running');
    }

    function quickProgressFinish(success) {
        quickAnalysisProgress.classList.toggle('complete', success);
        quickAnalysisProgress.classList.toggle('error', !success);
        quickProgressFill.classList.remove('running');
        quickProgressFill.style.width = success ? '100%' : '0%';
        quickProgressPercent.textContent = success ? '100%' : 'Stopped';
        quickProgressMessage.textContent = success ? 'Analysis complete. Results are ready below.' : 'The analysis could not be completed.';
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
        quickAnalysisProgress.classList.add('hidden');
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
        timelineContainer.innerHTML = `
            <div class="pipeline-progress" role="progressbar" aria-label="Analysis progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
                <div class="progress-heading">
                    <span id="timelineProgressCount">Stage 1 of ${STAGE_NAMES.length}</span>
                    <strong id="timelineProgressPercent">0%</strong>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" id="timelineProgressFill"></div>
                </div>
            </div>
            <div class="current-stage active" id="timelineCurrentStage" aria-live="polite">
                <div class="current-stage-icon"><span></span></div>
                <div class="current-stage-copy">
                    <span class="current-stage-label">Current stage</span>
                    <strong id="timelineCurrentTitle">${STAGE_NAMES[0]}</strong>
                    <p id="timelineCurrentMessage">Preparing analysis...</p>
                </div>
            </div>
        `;
    }

    async function pollJobStatus(jobId) {
        try {
            const resp = await fetch(`/api/jobs/${jobId}`);
            if (!resp.ok) return;

            const job = await resp.json();
            const currentStage = job.stage_index;

            const safeStage = Math.max(1, Math.min(Number(currentStage) || 1, STAGE_NAMES.length));
            const progress = job.status === 'completed'
                ? 100
                : Math.min(95, (safeStage / STAGE_NAMES.length) * 100);
            const progressBar = timelineContainer.querySelector('.pipeline-progress');
            const progressFill = document.getElementById('timelineProgressFill');
            const progressCount = document.getElementById('timelineProgressCount');
            const progressPercent = document.getElementById('timelineProgressPercent');
            const currentStageCard = document.getElementById('timelineCurrentStage');
            const currentTitle = document.getElementById('timelineCurrentTitle');
            const currentMessage = document.getElementById('timelineCurrentMessage');

            progressFill.style.width = `${progress}%`;
            const roundedProgress = Math.round(progress);
            progressBar.setAttribute('aria-valuenow', String(roundedProgress));
            progressPercent.textContent = `${roundedProgress}%`;
            progressCount.textContent = job.status === 'completed'
                ? `All ${STAGE_NAMES.length} stages complete`
                : `Stage ${safeStage} of ${STAGE_NAMES.length}`;
            currentTitle.textContent = job.status === 'completed' ? 'Analysis complete' : STAGE_NAMES[safeStage - 1];
            currentMessage.textContent = job.status === 'completed' ? 'Results are ready to review.' : (job.message || 'Processing...');
            currentStageCard.className = `current-stage ${job.status === 'error' ? 'error' : job.status === 'completed' ? 'completed' : 'active'}`;


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

        trees.forEach((tree, treeIdx) => {
            const card = document.createElement('div');
            card.className = 'card tree-card fade-in-up';
            card.style.animationDelay = `${treeIdx * 50}ms`;

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
                    ${renderSpeciesProfile(tree.raglo.final_species, comm)}
                </div>
                <div class="tree-card-footer">
                    <button type="button" class="btn btn-secondary inspect-btn" data-tree-id="${tree.tree_id}" aria-label="View identification process for ${tree.display_name}">
                        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/></svg>
                        <span>View identification process</span>
                        <span aria-hidden="true">&rarr;</span>
                    </button>
                </div>
            `;
            treesListContainer.appendChild(card);
        });
    }

    // Delegation keeps the action available when result cards are replaced or rerendered.
    treesListContainer.addEventListener('click', (event) => {
        const button = event.target.closest('.inspect-btn');
        if (!button || !treesListContainer.contains(button)) return;
        const rawTreeId = button.dataset.treeId;
        const numericTreeId = Number(rawTreeId);
        openProcessModal(Number.isNaN(numericTreeId) ? rawTreeId : numericTreeId);
    });

    // Section 26 & 27: Expandable Multi-Candidate Process View
    function openProcessModal(treeId) {
        if (!currentResultsData) return;
        const tree = currentResultsData.trees.find(t => String(t.tree_id) === String(treeId));
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

        openModal();
    }

    function openModal() {
        processModal.classList.remove('hidden');
        processModal.setAttribute('aria-hidden', 'false');
        document.body.classList.add('modal-open');
        // Force the initial opacity state to render before applying the open state.
        void processModal.offsetWidth;
        processModal.classList.add('open');
        closeModalBtn.focus();
    }

    function closeModal() {
        processModal.classList.remove('open');
        processModal.setAttribute('aria-hidden', 'true');
        document.body.classList.remove('modal-open');
        const onTransitionEnd = (e) => {
            if (e.target !== processModal) return;
            processModal.classList.add('hidden');
            processModal.removeEventListener('transitionend', onTransitionEnd);
        };
        processModal.addEventListener('transitionend', onTransitionEnd);
        // Fallback in case transitionend doesn't fire (e.g. reduced-motion 0-duration).
        // Guarded so a quick reopen within the window isn't hidden again afterward.
        setTimeout(() => {
            if (!processModal.classList.contains('open')) processModal.classList.add('hidden');
        }, 300);
    }

    closeModalBtn.addEventListener('click', closeModal);
    processModal.addEventListener('click', (e) => {
        if (e.target === processModal) closeModal();
    });
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && processModal.classList.contains('open')) closeModal();
    });

    // 2. Tree Detector Only Execution Mode
    async function startDetectorOnlyAnalysis() {
        analyzeBtn.disabled = true;
        analyzeBtnText.textContent = "Detecting Tree Trunks...";
        resultsWrapperDetector.classList.add('hidden');
        quickProgressStart('Detecting tree trunks', 'Running segmentation and spatial measurements...');

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
                data.trees.forEach((tree, treeIdx) => {
                    const card = document.createElement('div');
                    card.className = 'card tree-card fade-in-up';
                    card.style.animationDelay = `${treeIdx * 50}ms`;
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
            quickProgressFinish(true);
        } catch (err) {
            quickProgressFinish(false);
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
        quickProgressStart('Identifying tree species', 'Preparing bark evidence and running open-set classification...');

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
                clfSpeciesProfile.innerHTML = '';
            } else {
                clfDecisionBadge.textContent = "KNOWN SPECIES";
                clfDecisionBadge.className = "status-badge known";
                clfSpeciesName.textContent = formatSpeciesName(res.final_species);
                clfSpeciesProfile.innerHTML = renderSpeciesProfile(res.final_species, null);
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
                    card.className = 'attention-card fade-in-up';
                    card.style.animationDelay = `${idx * 40}ms`;
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
            quickProgressFinish(true);
        } catch (err) {
            quickProgressFinish(false);
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
            'pine': 'Pine (Pinus caribaea)',
            'rubber': 'Rubber (Hevea brasiliensis)',
            'teak': 'Teak (Tectona grandis)'
        };
        return map[name.toLowerCase()] || name;
    }

    // Sri Lanka-specific species facts, used wherever a classification result is
    // shown but no backend "commercial" data is available for that surface
    // (e.g. the standalone Tree Classification tab, which doesn't run the
    // Integrated Pipeline's commercial-flagging stage). Mirrors
    // app/data/commercial_species.json's scientific_name/growing_regions/
    // typical_uses fields for the same 4 species.
    const SPECIES_INFO = {
        'teak': {
            scientific_name: 'Tectona grandis',
            growing_regions: 'Dry and low-intermediate zones, notably Kurunegala and Nuwara Eliya districts',
            typical_uses: 'Boat building, furniture, veneer, exterior construction, and carving — valued for durability and termite/weather resistance',
        },
        'mahogany': {
            scientific_name: 'Swietenia macrophylla',
            growing_regions: 'Wet and intermediate zones, with plantations concentrated around Kurunegala and Kegalle',
            typical_uses: 'Furniture and high-end joinery — strong local and export demand',
        },
        'pine': {
            scientific_name: 'Pinus caribaea (Caribbean Pine)',
            growing_regions: 'Wet and intermediate zone hill country reforestation sites (e.g. Belihuloya), 100-2000m elevation',
            typical_uses: 'Timber, pulpwood, and resin/turpentine production; originally planted for erosion control on degraded land',
        },
        'rubber': {
            scientific_name: 'Hevea brasiliensis',
            growing_regions: 'Traditionally the wet-zone rubber belt (Kalutara, Ratnapura, Kegalle, Kurunegala); newer plantings in the Northern dry zone since 2010',
            typical_uses: 'Rubberwood harvested after the ~25-30 year latex-tapping cycle, used for furniture, chipboard, and MDF panel production',
        },
    };

    // Builds a compact "Species Profile" info block (scientific name, Sri Lanka
    // growing regions, typical commercial uses) for a classification result.
    // Prefers backend-supplied commercial data (already Sri Lanka-specific,
    // app/data/commercial_species.json) when available/evaluated, otherwise
    // falls back to the local SPECIES_INFO map above. Returns '' when nothing
    // is known about the species (e.g. open-set rejected/unknown).
    function renderSpeciesProfile(name, commercialData) {
        if (!name) return '';
        const key = name.toLowerCase();
        const fromBackend = commercialData && commercialData.evaluated ? commercialData : null;
        const info = {
            scientific_name: (fromBackend && fromBackend.scientific_name) || (SPECIES_INFO[key] && SPECIES_INFO[key].scientific_name),
            growing_regions: (fromBackend && fromBackend.growing_regions) || (SPECIES_INFO[key] && SPECIES_INFO[key].growing_regions),
            typical_uses: (fromBackend && fromBackend.typical_uses) || (SPECIES_INFO[key] && SPECIES_INFO[key].typical_uses),
        };
        if (!info.scientific_name && !info.growing_regions && !info.typical_uses) return '';

        return `
            <div class="species-profile-note">
                ${info.scientific_name ? `<p><strong>Scientific name:</strong> ${info.scientific_name}</p>` : ''}
                ${info.growing_regions ? `<p><strong>Grown in Sri Lanka:</strong> ${info.growing_regions}</p>` : ''}
                ${info.typical_uses ? `<p><strong>Typical uses:</strong> ${info.typical_uses}</p>` : ''}
            </div>
        `;
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
            .map((result, resultIdx) => {
                const delayStyle = `animation-delay:${resultIdx * 50}ms`;
                if (!result.dbh_success) {
                    return `
                        <article class="result-card error fade-in-up" style="${delayStyle}">
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
                    <article class="result-card fade-in-up" style="${delayStyle}">
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
        const heroSpeciesProfile = document.getElementById('autoFlowSpeciesProfile');

        if (tree) {
            const known = tree.status === 'known';
            heroBadge.textContent = known ? 'KNOWN SPECIES' : String(tree.status || 'unknown').toUpperCase().replaceAll('_', ' ');
            heroBadge.className = `status-badge ${known ? 'known' : 'open_set_rejected'}`;
            heroSpecies.textContent = tree.raglo && tree.raglo.final_species ? formatSpeciesName(tree.raglo.final_species) : 'Unknown';
            heroConfidence.textContent = `${((tree.detection?.confidence || 0) * 100).toFixed(1)}%`;
            const comm = tree.commercial || {};
            heroCommercial.textContent = comm.evaluated ? (comm.commercial_flag ? 'Commercial species' : 'Non-commercial') : 'Not evaluated';
            heroSpeciesProfile.innerHTML = renderSpeciesProfile(tree.raglo && tree.raglo.final_species, comm);
        } else {
            heroBadge.textContent = 'NO TREE DETECTED';
            heroBadge.className = 'status-badge open_set_rejected';
            heroSpecies.textContent = 'N/A';
            heroConfidence.textContent = '0.0%';
            heroCommercial.textContent = 'Not evaluated';
            heroSpeciesProfile.innerHTML = '';
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
