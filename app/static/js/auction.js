/**
 * Frontend JavaScript for the Auction Bid Price tabs (SmartTimber-LK Module 4):
 * "Grading + Bid Price" (tab 7) and "Auction Bid Price" (tab 8). Fully independent
 * backend from app.js/timber.js - its own /api/auction/* endpoints, its own model.
 *
 * The only integration point is a UI-layer convenience: on the "Grading + Bid Price"
 * tab, this file reads window.timberGradedCrops (populated by timber.js after a
 * crop is graded) to offer a "Use This Crop" prefill. Nothing here calls into
 * app/timber's or app/inference's backend, and grading itself is never touched.
 */

document.addEventListener('DOMContentLoaded', () => {
    const tabAuction = document.getElementById('tabAuction');
    const tabGradingBid = document.getElementById('tabGradingBid');
    if (!tabAuction && !tabGradingBid) return; // Auction panels not present on this page

    const GRADE_TO_QUALITY = { B: 1, A10: 2, A20: 3, A30: 3, A40: 4, A50: 5 };

    const FIELD_LABELS = {
        species: 'Species',
        region: 'Region',
        season: 'Harvest Season',
        auction_type: 'Auction Config',
        competition_level: 'Competition Strength',
    };

    const FIELD_SECTIONS = [
        {
            title: 'Wood Sourcing & Species',
            fields: [
                { name: 'species', type: 'select' },
                { name: 'region', type: 'select' },
                { name: 'season', type: 'select' },
            ],
        },
        {
            title: 'Physical Dimensions & Properties',
            fields: [
                { name: 'diameter_cm', label: 'Diameter (cm)', type: 'number', step: 0.01, min: 0.1 },
                { name: 'length_m', label: 'Length (m)', type: 'number', step: 0.01, min: 0.1 },
                { name: 'volume_m3', label: 'Volume (m³)', type: 'number', step: 0.0001, min: 0.0001 },
                { name: 'density_kg_m3', label: 'Density (kg/m³)', type: 'number', step: 1, min: 50 },
                { name: 'moisture_content', label: 'Moisture Content (%)', type: 'number', step: 0.1, min: 0, max: 100 },
            ],
        },
        {
            title: 'Quality & Defect Grades',
            fields: [
                { name: 'straightness_score', label: 'Straightness (1-10)', type: 'number', step: 0.1, min: 1, max: 10 },
                { name: 'taper_score', label: 'Taper Score (1-10)', type: 'number', step: 0.1, min: 1, max: 10 },
                { name: 'visible_defects_score', label: 'Visible Defects (0-10)', type: 'number', step: 0.1, min: 0, max: 10 },
                { name: 'internal_defect_risk', label: 'Internal Defect Risk (0-1)', type: 'number', step: 0.01, min: 0, max: 1 },
                { name: 'quality_grade', label: 'Overall Quality Grade (1-5)', type: 'number', step: 1, min: 1, max: 5 },
            ],
        },
        {
            title: 'Market Config & Price Indicators',
            fields: [
                { name: 'avg_market_price_species', label: 'Species Base Price (Rs./m³)', type: 'number', step: 0.01, min: 1 },
                { name: 'price_volatility', label: 'Price Volatility (0-1)', type: 'number', step: 0.01, min: 0, max: 1 },
                { name: 'market_demand_index', label: 'Demand Index', type: 'number', step: 0.01, min: 0.1 },
                { name: 'supply_index', label: 'Supply Index', type: 'number', step: 0.01, min: 0.1 },
                { name: 'export_demand_index', label: 'Export Index', type: 'number', step: 0.01, min: 0.1 },
                { name: 'auction_type', type: 'select' },
                { name: 'competition_level', type: 'select' },
                { name: 'num_expected_bidders', label: 'Expected Bidders', type: 'number', step: 1, min: 1 },
            ],
        },
    ];

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

    function formatRs(value) {
        const num = Number(value) || 0;
        return `Rs. ${num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }

    function fieldId(prefix, name) {
        return prefix + name.split('_').map((w) => w[0].toUpperCase() + w.slice(1)).join('');
    }

    function buildFormHtml(prefix) {
        const renderSection = (section) => {
            const fieldsHtml = section.fields.map((field) => {
                const id = fieldId(prefix, field.name);
                if (field.type === 'select') {
                    return `
                        <label>
                            <span>${escapeHtml(FIELD_LABELS[field.name] || field.name)}</span>
                            <select id="${id}"></select>
                        </label>
                    `;
                }
                const attrs = [
                    field.min !== undefined ? `min="${field.min}"` : '',
                    field.max !== undefined ? `max="${field.max}"` : '',
                    field.step !== undefined ? `step="${field.step}"` : '',
                ].filter(Boolean).join(' ');
                return `
                    <label>
                        <span>${escapeHtml(field.label)}</span>
                        <input type="number" id="${id}" ${attrs}>
                    </label>
                `;
            }).join('');
            return `
                <div class="valuation-section">
                    <div class="section-title">${escapeHtml(section.title)}</div>
                    <div class="mat-form-grid">${fieldsHtml}</div>
                </div>
            `;
        };
        const essentialHtml = FIELD_SECTIONS.slice(0, 3).map(renderSection).join('');
        const advancedHtml = FIELD_SECTIONS.slice(3).map(renderSection).join('');

        return `
            <section class="card valuation-form-card">
                <div class="valuation-form-heading">
                    <div><span class="workspace-kicker">Auction estimator</span><h2>Describe the log</h2><p>Start with the essential timber details. Model assumptions remain editable under Advanced.</p></div>
                    <span class="defaults-chip">Model defaults applied</span>
                </div>
                ${essentialHtml}
                <details class="valuation-advanced">
                    <summary><span><strong>Advanced market assumptions</strong><small>Demand, supply, volatility and auction settings</small></span><span>Review settings</span></summary>
                    <div class="valuation-advanced-body">${advancedHtml}</div>
                </details>
                <div class="action-bar">
                    <button type="button" class="btn btn-primary" id="${prefix}PredictBtn">Estimate auction value</button>
                </div>
                <div class="embedded-progress hidden" id="${prefix}Progress">
                    <div><span class="embedded-progress-label">Computing market value</span><strong class="embedded-progress-percent">10%</strong></div>
                    <div class="compact-progress-track"><span></span></div>
                </div>
                <div id="${prefix}Status" class="mat-status">Ready</div>
            </section>
        `;
    }

    function populateForm(prefix, config) {
        const options = config.categorical_options || {};
        const defaults = config.field_defaults || {};
        for (const section of FIELD_SECTIONS) {
            for (const field of section.fields) {
                const el = document.getElementById(fieldId(prefix, field.name));
                if (!el) continue;
                if (field.type === 'select') {
                    el.innerHTML = (options[field.name] || [])
                        .map((opt) => `<option value="${escapeHtml(opt)}">${escapeHtml(opt)}</option>`)
                        .join('');
                }
                if (defaults[field.name] !== undefined) {
                    el.value = defaults[field.name];
                }
            }
        }
        if (prefix === 'au') {
            const diameter = document.getElementById(fieldId(prefix, 'diameter_cm'));
            const length = document.getElementById(fieldId(prefix, 'length_m'));
            const volume = document.getElementById(fieldId(prefix, 'volume_m3'));
            const syncVolume = () => {
                const d = Number(diameter.value) / 100;
                const l = Number(length.value);
                if (d > 0 && l > 0) volume.value = (Math.PI * d * d * l / 4).toFixed(4);
            };
            diameter.addEventListener('input', syncVolume);
            length.addEventListener('input', syncVolume);
            volume.insertAdjacentHTML('afterend', '<small class="field-help">Calculated automatically from diameter and length; you can still edit it.</small>');
        }
    }

    function setFieldValue(prefix, fieldName, value) {
        const el = document.getElementById(fieldId(prefix, fieldName));
        if (el) el.value = value;
    }

    function collectFormValues(prefix) {
        const values = {};
        for (const section of FIELD_SECTIONS) {
            for (const field of section.fields) {
                const el = document.getElementById(fieldId(prefix, field.name));
                if (!el) continue;
                values[field.name] = field.type === 'select' ? el.value : parseFloat(el.value);
            }
        }
        return values;
    }

    function maxAbsShap(shapList) {
        return Math.max(...(shapList || []).map((s) => Math.abs(s.value)), 1);
    }

    function renderResults(prefix, inputs, data) {
        const container = document.getElementById(`${prefix}ResultsContainer`);
        if (!container) return;

        const maxAbs = maxAbsShap(data.shap);
        const shapRows = (data.shap || []).map((item) => {
            const isNeg = item.value < 0;
            const pct = (Math.abs(item.value) / maxAbs) * 100;
            return `
                <div class="prob-row">
                    <div class="prob-header">
                        <span class="prob-name">${escapeHtml(item.label)}</span>
                        <span class="prob-pct${isNeg ? ' negative' : ''}">${isNeg ? '-' : '+'}${formatRs(Math.abs(item.value))}</span>
                    </div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill${isNeg ? ' negative' : ''}" style="width: ${pct.toFixed(1)}%;"></div>
                    </div>
                </div>
            `;
        }).join('');

        const qp = data.quality_profile || { labels: [], user_values: [], market_avg: [] };
        const qualityRows = qp.labels.map((label, idx) => {
            const userVal = qp.user_values[idx] || 0;
            const marketVal = qp.market_avg[idx] || 0;
            return `
                <div class="paired-bar-row">
                    <span class="paired-bar-label">${escapeHtml(label)}</span>
                    <div class="paired-bar-line this-log">
                        <span style="width: 84px;">This Log</span>
                        <div class="paired-bar-track"><div class="paired-bar-fill" style="width: ${Math.min(100, (userVal / 10) * 100)}%;"></div></div>
                        <span>${userVal.toFixed(1)}</span>
                    </div>
                    <div class="paired-bar-line market-avg">
                        <span style="width: 84px;">Market Avg</span>
                        <div class="paired-bar-track"><div class="paired-bar-fill" style="width: ${Math.min(100, (marketVal / 10) * 100)}%;"></div></div>
                        <span>${marketVal.toFixed(1)}</span>
                    </div>
                </div>
            `;
        }).join('');

        const pc = data.price_comparison || { labels: [], values: [] };
        const priceCards = pc.labels.map((label, idx) => `
            <div class="metric-item">
                <span class="label">${escapeHtml(label)}</span>
                <span class="val bold">${formatRs(pc.values[idx])}</span>
            </div>
        `).join('');

        const bm = data.bar_metrics || {};

        const floorNote = data.below_price_floor ? `
            <div class="card" style="background: var(--warning-bg); border: 1px solid rgba(217, 140, 43, 0.35); color: #7a4a12;">
                <strong>Below the model's viable price floor.</strong>
                The model's raw estimate for this combination was ${formatRs(data.raw_predicted_value)} (negative), so it has been floored at Rs. 0.00 rather than shown as a loss.
                This model is heavily driven by log <strong>volume</strong> and <strong>quality grade</strong> &mdash; a small or modestly-sized log will often floor out this way. Try a larger volume/length, a higher quality grade, or a higher-demand market profile for a representative estimate.
            </div>
        ` : '';

        container.innerHTML = `
            <div class="card value-hero">
                <span class="value-hero-label">Commercial Bid Estimation</span>
                <div class="value-hero-amount">${formatRs(data.predicted_value)}</div>
            </div>
            ${floorNote}

            <div class="card">
                <div class="card-header">
                    <h2>90% Uncertainty Interval</h2>
                    <p>Lower and upper pricing confidence bounds based on volatility and defect risk.</p>
                </div>
                <div class="interval-bar-wrap">
                    <div class="interval-bar-labels">
                        <span>Low: ${formatRs(bm.low_bound)}</span>
                        <span>High: ${formatRs(bm.high_bound)}</span>
                    </div>
                    <div class="interval-bar-track">
                        <div class="interval-bar-range" style="left: ${bm.range_left_pct || 0}%; width: ${bm.range_width_pct || 0}%;"></div>
                        <div class="interval-bar-point" style="left: ${bm.point_pct || 0}%;"></div>
                    </div>
                </div>
            </div>

            <div class="summary-grid">
                <div class="metric-card">
                    <span class="metric-label">Rec. Starting Bid</span>
                    <span class="metric-value">${formatRs(data.starting_bid)}</span>
                </div>
                <div class="metric-card">
                    <span class="metric-label">Expected Final Price</span>
                    <span class="metric-value">${formatRs(data.expected_final_price)}</span>
                </div>
                <div class="metric-card metric-success">
                    <span class="metric-label">Sale Probability</span>
                    <span class="metric-value">${((data.sale_probability || 0) * 100).toFixed(1)}%</span>
                </div>
            </div>

            <div class="two-column-grid">
                <div class="card">
                    <div class="card-header">
                        <h2>Key Influencing Factors</h2>
                        <p>Positive (green) and negative (red) contributions vs. a typical baseline log.</p>
                    </div>
                    <div class="prob-bars-container">${shapRows}</div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <h2>Log Quality Profile</h2>
                        <p>This log vs. market baseline, 0-10 scale.</p>
                    </div>
                    ${qualityRows}
                </div>
            </div>

            <div class="card">
                <div class="card-header"><h2>Price Comparison</h2></div>
                <div class="tree-metrics">${priceCards}</div>
            </div>

            <div class="card">
                <div class="card-header"><h2>Parameter Summary</h2></div>
                <div class="mat-table-wrap">
                    <table>
                        <tbody>
                            <tr><td><strong>Species &amp; Origin</strong></td><td>${escapeHtml(inputs.species)} from ${escapeHtml(inputs.region)} (${escapeHtml(inputs.season)} season)</td></tr>
                            <tr><td><strong>Dimensions &amp; Volume</strong></td><td>${inputs.diameter_cm} cm &times; ${inputs.length_m} m (Volume: ${inputs.volume_m3} m&sup3;)</td></tr>
                            <tr><td><strong>Physical Characteristics</strong></td><td>Density: ${inputs.density_kg_m3} kg/m&sup3; | Moisture: ${inputs.moisture_content}%</td></tr>
                            <tr><td><strong>Quality Scores</strong></td><td>Grade ${inputs.quality_grade}/5 (Straightness: ${inputs.straightness_score}/10 | Taper: ${inputs.taper_score}/10)</td></tr>
                            <tr><td><strong>Defect Metrics</strong></td><td>Visible: ${inputs.visible_defects_score}/10 | Internal Risk: ${(inputs.internal_defect_risk * 100).toFixed(1)}%</td></tr>
                            <tr><td><strong>Market Indicators</strong></td><td>Avg Price: ${formatRs(inputs.avg_market_price_species)}/m&sup3; | Volatility: ${(inputs.price_volatility * 100).toFixed(1)}%</td></tr>
                            <tr><td><strong>Auction Configuration</strong></td><td>${escapeHtml(inputs.auction_type)} auction (${escapeHtml(inputs.competition_level)} competition | ${inputs.num_expected_bidders} bidders)</td></tr>
                        </tbody>
                    </table>
                </div>
                ${data.prediction_id ? `<a href="/api/auction/report/${data.prediction_id}" class="download" target="_blank">Download PDF Report</a>` : ''}
            </div>
        `;
    }

    async function renderHistory(prefix) {
        const container = document.getElementById(`${prefix}ResultsContainer`);
        if (!container) return;
        let historyHtml = '<p style="color: var(--text-secondary);">No predictions yet.</p>';
        try {
            const resp = await fetch('/api/auction/history');
            const data = await resp.json();
            if (data.history && data.history.length) {
                historyHtml = data.history.map((item) => `
                    <div class="metric-item">
                        <span class="label">${escapeHtml(item.species)} &middot; ${escapeHtml((item.timestamp || '').split(' ')[0])}</span>
                        <span class="val">${item.diameter_cm}cm &times; ${item.length_m}m &mdash; ${formatRs(item.predicted_value)}</span>
                    </div>
                `).join('');
            }
        } catch (err) {
            historyHtml = '<p style="color: var(--text-secondary);">Could not load recent estimations.</p>';
        }
        container.insertAdjacentHTML('beforeend', `
            <div class="card">
                <div class="card-header"><h2>Recent Estimations</h2></div>
                <div class="tree-metrics">${historyHtml}</div>
            </div>
        `);
    }

    async function submitPrediction(prefix) {
        const btn = document.getElementById(`${prefix}PredictBtn`);
        const statusEl = document.getElementById(`${prefix}Status`);
        const formCard = btn.closest('.valuation-form-card');
        const invalidField = Array.from(formCard.querySelectorAll('input, select')).find((field) => !field.checkValidity());
        if (invalidField) {
            invalidField.reportValidity();
            invalidField.focus();
            setStatus(statusEl, 'Review the highlighted field before estimating value.', true);
            return;
        }
        const inputs = collectFormValues(prefix);

        if (Object.values(inputs).some((v) => typeof v === 'number' && Number.isNaN(v))) {
            setStatus(statusEl, 'Please fill in every field with a valid number before computing.', true);
            return;
        }

        btn.disabled = true;
        setStatus(statusEl, 'Running XGBoost valuation pipeline...');
        const progress = document.getElementById(`${prefix}Progress`);
        progress.classList.remove('hidden', 'error');
        progress.classList.add('running');
        progress.querySelector('.embedded-progress-percent').textContent = '10%';
        progress.querySelector('.compact-progress-track span').style.width = '10%';

        try {
            const resp = await fetch('/api/auction/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputs),
            });
            const data = await resp.json();
            if (!resp.ok) {
                const detail = data.detail || data;
                throw new Error(typeof detail === 'string' ? detail : 'Valuation failed.');
            }
            renderResults(prefix, inputs, data);
            renderHistory(prefix);
            document.getElementById(`${prefix}ResultsContainer`).scrollIntoView({ behavior: 'smooth', block: 'start' });
            setStatus(statusEl, 'Valuation complete.');
            progress.classList.remove('running');
            progress.querySelector('.embedded-progress-label').textContent = 'Valuation complete';
            progress.querySelector('.embedded-progress-percent').textContent = '100%';
            progress.querySelector('.compact-progress-track span').style.width = '100%';
        } catch (err) {
            setStatus(statusEl, err.message || 'Valuation failed.', true);
            progress.classList.remove('running');
            progress.classList.add('error');
            progress.querySelector('.embedded-progress-percent').textContent = 'Stopped';
            progress.querySelector('.compact-progress-track span').style.width = '0%';
        } finally {
            btn.disabled = false;
        }
    }

    // --- Grading + Bid Price crop picker (UI-layer bridge from Timber Grading) ---

    function applyCropToForm(crop) {
        if (crop.Mid_Girth_M) {
            const diameterCm = (crop.Mid_Girth_M / Math.PI) * 100;
            setFieldValue('gb', 'diameter_cm', diameterCm.toFixed(2));
        }
        if (crop.Length_M !== undefined && crop.Length_M !== null) {
            setFieldValue('gb', 'length_m', Number(crop.Length_M).toFixed(2));
        }
        if (crop.Volume !== undefined && crop.Volume !== null) {
            setFieldValue('gb', 'volume_m3', Number(crop.Volume).toFixed(4));
        }
        const qualityGrade = GRADE_TO_QUALITY[crop.grade];
        if (qualityGrade !== undefined) {
            setFieldValue('gb', 'quality_grade', qualityGrade);
        }
    }

    function renderCropPicker() {
        const container = document.getElementById('gbCropPicker');
        if (!container) return;
        const crops = window.timberGradedCrops || [];
        if (!crops.length) {
            container.innerHTML = '<p style="color: var(--text-secondary);">No graded crops yet. Switch to the Timber Grading tab, detect and grade a photo, then come back here.</p>';
            return;
        }
        container.innerHTML = crops.map((crop, cropIdx) => `
            <div class="crop-picker-item fade-in-up" style="animation-delay:${cropIdx * 40}ms" data-crop-id="${escapeHtml(crop.crop_id)}">
                <div>
                    <strong>${escapeHtml(crop.label)}</strong> &mdash; ${escapeHtml(crop.source_filename)}
                    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 2px;">
                        Grade ${escapeHtml(crop.grade)} (${escapeHtml(crop.bucket || '-')}) &middot;
                        ${crop.Length_M !== undefined ? Number(crop.Length_M).toFixed(2) : '-'} m length &times;
                        ${crop.Mid_Girth_M !== undefined ? Number(crop.Mid_Girth_M).toFixed(2) : '-'} m girth
                    </div>
                </div>
                <button type="button" class="btn btn-secondary use-crop-btn" data-crop-id="${escapeHtml(crop.crop_id)}">Use This Crop</button>
            </div>
        `).join('');

        container.querySelectorAll('.use-crop-btn').forEach((btn) => {
            btn.addEventListener('click', () => {
                const cropId = btn.dataset.cropId;
                const crop = crops.find((c) => c.crop_id === cropId);
                if (!crop) return;
                applyCropToForm(crop);
                container.querySelectorAll('.crop-picker-item').forEach((el) => {
                    el.classList.toggle('selected', el.dataset.cropId === cropId);
                });
            });
        });
    }

    document.addEventListener('timber:cropGraded', renderCropPicker);

    // --- Self-contained Grade & Value workflow ---

    function initIntegratedWorkflow(config) {
        const host = document.getElementById('gbWorkflowContainer');
        if (!host) return;
        const defaults = config.field_defaults || {};
        const options = config.categorical_options || {};
        let files = [];
        let sessionId = null;
        let detectedCrops = [];

        const optionHtml = (name, selected) => (options[name] || []).map((value) =>
            `<option value="${escapeHtml(value)}"${value === selected ? ' selected' : ''}>${escapeHtml(value)}</option>`
        ).join('');

        host.innerHTML = `
            <section class="card gb-upload-card">
                <div class="card-header"><h2>1. Upload timber photos</h2><p>Use clear photos where the cut face is visible and reasonably sharp. Multiple files are supported.</p></div>
                <label class="gb-dropzone" id="gbDropzone" tabindex="0" role="button">
                    <input type="file" id="gbFiles" accept=".jpg,.jpeg,.png,.webp" multiple hidden>
                    <span class="upload-icon"><svg width="25" height="25" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 16V4m0 0L7 9m5-5 5 5"/><path d="M4 15v4h16v-4"/></svg></span>
                    <strong>Drop timber photos here</strong><small>JPEG, PNG or WEBP · maximum 20MB each</small>
                    <span class="btn btn-secondary">Choose photos</span>
                </label>
                <div class="gb-file-list" id="gbFileList"></div>
                <div class="action-bar"><button class="btn btn-primary" type="button" id="gbDetectBtn" disabled>Detect timber faces</button></div>
                <div class="embedded-progress hidden" id="gbWorkflowProgress"><div><span class="embedded-progress-label">Preparing detection</span><strong class="embedded-progress-percent">10%</strong></div><div class="compact-progress-track"><span></span></div></div>
                <div class="mat-status" id="gbWorkflowStatus">Ready</div>
            </section>
            <section class="hidden" id="gbReviewSection">
                <div class="section-title"><h2>2. Review and measure detected logs</h2><p>Select the logs to process and enter the whole log's length and mid-girth.</p></div>
                <div class="gb-crop-grid" id="gbCropGrid"></div>
                <section class="card gb-market-card">
                    <div class="card-header"><h2>3. Shared auction context</h2><p>These settings apply to every selected log. Species can be chosen separately on each log card.</p></div>
                    <div class="mat-form-grid">
                        <label><span>Region</span><select id="gbRegion">${optionHtml('region', defaults.region)}</select></label>
                        <label><span>Harvest season</span><select id="gbSeason">${optionHtml('season', defaults.season)}</select></label>
                        <label><span>Moisture content (%)</span><input id="gbMoisture" type="number" min="0" max="100" step="0.1" value="${defaults.moisture_content}"></label>
                        <label><span>Species base price (Rs./m³)</span><input id="gbBasePrice" type="number" min="1" step="0.01" value="${defaults.avg_market_price_species}"></label>
                    </div>
                    <details class="valuation-advanced"><summary><span><strong>Advanced market assumptions</strong><small>Defaults are suitable for a first estimate</small></span><span>Review settings</span></summary>
                        <div class="valuation-advanced-body"><div class="mat-form-grid">
                            <label><span>Density (kg/m³)</span><input id="gbDensity" type="number" min="50" value="${defaults.density_kg_m3}"></label>
                            <label><span>Price volatility (0–1)</span><input id="gbVolatility" type="number" min="0" max="1" step="0.01" value="${defaults.price_volatility}"></label>
                            <label><span>Demand index</span><input id="gbDemand" type="number" min="0.1" step="0.01" value="${defaults.market_demand_index}"></label>
                            <label><span>Supply index</span><input id="gbSupply" type="number" min="0.1" step="0.01" value="${defaults.supply_index}"></label>
                            <label><span>Export index</span><input id="gbExport" type="number" min="0.1" step="0.01" value="${defaults.export_demand_index}"></label>
                            <label><span>Auction type</span><select id="gbAuctionType">${optionHtml('auction_type', defaults.auction_type)}</select></label>
                            <label><span>Competition</span><select id="gbCompetition">${optionHtml('competition_level', defaults.competition_level)}</select></label>
                            <label><span>Expected bidders</span><input id="gbBidders" type="number" min="1" step="1" value="${defaults.num_expected_bidders}"></label>
                        </div></div>
                    </details>
                    <div class="action-bar"><span class="selection-count" id="gbSelectionCount">0 logs ready</span><button class="btn btn-primary" type="button" id="gbRunBtn">Grade and estimate selected logs</button></div>
                </section>
            </section>
        `;

        const input = document.getElementById('gbFiles');
        const dropzone = document.getElementById('gbDropzone');
        const fileList = document.getElementById('gbFileList');
        const detectBtn = document.getElementById('gbDetectBtn');
        const status = document.getElementById('gbWorkflowStatus');
        const progress = document.getElementById('gbWorkflowProgress');

        function setProgress(percent, label, error = false) {
            progress.classList.remove('hidden');
            progress.classList.toggle('running', percent > 0 && percent < 100);
            progress.classList.toggle('error', error);
            progress.querySelector('.embedded-progress-label').textContent = label;
            progress.querySelector('.embedded-progress-percent').textContent = error ? 'Stopped' : `${percent}%`;
            progress.querySelector('.compact-progress-track span').style.width = `${percent}%`;
        }

        function renderFiles() {
            fileList.innerHTML = files.map((file, idx) => `<div class="gb-file-row"><span><strong>${escapeHtml(file.name)}</strong><small>${(file.size / 1048576).toFixed(2)} MB</small></span><button type="button" data-index="${idx}" aria-label="Remove ${escapeHtml(file.name)}">&times;</button></div>`).join('');
            detectBtn.disabled = files.length === 0;
            fileList.querySelectorAll('button').forEach((btn) => btn.addEventListener('click', () => { files.splice(Number(btn.dataset.index), 1); renderFiles(); }));
        }

        function addFiles(fileCollection) {
            const incoming = Array.from(fileCollection || []);
            const rejected = incoming.filter((file) => !['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || file.size > 20 * 1024 * 1024);
            files.push(...incoming.filter((file) => !rejected.includes(file)));
            renderFiles();
            status.textContent = rejected.length ? `${rejected.length} unsupported or oversized file(s) were skipped.` : `${files.length} photo(s) selected.`;
        }
        input.addEventListener('change', () => { addFiles(input.files); input.value = ''; });
        dropzone.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                input.click();
            }
        });
        dropzone.addEventListener('dragover', (event) => { event.preventDefault(); dropzone.classList.add('drag-over'); });
        dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
        dropzone.addEventListener('drop', (event) => { event.preventDefault(); dropzone.classList.remove('drag-over'); addFiles(event.dataTransfer.files); });

        function volume(length, girth) { return length > 0 && girth > 0 ? length * girth * girth / (4 * Math.PI) : 0; }
        function updateReadyCount() {
            const ready = detectedCrops.filter((crop) => {
                const card = document.querySelector(`[data-gb-crop="${crop.crop_id}"]`);
                return card?.querySelector('.gb-select').checked && Number(card.querySelector('.gb-length').value) > 0 && Number(card.querySelector('.gb-girth').value) > 0;
            }).length;
            document.getElementById('gbSelectionCount').textContent = `${ready} ${ready === 1 ? 'log' : 'logs'} ready`;
        }

        function renderCrops(data) {
            detectedCrops = data.images.flatMap((image) => (image.crops || []).map((crop) => ({ ...crop, source_filename: image.filename })));
            const grid = document.getElementById('gbCropGrid');
            grid.innerHTML = detectedCrops.map((crop, idx) => `
                <article class="gb-crop-card selected" data-gb-crop="${escapeHtml(crop.crop_id)}">
                    <div class="gb-crop-image"><img src="${crop.preview}" alt="${escapeHtml(crop.label)}"><label><input class="gb-select" type="checkbox" checked> Include log</label></div>
                    <div class="gb-crop-content"><div class="gb-crop-heading"><span><small>Detected log ${idx + 1}</small><strong>${escapeHtml(crop.label)} · ${escapeHtml(crop.source_filename)}</strong></span><span class="tree-badge badge-success">${((crop.confidence || 0) * 100).toFixed(1)}% confidence</span></div>
                    <div class="mat-form-grid"><label><span>Species</span><select class="gb-species">${optionHtml('species', defaults.species)}</select></label><label><span>Whole log length (m)</span><input class="gb-length" type="number" min="0.01" step="0.01"></label><label><span>Mid-girth (m)</span><input class="gb-girth" type="number" min="0.01" step="0.01"></label><label><span>Calculated volume (m³)</span><input class="gb-volume" readonly value="—"></label></div>
                    <div class="crop-card-disclosures"><details class="crop-detection-details"><summary>Detection details</summary><div><span>Method <b>${escapeHtml(crop.method || 'N/A')}</b></span><span>Visible face <b>${((crop.visible_ratio || 0) * 100).toFixed(1)}%</b></span><span>Sharpness <b>${Number(crop.sharpness_score || 0).toFixed(1)}</b></span></div></details><details class="crop-detection-details per-log-overrides"><summary>Per-log market overrides</summary><div class="mat-form-grid"><label><span>Moisture override (%)</span><input class="gb-moisture-override" type="number" min="0" max="100" step="0.1" placeholder="Use shared"></label><label><span>Base price override</span><input class="gb-price-override" type="number" min="1" step="0.01" placeholder="Use shared"></label></div></details></div></div>
                </article>`).join('');
            grid.querySelectorAll('.gb-crop-card').forEach((card) => {
                const length = card.querySelector('.gb-length'); const girth = card.querySelector('.gb-girth'); const output = card.querySelector('.gb-volume');
                const refresh = () => { const v = volume(Number(length.value), Number(girth.value)); output.value = v ? v.toFixed(4) : '—'; updateReadyCount(); };
                length.addEventListener('input', refresh); girth.addEventListener('input', refresh);
                card.querySelector('.gb-select').addEventListener('change', (event) => { card.classList.toggle('selected', event.target.checked); updateReadyCount(); });
            });
            document.getElementById('gbReviewSection').classList.remove('hidden');
            document.getElementById('gbStageNav').querySelectorAll('span')[1].classList.add('active');
            updateReadyCount();
        }

        detectBtn.addEventListener('click', async () => {
            detectBtn.disabled = true; status.classList.remove('error'); setProgress(15, 'Detecting timber faces'); status.textContent = 'Analyzing uploaded photos...';
            const form = new FormData(); files.forEach((file) => form.append('files', file, file.name));
            try {
                const response = await fetch('/api/timber/detect', { method: 'POST', body: form }); const data = await response.json();
                if (!response.ok) throw new Error(data.detail || 'Detection failed.');
                sessionId = data.session_id; renderCrops(data); setProgress(100, 'Detection complete'); status.textContent = `${detectedCrops.length} timber face(s) ready for review.`;
            } catch (error) { setProgress(0, 'Detection stopped', true); status.textContent = error.message; status.classList.add('error'); }
            finally { detectBtn.disabled = false; }
        });

        document.getElementById('gbRunBtn').addEventListener('click', async (event) => {
            const selected = detectedCrops.map((crop) => ({ crop, card: document.querySelector(`[data-gb-crop="${crop.crop_id}"]`) })).filter(({ card }) => card.querySelector('.gb-select').checked);
            if (!selected.length || selected.some(({ card }) => !(Number(card.querySelector('.gb-length').value) > 0 && Number(card.querySelector('.gb-girth').value) > 0))) { status.textContent = 'Select at least one log and enter valid length and mid-girth measurements for every selected log.'; status.classList.add('error'); return; }
            const invalidField = Array.from(document.querySelectorAll('#gbReviewSection input, #gbReviewSection select')).find((field) => !field.checkValidity());
            if (invalidField) { invalidField.reportValidity(); invalidField.focus(); status.textContent = 'Review the highlighted auction field before continuing.'; status.classList.add('error'); return; }
            status.classList.remove('error'); event.currentTarget.disabled = true; setProgress(35, 'Grading selected logs');
            const measurements = selected.map(({ crop, card }) => ({ crop_id: crop.crop_id, Length_M: Number(card.querySelector('.gb-length').value), Mid_Girth_M: Number(card.querySelector('.gb-girth').value) }));
            try {
                const gradeResponse = await fetch('/api/timber/grade', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: sessionId, measurements }) });
                const gradeData = await gradeResponse.json(); if (!gradeResponse.ok) throw new Error(gradeData.detail || 'Grading failed.');
                document.getElementById('gbStageNav').querySelectorAll('span')[2].classList.add('active'); setProgress(70, 'Calculating auction values');
                const successful = (gradeData.results || []).filter((item) => item.prediction?.status === 'ok');
                const valuationAttempts = await Promise.allSettled(successful.map(async (graded) => {
                    const match = selected.find(({ crop }) => crop.crop_id === graded.crop_id); const card = match.card; const m = graded.measurements; const d = defaults;
                    const moistureField = card.querySelector('.gb-moisture-override'); const priceField = card.querySelector('.gb-price-override');
                    const payload = { species: card.querySelector('.gb-species').value, region: document.getElementById('gbRegion').value, season: document.getElementById('gbSeason').value, auction_type: document.getElementById('gbAuctionType').value, competition_level: document.getElementById('gbCompetition').value, diameter_cm: (m.Mid_Girth_M / Math.PI) * 100, length_m: m.Length_M, volume_m3: m.Volume, density_kg_m3: Number(document.getElementById('gbDensity').value), moisture_content: moistureField.value !== '' ? Number(moistureField.value) : Number(document.getElementById('gbMoisture').value), quality_grade: GRADE_TO_QUALITY[graded.prediction.grade] || d.quality_grade, straightness_score: d.straightness_score, taper_score: d.taper_score, visible_defects_score: d.visible_defects_score, internal_defect_risk: d.internal_defect_risk, avg_market_price_species: priceField.value !== '' ? Number(priceField.value) : Number(document.getElementById('gbBasePrice').value), price_volatility: Number(document.getElementById('gbVolatility').value), market_demand_index: Number(document.getElementById('gbDemand').value), supply_index: Number(document.getElementById('gbSupply').value), export_demand_index: Number(document.getElementById('gbExport').value), num_expected_bidders: Number(document.getElementById('gbBidders').value) };
                    const response = await fetch('/api/auction/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }); const value = await response.json();
                    if (!response.ok) throw new Error(value.detail || 'Valuation failed.'); return { graded, payload, value, preview: match.crop.preview };
                }));
                const valuations = valuationAttempts.filter((attempt) => attempt.status === 'fulfilled').map((attempt) => attempt.value);
                const failed = (gradeData.results || []).length - successful.length + valuationAttempts.filter((attempt) => attempt.status === 'rejected').length;
                renderIntegratedResults(valuations, failed); setProgress(100, 'Integrated analysis complete'); document.getElementById('gbStageNav').querySelectorAll('span').forEach((step) => step.classList.add('active')); status.textContent = `${valuations.length} log valuation(s) completed${failed ? `; ${failed} log(s) need attention.` : '.'}`;
            } catch (error) { setProgress(0, 'Analysis stopped', true); status.textContent = error.message; status.classList.add('error'); }
            finally { event.currentTarget.disabled = false; }
        });
    }

    function renderIntegratedResults(items, failedCount) {
        const container = document.getElementById('gbResultsContainer');
        container.innerHTML = `<section class="integrated-results"><div class="section-title"><span class="workspace-kicker">Batch results</span><h2>Grade and auction summary</h2><p>${items.length} valued${failedCount ? ` · ${failedCount} could not be processed` : ''}</p></div><div class="integrated-result-grid">${items.map(({ graded, payload, value, preview }) => `
            <article class="integrated-result-card"><img src="${preview}" alt="${escapeHtml(graded.label)}"><div class="integrated-result-body"><div class="gb-crop-heading"><span><small>${escapeHtml(graded.source_filename)}</small><strong>${escapeHtml(graded.label)}</strong></span><span class="grade-seal">${escapeHtml(graded.prediction.grade)}</span></div><div class="integrated-price"><small>Estimated auction value</small><strong>${formatRs(value.predicted_value)}</strong><span>${escapeHtml(graded.prediction.bucket || '')} · ${(graded.prediction.confidence * 100).toFixed(1)}% grading confidence</span></div><div class="integrated-metrics"><span>Volume <b>${Number(graded.measurements.Volume).toFixed(3)} m³</b></span><span>Starting bid <b>${formatRs(value.starting_bid)}</b></span><span>Expected final <b>${formatRs(value.expected_final_price)}</b></span><span>Sale probability <b>${((value.sale_probability || 0) * 100).toFixed(1)}%</b></span></div>${value.prediction_id ? `<a class="download" href="/api/auction/report/${value.prediction_id}" target="_blank">Download valuation report</a>` : ''}</div></article>`).join('')}</div></section>`;
        container.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // --- Init ---

    async function init() {
        const auFormContainer = document.getElementById('auFormContainer');
        const gbFormContainer = document.getElementById('gbFormContainer');
        if (auFormContainer) auFormContainer.innerHTML = buildFormHtml('au');

        let config = { categorical_options: {}, field_defaults: {} };
        try {
            const resp = await fetch('/api/auction/config');
            config = await resp.json();
        } catch (err) {
            console.error('Failed to load auction config', err);
        }

        if (auFormContainer) populateForm('au', config);
        initIntegratedWorkflow(config);

        const auPredictBtn = document.getElementById('auPredictBtn');
        if (auPredictBtn) auPredictBtn.addEventListener('click', () => submitPrediction('au'));
    }

    init();
});
