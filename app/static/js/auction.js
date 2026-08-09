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
        const sectionsHtml = FIELD_SECTIONS.map((section) => {
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
                <div class="section-title">${escapeHtml(section.title)}</div>
                <div class="mat-form-grid">${fieldsHtml}</div>
            `;
        }).join('');

        return `
            <section class="card">
                <div class="card-header">
                    <h2>Log Specification Input</h2>
                    <p>21 characteristics feed the XGBoost valuation model. Every field is editable.</p>
                </div>
                ${sectionsHtml}
                <div class="action-bar">
                    <button type="button" class="btn btn-primary" id="${prefix}PredictBtn">Compute Expected Bid Value</button>
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
        const inputs = collectFormValues(prefix);

        if (Object.values(inputs).some((v) => typeof v === 'number' && Number.isNaN(v))) {
            setStatus(statusEl, 'Please fill in every field with a valid number before computing.', true);
            return;
        }

        btn.disabled = true;
        setStatus(statusEl, 'Running XGBoost valuation pipeline...');

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
            setStatus(statusEl, 'Valuation complete.');
        } catch (err) {
            setStatus(statusEl, err.message || 'Valuation failed.', true);
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
        container.innerHTML = crops.map((crop) => `
            <div class="crop-picker-item" data-crop-id="${escapeHtml(crop.crop_id)}">
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

    // --- Init ---

    async function init() {
        const auFormContainer = document.getElementById('auFormContainer');
        const gbFormContainer = document.getElementById('gbFormContainer');
        if (auFormContainer) auFormContainer.innerHTML = buildFormHtml('au');
        if (gbFormContainer) gbFormContainer.innerHTML = buildFormHtml('gb');

        let config = { categorical_options: {}, field_defaults: {} };
        try {
            const resp = await fetch('/api/auction/config');
            config = await resp.json();
        } catch (err) {
            console.error('Failed to load auction config', err);
        }

        if (auFormContainer) populateForm('au', config);
        if (gbFormContainer) populateForm('gb', config);

        const auPredictBtn = document.getElementById('auPredictBtn');
        const gbPredictBtn = document.getElementById('gbPredictBtn');
        if (auPredictBtn) auPredictBtn.addEventListener('click', () => submitPrediction('au'));
        if (gbPredictBtn) gbPredictBtn.addEventListener('click', () => submitPrediction('gb'));

        renderCropPicker();
    }

    init();
});
