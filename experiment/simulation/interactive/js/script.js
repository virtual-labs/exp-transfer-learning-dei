/* ============================================
   Transfer Learning Virtual Lab – Main Script
   ============================================ */

// =========== GLOBAL DATA & CONFIG ===========

const FLOWERS = ['rose', 'sunflower', 'lotus', 'marigold', 'bougainvillea'];
const FLOWER_IMAGES = {
    rose: 'images/rose.jpg',
    sunflower: 'images/sunflower.jpg',
    lotus: 'images/lotus.jpg',
    marigold: 'images/marigold.jpg',
    bougainvillea: 'images/bougainvillea.jpg'
};

const FLOWER_COLORS = {
    rose: '#e74c3c',
    sunflower: '#f39c12',
    lotus: '#e84393',
    marigold: '#fdcb6e',
    bougainvillea: '#6c5ce7'
};

const LR_VALUES = [1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3];
const LR_LABELS = ['1e-6', '1e-5', '3e-5', '1e-4', '3e-4', '1e-3'];

const MODEL_DATA = {
    vgg19: {
        name: 'VGG19',
        // From notebook: 4-vgg19-0-base-freezing.ipynb
        // Keras VGG19(include_top=False) base model: 22 layer objects
        //   (16 Conv2D + 5 MaxPool + 1 Input = 22 Keras layers)
        // Final model: 27 layers (base + GAP + dropout + dense + dropout + dense)
        // 0% unfreeze config: all 22 base layers frozen
        // Note: "VGG19" is named for its 19 learnable layers (16 conv + 3 FC);
        //       here we use include_top=False so only 16 conv layers are present.
        totalLayers: 22,          // Keras base model layer count (incl. pooling + input)
        finalLayers: 27,          // total layers in final model
        // These are the FINE-TUNED model totals (base + custom head), not standalone VGG19 (~143.7M)
        totalParams: 20181926,    // Total params: 20,181,926 (base 20.0M + head 0.2M)
        trainableParams: 157542,  // Trainable params: 157,542 (only classifier head at 0% unfreeze)
        nonTrainableParams: 20024384, // Non-trainable params: 20,024,384 (= base model)
        baseLR: 1e-4,             // learning_rate = 1e-4
        layerNames: [
            'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
            'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
            'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
            'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
            'GAP', 'drop1', 'dense1', 'drop2', 'bn', 'dense_out'
        ],
        layerTypes: [
            'Conv', 'Conv', 'Conv', 'Conv',
            'Conv', 'Conv', 'Conv', 'Conv',
            'Conv', 'Conv', 'Conv', 'Conv',
            'Conv', 'Conv', 'Conv', 'Conv',
            'GAP', 'Drop', 'FC', 'Drop', 'BN', 'FC'
        ],
        // Experiment results (0% unfreeze)
        bestFreeze: 100,          // notebook used 0% unfreeze = 100% freeze
        bestLR: 3,                // index into LR_VALUES → 1e-4
        trainAccFinal: 99.88,     // Train Accuracy: 99.88%
        valAccFinal: 88.51,       // Validation Accuracy: 88.51%
        testAccFinal: 88.28,      // Test Accuracy: 88.28%
        epochs: 150,
        batchSize: 32,
        patience: 15,
        convergenceSpeed: 0.65,   // VGG converges slower
        stabilityBase: 0.55       // less stable with high LR
    },
    mobilenet: {
        name: 'MobileNetV2',
        // From notebook: 2-mobilenetv2-10-unfreeze.ipynb
        // Base model: 154 layers, Final model: 159 layers
        // 10% unfreeze: 15 layers unfrozen, 139 frozen
        // Note: 10% of layers ≈ 59% of params because deeper layers have more parameters
        totalLayers: 17,          // grouped stage count for architecture display
        realLayers: 154,          // actual base model layers
        finalLayers: 159,         // total layers in final model
        totalParams: 2966182,     // Total params: 2,966,182 (base 2.26M + head 0.71M)
        trainableParams: 1748198, // Trainable params: 1,748,198 (at 10% layer unfreeze)
        nonTrainableParams: 1217984, // Non-trainable params: 1,217,984
        // At 0% unfreeze (head-only): trainableParams = 708,198 (see notebook-1)
        baseLR: 1e-5,             // learning_rate = 1e-5
        layerNames: [
            'conv_stem', 'dw_blk1', 'dw_blk2', 'dw_blk3', 'dw_blk4',
            'dw_blk5', 'dw_blk6', 'dw_blk7', 'dw_blk8', 'dw_blk9',
            'dw_blk10', 'dw_blk11', 'conv_head', 'GAP', 'drop1', 'dense1', 'dense_out'
        ],
        layerTypes: [
            'Conv', 'DW', 'DW', 'DW', 'DW',
            'DW', 'DW', 'DW', 'DW', 'DW',
            'DW', 'DW', 'Conv', 'GAP', 'Drop', 'FC', 'FC'
        ],
        // Experiment results (10% unfreeze)
        bestFreeze: 90,           // 10% unfreeze = 90% freeze
        bestLR: 1,                // index into LR_VALUES → 1e-5
        trainAccFinal: 100.00,    // Train Accuracy: 100.00%
        valAccFinal: 94.99,       // Validation Accuracy: 94.99%
        testAccFinal: 94.87,      // Test Accuracy: 94.87%
        epochs: 150,
        batchSize: 32,
        patience: 15,
        convergenceSpeed: 0.85,   // MobileNet converges faster
        stabilityBase: 0.8        // more stable architecture
    }
};

// =========== TAB NAVIGATION ===========

const TAB_ORDER = ['sim1', 'sim2', 'sim3', 'sim4', 'sim5', 'sim6'];

function switchTab(tabId) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
    const targetTab = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
    if (targetTab) targetTab.classList.add('active');

    // Update tab content panels
    document.querySelectorAll('.tab-content').forEach(p => p.classList.remove('active'));
    const targetPanel = document.getElementById(tabId);
    if (targetPanel) targetPanel.classList.add('active');

    // Initialize simulation when first opened
    if (tabId === 'sim1') initSim1();
    if (tabId === 'sim3') initSim3Charts();
    if (tabId === 'sim4') initSim4();
    if (tabId === 'sim5') initSim5Charts();
    if (tabId === 'sim6') updateSim6();

    // Re-trigger canvas sizing after tab becomes visible
    requestAnimationFrame(() => {
        window.dispatchEvent(new Event('resize'));
    });
}

// Tab button clicks
document.querySelectorAll('.tab-btn').forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

// Initialize Sim 1 on page load (it's the default active tab)
window.addEventListener('DOMContentLoaded', () => initSim1());

// ================================================
// SIM 1: MODEL ARCHITECTURE ANIMATION
// ================================================

let sim1Initialized = false;

function initSim1() {
    buildArchPipeline();
    updateSim1Info();
    sim1Initialized = true;
}

function buildArchPipeline() {
    const model = document.getElementById('sim1-model').value;
    const freezePct = parseInt(document.getElementById('sim1-freeze').value);
    const md = MODEL_DATA[model];
    const pipeline = document.getElementById('sim1-pipeline');
    pipeline.innerHTML = '';

    // Input block
    const flower = document.getElementById('sim1-flower').value;
    const inputBlock = document.createElement('div');
    inputBlock.className = 'input-block';
    inputBlock.innerHTML = `<img src="${FLOWER_IMAGES[flower]}" style="width:33px;height:33px;border-radius:4px;object-fit:cover;margin-bottom:2px"><span>Input</span>`;
    pipeline.appendChild(inputBlock);

    const arrow1 = document.createElement('span');
    arrow1.className = 'pipeline-arrow';
    arrow1.textContent = '→';
    pipeline.appendChild(arrow1);

    const numLayers = md.totalLayers;
    const frozenCount = Math.round(numLayers * (freezePct / 100));

    for (let i = 0; i < numLayers; i++) {
        const block = document.createElement('div');
        const isFrozen = i < frozenCount;
        block.className = `layer-block ${isFrozen ? 'frozen' : 'trainable'}`;
        block.dataset.index = i;
        block.dataset.layerName = md.layerNames[i] || `Layer ${i + 1}`;
        block.dataset.layerType = md.layerTypes[i] || 'Layer';
        block.dataset.status = isFrozen ? 'Frozen' : 'Trainable';

        const typeBadge = md.layerTypes[i] || 'L';
        const name = md.layerNames[i] || `L${i + 1}`;
        block.innerHTML = `
            <span class="layer-name">${typeBadge}</span>
            <span style="font-size:0.45rem;opacity:0.7">${name}</span>
            ${isFrozen ? '<span class="lock-icon">🔒</span>' : '<span class="lock-icon">🔓</span>'}
        `;

        // Tooltip on hover
        block.addEventListener('mouseenter', showLayerTooltip);
        block.addEventListener('mouseleave', hideLayerTooltip);

        pipeline.appendChild(block);

        if (i < numLayers - 1) {
            const a = document.createElement('span');
            a.className = 'pipeline-arrow';
            a.textContent = '→';
            pipeline.appendChild(a);
        }
    }

    // Output block
    const arrowOut = document.createElement('span');
    arrowOut.className = 'pipeline-arrow';
    arrowOut.textContent = '→';
    pipeline.appendChild(arrowOut);

    const outputBlock = document.createElement('div');
    outputBlock.className = 'output-block';
    outputBlock.innerHTML = `<span style="font-size:0.7rem">Softmax</span><span style="font-size:0.5rem">102 classes</span>`;
    pipeline.appendChild(outputBlock);
}

function showLayerTooltip(e) {
    const block = e.currentTarget;
    const tooltip = document.getElementById('sim1-tooltip');
    const titleEl = document.getElementById('sim1-tooltip-title');
    const bodyEl = document.getElementById('sim1-tooltip-body');
    const status = block.dataset.status;
    const statusColor = status === 'Frozen' ? '#636e72' : '#6c5ce7';

    titleEl.textContent = block.dataset.layerName;
    bodyEl.innerHTML = `
        <strong>Type:</strong> ${block.dataset.layerType}<br>
        <strong>Index:</strong> ${parseInt(block.dataset.index) + 1}<br>
        <strong>Status:</strong> <span style="color:${statusColor};font-weight:600">${status}</span>
    `;
    tooltip.style.display = 'block';
}

function hideLayerTooltip() {
    document.getElementById('sim1-tooltip').style.display = 'none';
}

function updateSim1Info() {
    const model = document.getElementById('sim1-model').value;
    const freezePct = parseInt(document.getElementById('sim1-freeze').value);
    const md = MODEL_DATA[model];
    const displayLayers = model === 'mobilenet' ? md.realLayers : md.totalLayers;
    const frozenCount = Math.round(displayLayers * (freezePct / 100));
    const trainableCount = displayLayers - frozenCount;

    // Use exact notebook params, scale trainable based on freeze slider
    const totalParams = md.totalParams;
    // Classifier head params (always trainable, from 0%-unfreeze notebooks)
    // MobileNetV2: 708,198 (notebook-1), VGG19: 157,542 (notebook-4)
    const baseTrainable = model === 'mobilenet' ? 708198 : 157542;
    const baseBodyParams = totalParams - baseTrainable;
    const unfrozenBodyParams = Math.round(baseBodyParams * (1 - freezePct / 100));
    const trainableParams = baseTrainable + unfrozenBodyParams;
    const nonTrainableParams = totalParams - trainableParams;

    // Expected accuracy: Higher for unfreezing (fine-tuning), lower for heavy freezing
    const testAcc = md.testAccFinal;

    // Penalty logic: 0% freeze gets ~max accuracy, 100% freeze gets a significant penalty
    // We use a non-linear penalty that increases as freezing depth goes up
    const rigidityPenalty = Math.pow(freezePct / 100, 1.5) * 12; // up to 12% loss for full freeze
    const expectedAcc = Math.max(50, testAcc - rigidityPenalty).toFixed(1);

    document.getElementById('sim1-total-layers').textContent = displayLayers;
    document.getElementById('sim1-frozen-count').textContent = frozenCount;
    document.getElementById('sim1-trainable-count').textContent = trainableCount;
    document.getElementById('sim1-total-params').textContent = (totalParams / 1e6).toFixed(1) + 'M';
    document.getElementById('sim1-trainable-params').textContent = (trainableParams / 1e6).toFixed(1) + 'M';
    document.getElementById('sim1-nontrain-params').textContent = (nonTrainableParams / 1e6).toFixed(1) + 'M';
    document.getElementById('sim1-expected-acc').textContent = expectedAcc + '%';
    document.getElementById('sim1-prediction').textContent = '—';
}

// Forward pass animation with progress bar
document.getElementById('sim1-start').addEventListener('click', () => {
    const blocks = document.querySelectorAll('#sim1-pipeline .layer-block');
    const flower = document.getElementById('sim1-flower').value;
    const model = document.getElementById('sim1-model').value;
    const md = MODEL_DATA[model];
    const freezePct = parseInt(document.getElementById('sim1-freeze').value);
    const total = blocks.length;
    let i = 0;

    // Show progress bar
    const progressWrap = document.getElementById('sim1-progress-wrap');
    const progressBar = document.getElementById('sim1-progress-bar');
    const progressLabel = document.getElementById('sim1-progress-label');
    progressWrap.style.display = 'block';
    progressBar.style.width = '0%';
    progressLabel.textContent = '0%';

    function animateBlock() {
        if (i > 0) blocks[i - 1].classList.remove('active-signal');
        if (i < total) {
            blocks[i].classList.add('active-signal');
            const pct = Math.round(((i + 1) / total) * 100);
            progressBar.style.width = pct + '%';
            progressLabel.textContent = pct + '%';
            i++;
            setTimeout(animateBlock, 180);
        } else {
            // Show prediction with confidence
            const flowerName = flower.charAt(0).toUpperCase() + flower.slice(1);
            const testAcc = md.testAccFinal;
            const rigidityPenalty = Math.pow(freezePct / 100, 1.5) * 12;
            const confidence = Math.max(50, testAcc - rigidityPenalty).toFixed(1);

            document.getElementById('sim1-prediction').textContent = `${flowerName} (${confidence}%)`;

            const outputBlock = document.querySelector('#sim1-pipeline .output-block');
            outputBlock.innerHTML = `<span style="font-size:0.7rem;font-weight:700">${flowerName}</span><span style="font-size:0.5rem">${confidence}%</span>`;

            progressBar.style.width = '100%';
            progressLabel.textContent = '100% ✓';
        }
    }
    // Reset
    blocks.forEach(b => b.classList.remove('active-signal'));
    document.getElementById('sim1-prediction').textContent = '—';
    animateBlock();
});

// Live slider update
document.getElementById('sim1-freeze').addEventListener('input', (e) => {
    document.getElementById('sim1-freeze-val').textContent = e.target.value + '%';
    buildArchPipeline();
    updateSim1Info();
});

document.getElementById('sim1-model').addEventListener('change', () => {
    buildArchPipeline();
    updateSim1Info();
});

document.getElementById('sim1-flower').addEventListener('change', () => {
    buildArchPipeline();
});

// Reset
document.getElementById('sim1-reset').addEventListener('click', () => {
    document.getElementById('sim1-freeze').value = 70;
    document.getElementById('sim1-freeze-val').textContent = '70%';
    document.getElementById('sim1-model').value = 'vgg19';
    document.getElementById('sim1-flower').value = 'rose';
    document.getElementById('sim1-progress-wrap').style.display = 'none';
    document.getElementById('sim1-tooltip').style.display = 'none';
    buildArchPipeline();
    updateSim1Info();
});


// ================================================
// SIM 2: FEATURE EXTRACTION ANIMATION
// ================================================

function getFilterForMode(mode, stage) {
    const filters = {
        random: {
            edges: 'grayscale(100%) contrast(50%) blur(5px)',
            blobs: 'grayscale(100%) contrast(30%) blur(10px) opacity(0.6)',
            textures: 'grayscale(80%) blur(6px) contrast(40%) opacity(0.5)',
            parts: 'grayscale(100%) blur(12px) brightness(0.6) opacity(0.4)',
            shapes: 'blur(11px) contrast(30%) saturate(0.2) opacity(0.3)'
        },
        pretrained: {
            edges: 'grayscale(100%) contrast(200%) brightness(1.2)',
            blobs: 'grayscale(30%) blur(4px) contrast(140%) saturate(1.5)',
            textures: 'contrast(250%) saturate(0.8) hue-rotate(15deg) brightness(0.9)',
            parts: 'contrast(180%) saturate(2.5) brightness(1.2)',
            shapes: 'contrast(120%) saturate(1.3) brightness(1.05)'
        },
        finetuned: {
            edges: 'grayscale(100%) contrast(400%) invert(0.1)',
            blobs: 'contrast(250%) saturate(2.5) blur(3px) brightness(1.1)',
            textures: 'contrast(300%) saturate(0.5) hue-rotate(-10deg) brightness(0.85)',
            parts: 'contrast(220%) saturate(3) hue-rotate(5deg) brightness(1.2)',
            shapes: 'contrast(180%) saturate(2) brightness(1.15) sepia(0.1)'
        }
    };
    return filters[mode]?.[stage] || 'none';
}

const FLOWER_DIFFICULTY = {
    rose: 0.2,
    sunflower: 1.5,
    lotus: -1.2,
    marigold: -2.8,
    bougainvillea: 0.8
};

function getConfidence(mode, flower) {
    const base = { random: 12, pretrained: 65, finetuned: 95 }[mode] || 0;
    const diff = FLOWER_DIFFICULTY[flower] || 0;
    // Pseudo-random jitter based on flower name length and characters
    const jitter = Math.sin(flower.charCodeAt(0) + flower.length) * 2.2;
    let val = base + diff + jitter;

    // Realistic Clamping
    if (mode === 'finetuned') val = Math.min(99.6, Math.max(90.5, val));
    if (mode === 'pretrained') val = Math.min(76.4, Math.max(52.1, val));
    if (mode === 'random') val = Math.min(16.5, Math.max(3.2, val));

    return val.toFixed(1) + '%';
}

function getQuality(mode, flower) {
    const conf = parseFloat(getConfidence(mode, flower));
    return conf;
}

document.getElementById('sim2-start').addEventListener('click', () => {
    const mode = document.getElementById('sim2-mode').value;
    const flower = document.getElementById('sim2-flower').value;
    const imgSrc = FLOWER_IMAGES[flower];
    const stages = ['input', 'edges', 'blobs', 'textures', 'parts', 'shapes', 'predict'];

    // Set all images
    document.getElementById('sim2-img-input').src = imgSrc;
    document.getElementById('sim2-img-edges').src = imgSrc;
    document.getElementById('sim2-img-blobs').src = imgSrc;
    document.getElementById('sim2-img-textures').src = imgSrc;
    document.getElementById('sim2-img-parts').src = imgSrc;
    document.getElementById('sim2-img-shapes').src = imgSrc;

    // Apply initial filters
    document.getElementById('sim2-img-edges').style.filter = getFilterForMode(mode, 'edges');
    document.getElementById('sim2-img-blobs').style.filter = getFilterForMode(mode, 'blobs');
    document.getElementById('sim2-img-textures').style.filter = getFilterForMode(mode, 'textures');
    document.getElementById('sim2-img-parts').style.filter = getFilterForMode(mode, 'parts');
    document.getElementById('sim2-img-shapes').style.filter = getFilterForMode(mode, 'shapes');

    // Animate stages sequentially
    document.querySelectorAll('.feature-stage').forEach(s => s.classList.remove('active'));
    let i = 0;

    const stageDescriptions = {
        input: "Starting point: Raw pixel data is fed into the network.",
        edges: "Early Layers: Filters respond to sharp transitions and primitive edges.",
        blobs: "Early-Mid: Simple edges begin to form color gradients and localized blobs.",
        textures: "Middle Layers: Blobs and edges combine into complex texture patterns.",
        parts: "Mid-Deep: Textures merge into recognizable geometric parts and motifs.",
        shapes: "Deep Layers: Parts assemble into high-level object shapes and flower structures.",
        predict: "Final Layer: Softmax assigns class probabilities based on global abstract features."
    };

    function showStage() {
        if (i < stages.length) {
            const stageKey = stages[i];
            const stg = document.getElementById(`sim2-stage-${stageKey}`);
            if (stg) stg.classList.add('active');

            const img = document.getElementById(`sim2-img-${stageKey}`);
            if (img && stageKey !== 'input') {
                // Start blurred then clarify
                img.style.filter = 'blur(9px) grayscale(100%)';
                setTimeout(() => {
                    img.style.filter = getFilterForMode(mode, stageKey);
                }, 100);
            }

            // Update explanatory text
            const descEl = document.getElementById('sim2-stage-desc');
            descEl.textContent = stageDescriptions[stageKey];
            descEl.style.color = 'var(--primary)';
            descEl.style.fontWeight = '600';

            if (stageKey === 'predict') {
                const flowerName = flower.charAt(0).toUpperCase() + flower.slice(1);
                document.getElementById('sim2-prediction').textContent = mode === 'random' ? '???' : flowerName;
                document.getElementById('sim2-confidence').textContent = getConfidence(mode, flower);
                descEl.style.color = 'var(--text-light)';
                descEl.style.fontWeight = '400';
            }
            i++;
            setTimeout(showStage, 2000);
        } else {
            // Update quality bar
            const q = getQuality(mode, flower);
            document.getElementById('sim2-quality').style.width = q + '%';
            document.getElementById('sim2-quality-label').textContent =
                q < 40 ? 'Poor' : q < 75 ? 'Good' : 'Excellent';
        }
    }
    showStage();
});

document.getElementById('sim2-reset').addEventListener('click', () => {
    document.getElementById('sim2-mode').value = 'random';
    document.getElementById('sim2-flower').value = 'rose';
    document.querySelectorAll('.feature-stage').forEach(s => s.classList.remove('active'));
    document.getElementById('sim2-prediction').textContent = '—';
    document.getElementById('sim2-confidence').textContent = '—';
    document.getElementById('sim2-quality').style.width = '0%';
    document.getElementById('sim2-quality-label').textContent = '—';
    document.getElementById('sim2-stage-desc').textContent = 'Select a mode and run the extraction to see how the CNN processes the image.';
    document.getElementById('sim2-stage-desc').style.color = 'var(--text-light)';
    document.getElementById('sim2-stage-desc').style.fontWeight = '400';
    ['input', 'edges', 'blobs', 'textures', 'parts', 'shapes'].forEach(id => {
        const img = document.getElementById(`sim2-img-${id}`);
        if (img) { img.src = ''; img.style.filter = 'none'; }
    });
});


// ================================================
// SIM 3: TRAINING DYNAMICS
// ================================================

let sim3Charts = {};

function initSim3Charts() {
    if (sim3Charts.trainAcc) return; // already initialized

    const commonOpts = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 800 },
        scales: {
            x: { title: { display: true, text: 'Epoch', font: { size: 10 } }, ticks: { font: { size: 9 } } },
            y: { ticks: { font: { size: 9 } } }
        },
        plugins: { legend: { display: false } }
    };

    sim3Charts.trainAcc = new Chart(document.getElementById('sim3-chart-trainAcc'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Train Acc', data: [], borderColor: '#6c5ce7', backgroundColor: 'rgba(108,92,231,0.1)', fill: true, tension: 0.3, pointRadius: 0 }] },
        options: { ...commonOpts, scales: { ...commonOpts.scales, y: { min: 0, max: 100, ticks: { font: { size: 9 } } } } }
    });

    sim3Charts.valAcc = new Chart(document.getElementById('sim3-chart-valAcc'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Val Acc', data: [], borderColor: '#00cec9', backgroundColor: 'rgba(0,206,201,0.1)', fill: true, tension: 0.3, pointRadius: 0 }] },
        options: { ...commonOpts, scales: { ...commonOpts.scales, y: { min: 0, max: 100, ticks: { font: { size: 9 } } } } }
    });

    sim3Charts.trainLoss = new Chart(document.getElementById('sim3-chart-trainLoss'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Train Loss', data: [], borderColor: '#e17055', backgroundColor: 'rgba(225,112,85,0.1)', fill: true, tension: 0.3, pointRadius: 0 }] },
        options: { ...commonOpts, scales: { ...commonOpts.scales, y: { min: 0, ticks: { font: { size: 9 } } } } }
    });

    sim3Charts.valLoss = new Chart(document.getElementById('sim3-chart-valLoss'), {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Val Loss', data: [], borderColor: '#fdcb6e', backgroundColor: 'rgba(253,203,110,0.1)', fill: true, tension: 0.3, pointRadius: 0 }] },
        options: { ...commonOpts, scales: { ...commonOpts.scales, y: { min: 0, ticks: { font: { size: 9 } } } } }
    });
}

function generateTrainingCurves(model, freezePct, lrIdx, dataPct) {
    const md = MODEL_DATA[model];
    const epochs = 30;
    const labels = Array.from({ length: epochs }, (_, i) => i + 1);
    const lr = LR_VALUES[lrIdx];

    // Base final values from experiment data
    let baseTrainAcc = md.trainAccFinal;
    let baseValAcc = md.valAccFinal;

    // Freeze effects: More unfreezing (linear bonus) vs Heavy freezing (Rigidity Penalty)
    // We drop accuracy linearly as freezing increases
    baseTrainAcc -= (freezePct / 100) * 10;
    baseValAcc -= (freezePct / 100) * 15;

    // Small dataset penalty for unfreezing (Overfitting check)
    if (freezePct < 30 && dataPct < 40) {
        baseValAcc -= (30 - freezePct) * 0.4; // Unfreezing with very little data hurts val acc
    }

    // LR effects
    const optimalLRIdx = md.bestLR;
    const lrRatio = lr / LR_VALUES[optimalLRIdx];

    let instability = 0;
    if (lrRatio > 10) {
        instability = 0.4;
        baseValAcc -= 15;
        if (model === 'vgg19') {
            instability = 0.6;
            baseValAcc -= 25;
        }
    } else if (lrRatio > 3) {
        instability = 0.15;
        baseValAcc -= 5;
        if (model === 'vgg19') instability = 0.25;
    } else if (lrRatio < 0.1) {
        // Too small LR -> slow convergence
        baseTrainAcc -= 10;
        baseValAcc -= 8;
    }

    // Dataset size effect
    const dataFactor = dataPct / 100;
    if (dataFactor < 0.5) {
        baseValAcc -= (1 - dataFactor) * 20;
    } else if (dataFactor < 0.8) {
        baseValAcc -= (1 - dataFactor) * 8;
    }

    baseTrainAcc = Math.min(100, Math.max(30, baseTrainAcc));
    baseValAcc = Math.min(baseTrainAcc, Math.max(25, baseValAcc));

    // Convergence speed
    const convSpeed = md.convergenceSpeed * (lrRatio > 1 ? 1.2 : lrRatio < 0.5 ? 0.6 : 1);

    const trainAcc = [], valAcc = [], trainLoss = [], valLoss = [];

    for (let e = 0; e < epochs; e++) {
        const t = (e + 1) / epochs;
        const curve = 1 - Math.exp(-3.5 * convSpeed * t);

        let tAcc = 30 + (baseTrainAcc - 30) * curve;
        let vAcc = 25 + (baseValAcc - 25) * curve * 0.9;

        // Add instability noise
        if (instability > 0) {
            tAcc += (Math.random() - 0.5) * instability * 30;
            vAcc += (Math.random() - 0.5) * instability * 40;
        }

        // Overfitting: train acc pulls away from val acc in later epochs
        if (e > epochs * 0.5 && dataFactor < 0.6) {
            tAcc = Math.min(100, tAcc + (e - epochs * 0.5) * 0.5);
            vAcc = Math.max(vAcc - (e - epochs * 0.5) * 0.3, baseValAcc * 0.85);
        }

        trainAcc.push(Math.min(100, Math.max(0, tAcc)));
        valAcc.push(Math.min(100, Math.max(0, vAcc)));

        // Loss (inverse of accuracy, roughly)
        let tLoss = 4.5 * (1 - curve) + 0.01;
        let vLoss = 4.5 * (1 - curve * 0.85) + 0.05;

        if (instability > 0) {
            tLoss += Math.random() * instability * 2;
            vLoss += Math.random() * instability * 3;
        }

        if (e > epochs * 0.5 && dataFactor < 0.6) {
            vLoss += (e - epochs * 0.5) * 0.05;
        }

        trainLoss.push(Math.max(0, tLoss));
        valLoss.push(Math.max(0, vLoss));
    }

    return { labels, trainAcc, valAcc, trainLoss, valLoss };
}

let sim3AnimFrame = null;

document.getElementById('sim3-train').addEventListener('click', () => {
    const model = document.getElementById('sim3-model').value;
    const freezePct = parseInt(document.getElementById('sim3-freeze').value);
    const lrIdx = parseInt(document.getElementById('sim3-lr').value);
    const dataPct = parseInt(document.getElementById('sim3-data').value);

    const curves = generateTrainingCurves(model, freezePct, lrIdx, dataPct);

    // Animate chart data
    if (sim3AnimFrame) cancelAnimationFrame(sim3AnimFrame);

    let step = 0;
    let bestAcc = 0;
    let minLoss = 999;
    let convEpoch = "—";

    function animate() {
        if (step > curves.labels.length) return;

        const sl = curves.labels.slice(0, step);
        sim3Charts.trainAcc.data.labels = sl;
        sim3Charts.trainAcc.data.datasets[0].data = curves.trainAcc.slice(0, step);
        sim3Charts.trainAcc.update('none');

        sim3Charts.valAcc.data.labels = sl;
        sim3Charts.valAcc.data.datasets[0].data = curves.valAcc.slice(0, step);
        sim3Charts.valAcc.update('none');

        sim3Charts.trainLoss.data.labels = sl;
        sim3Charts.trainLoss.data.datasets[0].data = curves.trainLoss.slice(0, step);
        sim3Charts.trainLoss.update('none');

        sim3Charts.valLoss.data.labels = sl;
        sim3Charts.valLoss.data.datasets[0].data = curves.valLoss.slice(0, step);
        sim3Charts.valLoss.update('none');

        // Update real-time metrics
        if (step > 0) {
            const currentValAcc = curves.valAcc[step - 1];
            const currentValLoss = curves.valLoss[step - 1];
            const currentTrainAcc = curves.trainAcc[step - 1];

            if (currentValAcc > bestAcc) bestAcc = currentValAcc;
            if (currentValLoss < minLoss) {
                minLoss = currentValLoss;
                // Simple convergence detection: if loss is low and step is > 5
                if (step > 5 && convEpoch === "—") convEpoch = step;
            }

            document.getElementById('sim3-best-acc').textContent = bestAcc.toFixed(1) + '%';
            document.getElementById('sim3-min-loss').textContent = minLoss.toFixed(3);
            document.getElementById('sim3-convergence').textContent = convEpoch === "—" ? "..." : `Epoch ${convEpoch}`;

            // Overfitting indicator
            const gap = currentTrainAcc - currentValAcc;
            const fill = document.getElementById('sim3-overfit-fill');
            const pct = Math.min(100, Math.max(0, gap * 3));
            fill.style.width = pct + '%';
            fill.style.background = pct < 30 ? '#00b894' : pct < 60 ? '#fdcb6e' : '#e17055';
            document.getElementById('sim3-overfit-label').textContent =
                pct < 30 ? 'None' : pct < 60 ? 'Moderate' : 'Severe';

            document.getElementById('sim3-overfit-risk').textContent = pct.toFixed(1) + '%';
        }

        step++;
        sim3AnimFrame = setTimeout(animate, 60);
    }
    animate();
});

// Slider labels
document.getElementById('sim3-freeze').addEventListener('input', e => {
    document.getElementById('sim3-freeze-val').textContent = e.target.value + '%';
});
document.getElementById('sim3-lr').addEventListener('input', e => {
    document.getElementById('sim3-lr-val').textContent = LR_LABELS[e.target.value];
});
document.getElementById('sim3-data').addEventListener('input', e => {
    document.getElementById('sim3-data-val').textContent = e.target.value + '%';
});

document.getElementById('sim3-reset').addEventListener('click', () => {
    if (sim3AnimFrame) clearTimeout(sim3AnimFrame);
    document.getElementById('sim3-model').value = 'vgg19';
    document.getElementById('sim3-freeze').value = 70;
    document.getElementById('sim3-freeze-val').textContent = '70%';
    document.getElementById('sim3-lr').value = 1;
    document.getElementById('sim3-lr-val').textContent = '1e-5';
    document.getElementById('sim3-data').value = 100;
    document.getElementById('sim3-data-val').textContent = '100%';
    document.getElementById('sim3-overfit-fill').style.width = '0%';
    document.getElementById('sim3-overfit-label').textContent = 'None';
    document.getElementById('sim3-best-acc').textContent = '—';
    document.getElementById('sim3-min-loss').textContent = '—';
    document.getElementById('sim3-convergence').textContent = '—';
    document.getElementById('sim3-overfit-risk').textContent = '—';

    Object.values(sim3Charts).forEach(c => {
        if (c) {
            c.data.labels = [];
            c.data.datasets[0].data = [];
            c.update();
        }
    });
});


// ================================================
// SIM 4: REPRESENTATION SPACE (D3.js)
// ================================================

let sim4Initialized = false;
let sim4AnimTimer = null;
let sim4ShowHulls = false;

function initSim4() {
    if (sim4Initialized) return;
    sim4Initialized = true;
    renderSim4Scatter();
    buildSim4Legend();
}

function buildSim4Legend() {
    const legend = document.getElementById('sim4-legend');
    legend.innerHTML = '';
    FLOWERS.forEach(f => {
        const item = document.createElement('span');
        item.className = 'legend-item';
        item.innerHTML = `<span class="legend-dot" style="background:${FLOWER_COLORS[f]}"></span>${f.charAt(0).toUpperCase() + f.slice(1)}`;

        item.addEventListener('mouseenter', () => highlightCluster(f));
        item.addEventListener('mouseleave', () => resetHighlight());

        legend.appendChild(item);
    });
}

function highlightCluster(flower) {
    d3.selectAll('.scatter-point')
        .classed('faded', d => d.flower !== flower)
        .classed('highlighted', d => d.flower === flower);

    d3.selectAll('.hull')
        .style('fill-opacity', d => d.flower === flower ? 0.25 : 0.02)
        .style('stroke-opacity', d => d.flower === flower ? 1 : 0.2);
}

function resetHighlight() {
    d3.selectAll('.scatter-point').classed('faded', false).classed('highlighted', false);
    d3.selectAll('.hull').style('fill-opacity', 0.08).style('stroke-opacity', 1);
}

function generateClusterData(mode, epoch, model) {
    const points = [];
    const md = MODEL_DATA[model];
    const convFactor = md.convergenceSpeed;

    const centers = {
        rose: [0.2, 0.8],
        sunflower: [0.8, 0.8],
        lotus: [0.5, 0.2],
        marigold: [0.8, 0.3],
        bougainvillea: [0.2, 0.3]
    };

    const randomCenters = {
        rose: [0.45, 0.55],
        sunflower: [0.55, 0.52],
        lotus: [0.48, 0.48],
        marigold: [0.52, 0.45],
        bougainvillea: [0.5, 0.5]
    };

    FLOWERS.forEach(flower => {
        const finalCenter = centers[flower];
        const randCenter = randomCenters[flower];
        const pointsPerClass = 25; // Increased for better hulls

        for (let i = 0; i < pointsPerClass; i++) {
            let spread, cx, cy;

            if (mode === 'random') {
                spread = 0.22;
                cx = randCenter[0];
                cy = randCenter[1];
            } else if (mode === 'pretrained') {
                const t = Math.min(1, (epoch / 30) * convFactor);
                cx = randCenter[0] + (finalCenter[0] - randCenter[0]) * 0.5;
                cy = randCenter[1] + (finalCenter[1] - randCenter[1]) * 0.5;
                spread = 0.16 - t * 0.05;
            } else { // finetuned
                const t = Math.min(1, (epoch / 30) * convFactor * 1.3);
                cx = randCenter[0] + (finalCenter[0] - randCenter[0]) * t;
                cy = randCenter[1] + (finalCenter[1] - randCenter[1]) * t;
                spread = 0.18 - t * 0.12;
            }

            points.push({
                x: cx + (Math.random() - 0.5) * spread * 2,
                y: cy + (Math.random() - 0.5) * spread * 2,
                flower: flower
            });
        }
    });

    return points;
}

function updateSim4Concepts(mode, epoch, metrics) {
    const titleEl = document.getElementById('sim4-concept-title');
    const descEl = document.getElementById('sim4-concept-desc');

    if (mode === 'random') {
        titleEl.textContent = 'Random Representation';
        descEl.textContent = 'In a randomly initialized network, features have no semantic meaning. All classes collapse into a single "blob" at the center, sharing nearly identical feature vectors.';
    } else if (mode === 'pretrained') {
        titleEl.textContent = 'Generic Feature Space';
        descEl.textContent = 'Deep networks pretrained on ImageNet have learned robust general features (edges, textures). Flowers begin to cluster, but without fine-tuning, the boundaries remain blurry and overlapping.';
    } else {
        if (parseFloat(metrics.separability) < 0.8) {
            titleEl.textContent = 'Manifold Convergence';
            descEl.textContent = 'The network is learning to separate flower-specific features. Data points are migrating towards their class-specific manifolds in the embedding space.';
        } else {
            titleEl.textContent = 'Linear Separability';
            descEl.textContent = 'Success! The clusters are now highly distinct. In this space, a simple linear classifier (like the final Softmax layer) can easily achieve high accuracy.';
        }
    }
}


function renderSim4Scatter() {
    const svgEl = document.getElementById('sim4-scatter');
    const rect = svgEl.getBoundingClientRect();
    const w = rect.width || 450;
    const h = rect.height || 300;

    const model = document.getElementById('sim4-model').value;
    const mode = document.getElementById('sim4-mode').value;
    const epoch = parseInt(document.getElementById('sim4-epoch').value);

    const data = generateClusterData(mode, epoch, model);

    const svg = d3.select('#sim4-scatter');
    // Don't remove all every time to allow transitions
    if (svg.select('.main-group').empty()) {
        svg.append('g').attr('class', 'hull-group');
        svg.append('g').attr('class', 'main-group');
        svg.append('g').attr('class', 'axis-group');
    }

    svg.attr('viewBox', `0 0 ${w} ${h}`);

    const xScale = d3.scaleLinear().domain([0, 1]).range([40, w - 20]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([h - 30, 20]);

    // Metrics calculation
    let totalIntraSpread = 0;
    let totalInterGap = 0;
    const centerCoords = {};
    const hullData = [];

    const hullGenerator = d3.line().curve(d3.curveCardinalClosed);

    FLOWERS.forEach(flower => {
        const classPoints = data.filter(d => d.flower === flower);
        const meanX = d3.mean(classPoints, d => d.x);
        const meanY = d3.mean(classPoints, d => d.y);
        centerCoords[flower] = { x: meanX, y: meanY };

        const spread = d3.mean(classPoints, d => Math.sqrt(Math.pow(d.x - meanX, 2) + Math.pow(d.y - meanY, 2)));
        totalIntraSpread += spread;

        // Prep Smooth Hull
        if (classPoints.length >= 3) {
            const hullPoints = d3.polygonHull(classPoints.map(d => [xScale(d.x), yScale(d.y)]));
            if (hullPoints) {
                hullData.push({ flower: flower, path: hullGenerator(hullPoints) });
            }
        }
    });

    // Update Hulls (only if sim4ShowHulls is true)
    svg.select('.hull-group').selectAll('path.hull')
        .data(sim4ShowHulls ? hullData : [], d => d.flower)
        .join(
            enter => enter.append('path').attr('class', 'hull').attr('opacity', 0).call(e => e.transition().duration(500).attr('opacity', 1)),
            update => update.transition().duration(800).ease(d3.easeCubicInOut).attr('d', d => d.path),
            exit => exit.transition().duration(500).attr('opacity', 0).remove()
        )
        .attr('fill', d => FLOWER_COLORS[d.flower])
        .attr('stroke', d => FLOWER_COLORS[d.flower]);

    // Calculate average Inter-Cluster Gap
    let pairs = 0;
    FLOWERS.forEach((f1, i) => {
        FLOWERS.forEach((f2, j) => {
            if (i < j) {
                const dist = Math.sqrt(Math.pow(centerCoords[f1].x - centerCoords[f2].x, 2) + Math.pow(centerCoords[f1].y - centerCoords[f2].y, 2));
                totalInterGap += dist;
                pairs++;
            }
        });
    });

    const avgSpread = totalIntraSpread / FLOWERS.length;
    const avgGap = totalInterGap / pairs;
    const separability = Math.min(0.99, (avgGap / (avgSpread + 0.05)) * 0.4);

    document.getElementById('sim4-separability').textContent = (separability * 100).toFixed(1) + '%';
    document.getElementById('sim4-spread').textContent = avgSpread.toFixed(3);
    document.getElementById('sim4-gap').textContent = avgGap.toFixed(3);

    updateSim4Concepts(mode, epoch, { separability });

    // Axes
    const axisGroup = svg.select('.axis-group');
    axisGroup.selectAll('*').remove();
    axisGroup.append('g')
        .attr('transform', `translate(0,${h - 30})`)
        .call(d3.axisBottom(xScale).ticks(5).tickSize(3))
        .selectAll('text').style('font-size', '9px');

    axisGroup.append('g')
        .attr('transform', 'translate(40,0)')
        .call(d3.axisLeft(yScale).ticks(5).tickSize(3))
        .selectAll('text').style('font-size', '9px');

    // Tooltip setup
    let tooltip = d3.select('.sim4-tooltip');
    if (tooltip.empty()) {
        tooltip = d3.select('body').append('div').attr('class', 'tooltip sim4-tooltip').style('opacity', 0);
    }

    // Points with Transition
    svg.select('.main-group').selectAll('circle.scatter-point')
        .data(data)
        .join('circle')
        .attr('class', 'scatter-point')
        .attr('fill', d => FLOWER_COLORS[d.flower])
        .attr('stroke', '#fff')
        .attr('stroke-width', 1)
        .on('mouseover', (event, d) => {
            tooltip.transition().duration(200).style('opacity', 0.95);
            tooltip.html(`<strong>${d.flower.charAt(0).toUpperCase() + d.flower.slice(1)}</strong>`)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px');
        })
        .on('mouseout', () => {
            tooltip.transition().duration(500).style('opacity', 0);
        })
        .transition().duration(800).ease(d3.easeCubicInOut)
        .attr('cx', d => xScale(d.x))
        .attr('cy', d => yScale(d.y))
        .attr('r', 4)
        .attr('opacity', 0.8);
}

document.getElementById('sim4-epoch').addEventListener('input', e => {
    document.getElementById('sim4-epoch-val').textContent = e.target.value;
    sim4ShowHulls = false; // Hide on manual scrub
    renderSim4Scatter();
});

document.getElementById('sim4-model').addEventListener('change', () => {
    sim4ShowHulls = false;
    renderSim4Scatter();
});

document.getElementById('sim4-mode').addEventListener('change', () => {
    sim4ShowHulls = false;
    renderSim4Scatter();
});

document.getElementById('sim4-animate').addEventListener('click', () => {
    if (sim4AnimTimer) clearInterval(sim4AnimTimer);
    const slider = document.getElementById('sim4-epoch');
    slider.value = 1;
    document.getElementById('sim4-epoch-val').textContent = '1';

    sim4AnimTimer = setInterval(() => {
        let v = parseInt(slider.value);
        if (v >= 30) {
            clearInterval(sim4AnimTimer);
            sim4ShowHulls = true; // Show at the end
            renderSim4Scatter();
            return;
        }
        slider.value = v + 1;
        document.getElementById('sim4-epoch-val').textContent = v + 1;
        sim4ShowHulls = false; // Keep hidden during animation
        renderSim4Scatter();
    }, 200);
});

document.getElementById('sim4-reset').addEventListener('click', () => {
    if (sim4AnimTimer) clearInterval(sim4AnimTimer);
    sim4ShowHulls = false;
    document.getElementById('sim4-model').value = 'vgg19';
    document.getElementById('sim4-mode').value = 'random';
    document.getElementById('sim4-epoch').value = 1;
    document.getElementById('sim4-epoch-val').textContent = '1';
    renderSim4Scatter();
});


// ================================================
// SIM 5: GRADIENT FLOW & STABILITY
// ================================================

let sim5Charts = {};

function initSim5Charts() {
    if (sim5Charts.grad) return;

    const commonOpts = {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: { legend: { display: false } },
        scales: {
            x: { ticks: { font: { size: 9 } } },
            y: { ticks: { font: { size: 8 } } }
        }
    };

    sim5Charts.grad = new Chart(document.getElementById('sim5-chart-grad'), {
        type: 'bar',
        data: { labels: [], datasets: [{ data: [], backgroundColor: [] }] },
        options: commonOpts
    });

    sim5Charts.weight = new Chart(document.getElementById('sim5-chart-weight'), {
        type: 'bar',
        data: { labels: [], datasets: [{ data: [], backgroundColor: [] }] },
        options: commonOpts
    });

    updateSim5();
}

function updateSim5() {
    const model = document.getElementById('sim5-model').value;
    const freezePct = parseInt(document.getElementById('sim5-freeze').value);
    const lrIdx = parseInt(document.getElementById('sim5-lr').value);
    const epoch = parseInt(document.getElementById('sim5-epoch').value);
    const md = MODEL_DATA[model];
    const lr = LR_VALUES[lrIdx];

    const displayLayers = md.totalLayers;
    const frozenCount = Math.round(displayLayers * (freezePct / 100));

    const labels = [];
    const gradients = [];
    const weights = [];
    const gradColors = [];
    const weightColors = [];

    for (let i = 0; i < displayLayers; i++) {
        labels.push(md.layerNames[i] || `L${i + 1}`);
        const isFrozen = i < frozenCount;

        if (isFrozen) {
            gradients.push(0);
            weights.push(0);
            gradColors.push('#dfe6e9');
            weightColors.push('#dfe6e9');
        } else {
            const depthFactor = (i - frozenCount) / (displayLayers - frozenCount);
            const baseMag = 0.01 + depthFactor * 0.05;
            const lrMult = lr / 1e-5;
            let grad = baseMag * lrMult * (1 + Math.random() * 0.3);

            // VGG is less stable with high LR
            if (model === 'vgg19' && lrMult > 10) {
                grad *= 2 + Math.random() * 3;
            }

            // Epoch decay
            grad *= 1 / (1 + epoch * 0.03);

            gradients.push(Math.min(2, grad));
            weights.push(Math.min(1.5, grad * 0.6 * (1 + Math.random() * 0.2)));

            const isSpike = grad > 0.3;
            gradColors.push(isSpike ? '#e17055' : '#6c5ce7');
            weightColors.push(isSpike ? '#e17055' : '#00cec9');
        }
    }

    if (sim5Charts.grad) {
        sim5Charts.grad.data.labels = labels;
        sim5Charts.grad.data.datasets[0].data = gradients;
        sim5Charts.grad.data.datasets[0].backgroundColor = gradColors;
        sim5Charts.grad.update();
    }

    if (sim5Charts.weight) {
        sim5Charts.weight.data.labels = labels;
        sim5Charts.weight.data.datasets[0].data = weights;
        sim5Charts.weight.data.datasets[0].backgroundColor = weightColors;
        sim5Charts.weight.update();
    }

    // Stability meter calculation
    const activeGradients = gradients.filter(g => g > 0);
    const maxGrad = activeGradients.length ? Math.max(...activeGradients) : 0;
    const avgUpdate = d3.mean(weights.filter(w => w > 0)) || 0;

    let stability;
    if (maxGrad < 0.1) stability = 95 - (lrIdx * 2);
    else if (maxGrad < 0.3) stability = 75 - (lrIdx * 5);
    else if (maxGrad < 0.8) stability = 45 - (lrIdx * 8);
    else stability = Math.max(5, 20 - (lrIdx * 10));

    // Update Metrics Row
    document.getElementById('sim5-max-grad').textContent = maxGrad.toFixed(4);
    document.getElementById('sim5-avg-update').textContent = avgUpdate.toFixed(4);
    document.getElementById('sim5-stability-score').textContent = stability.toFixed(0) + '/100';

    const risk = (lrIdx > 3 && freezePct < 30) ? 'High' : (lrIdx > 2 || freezePct < 20) ? 'Moderate' : 'Low';
    const riskLabel = document.getElementById('sim5-risk-label');
    riskLabel.textContent = risk;
    riskLabel.style.color = risk === 'High' ? 'var(--danger)' : risk === 'Moderate' ? 'var(--warning)' : 'var(--success)';

    // Update Stability UI
    const gaugeFill = document.getElementById('sim5-gauge-fill');
    gaugeFill.style.width = stability + '%';
    gaugeFill.style.background = stability > 70 ? '#00b894' : stability > 40 ? '#fdcb6e' : '#e17055';

    const statusLabel = document.getElementById('sim5-gauge-label');
    statusLabel.textContent = stability > 70 ? 'Stable' : stability > 40 ? 'Moderate' : 'Unstable';
    statusLabel.style.color = stability > 70 ? '#00b894' : stability > 40 ? '#fdcb6e' : '#e17055';

    document.getElementById('sim5-forgetting').innerHTML = `Catastrophic Forgetting Risk: <strong style="color:${risk === 'High' ? 'var(--danger)' : risk === 'Moderate' ? 'var(--warning)' : 'var(--success)'}">${risk}</strong>`;

    // Render Schematic
    const schematic = document.getElementById('sim5-schematic');
    if (schematic) {
        const pulse = schematic.querySelector('.backprop-pulse');
        schematic.innerHTML = '';
        if (pulse) schematic.appendChild(pulse);

        const stepSize = Math.max(1, Math.floor(displayLayers / 10));
        for (let i = 0; i < displayLayers; i += stepSize) {
            const div = document.createElement('div');
            div.className = 'schematic-layer';
            const isFrozen = i < frozenCount;
            const gradNorm = gradients[i];

            if (isFrozen) {
                div.style.background = '#4b6584';
                div.style.boxShadow = 'none';
            } else {
                const intensity = Math.min(1, gradNorm * 3);
                div.style.background = d3.interpolatePurples(0.3 + intensity * 0.7);
                div.style.boxShadow = `0 0 ${10 + intensity * 15}px ${d3.interpolatePurples(0.5 + intensity * 0.5)}`;
                if (gradNorm > 0.5) div.style.transform = `scale(${1 + gradNorm * 0.1})`;
            }
            schematic.appendChild(div);
        }
    }

    // Educational Context
    const title = document.getElementById('sim5-concept-title');
    const desc = document.getElementById('sim5-concept-desc');
    if (stability < 40) {
        title.textContent = 'Exploding Gradients';
        desc.textContent = 'Gradients are now unstable! Large weight updates can cause the loss to oscillate or diverge. High learning rates with few frozen layers are dangerous for pretrained models.';
    } else if (stability > 80) {
        title.textContent = 'Stable Convergence';
        desc.textContent = 'Gradient flow is healthy. The network is learning efficiently without overwriting existing feature extraction knowledge.';
    } else {
        title.textContent = 'Balanced Tuning';
        desc.textContent = 'Partial freezing and a moderate learning rate provide a good trade-off between speed and stability.';
    }
}

['sim5-model', 'sim5-freeze', 'sim5-lr', 'sim5-epoch'].forEach(id => {
    const el = document.getElementById(id);
    el.addEventListener(el.tagName === 'SELECT' ? 'change' : 'input', () => {
        if (id === 'sim5-freeze') document.getElementById('sim5-freeze-val').textContent = el.value + '%';
        if (id === 'sim5-lr') document.getElementById('sim5-lr-val').textContent = LR_LABELS[el.value];
        if (id === 'sim5-epoch') document.getElementById('sim5-epoch-val').textContent = el.value;
        updateSim5();
    });
});

document.getElementById('sim5-reset').addEventListener('click', () => {
    document.getElementById('sim5-model').value = 'vgg19';
    document.getElementById('sim5-freeze').value = 70;
    document.getElementById('sim5-freeze-val').textContent = '70%';
    document.getElementById('sim5-lr').value = 1;
    document.getElementById('sim5-lr-val').textContent = '1e-5';
    document.getElementById('sim5-epoch').value = 1;
    document.getElementById('sim5-epoch-val').textContent = '1';
    updateSim5();
});


// ================================================
// SIM 6: DOMAIN SIMILARITY & STRATEGY
// ================================================

function updateSim6() {
    const similarity = parseInt(document.getElementById('sim6-sim').value);
    const dataSize = parseInt(document.getElementById('sim6-data').value);
    const freezeDepth = parseInt(document.getElementById('sim6-freeze').value);
    const lr = document.getElementById('sim6-lr').value;

    // 1. Domain Visualizer
    const targetIcon = document.getElementById('sim6-target-icon');
    const targetLabel = document.getElementById('sim6-target-label');
    if (similarity > 80) {
        targetIcon.textContent = '🌸'; targetLabel.textContent = 'Target: Oxford Flower';
    } else if (similarity > 50) {
        targetIcon.textContent = '🍎'; targetLabel.textContent = 'Target: Food (Fruit Species)';
    } else if (similarity > 25) {
        targetIcon.textContent = '🛰️'; targetLabel.textContent = 'Target: Satellite (Land Use)';
    } else {
        targetIcon.textContent = '🏥'; targetLabel.textContent = 'Target: Medical (X-Ray)';
    }

    // 2. Strategy Matrix Projection
    const point = document.getElementById('sim6-matrix-point');
    // x: similarity (0-100), y: dataSize (100-0 reversed)
    point.style.left = (similarity * 2.5) + 'px';
    point.style.top = (250 - dataSize * 2.5) + 'px';

    // 3. Performance Metrics
    // User requested higher accuracy for unfreezing than freezing.
    let effectiveness = (similarity * 0.45) + (dataSize * 0.25) + (100 - Math.abs(freezeDepth - 40)) * 0.15 + (100 - freezeDepth) * 0.05;
    if (lr === '1e-5' || lr === '3e-5') effectiveness += 10;

    const accuracy = Math.min(98, 35 + effectiveness * 0.6);
    const stability = Math.min(100, (freezeDepth * 0.6) + (similarity * 0.2) + 20);
    const speed = Math.min(100, (similarity * 0.5) + (freezeDepth * 0.3) + 20);
    const efficiency = Math.min(100, (freezeDepth * 0.7) + 30);

    const metrics = [
        { axis: "Accuracy", value: accuracy / 100 },
        { axis: "Stability", value: stability / 100 },
        { axis: "Speed", value: speed / 100 },
        { axis: "Efficiency", value: efficiency / 100 }
    ];

    renderSim6Radar(metrics);

    // 4. Badges & Recommendations
    const badge = document.getElementById('sim6-status-badge');
    if (similarity > 60 && dataSize > 60) {
        badge.textContent = 'Optimal'; badge.className = 'status-badge badge-optimal';
    } else if (similarity < 30 || dataSize < 30) {
        badge.textContent = 'High Risk'; badge.className = 'status-badge badge-danger';
    } else {
        badge.textContent = 'Moderate'; badge.className = 'status-badge badge-caution';
    }

    let rec = '';
    if (similarity > 75) {
        if (freezeDepth < 40) {
            rec = '✅ <strong>Optimal Adaptation:</strong> With high similarity and partial unfreezing, you are capturing both general features and domain nuances. Expected accuracy is at its peak.';
        } else {
            rec = '👍 <strong>Safe & Fast:</strong> High similarity allows for heavy freezing. It is efficient and stable, though unfreezing a few more layers could further boost your accuracy.';
        }
    } else if (similarity < 30) {
        if (freezeDepth > 70) {
            rec = '⚠️ <strong>Feature Mismatch:</strong> Heavy freezing on a dissimilar domain (e.g. Medical) prevents the model from adapting. Accuracy will be significantly limited.';
        } else {
            rec = '🔍 <strong>Deep Fine-Tuning:</strong> Domain mismatch requires unfreezing more layers to re-learn low-level features. This is compute-heavy but necessary for accuracy.';
        }
    } else {
        rec = '💡 <strong>Balanced Strategy:</strong> A hybrid approach is best. Keep mid-level layers frozen to retain generic shapes, but unfreeze top blocks to adapt to your specific dataset.';
    }

    if (dataSize < 30 && freezeDepth < 40) {
        rec += ' <br><strong>Caution:</strong> Low data with high unfreezing increases <strong>Overfitting Risk</strong>. Consider increasing freeze depth or adding regularization.';
    }

    document.getElementById('sim6-rec-text').innerHTML = rec;
}

function renderSim6Radar(data) {
    const svg = d3.select('#sim6-radar-svg');
    svg.selectAll('*').remove();

    const w = 200, h = 200;
    const radius = 70;
    const centerX = w / 2, centerY = h / 2;

    // Grid circles
    const levels = 4;
    for (let i = 1; i <= levels; i++) {
        svg.append('circle')
            .attr('cx', centerX).attr('cy', centerY)
            .attr('r', (radius / levels) * i)
            .attr('class', 'radar-grid');
    }

    const angleSlice = (Math.PI * 2) / data.length;

    // Axes
    data.forEach((d, i) => {
        const x = centerX + radius * Math.cos(angleSlice * i - Math.PI / 2);
        const y = centerY + radius * Math.sin(angleSlice * i - Math.PI / 2);

        svg.append('line')
            .attr('x1', centerX).attr('y1', centerY)
            .attr('x2', x).attr('y2', y)
            .attr('class', 'radar-axis');

        svg.append('text')
            .attr('x', x).attr('y', y > centerY ? y + 15 : y - 10)
            .attr('text-anchor', 'middle')
            .style('font-size', '9px')
            .style('font-weight', '700')
            .text(d.axis);
    });

    // Polygon area
    const line = d3.lineRadial()
        .radius(d => d.value * radius)
        .angle((d, i) => i * angleSlice);

    const pathData = data.map(d => ({ ...d }));
    pathData.push(pathData[0]); // Close path

    svg.append('g')
        .attr('transform', `translate(${centerX}, ${centerY})`)
        .append('path')
        .datum(pathData)
        .attr('d', line)
        .attr('class', 'radar-area');
}

['sim6-sim', 'sim6-data', 'sim6-freeze'].forEach(id => {
    document.getElementById(id).addEventListener('input', (e) => {
        document.getElementById(`${id}-val`).textContent = e.target.value + '%';
        updateSim6();
    });
});

document.getElementById('sim6-lr').addEventListener('change', updateSim6);
document.getElementById('sim6-home').addEventListener('click', () => switchTab('landing'));

document.getElementById('sim6-reset').addEventListener('click', () => {
    document.getElementById('sim6-sim').value = 50;
    document.getElementById('sim6-sim-val').textContent = '50%';
    document.getElementById('sim6-data').value = 50;
    document.getElementById('sim6-data-val').textContent = '50%';
    document.getElementById('sim6-freeze').value = 50;
    document.getElementById('sim6-freeze-val').textContent = '50%';
    document.getElementById('sim6-lr').value = '1e-5';
    updateSim6();
});
