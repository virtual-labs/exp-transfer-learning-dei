/**
 * Transfer Learning CNN Simulation
 * Interactive Jupyter notebook-style simulation for Transfer Learning experiments
 */

// ============================================
// Configuration
// ============================================

const CONFIG = {
    executionDelay: 1500,
    typingDelay: 30,
    totalSteps: 7
};

// ============================================
// State Management
// ============================================

const state = {
    currentStep: 1,
    completedSteps: new Set(),
    runningStep: null,
    isRunningAll: false,
    model: 'mobilenetv2',
    freeze: '0'
};

// ============================================
// DOM Elements
// ============================================

const elements = {
    stepItems: document.querySelectorAll('.step-item'),
    cells: document.querySelectorAll('.notebook-cell'),
    downloadBtn: document.getElementById('downloadBtn'),
    resetBtn: document.getElementById('resetBtn'),
    completionMessage: document.getElementById('completionMessage'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    modelSelect: document.getElementById('modelSelect'),
    freezeSelect: document.getElementById('freezeSelect')
};

// ============================================
// Data Access
// ============================================

function getDataKey() {
    return `${state.model}_${state.freeze}`;
}

function getCurrentData() {
    const key = getDataKey();
    return EXPERIMENT_DATA[key];
}

// ============================================
// UI Update Functions
// ============================================

function downloadExperiment() {
    const link = document.createElement('a');
    link.href = './assets/Exp-5_ Transfer_ Learnig.pdf';
    link.download = 'Transfer_Learning_Experiment.pdf';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function updateFreezeOptions() {
    const freezeSelect = elements.freezeSelect;
    const currentFreeze = state.freeze;
    
    // Clear options
    freezeSelect.innerHTML = '';
    
    if (state.model === 'mobilenetv2') {
        freezeSelect.innerHTML = `
            <option value="0">Base Freeze (0% Unfreeze)</option>
            <option value="10">10% Unfreeze</option>
            <option value="20">20% Unfreeze</option>
        `;
    } else if (state.model === 'vgg19') {
        freezeSelect.innerHTML = `
            <option value="0">Base Freeze (0% Unfreeze)</option>
            <option value="20">20% Unfreeze</option>
            <option value="30">30% Unfreeze</option>
        `;
    }
    
    // Try to preserve selection, or default to first
    const options = Array.from(freezeSelect.options);
    const matchingOption = options.find(opt => opt.value === currentFreeze);
    if (matchingOption) {
        freezeSelect.value = currentFreeze;
    } else {
        freezeSelect.value = options[0].value;
        state.freeze = options[0].value;
    }
}

function updateCodeDisplay() {
    // Check if CODE_TEMPLATES exists
    if (typeof CODE_TEMPLATES === 'undefined') {
        console.warn('CODE_TEMPLATES not loaded');
        return;
    }
    
    const templates = CODE_TEMPLATES[state.model];
    if (!templates) {
        console.warn('No templates for model:', state.model);
        return;
    }
    
    // Learning rate mapping based on model and freeze percentage
    const learningRates = {
        mobilenetv2: {
            '0': '1e-4',
            '10': '1e-5',
            '20': '1e-5'
        },
        vgg19: {
            '0': '1e-4',
            '20': '1e-5',
            '30': '1e-5'
        }
    };
    
    // Update all code blocks with the model-specific code
    const codeBlocks = document.querySelectorAll('.cell-code pre code');
    
    codeBlocks.forEach((block, index) => {
        const cellKey = `cell${index}`;
        if (templates[cellKey]) {
            let code = templates[cellKey];
            
            // Update cell 1 (parameters cell) - learning rate
            if (index === 1) {
                const lr = learningRates[state.model][state.freeze];
                if (lr) {
                    code = code.replace(/learning_rate = [\de.-]+/, `learning_rate = ${lr}`);
                }
            }
            
            // Update cell 3 (training cell) - unfreeze percentage
            if (index === 3) {
                // Replace the unfreeze percentage value in the function call
                const unfreezeValue = (parseInt(state.freeze) / 100).toFixed(2);
                code = code.replace(/set_trainable_layers\(base_model,\s*[\d.]+\)/, 
                    `set_trainable_layers(base_model, ${unfreezeValue})`);
                
                // Replace the comment above the set_trainable_layers call
                code = code.replace(/# Unfreeze last \d+% of layers/, 
                    `# Unfreeze last ${state.freeze}% of layers`);
            }
            
            block.textContent = code;
        }
    });
}

function updateStepState(stepNum, status) {
    const stepItem = document.querySelector(`.step-item[data-step="${stepNum}"]`);
    const cell = document.querySelector(`.notebook-cell[data-step="${stepNum}"]`);
    
    if (!stepItem || !cell) return;
    
    stepItem.classList.remove('active', 'running', 'completed');
    cell.classList.remove('running', 'completed');
    
    switch (status) {
        case 'active':
            stepItem.classList.add('active');
            break;
        case 'running':
            stepItem.classList.add('running');
            cell.classList.add('running');
            break;
        case 'completed':
            stepItem.classList.add('completed');
            cell.classList.add('completed');
            break;
    }
}

function updateRunButtonStates() {
    elements.cells.forEach((cell, index) => {
        const stepNum = index + 1;
        const runBtn = cell.querySelector('.run-btn');
        if (!runBtn) return;
        
        if (stepNum === 1) {
            runBtn.disabled = state.completedSteps.has(1);
        } else {
            runBtn.disabled = !state.completedSteps.has(stepNum - 1) || state.completedSteps.has(stepNum);
        }
    });
}

// ============================================
// Cell Execution
// ============================================

async function executeCell(stepNum) {
    if (state.runningStep) return;
    
    state.runningStep = stepNum;
    updateStepState(stepNum, 'running');
    
    const runBtn = document.querySelector(`.run-btn[data-step="${stepNum}"]`);
    if (runBtn) {
        runBtn.disabled = true;
        runBtn.classList.add('running');
        runBtn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"/>
            </svg>
            Running...
        `;
    }
    
    // Show output area
    const cell = document.querySelector(`.notebook-cell[data-step="${stepNum}"]`);
    const output = cell.querySelector('.cell-output');
    if (output) {
        output.classList.remove('hidden');
    }
    
    // Simulate execution delay
    await new Promise(resolve => setTimeout(resolve, CONFIG.executionDelay));
    
    // Populate output based on step
    await populateOutput(stepNum);
    
    // Mark as completed
    state.completedSteps.add(stepNum);
    state.runningStep = null;
    
    updateStepState(stepNum, 'completed');
    
    if (runBtn) {
        runBtn.classList.remove('running');
        runBtn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
            </svg>
            Done
        `;
    }
    
    // Update next step as active
    if (stepNum < CONFIG.totalSteps) {
        updateStepState(stepNum + 1, 'active');
    }
    
    updateRunButtonStates();
    
    // Check if all steps completed
    if (state.completedSteps.size === CONFIG.totalSteps) {
        showCompletionMessage();
    }
}

async function populateOutput(stepNum) {
    const data = getCurrentData();
    if (!data || !data.cells[stepNum - 1]) return;
    
    const cellData = data.cells[stepNum - 1];
    const key = getDataKey();
    
    switch(stepNum) {
        case 1:
            // Import libraries - clean success message
            document.getElementById('output1').textContent = 'All the required libraries are imported successfully.';
            break;
            
        case 2:
            // Parameters
            const params = cellData.outputs.join('\n');
            await typeText('output2', params);
            break;
            
        case 3:
            // Dataset visualization
            const datasetOutput = cellData.outputs.join('\n');
            // Truncate class mapping to first 20 classes
            const lines = datasetOutput.split('\n');
            const truncatedOutput = lines.slice(0, 30).join('\n') + '\n... (showing first 24 of 102 classes)\n\nShowing random sample images from the dataset...';
            await typeText('output3', truncatedOutput);
            
            // Show sample image
            const datasetImg = document.getElementById('datasetImage');
            datasetImg.src = `images/${key}_cell2.png`;
            datasetImg.style.display = 'block';
            break;
            
        case 4:
            // Training output
            const trainingOutput = cellData.outputs.join('\n');
            // Truncate training log
            const trainingLines = trainingOutput.split('\n');
            let truncatedTraining = '';
            if (trainingLines.length > 30) {
                truncatedTraining = trainingLines.slice(0, 15).join('\n') + 
                    '\n\n...\n\n' + 
                    trainingLines.slice(-10).join('\n');
            } else {
                truncatedTraining = trainingOutput;
            }
            await typeText('output4', truncatedTraining);
            break;
            
        case 5:
            // Evaluation
            const evalOutput = cellData.outputs.join('\n');
            await typeText('output5', evalOutput);
            
            // Extract and show metrics
            const trainMatch = evalOutput.match(/Train Accuracy: (\d+\.\d+)%/);
            const valMatch = evalOutput.match(/Validation Accuracy: (\d+\.\d+)%/);
            const testMatch = evalOutput.match(/Test Accuracy: (\d+\.\d+)%/);
            
            if (trainMatch && valMatch && testMatch) {
                document.getElementById('trainAcc').textContent = trainMatch[1] + '%';
                document.getElementById('valAcc').textContent = valMatch[1] + '%';
                document.getElementById('testAcc').textContent = testMatch[1] + '%';
                document.getElementById('metricsSummary').style.display = 'block';
            }
            break;
            
        case 6:
            // Results visualization
            let resultsOutput = cellData.outputs.join('\n');
            // Truncate classification report
            const resultLines = resultsOutput.split('\n');
            if (resultLines.length > 25) {
                resultsOutput = resultLines.slice(0, 20).join('\n') + '\n... (102 classes total)\n\n' + 
                    resultLines.slice(-5).join('\n');
            }
            await typeText('output6', resultsOutput);
            
            // Show training curves image
            const curvesImg = document.getElementById('trainingCurvesImage');
            curvesImg.src = `images/${key}_cell5.png`;
            curvesImg.style.display = 'block';
            break;
            
        case 7:
            // Confusion matrix
            const cmImg = document.getElementById('confusionMatrixImage');
            cmImg.src = `images/${key}_cell6.png`;
            cmImg.style.display = 'block';
            break;
    }
}

async function typeText(elementId, text, delay = CONFIG.typingDelay) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    element.textContent = '';
    
    // For faster display, add chunks instead of single characters
    const chunkSize = 10;
    for (let i = 0; i < text.length; i += chunkSize) {
        element.textContent += text.substring(i, i + chunkSize);
        if (i % 100 === 0) {
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
}

// ============================================
// Download / Reset
// ============================================

function resetSimulation() {
    state.currentStep = 1;
    state.completedSteps.clear();
    state.runningStep = null;
    state.isRunningAll = false;
    
    // Reset step items
    elements.stepItems.forEach((item, index) => {
        item.classList.remove('active', 'running', 'completed');
        if (index === 0) {
            item.classList.add('active');
        }
    });
    
    // Reset cells
    elements.cells.forEach(cell => {
        cell.classList.remove('running', 'completed');
        const output = cell.querySelector('.cell-output');
        const runBtn = cell.querySelector('.run-btn');
        
        if (output) output.classList.add('hidden');
        if (runBtn) {
            runBtn.disabled = false;
            runBtn.classList.remove('running');
            runBtn.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M8 5v14l11-7z"/>
                </svg>
                Run
            `;
        }
    });
    
    // Reset output elements
    for (let i = 1; i <= 7; i++) {
        const output = document.getElementById(`output${i}`);
        if (output) output.textContent = '';
    }
    
    // Reset images
    ['datasetImage', 'trainingCurvesImage', 'confusionMatrixImage'].forEach(id => {
        const img = document.getElementById(id);
        if (img) {
            img.style.display = 'none';
            img.src = '';
        }
    });
    
    // Reset metrics summary
    document.getElementById('metricsSummary').style.display = 'none';
    
    // Hide completion message
    elements.completionMessage.classList.add('hidden');
    
    // Update code display for current settings
    updateCodeDisplay();
    
    // Update button states
    updateRunButtonStates();
}

function showCompletionMessage() {
    elements.completionMessage.classList.remove('hidden');
    elements.completionMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ============================================
// Event Handlers
// ============================================

function handleModelChange(e) {
    const newModel = e.target.value;
    if (newModel !== state.model) {
        state.model = newModel;
        updateFreezeOptions();
        updateCodeDisplay();
        
        // Reset if any step was completed
        if (state.completedSteps.size > 0) {
            resetSimulation();
        }
    }
}

function handleFreezeChange(e) {
    const newFreeze = e.target.value;
    if (newFreeze !== state.freeze) {
        state.freeze = newFreeze;
        updateCodeDisplay();
        
        // Reset if cell 1 (parameters) or any training-related steps were completed
        // Since learning rate changes with freeze percentage
        if (state.completedSteps.has(1) || state.completedSteps.has(4) || 
            state.completedSteps.has(5) || state.completedSteps.has(6) || 
            state.completedSteps.has(7)) {
            resetSimulation();
        }
    }
}

// ============================================
// Initialization
// ============================================

function init() {
    // Event listeners for model/freeze selection
    elements.modelSelect.addEventListener('change', handleModelChange);
    elements.freezeSelect.addEventListener('change', handleFreezeChange);
    
    // Event listeners for run buttons
    document.querySelectorAll('.run-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const step = parseInt(e.target.closest('.run-btn').dataset.step);
            executeCell(step);
        });
    });
    
    // Download Experiment button
    if (elements.downloadBtn) {
        elements.downloadBtn.addEventListener('click', downloadExperiment);
    }
    
    // Reset button
    elements.resetBtn.addEventListener('click', resetSimulation);
    
    // Step item click handlers
    elements.stepItems.forEach(item => {
        item.addEventListener('click', () => {
            const step = parseInt(item.dataset.step);
            const cell = document.querySelector(`.notebook-cell[data-step="${step}"]`);
            if (cell) {
                cell.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        });
    });
    
    // Initialize UI
    updateFreezeOptions();
    updateCodeDisplay();
    updateRunButtonStates();
    
    console.log('Transfer Learning Simulation initialized');
}

// Start when DOM is ready
document.addEventListener('DOMContentLoaded', init);
