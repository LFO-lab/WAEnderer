/**
 * Stable Audio Wanderer - Real-time Visualization
 * p5.js sketch for visualizing navigation trajectories
 */

// WebSocket connection
let ws = null;
let wsConnected = false;
const WS_URL = 'ws://127.0.0.1:8765';

// Corpus data
let corpusPoints = [];
let corpusFileIds = [];
let corpusRegimes = [];
let totalCorpusPoints = 0;

// Navigation state
let currentPosition = [0.5, 0.5];
let trajectory = [];
let currentRegime = 0;
let currentIndex = 0;
let currentVelocity = 0;
let currentFileId = 0;

// Visualization settings
let vizMode = 'scatter';
let trailLength = 64;
let heatmapData = null;
let heatmapResolution = 50;

// Colors
const COLORS = {
    background: [13, 13, 20],
    corpusPoint: [60, 60, 80, 100],
    trajectory: [74, 158, 255],
    cursor: [255, 255, 255],
    regimes: [
        [74, 255, 106],   // Drift - green
        [255, 166, 74],   // Turn - orange
        [74, 158, 255],   // Linger - blue
    ],
    files: [],
};

// Canvas dimensions
let canvasSize;

function setup() {
    const container = document.getElementById('canvas-container');
    canvasSize = Math.min(container.clientWidth, container.clientHeight) - 40;
    
    const canvas = createCanvas(canvasSize, canvasSize);
    canvas.parent('canvas-container');
    
    // Generate file colors
    for (let i = 0; i < 20; i++) {
        const hue = (i * 137.5) % 360;
        COLORS.files.push(hslToRgb(hue, 60, 50));
    }
    
    // Initialize heatmap
    heatmapData = new Array(heatmapResolution).fill(0).map(() => 
        new Array(heatmapResolution).fill(0)
    );
    
    connectWebSocket();
    setupControls();
    
    frameRate(30);
}

function draw() {
    background(...COLORS.background);
    
    if (vizMode === 'scatter') {
        drawCorpusScatter();
        drawTrajectory();
        drawCursor();
    } else if (vizMode === 'trail') {
        drawTrajectory();
        drawCursor();
    } else if (vizMode === 'heatmap') {
        drawHeatmap();
        drawTrajectory();
        drawCursor();
    }
    
    noFill();
    stroke(40, 40, 60);
    strokeWeight(2);
    rect(0, 0, width, height);
}

function drawCorpusScatter() {
    noStroke();
    
    for (let i = 0; i < corpusPoints.length; i++) {
        const [x, y] = corpusPoints[i];
        const fileId = corpusFileIds[i] || 0;
        
        const fileColor = COLORS.files[fileId % COLORS.files.length];
        fill(fileColor[0], fileColor[1], fileColor[2], 50);
        
        const screenX = x * width;
        const screenY = (1 - y) * height;
        
        ellipse(screenX, screenY, 4, 4);
    }
}

function drawTrajectory() {
    if (trajectory.length < 2) return;
    
    noFill();
    
    for (let i = 1; i < trajectory.length; i++) {
        const alpha = map(i, 0, trajectory.length, 50, 255);
        const weight = map(i, 0, trajectory.length, 1, 3);
        
        const [x1, y1] = trajectory[i - 1];
        const [x2, y2] = trajectory[i];
        
        const regimeColor = COLORS.regimes[currentRegime] || COLORS.trajectory;
        stroke(regimeColor[0], regimeColor[1], regimeColor[2], alpha);
        strokeWeight(weight);
        
        const sx1 = x1 * width;
        const sy1 = (1 - y1) * height;
        const sx2 = x2 * width;
        const sy2 = (1 - y2) * height;
        
        line(sx1, sy1, sx2, sy2);
    }
}

function drawCursor() {
    if (!currentPosition) return;
    
    const [x, y] = currentPosition;
    const screenX = x * width;
    const screenY = (1 - y) * height;
    
    const regimeColor = COLORS.regimes[currentRegime] || COLORS.cursor;
    
    noStroke();
    for (let r = 30; r > 0; r -= 5) {
        const alpha = map(r, 30, 0, 10, 50);
        fill(regimeColor[0], regimeColor[1], regimeColor[2], alpha);
        ellipse(screenX, screenY, r, r);
    }
    
    noFill();
    stroke(regimeColor[0], regimeColor[1], regimeColor[2]);
    strokeWeight(2);
    ellipse(screenX, screenY, 20, 20);
    
    fill(255);
    noStroke();
    ellipse(screenX, screenY, 6, 6);
}

function drawHeatmap() {
    noStroke();
    const cellW = width / heatmapResolution;
    const cellH = height / heatmapResolution;
    
    let maxVal = 1;
    for (let i = 0; i < heatmapResolution; i++) {
        for (let j = 0; j < heatmapResolution; j++) {
            maxVal = max(maxVal, heatmapData[i][j]);
        }
    }
    
    for (let i = 0; i < heatmapResolution; i++) {
        for (let j = 0; j < heatmapResolution; j++) {
            const val = heatmapData[i][j] / maxVal;
            if (val > 0.01) {
                const r = map(val, 0, 1, 20, 255);
                const g = map(val, 0, 1, 30, 50);
                const b = map(val, 0, 1, 80, 50);
                fill(r, g, b, map(val, 0, 1, 50, 200));
                rect(i * cellW, (heatmapResolution - 1 - j) * cellH, cellW, cellH);
            }
        }
    }
}

function updateHeatmap() {
    if (!currentPosition) return;
    
    const [x, y] = currentPosition;
    const i = Math.floor(x * heatmapResolution);
    const j = Math.floor(y * heatmapResolution);
    
    if (i >= 0 && i < heatmapResolution && j >= 0 && j < heatmapResolution) {
        heatmapData[i][j] += 1;
    }
    
    for (let a = 0; a < heatmapResolution; a++) {
        for (let b = 0; b < heatmapResolution; b++) {
            heatmapData[a][b] *= 0.999;
        }
    }
}

// WebSocket handling
function connectWebSocket() {
    try {
        ws = new WebSocket(WS_URL);
        
        ws.onopen = () => {
            wsConnected = true;
            updateConnectionStatus(true);
            console.log('WebSocket connected');
        };
        
        ws.onclose = () => {
            wsConnected = false;
            updateConnectionStatus(false);
            console.log('WebSocket disconnected');
            setTimeout(connectWebSocket, 3000);
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleMessage(data);
            } catch (e) {
                console.error('Failed to parse message:', e);
            }
        };
    } catch (e) {
        console.error('Failed to connect WebSocket:', e);
        setTimeout(connectWebSocket, 3000);
    }
}

function handleMessage(data) {
    if (data.type === 'corpus') {
        corpusPoints = data.positions_2d || [];
        corpusFileIds = data.file_ids || [];
        corpusRegimes = data.regimes || [];
        totalCorpusPoints = data.total_points || 0;
        console.log(`Received corpus: ${corpusPoints.length} points`);
        
    } else if (data.type === 'state') {
        const nav = data.navigation || {};
        
        currentPosition = nav.position_2d || currentPosition;
        currentRegime = nav.regime || 0;
        currentIndex = nav.index || 0;
        currentVelocity = nav.velocity || 0;
        currentFileId = nav.file_id || 0;
        
        if (nav.trajectory_2d && nav.trajectory_2d.length > 0) {
            trajectory = nav.trajectory_2d;
        }
        
        updateHeatmap();
        updateInfoDisplay(data);
        
        if (data.controls) {
            updateControlDisplays('ctrl', data.controls);
        }
        
        if (data.grain) {
            updateControlDisplays('grain', data.grain);
        }
    }
}

function updateConnectionStatus(connected) {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    
    if (connected) {
        dot.classList.add('connected');
        text.textContent = 'Connected';
    } else {
        dot.classList.remove('connected');
        text.textContent = 'Disconnected';
    }
}

function updateInfoDisplay(data) {
    const nav = data.navigation || {};
    
    document.getElementById('info-index').textContent = nav.index || 0;
    document.getElementById('info-velocity').textContent = (nav.velocity || 0).toFixed(2);
    document.getElementById('info-file').textContent = nav.file_id || 0;
    
    const regimeNames = ['Drift', 'Turn', 'Linger'];
    document.getElementById('info-regime').textContent = regimeNames[nav.regime] || 'Unknown';
    
    document.getElementById('regime-drift').className = 'regime-dot' + (nav.regime === 0 ? ' active-drift' : '');
    document.getElementById('regime-turn').className = 'regime-dot' + (nav.regime === 1 ? ' active-turn' : '');
    document.getElementById('regime-linger').className = 'regime-dot' + (nav.regime === 2 ? ' active-linger' : '');
}

function updateControlDisplays(prefix, values) {
    for (const [key, value] of Object.entries(values)) {
        const display = document.getElementById(`val-${key}`);
        if (display && typeof value === 'number') {
            if (key === 'filter_freq') {
                display.textContent = value.toFixed(0);
            } else {
                display.textContent = value.toFixed(2);
            }
        }
    }
}

// UI Control handlers
function setupControls() {
    // Policy controls
    const policyControls = ['width', 'energy', 'gravity', 'memory', 'coherence', 'exploration', 'regime_bias'];
    
    policyControls.forEach(ctrl => {
        const input = document.getElementById(`ctrl-${ctrl}`);
        const display = document.getElementById(`val-${ctrl}`);
        
        if (input) {
            input.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                if (display) display.textContent = value.toFixed(2);
                sendControl(ctrl, value);
            });
        }
    });
    
    // Grain controls - all parameters
    const grainControls = [
        'trigger_rate',
        'trigger_jitter',
        'pitch',
        'pitch_spread',
        'grain_dur',
        'grain_dur_spread',
        'position_spread',
        'pan',
        'pan_spread',
        'filter_freq',
        'filter_q',
        'reverse_prob',
        'master_amp',
    ];
    
    grainControls.forEach(param => {
        const input = document.getElementById(`grain-${param}`);
        const display = document.getElementById(`val-${param}`);
        
        if (input) {
            input.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                if (display) {
                    if (param === 'filter_freq') {
                        display.textContent = value.toFixed(0);
                    } else {
                        display.textContent = value.toFixed(2);
                    }
                }
                sendGrainParam(param, value);
            });
        }
    });
    
    // Envelope select
    const envSelect = document.getElementById('grain-envelope');
    if (envSelect) {
        envSelect.addEventListener('change', (e) => {
            sendGrainParam('envelope', e.target.value);
        });
    }
    
    // Visualization mode buttons
    document.querySelectorAll('.viz-option').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.viz-option').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            vizMode = e.target.dataset.mode;
        });
    });
    
    // Reset button
    document.getElementById('btn-reset').addEventListener('click', () => {
        sendReset();
        for (let i = 0; i < heatmapResolution; i++) {
            for (let j = 0; j < heatmapResolution; j++) {
                heatmapData[i][j] = 0;
            }
        }
    });
    
    // Reconnect button
    document.getElementById('btn-reconnect').addEventListener('click', () => {
        if (ws) {
            ws.close();
        }
        connectWebSocket();
    });
}

function sendControl(name, value) {
    if (ws && wsConnected) {
        ws.send(JSON.stringify({
            type: 'control',
            controls: { [name]: value }
        }));
    }
}

function sendGrainParam(name, value) {
    if (ws && wsConnected) {
        ws.send(JSON.stringify({
            type: 'grain',
            params: { [name]: value }
        }));
    }
}

function sendReset() {
    if (ws && wsConnected) {
        ws.send(JSON.stringify({
            type: 'reset'
        }));
    }
}

// Handle mouse clicks on canvas
function mousePressed() {
    if (mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height) {
        const x = mouseX / width;
        const y = 1 - (mouseY / height);
        
        if (ws && wsConnected) {
            ws.send(JSON.stringify({
                type: 'cursor',
                coords: [x, y]
            }));
        }
    }
}

// Utility: HSL to RGB conversion
function hslToRgb(h, s, l) {
    h /= 360;
    s /= 100;
    l /= 100;
    
    let r, g, b;
    
    if (s === 0) {
        r = g = b = l;
    } else {
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1/6) return p + (q - p) * 6 * t;
            if (t < 1/2) return q;
            if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
        };
        
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1/3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1/3);
    }
    
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function windowResized() {
    const container = document.getElementById('canvas-container');
    canvasSize = Math.min(container.clientWidth, container.clientHeight) - 40;
    resizeCanvas(canvasSize, canvasSize);
}
