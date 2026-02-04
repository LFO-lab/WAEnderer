/**
 * Stable Audio Wanderer - Real-time Visualization
 * p5.js sketch for visualizing navigation trajectories
 */

// WebSocket connection
let ws = null;
let wsConnected = false;
const WS_URL = 'ws://127.0.0.1:8765';

// Track which controls are currently being adjusted by the user
let activeControls = new Set();

// Corpus data
let corpusPoints = [];
let corpusFileIds = [];
let totalCorpusPoints = 0;

// Navigation state
let currentPosition = [0.5, 0.5];
let trajectory = [];
let currentIndex = 0;
let currentVelocity = 0;
let currentFileId = 0;

// Visualization settings
let vizMode = 'scatter';
let trailLength = 64;
let heatmapData = null;
let heatmapResolution = 100; // Will be set from VIZ_CONFIG in setup()

// Colors
const COLORS = {
    background: [13, 13, 20],
    corpusPoint: [60, 60, 80, 100],
    trajectory: [74, 158, 255],
    cursor: [255, 255, 255],
    files: [],
};

// Visualization configuration
const VIZ_CONFIG = {
    // Corpus points
    pointSize: 6,
    pointOpacity: 80,
    pointGlow: true,
    glowSize: 12,
    glowOpacity: 25,

    // Trajectory
    trailFadeExponent: 2.5,
    minLineWeight: 1,
    maxLineWeight: 4,
    smoothTrajectory: true,

    // Heatmap
    heatmapResolution: 100,

    // Cursor
    showVelocityVector: true,
    velocityArrowScale: 50,
    pulseEnabled: true,
    pulseSpeed: 0.05,
    pulseAmount: 0.3,

    // Grid
    showGrid: true,
    gridDivisions: 10,
    gridOpacity: 20,
};

// Cached corpus colors (computed once on corpus load)
let cachedCorpusColors = [];

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

    // Initialize heatmap with configured resolution
    heatmapResolution = VIZ_CONFIG.heatmapResolution;
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
    // Draw optional grid first (behind points)
    if (VIZ_CONFIG.showGrid) {
        drawGrid();
    }

    noStroke();

    for (let i = 0; i < corpusPoints.length; i++) {
        const [x, y] = corpusPoints[i];
        const fileColor = cachedCorpusColors[i] || COLORS.files[0];

        const screenX = x * width;
        const screenY = (1 - y) * height;

        // Draw glow effect (larger semi-transparent point behind)
        if (VIZ_CONFIG.pointGlow) {
            fill(fileColor[0], fileColor[1], fileColor[2], VIZ_CONFIG.glowOpacity);
            ellipse(screenX, screenY, VIZ_CONFIG.glowSize, VIZ_CONFIG.glowSize);
        }

        // Draw main point
        fill(fileColor[0], fileColor[1], fileColor[2], VIZ_CONFIG.pointOpacity);
        ellipse(screenX, screenY, VIZ_CONFIG.pointSize, VIZ_CONFIG.pointSize);
    }
}

// Draw subtle reference grid
function drawGrid() {
    const divisions = VIZ_CONFIG.gridDivisions;
    const opacity = VIZ_CONFIG.gridOpacity;

    stroke(255, 255, 255, opacity);
    strokeWeight(1);

    // Vertical lines
    for (let i = 1; i < divisions; i++) {
        const x = (i / divisions) * width;
        line(x, 0, x, height);
    }

    // Horizontal lines
    for (let i = 1; i < divisions; i++) {
        const y = (i / divisions) * height;
        line(0, y, width, y);
    }

    // Draw coordinate labels at edges (very subtle)
    fill(255, 255, 255, opacity + 10);
    noStroke();
    textSize(9);
    textAlign(CENTER, TOP);

    // X-axis labels (bottom)
    for (let i = 0; i <= divisions; i += 2) {
        const x = (i / divisions) * width;
        const val = (i / divisions).toFixed(1);
        text(val, x, height - 12);
    }

    // Y-axis labels (left)
    textAlign(LEFT, CENTER);
    for (let i = 0; i <= divisions; i += 2) {
        const y = height - (i / divisions) * height;
        const val = (i / divisions).toFixed(1);
        text(val, 4, y);
    }
}

function drawTrajectory() {
    if (trajectory.length < 2) return;

    noFill();
    const len = trajectory.length;
    const trajectoryColor = COLORS.trajectory;

    if (VIZ_CONFIG.smoothTrajectory && len >= 4) {
        // Draw smooth Catmull-Rom curve
        drawSmoothTrajectory(trajectoryColor);
    } else {
        // Draw standard line segments with exponential fade
        for (let i = 1; i < len; i++) {
            const t = i / len;
            const alpha = pow(t, VIZ_CONFIG.trailFadeExponent) * 255;
            const weight = map(t, 0, 1, VIZ_CONFIG.minLineWeight, VIZ_CONFIG.maxLineWeight);

            const [x1, y1] = trajectory[i - 1];
            const [x2, y2] = trajectory[i];

            stroke(trajectoryColor[0], trajectoryColor[1], trajectoryColor[2], alpha);
            strokeWeight(weight);

            const sx1 = x1 * width;
            const sy1 = (1 - y1) * height;
            const sx2 = x2 * width;
            const sy2 = (1 - y2) * height;

            line(sx1, sy1, sx2, sy2);
        }
    }
}

// Catmull-Rom spline interpolation for smooth trajectory
function drawSmoothTrajectory(trajectoryColor) {
    const len = trajectory.length;
    const steps = 4; // interpolation steps between points

    for (let i = 0; i < len - 1; i++) {
        // Get 4 control points for Catmull-Rom
        const p0 = trajectory[max(0, i - 1)];
        const p1 = trajectory[i];
        const p2 = trajectory[min(len - 1, i + 1)];
        const p3 = trajectory[min(len - 1, i + 2)];

        for (let s = 0; s < steps; s++) {
            const t1 = s / steps;
            const t2 = (s + 1) / steps;

            // Calculate progress along entire trajectory for alpha/weight
            const progress1 = (i + t1) / len;
            const progress2 = (i + t2) / len;

            const alpha = pow(progress2, VIZ_CONFIG.trailFadeExponent) * 255;
            const weight = map(progress2, 0, 1, VIZ_CONFIG.minLineWeight, VIZ_CONFIG.maxLineWeight);

            // Catmull-Rom interpolation
            const pt1 = catmullRom(p0, p1, p2, p3, t1);
            const pt2 = catmullRom(p0, p1, p2, p3, t2);

            stroke(trajectoryColor[0], trajectoryColor[1], trajectoryColor[2], alpha);
            strokeWeight(weight);

            const sx1 = pt1[0] * width;
            const sy1 = (1 - pt1[1]) * height;
            const sx2 = pt2[0] * width;
            const sy2 = (1 - pt2[1]) * height;

            line(sx1, sy1, sx2, sy2);
        }
    }
}

// Catmull-Rom spline point calculation
function catmullRom(p0, p1, p2, p3, t) {
    const t2 = t * t;
    const t3 = t2 * t;

    const x = 0.5 * (
        (2 * p1[0]) +
        (-p0[0] + p2[0]) * t +
        (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
        (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
    );

    const y = 0.5 * (
        (2 * p1[1]) +
        (-p0[1] + p2[1]) * t +
        (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
        (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
    );

    return [x, y];
}

function drawCursor() {
    if (!currentPosition) return;

    const [x, y] = currentPosition;
    const screenX = x * width;
    const screenY = (1 - y) * height;

    const cursorColor = COLORS.trajectory;
    const fileColor = COLORS.files[currentFileId % COLORS.files.length];

    // Calculate pulse effect
    let pulseScale = 1.0;
    if (VIZ_CONFIG.pulseEnabled) {
        pulseScale = 1.0 + sin(frameCount * VIZ_CONFIG.pulseSpeed) * VIZ_CONFIG.pulseAmount;
    }

    // Draw glow rings with pulse
    noStroke();
    for (let r = 30 * pulseScale; r > 0; r -= 5) {
        const alpha = map(r, 30 * pulseScale, 0, 10, 50);
        fill(cursorColor[0], cursorColor[1], cursorColor[2], alpha);
        ellipse(screenX, screenY, r, r);
    }

    // Draw file ID colored ring
    noFill();
    stroke(fileColor[0], fileColor[1], fileColor[2], 150);
    strokeWeight(3);
    ellipse(screenX, screenY, 26 * pulseScale, 26 * pulseScale);

    // Draw main cursor ring
    stroke(cursorColor[0], cursorColor[1], cursorColor[2]);
    strokeWeight(2);
    ellipse(screenX, screenY, 20 * pulseScale, 20 * pulseScale);

    // Draw velocity vector arrow
    if (VIZ_CONFIG.showVelocityVector && trajectory.length >= 2) {
        const len = trajectory.length;
        const [prevX, prevY] = trajectory[len - 2];
        const [currX, currY] = trajectory[len - 1];

        const dx = currX - prevX;
        const dy = currY - prevY;
        const mag = sqrt(dx * dx + dy * dy);

        if (mag > 0.001) {
            // Normalize and scale
            const scale = VIZ_CONFIG.velocityArrowScale * min(mag * 100, 1);
            const arrowX = (dx / mag) * scale;
            const arrowY = -(dy / mag) * scale; // Flip Y for screen coords

            // Draw arrow line
            stroke(255, 255, 255, 200);
            strokeWeight(2);
            line(screenX, screenY, screenX + arrowX, screenY + arrowY);

            // Draw arrowhead
            const angle = atan2(arrowY, arrowX);
            const arrowHeadSize = 8;
            const endX = screenX + arrowX;
            const endY = screenY + arrowY;

            fill(255, 255, 255, 200);
            noStroke();
            push();
            translate(endX, endY);
            rotate(angle);
            triangle(0, 0, -arrowHeadSize, -arrowHeadSize / 2, -arrowHeadSize, arrowHeadSize / 2);
            pop();
        }
    }

    // Draw center point
    fill(255);
    noStroke();
    ellipse(screenX, screenY, 6, 6);
}

function drawHeatmap() {
    noStroke();
    const res = heatmapResolution;
    const cellW = width / res;
    const cellH = height / res;

    // Find max value (skip zero cells for efficiency)
    let maxVal = 1;
    for (let i = 0; i < res; i++) {
        for (let j = 0; j < res; j++) {
            if (heatmapData[i][j] > 0) {
                maxVal = max(maxVal, heatmapData[i][j]);
            }
        }
    }

    // Draw cells with perceptually uniform viridis-like colormap
    for (let i = 0; i < res; i++) {
        for (let j = 0; j < res; j++) {
            const rawVal = heatmapData[i][j];
            if (rawVal < 0.01) continue; // Skip empty cells

            const val = rawVal / maxVal;
            const color = viridisColor(val);
            fill(color[0], color[1], color[2], map(val, 0, 1, 80, 220));
            rect(i * cellW, (res - 1 - j) * cellH, cellW + 1, cellH + 1); // +1 to avoid gaps
        }
    }
}

// Viridis-like perceptually uniform colormap
function viridisColor(t) {
    // Simplified viridis approximation (dark purple -> blue -> teal -> green -> yellow)
    t = constrain(t, 0, 1);

    let r, g, b;
    if (t < 0.25) {
        // Dark purple to blue
        const s = t / 0.25;
        r = map(s, 0, 1, 68, 59);
        g = map(s, 0, 1, 1, 82);
        b = map(s, 0, 1, 84, 139);
    } else if (t < 0.5) {
        // Blue to teal
        const s = (t - 0.25) / 0.25;
        r = map(s, 0, 1, 59, 33);
        g = map(s, 0, 1, 82, 145);
        b = map(s, 0, 1, 139, 140);
    } else if (t < 0.75) {
        // Teal to green
        const s = (t - 0.5) / 0.25;
        r = map(s, 0, 1, 33, 94);
        g = map(s, 0, 1, 145, 201);
        b = map(s, 0, 1, 140, 98);
    } else {
        // Green to yellow
        const s = (t - 0.75) / 0.25;
        r = map(s, 0, 1, 94, 253);
        g = map(s, 0, 1, 201, 231);
        b = map(s, 0, 1, 98, 37);
    }

    return [r, g, b];
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
        totalCorpusPoints = data.total_points || 0;

        // Pre-compute and cache colors for all corpus points (performance optimization)
        cachedCorpusColors = corpusFileIds.map(fileId =>
            COLORS.files[fileId % COLORS.files.length]
        );

        console.log(`Received corpus: ${corpusPoints.length} points`);

    } else if (data.type === 'state') {
        const nav = data.navigation || {};

        currentPosition = nav.position_2d || currentPosition;
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

        if (data.decoder) {
            updateControlDisplays('decoder', data.decoder);
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
}

function updateControlDisplays(prefix, values) {
    for (const [key, value] of Object.entries(values)) {
        // Skip updating controls that are currently being adjusted by the user
        const inputId = `${prefix}-${key}`;
        if (activeControls.has(inputId)) {
            continue;
        }

        const input = document.getElementById(inputId);

        const display = document.getElementById(`val-${key}`);

        // Handle boolean values (toggle switches)
        if (typeof value === 'boolean') {
            if (display) {
                display.textContent = value ? 'on' : 'off';
            }
            if (input && input.type === 'checkbox') {
                input.checked = value;
            }
            continue;
        }

        // Handle numeric values
        if (display && typeof value === 'number') {
            if (key === 'frame_samples' || key === 'underruns') {
                display.textContent = value.toFixed(0);
            } else {
                display.textContent = value.toFixed(2);
            }
        }

        // Also update the slider position to match server state
        if (input && typeof value === 'number' && input.type === 'range') {
            input.value = value;
        }
    }
}

// UI Control handlers
function setupControls() {
    // Policy controls
    const policyControls = ['width', 'energy', 'gravity', 'memory', 'coherence', 'exploration'];
    
    policyControls.forEach(ctrl => {
        const input = document.getElementById(`ctrl-${ctrl}`);
        const display = document.getElementById(`val-${ctrl}`);
        const inputId = `ctrl-${ctrl}`;

        if (input) {
            // Track when user starts adjusting
            input.addEventListener('mousedown', () => activeControls.add(inputId));
            input.addEventListener('touchstart', () => activeControls.add(inputId));

            // Track when user stops adjusting
            input.addEventListener('mouseup', () => setTimeout(() => activeControls.delete(inputId), 100));
            input.addEventListener('touchend', () => setTimeout(() => activeControls.delete(inputId), 100));
            input.addEventListener('mouseleave', () => setTimeout(() => activeControls.delete(inputId), 100));

            input.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                if (display) display.textContent = value.toFixed(2);
                sendControl(ctrl, value);
            });
        }
    });
    
    // Decoder controls
    const decoderControls = [
        { param: 'gain', format: 2 },
        { param: 'smoothing', format: 2 },
    ];

    decoderControls.forEach(({ param, format }) => {
        const input = document.getElementById(`decoder-${param}`);
        const display = document.getElementById(`val-${param}`);
        const inputId = `decoder-${param}`;

        if (input) {
            // Track when user starts adjusting
            input.addEventListener('mousedown', () => activeControls.add(inputId));
            input.addEventListener('touchstart', () => activeControls.add(inputId));

            // Track when user stops adjusting
            input.addEventListener('mouseup', () => setTimeout(() => activeControls.delete(inputId), 100));
            input.addEventListener('touchend', () => setTimeout(() => activeControls.delete(inputId), 100));
            input.addEventListener('mouseleave', () => setTimeout(() => activeControls.delete(inputId), 100));

            input.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                if (display) {
                    display.textContent = value.toFixed(format);
                }
                sendDecoderParam(param, value);
            });
        }
    });

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

function sendDecoderParam(name, value) {
    if (ws && wsConnected) {
        ws.send(JSON.stringify({
            type: 'decoder',
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
