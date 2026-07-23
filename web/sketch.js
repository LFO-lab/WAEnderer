/**
 * Stable Audio Wanderer - Real-time Visualization
 * p5.js sketch for visualizing navigation trajectories
 */

// Draw mode: 'perform' (existing 3D) or 'training' (loss curves)
let drawMode = 'perform';

function setDrawMode(mode) {
    drawMode = mode;
}

// WebSocket connection
let ws = null;
let wsConnected = false;
const WS_URL = 'ws://127.0.0.1:8765';

// Track which controls are currently being adjusted by the user
let activeControls = new Set();

// Corpus data
let manualCorpusPoints3D = [];
let manualCorpusColorValues = [];
let manualCorpusFileIds = [];
let manualCorpusIndices = [];

// Navigation state
let manualPosition3D = [0.5, 0.5, 0.5];
let manualTrajectory3D = [];
let currentIndex = 0;
let currentFileId = 0;
let selectedNavigationMode = 'random';
let transportRunning = false;
let manualNearestIndex = 0;
let manualNearestDistance = 0;
let manualFaders = new Array(4).fill(0.5);
let manualDecodeWindow = 6;
let manualDecodeWindowMin = 1;
let manualDecodeWindowMax = 64;
let manualWindowLastSent = 6;
let manualBufferRatio = 0.15;
let manualWanderK = 1;
let manualControlDim = 4;
const WANDER_FRAME_SOURCE_LABELS = {
    k_nearest: 'K Nearest',
    contiguous: 'Contiguous',
    morphology_graph: 'Morphology Graph',
};

// Manual 3D camera controls
let manualViewYawDeg = -34.4;
let manualViewPitchDeg = 20.1;
let manualViewDistance = 2.6;
let manualCameraSpeed = 1.0;
let manualPickDragging = false;
let manualColorA = [74, 158, 255];
let manualColorB = [255, 141, 74];
const MANUAL_CAMERA_STANDARD_DISTANCE = 2.6;
const MANUAL_CAMERA_MIN_DISTANCE = 0.08;
const MANUAL_CAMERA_MAX_DISTANCE = 8.0;
const MANUAL_CAMERA_ANGULAR_SPEED_DEG = 63.0;
const MANUAL_CAMERA_RADIAL_SPEED = 2.0;
const MANUAL_CAMERA_TOGGLE_DEFAULTS = {
    left: false,
    right: false,
    over: false,
    under: false,
    forward: false,
    backward: false,
};
let manualCameraToggles = { ...MANUAL_CAMERA_TOGGLE_DEFAULTS };

// Visualization settings
let trailLength = 64;

// Colors
const COLORS = {
    background: [13, 13, 20],
    trajectory: [74, 158, 255],
    files: [],
};

// Visualization configuration
const VIZ_CONFIG = {
    pointSize: 6,
    trailFadeExponent: 2.5,
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

    connectWebSocket();
    setupControls();
    applyNavigationModeUI();

    frameRate(30);
}

function currentModeUsesShared3DView() {
    return manualCorpusPoints3D.length > 0;
}

function draw() {
    background(...COLORS.background);

    if (drawMode === 'training') {
        drawTrainingCurves();
    } else if (currentModeUsesShared3DView()) {
        updateManualCameraMotion();
        drawManual3DScene();
    }

    noFill();
    stroke(40, 40, 60);
    strokeWeight(2);
    rect(0, 0, width, height);
}

function drawTrainingCurves() {
    const margin = { top: 40, right: 20, bottom: 50, left: 60 };
    const plotW = width - margin.left - margin.right;
    const plotH = (height - margin.top - margin.bottom - 30) / 2; // two charts stacked

    // Gather data from pipeline.js trainingHistory
    const rHist = (typeof trainingHistory !== 'undefined') ? trainingHistory : null;
    if (!rHist) {
        fill(160);
        noStroke();
        textSize(14);
        textAlign(CENTER, CENTER);
        text('Waiting for training data...', width / 2, height / 2);
        return;
    }

    const randomTrain = rHist.random.train_loss;
    const randomVal = rHist.random.val_loss;
    const windowMAE = rHist.random.window_mae;
    const reorgTrain = rHist.reorganized.train_loss;
    const reorgVal = rHist.reorganized.val_loss;
    const reorgAcc = rHist.reorganized.accuracy;

    const hasRandom = randomTrain.length > 0;
    const hasReorg = reorgTrain.length > 0;

    if (!hasRandom && !hasReorg) {
        fill(160);
        noStroke();
        textSize(14);
        textAlign(CENTER, CENTER);
        text('Waiting for training data...', width / 2, height / 2);
        return;
    }

    // Helper: draw a chart in a given region
    function drawChart(ox, oy, w, h, title, series, yLabel) {
        // Title
        fill(200);
        noStroke();
        textSize(12);
        textAlign(LEFT, TOP);
        text(title, ox, oy - 16);

        // Background
        fill(20, 20, 30);
        noStroke();
        rect(ox, oy, w, h, 4);

        if (series.length === 0) return;

        // Compute Y range across all series
        let yMin = Infinity, yMax = -Infinity;
        for (const s of series) {
            for (const v of s.data) {
                if (v < yMin) yMin = v;
                if (v > yMax) yMax = v;
            }
        }
        if (yMin === yMax) { yMin -= 0.1; yMax += 0.1; }
        const yPad = (yMax - yMin) * 0.08;
        yMin -= yPad;
        yMax += yPad;

        // Longest series for X axis
        let maxLen = 0;
        for (const s of series) {
            if (s.data.length > maxLen) maxLen = s.data.length;
        }

        // Grid lines
        stroke(40, 40, 55);
        strokeWeight(0.5);
        const nGridY = 4;
        for (let i = 0; i <= nGridY; i++) {
            const gy = oy + h - (i / nGridY) * h;
            line(ox, gy, ox + w, gy);
            // Y label
            fill(100);
            noStroke();
            textSize(9);
            textAlign(RIGHT, CENTER);
            const val = yMin + (i / nGridY) * (yMax - yMin);
            text(val.toFixed(3), ox - 4, gy);
            stroke(40, 40, 55);
            strokeWeight(0.5);
        }

        // Y axis label
        push();
        fill(120);
        noStroke();
        textSize(10);
        textAlign(CENTER, CENTER);
        translate(ox - 40, oy + h / 2);
        rotate(-HALF_PI);
        text(yLabel || '', 0, 0);
        pop();

        // X axis label
        fill(100);
        noStroke();
        textSize(9);
        textAlign(CENTER, TOP);
        text(`Epoch (${maxLen})`, ox + w / 2, oy + h + 4);

        // Draw each series
        for (const s of series) {
            if (s.data.length < 2) continue;
            stroke(s.color[0], s.color[1], s.color[2]);
            strokeWeight(1.5);
            noFill();
            beginShape();
            for (let i = 0; i < s.data.length; i++) {
                const px = ox + (i / (maxLen - 1)) * w;
                const py = oy + h - ((s.data[i] - yMin) / (yMax - yMin)) * h;
                vertex(px, py);
            }
            endShape();
        }

        // Legend
        const legendX = ox + w - 10;
        let legendY = oy + 10;
        textAlign(RIGHT, TOP);
        textSize(10);
        for (const s of series) {
            fill(s.color[0], s.color[1], s.color[2]);
            noStroke();
            text(s.label, legendX, legendY);
            legendY += 14;
        }
    }

    // Top chart: loss curves
    const lossSeries = [];
    if (hasRandom) {
        lossSeries.push({ label: 'Train Loss', data: randomTrain, color: [74, 158, 255] });
        lossSeries.push({ label: 'Val Loss', data: randomVal, color: [255, 100, 100] });
    }
    if (hasReorg) {
        lossSeries.push({ label: 'Reorg Train', data: reorgTrain, color: [100, 255, 180] });
        lossSeries.push({ label: 'Reorg Val', data: reorgVal, color: [255, 200, 100] });
    }
    drawChart(margin.left, margin.top, plotW, plotH, 'Loss', lossSeries, 'Loss');

    // Bottom chart: secondary metrics (window MAE or accuracy)
    const secSeries = [];
    if (windowMAE.length > 0) {
        secSeries.push({ label: 'Window MAE (frames)', data: windowMAE, color: [180, 130, 255] });
    }
    if (reorgAcc.length > 0) {
        secSeries.push({ label: 'Reorg Accuracy', data: reorgAcc, color: [255, 220, 100] });
    }
    if (secSeries.length > 0) {
        const secY = margin.top + plotH + 30;
        drawChart(margin.left, secY, plotW, plotH, 'Metrics', secSeries, 'Value');
    }
}

function projectManualPoint3D(point) {
    const p = point || [0.5, 0.5, 0.5];
    const x = (p[0] - 0.5) * 2.0;
    const y = (p[1] - 0.5) * 2.0;
    const z = (p[2] - 0.5) * 2.0;

    const yaw = radians(manualViewYawDeg);
    const pitch = radians(manualViewPitchDeg);
    const cy = cos(yaw);
    const sy = sin(yaw);
    const cp = cos(pitch);
    const sp = sin(pitch);

    const x1 = x * cy + z * sy;
    const z1 = -x * sy + z * cy;

    const y2 = y * cp - z1 * sp;
    const z2 = y * sp + z1 * cp;

    const depth = z2 + manualViewDistance;
    const perspective = 1.0 / max(0.25, depth);

    const screenX = width * 0.5 + x1 * perspective * width * 0.9;
    const screenY = height * 0.5 - y2 * perspective * height * 0.9;

    return {
        x: screenX,
        y: screenY,
        depth,
        perspective,
    };
}

function cameraToWorldVector(vx, vy, vz) {
    const yaw = radians(manualViewYawDeg);
    const pitch = radians(manualViewPitchDeg);
    const cy = cos(yaw);
    const sy = sin(yaw);
    const cp = cos(pitch);
    const sp = sin(pitch);

    const y1 = vy * cp + vz * sp;
    const z1 = -vy * sp + vz * cp;

    const xw = vx * cy - z1 * sy;
    const zw = vx * sy + z1 * cy;
    return [xw, y1, zw];
}

function getManualRayFromScreen(mx, my) {
    const u = (mx - width * 0.5) / (width * 0.9);
    const v = -(my - height * 0.5) / (height * 0.9);

    const origin = cameraToWorldVector(0, 0, -manualViewDistance);
    const dirRaw = cameraToWorldVector(u, v, 1.0);
    const mag = sqrt(dirRaw[0] * dirRaw[0] + dirRaw[1] * dirRaw[1] + dirRaw[2] * dirRaw[2]);
    const dir = mag > 1e-6 ? [dirRaw[0] / mag, dirRaw[1] / mag, dirRaw[2] / mag] : [0, 0, 1];
    return { origin, dir };
}

function findClosestManualPointIndex(mx, my) {
    if (!manualCorpusPoints3D || manualCorpusPoints3D.length === 0) {
        return -1;
    }
    const ray = getManualRayFromScreen(mx, my);

    let bestIdx = -1;
    let bestScore = Infinity;
    for (let i = 0; i < manualCorpusPoints3D.length; i++) {
        const pt = manualCorpusPoints3D[i];
        const px = (pt[0] - 0.5) * 2.0;
        const py = (pt[1] - 0.5) * 2.0;
        const pz = (pt[2] - 0.5) * 2.0;

        const dx = px - ray.origin[0];
        const dy = py - ray.origin[1];
        const dz = pz - ray.origin[2];
        const t = dx * ray.dir[0] + dy * ray.dir[1] + dz * ray.dir[2];
        if (t <= 0.0) continue;

        const cx = ray.origin[0] + t * ray.dir[0];
        const cy = ray.origin[1] + t * ray.dir[1];
        const cz = ray.origin[2] + t * ray.dir[2];
        const ex = px - cx;
        const ey = py - cy;
        const ez = pz - cz;
        const score = ex * ex + ey * ey + ez * ez;

        if (score < bestScore) {
            bestScore = score;
            bestIdx = i;
        }
    }

    return bestIdx;
}

function applyManualPointSelection(pointIdx) {
    if (
        !Number.isInteger(pointIdx)
        || pointIdx < 0
        || pointIdx >= manualCorpusPoints3D.length
    ) {
        return;
    }

    const selected = manualCorpusPoints3D[pointIdx];
    const nextFaders = new Array(manualControlDim).fill(0.5);
    for (let i = 0; i < Math.min(3, manualControlDim); i++) {
        nextFaders[i] = constrain(selected[i], 0, 1);
    }
    if (manualControlDim >= 4 && pointIdx < manualCorpusColorValues.length) {
        const colorValue = manualCorpusColorValues[pointIdx];
        if (Number.isFinite(colorValue)) {
            nextFaders[3] = constrain(colorValue, 0, 1);
        }
    }
    manualFaders = nextFaders;
    for (let i = 0; i < manualControlDim; i++) {
        const display = document.getElementById(`val-manual-${i}`);
        if (display) display.textContent = manualFaders[i].toFixed(2);
        const input = document.getElementById(`manual-${i}`);
        if (input && !activeControls.has(`manual-${i}`)) input.value = manualFaders[i];
    }
    sendManualControls();
}

function sendCursorFor3DPoint(pointIdx) {
    if (!(ws && wsConnected)) {
        return;
    }
    if (!Number.isInteger(pointIdx) || pointIdx < 0 || pointIdx >= manualCorpusPoints3D.length) {
        return;
    }
    const corpusIndex = (
        pointIdx < manualCorpusIndices.length
            ? Number(manualCorpusIndices[pointIdx])
            : pointIdx
    );
    ws.send(JSON.stringify({
        type: 'cursor_index',
        index: corpusIndex,
    }));
}

function updateManualCameraButtonStates() {
    const cameraButtons = [
        { id: 'cam-left', dir: 'left' },
        { id: 'cam-right', dir: 'right' },
        { id: 'cam-over', dir: 'over' },
        { id: 'cam-under', dir: 'under' },
        { id: 'cam-forward', dir: 'forward' },
        { id: 'cam-backward', dir: 'backward' },
    ];
    for (const button of cameraButtons) {
        const el = document.getElementById(button.id);
        if (!el) continue;
        el.classList.toggle('active', Boolean(manualCameraToggles[button.dir]));
    }
}

function resetManualCameraToggles() {
    manualCameraToggles = { ...MANUAL_CAMERA_TOGGLE_DEFAULTS };
    updateManualCameraButtonStates();
}

function setManualCameraToggle(direction, enabled) {
    const opposite = {
        left: 'right',
        right: 'left',
        over: 'under',
        under: 'over',
        forward: 'backward',
        backward: 'forward',
    };
    if (!(direction in manualCameraToggles)) {
        return;
    }
    if (enabled) {
        manualCameraToggles[direction] = true;
        const oppositeDirection = opposite[direction];
        if (oppositeDirection) {
            manualCameraToggles[oppositeDirection] = false;
        }
    } else {
        manualCameraToggles[direction] = false;
    }
    updateManualCameraButtonStates();
}

function wrapManualCameraDistance() {
    if (!Number.isFinite(manualViewDistance)) {
        manualViewDistance = MANUAL_CAMERA_STANDARD_DISTANCE;
    }
    if (
        manualViewDistance <= MANUAL_CAMERA_MIN_DISTANCE
        || manualViewDistance >= MANUAL_CAMERA_MAX_DISTANCE
    ) {
        manualViewDistance = MANUAL_CAMERA_STANDARD_DISTANCE;
    }
}

function wrapAngleDegrees(value) {
    let wrapped = value % 360.0;
    if (wrapped < 0.0) {
        wrapped += 360.0;
    }
    return wrapped;
}

function updateManualCameraMotion() {
    const dt = max(0, deltaTime) / 1000.0;
    if (dt <= 0) {
        return;
    }

    const yawDir = (manualCameraToggles.right ? 1 : 0) - (manualCameraToggles.left ? 1 : 0);
    const pitchDir = (manualCameraToggles.over ? 1 : 0) - (manualCameraToggles.under ? 1 : 0);
    const radialDir = (manualCameraToggles.backward ? 1 : 0) - (manualCameraToggles.forward ? 1 : 0);
    if (yawDir === 0 && pitchDir === 0 && radialDir === 0) {
        return;
    }

    const speed = max(0, manualCameraSpeed);
    manualViewYawDeg = wrapAngleDegrees(
        manualViewYawDeg + yawDir * MANUAL_CAMERA_ANGULAR_SPEED_DEG * speed * dt
    );
    manualViewPitchDeg = wrapAngleDegrees(
        manualViewPitchDeg + pitchDir * MANUAL_CAMERA_ANGULAR_SPEED_DEG * speed * dt
    );
    manualViewDistance += radialDir * MANUAL_CAMERA_RADIAL_SPEED * speed * dt;
    wrapManualCameraDistance();
}

function drawManual3DScene() {
    drawManualAxes();

    if (manualCorpusPoints3D.length > 0) {
        const hasColorAxis = manualCorpusColorValues.length === manualCorpusPoints3D.length;
        const projected = [];
        for (let i = 0; i < manualCorpusPoints3D.length; i++) {
            projected.push({
                i,
                p: projectManualPoint3D(manualCorpusPoints3D[i]),
            });
        }
        projected.sort((a, b) => b.p.depth - a.p.depth);

        noStroke();
        for (const item of projected) {
            let fileColor;
            if (hasColorAxis) {
                const colorValue = manualCorpusColorValues[item.i];
                const blend = constrain(
                    Number.isFinite(colorValue) ? colorValue : 0.5,
                    0.0,
                    1.0
                );
                fileColor = [
                    lerp(manualColorA[0], manualColorB[0], blend),
                    lerp(manualColorA[1], manualColorB[1], blend),
                    lerp(manualColorA[2], manualColorB[2], blend),
                ];
            } else {
                const fileId = manualCorpusFileIds[item.i] || 0;
                fileColor = COLORS.files[fileId % COLORS.files.length] || COLORS.files[0];
            }
            const pointSize = constrain(VIZ_CONFIG.pointSize * item.p.perspective * 4.0, 1.0, 12.0);
            const alpha = map(pointSize, 1.0, 12.0, 30, 160);
            fill(fileColor[0], fileColor[1], fileColor[2], alpha);
            ellipse(item.p.x, item.p.y, pointSize, pointSize);
        }
    }

    drawManualTrajectory3D();
    drawManualCursor3D();
}

function drawManualAxes() {
    const axisPoints = {
        x0: projectManualPoint3D([0.0, 0.5, 0.5]),
        x1: projectManualPoint3D([1.0, 0.5, 0.5]),
        y0: projectManualPoint3D([0.5, 0.0, 0.5]),
        y1: projectManualPoint3D([0.5, 1.0, 0.5]),
        z0: projectManualPoint3D([0.5, 0.5, 0.0]),
        z1: projectManualPoint3D([0.5, 0.5, 1.0]),
    };

    strokeWeight(1.5);
    stroke(245, 110, 110, 120);
    line(axisPoints.x0.x, axisPoints.x0.y, axisPoints.x1.x, axisPoints.x1.y);
    stroke(110, 235, 180, 120);
    line(axisPoints.y0.x, axisPoints.y0.y, axisPoints.y1.x, axisPoints.y1.y);
    stroke(110, 170, 245, 120);
    line(axisPoints.z0.x, axisPoints.z0.y, axisPoints.z1.x, axisPoints.z1.y);
}

function drawManualTrajectory3D() {
    if (manualTrajectory3D.length < 2) return;
    const color = COLORS.trajectory;
    noFill();
    for (let i = 1; i < manualTrajectory3D.length; i++) {
        const a = projectManualPoint3D(manualTrajectory3D[i - 1]);
        const b = projectManualPoint3D(manualTrajectory3D[i]);
        const t = i / manualTrajectory3D.length;
        const alpha = pow(t, VIZ_CONFIG.trailFadeExponent) * 180;
        const weight = map(t, 0, 1, 1, 3);
        stroke(color[0], color[1], color[2], alpha);
        strokeWeight(weight);
        line(a.x, a.y, b.x, b.y);
    }
}

function drawManualCursor3D() {
    if (!manualPosition3D) return;
    const proj = projectManualPoint3D(manualPosition3D);
    const pulse = 1.0 + sin(frameCount * 0.05) * 0.2;
    const ring = constrain(24 * proj.perspective * pulse, 10, 28);

    noFill();
    stroke(COLORS.trajectory[0], COLORS.trajectory[1], COLORS.trajectory[2], 220);
    strokeWeight(2.5);
    ellipse(proj.x, proj.y, ring, ring);

    noStroke();
    fill(255, 255, 255, 230);
    ellipse(proj.x, proj.y, max(4, ring * 0.2), max(4, ring * 0.2));
}

// WebSocket handling
function connectWebSocket() {
    try {
        ws = new WebSocket(WS_URL);
        
        ws.onopen = () => {
            wsConnected = true;
            updateConnectionStatus(true);
            console.log('WebSocket connected');
            sendTransportSetMode(selectedNavigationMode);
            sendManualControls();
            // Request pipeline state if running via serve.py
            if (typeof requestCorpusList === 'function') {
                requestCorpusList();
            }
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

function applyNavigationCursor(nav, authoritativeTrajectory = false) {
    const nextIndex = Number(nav.index);
    currentIndex = Number.isFinite(nextIndex) ? nextIndex : 0;
    const nextFileId = Number(nav.file_id);
    currentFileId = Number.isFinite(nextFileId) ? nextFileId : 0;

    const clampedIdx = constrain(
        Math.round(currentIndex),
        0,
        Math.max(manualCorpusPoints3D.length - 1, 0)
    );
    if (Array.isArray(nav.position_3d) && nav.position_3d.length === 3) {
        manualPosition3D = nav.position_3d.slice(0, 3);
    } else if (manualCorpusPoints3D.length > 0) {
        manualPosition3D = manualCorpusPoints3D[clampedIdx];
    }

    if (authoritativeTrajectory) {
        const trajectory = Array.isArray(nav.trajectory_3d) ? nav.trajectory_3d : [];
        manualTrajectory3D = trajectory
            .filter(pt => Array.isArray(pt) && pt.length >= 3)
            .map(pt => pt.slice(0, 3));
    } else if (Array.isArray(nav.trajectory_3d) && nav.trajectory_3d.length > 0) {
        manualTrajectory3D = nav.trajectory_3d
            .filter(pt => Array.isArray(pt) && pt.length >= 3)
            .map(pt => pt.slice(0, 3));
    } else if (manualPosition3D) {
        const last = manualTrajectory3D[manualTrajectory3D.length - 1];
        const cursorChanged = !last || last.some(
            (value, axis) => value !== manualPosition3D[axis]
        );
        if (cursorChanged) {
            manualTrajectory3D.push(manualPosition3D.slice(0, 3));
            if (manualTrajectory3D.length > trailLength) {
                manualTrajectory3D = manualTrajectory3D.slice(-trailLength);
            }
        }
    }
}

function handleMessage(data) {
    // Route pipeline messages to pipeline.js handler
    if (data.type && data.type.startsWith('pipeline_')) {
        if (typeof handlePipelineMessage === 'function') {
            handlePipelineMessage(data);
        }
        return;
    }

    if (data.type === 'corpus') {
        manualCorpusPoints3D = data.manual_positions_3d || [];
        manualCorpusColorValues = data.manual_color_values || [];
        manualCorpusFileIds = data.manual_file_ids || data.file_ids || [];
        manualCorpusIndices = data.point_indices || [];
        if (typeof data.navigation_mode === 'string') {
            selectedNavigationMode = data.navigation_mode;
            applyNavigationModeUI();
        }

        console.log(
            `Received corpus: manual3d=${manualCorpusPoints3D.length}, manualColor=${manualCorpusColorValues.length}`
        );

        // Auto-switch to perform tab when corpus arrives in idle state (legacy server)
        if (typeof selectTab === 'function' && pipelinePhase === 'idle') {
            selectTab('perform');
        }

    } else if (data.type === 'state') {
        const nav = data.navigation || {};
        const transport = data.transport || {};
        const manual = data.manual || {};
        const hasAudioRenderCursor = nav.clock === 'audio_render';

        if (typeof nav.mode === 'string') {
            selectedNavigationMode = nav.mode;
        } else if (typeof transport.selected_mode === 'string') {
            selectedNavigationMode = transport.selected_mode;
        }
        transportRunning = Boolean(transport.running);
        applyNavigationModeUI();
        updateConnectionStatus(wsConnected);

        if (selectedNavigationMode === 'manual') {
            manualNearestIndex = manual.nearest_index || 0;
            manualNearestDistance = manual.distance || 0;
            manualControlDim = Number(manual.control_dim || 4);
            if (hasAudioRenderCursor) {
                applyNavigationCursor(nav, true);
            } else {
                const clampedIdx = constrain(
                    manualNearestIndex,
                    0,
                    Math.max(manualCorpusPoints3D.length - 1, 0)
                );
                currentIndex = clampedIdx;
                if (manualCorpusPoints3D.length > 0) {
                    manualPosition3D = manualCorpusPoints3D[clampedIdx];
                }
                if (Array.isArray(manual.position_3d) && manual.position_3d.length === 3) {
                    manualPosition3D = manual.position_3d.slice(0, 3);
                }
                currentFileId = manual.current_file_id ?? manualCorpusFileIds[clampedIdx] ?? 0;
                const last = manualTrajectory3D[manualTrajectory3D.length - 1];
                const cursorChanged = !last || last.some(
                    (value, axis) => value !== manualPosition3D[axis]
                );
                if (cursorChanged) {
                    manualTrajectory3D.push(manualPosition3D.slice(0, 3));
                    if (manualTrajectory3D.length > trailLength) {
                        manualTrajectory3D = manualTrajectory3D.slice(-trailLength);
                    }
                }
            }
            if (Array.isArray(manual.faders) && manual.faders.length === manualControlDim) {
                manualFaders = manual.faders.slice(0, manualControlDim);
                for (let i = 0; i < manualControlDim; i++) {
                    const display = document.getElementById(`val-manual-${i}`);
                    if (display) display.textContent = manualFaders[i].toFixed(2);
                    const input = document.getElementById(`manual-${i}`);
                    if (input && !activeControls.has(`manual-${i}`)) input.value = manualFaders[i];
                }
            }
            if (typeof manual.decode_window_min === 'number') {
                manualDecodeWindowMin = Math.round(manual.decode_window_min);
            }
            if (typeof manual.decode_window_max === 'number') {
                manualDecodeWindowMax = Math.round(manual.decode_window_max);
            }
            if (typeof manual.decode_window === 'number') {
                manualDecodeWindow = Math.round(manual.decode_window);
                manualWindowLastSent = manualDecodeWindow;
                const display = document.getElementById('val-manual-window');
                if (display) display.textContent = String(manualDecodeWindow);
            }
            if (typeof manual.buffer_ratio === 'number') {
                manualBufferRatio = Number(manual.buffer_ratio);
                const input = document.getElementById('manual-buffer-ratio');
                if (input && !activeControls.has('manual-buffer-ratio')) {
                    input.value = manualBufferRatio.toFixed(2);
                }
            }
            if (typeof manual.dither === 'number') {
                const ditherVal = Number(manual.dither);
                if (!activeControls.has('manual-dither')) {
                    const display = document.getElementById('val-manual-dither');
                    if (display) display.textContent = ditherVal.toFixed(2);
                    const input = document.getElementById('manual-dither');
                    if (input) input.value = String(ditherVal);
                }
            }
            if (typeof manual.wander_k === 'number') {
                manualWanderK = Math.round(Number(manual.wander_k));
                const display = document.getElementById('val-manual-wander-k');
                if (display) display.textContent = String(manualWanderK);
                const input = document.getElementById('manual-wander-k');
                if (input && !activeControls.has('manual-wander-k')) {
                    input.value = String(manualWanderK);
                }
            }
            const windowInput = document.getElementById('manual-window');
            if (windowInput) {
                windowInput.min = String(manualDecodeWindowMin);
                windowInput.max = String(manualDecodeWindowMax);
                if (!activeControls.has('manual-window')) {
                    windowInput.value = String(manualDecodeWindow);
                }
            }
        } else {
            applyNavigationCursor(nav, hasAudioRenderCursor);
        }

        updateInfoDisplay(data);
        
        if (data.controls) {
            if (data.controls.random) {
                updateControlDisplays('random', data.controls.random, 'random');
            }
            if (data.controls.reorganized) {
                updateControlDisplays(
                    'reorganized',
                    data.controls.reorganized,
                    'reorganized'
                );
            }
        }

        if (data.wander_render) {
            updateWanderRenderControls(data.wander_render);
        }

        if (data.decoder) {
            updateControlDisplays('decoder', data.decoder);
            updateDecoderRuntimeDisplay(data.decoder);
            if (typeof updateDecoderWindowState === 'function') {
                updateDecoderWindowState(data.decoder);
            }
        }
    }
}

function updateConnectionStatus(connected) {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    
    if (connected) {
        dot.classList.add('connected');
        const runStatus = transportRunning ? 'Running' : 'Idle';
        const modeLabel = selectedNavigationMode === 'random'
            ? 'Wander'
            : selectedNavigationMode;
        text.textContent = `Connected (${modeLabel}, ${runStatus})`;
    } else {
        dot.classList.remove('connected');
        text.textContent = 'Disconnected';
    }
}

function updateInfoDisplay(data) {
    const nav = data.navigation || {};
    if (selectedNavigationMode === 'manual') {
        const audibleIndex = nav.clock === 'audio_render'
            ? nav.index
            : manualNearestIndex;
        const audibleFileId = nav.clock === 'audio_render'
            ? nav.file_id
            : currentFileId;
        document.getElementById('info-index').textContent = audibleIndex ?? 0;
        document.getElementById('info-velocity').textContent = Number(manualNearestDistance || 0).toFixed(2);
        document.getElementById('info-file').textContent = audibleFileId ?? 0;
        return;
    }

    document.getElementById('info-index').textContent = nav.index ?? 0;
    document.getElementById('info-velocity').textContent = (nav.velocity || 0).toFixed(2);
    document.getElementById('info-file').textContent = nav.file_id ?? 0;
}

function updateDecoderRuntimeDisplay(decoder) {
    const setText = (id, value, title = null) => {
        const element = document.getElementById(id);
        if (!element) return;
        element.textContent = value;
        if (title !== null) element.title = title;
    };
    const firstValue = (...values) => values.find(value => value !== undefined && value !== null);
    const windows = Array.isArray(decoder.supported_windows)
        ? decoder.supported_windows
            .map(value => Number(String(value).replace(/^T/i, '')))
            .filter(value => Number.isInteger(value) && value > 0)
        : [];
    const selectedWindow = Number(firstValue(
        decoder.requested_window,
        decoder.pending_window,
        decoder.transition?.target_window,
        decoder.selected_window,
        decoder.decoder_window,
        decoder.window_size
    ));

    setText('info-decoder-backend', String(firstValue(decoder.backend, '--')));
    setText('info-decoder-provider', String(firstValue(decoder.provider, '--')));

    let windowText = Number.isInteger(selectedWindow) ? `T${selectedWindow}` : '--';
    if (windows.length > 0) {
        windowText += ` / ${windows.map(value => `T${value}`).join(',')}`;
    }
    setText('info-decoder-window', windowText);

    const latentHop = Number(firstValue(decoder.latent_hop, decoder.latent_hop_frames));
    const audioHop = Number(firstValue(decoder.audio_hop_samples, decoder.hop_samples));
    const hopParts = [];
    if (Number.isFinite(latentHop)) hopParts.push(`L${latentHop}`);
    if (Number.isFinite(audioHop)) hopParts.push(`A${Math.round(audioHop)}`);
    setText('info-decoder-hop', hopParts.length > 0 ? hopParts.join(' · ') : '--');

    const timing = firstValue(
        decoder.decode_timing,
        decoder.decode_timing_ms,
        decoder.last_decode_ms,
        decoder.decode_ms
    );
    let timingText = '--';
    if (typeof timing === 'number' && Number.isFinite(timing)) {
        timingText = `${timing.toFixed(1)} ms`;
    } else if (timing && typeof timing === 'object') {
        const lastMs = Number(firstValue(timing.last_ms, timing.latest_ms, timing.decode_ms));
        const p99Ms = Number(firstValue(timing.p99_ms, timing.p99));
        const parts = [];
        if (Number.isFinite(lastMs)) parts.push(`${lastMs.toFixed(1)} ms`);
        if (Number.isFinite(p99Ms)) parts.push(`p99 ${p99Ms.toFixed(1)}`);
        if (parts.length > 0) timingText = parts.join(' · ');
    }
    setText('info-decoder-timing', timingText);

    const transition = firstValue(decoder.transition_status, decoder.transition);
    let transitionText = 'idle';
    if (typeof transition === 'string') {
        transitionText = transition;
    } else if (typeof transition === 'boolean') {
        transitionText = transition ? 'staging' : 'idle';
    } else if (transition && typeof transition === 'object') {
        transitionText = String(firstValue(transition.status, transition.state, 'staging'));
        const target = Number(firstValue(transition.target_window, transition.window));
        if (Number.isInteger(target)) transitionText += ` T${target}`;
    }
    setText('info-decoder-transition', transitionText);

    const underruns = Number(firstValue(decoder.underruns, 0));
    setText('info-decoder-underruns', Number.isFinite(underruns) ? String(Math.round(underruns)) : '--');

    const resourcePath = String(firstValue(
        decoder.resource_path,
        decoder.model_path,
        ''
    ));
    const resourceName = resourcePath
        ? (resourcePath.split(/[\\/]/).pop() || resourcePath)
        : 'app-owned';
    setText('info-decoder-resource', resourceName, resourcePath);

    const decoderError = firstValue(decoder.error, decoder.runtime_error, '');
    setText('info-decoder-error', decoderError ? String(decoderError) : '--');

    // The unified presentation pipeline uses the manifest window globally;
    // standalone perform.py keeps its legacy manual-window control.
    const legacyManualWindow = document.getElementById('manual-window');
    if (legacyManualWindow) {
        const webOnnxMode = decoder.backend === 'onnxruntime' || Boolean(resourcePath);
        legacyManualWindow.disabled = webOnnxMode;
        legacyManualWindow.title = webOnnxMode
            ? 'Managed by the ONNX decoder window selector'
            : '';
    }
}

function updateControlDisplays(prefix, values, displayPrefix = null) {
    for (const [key, value] of Object.entries(values)) {
        // Skip updating controls that are currently being adjusted by the user
        const inputId = `${prefix}-${key}`;
        if (activeControls.has(inputId)) {
            continue;
        }

        const input = document.getElementById(inputId);
        const displayId = displayPrefix ? `val-${displayPrefix}-${key}` : `val-${key}`;
        const display = document.getElementById(displayId);

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

function updateWanderRenderControls(values) {
    if (!values || typeof values !== 'object') return;

    const requestedSource = (
        values.requested_frame_source ??
        values.requested_source ??
        values.frame_source
    );
    const effectiveSource = (
        values.effective_frame_source ??
        values.effective_source ??
        requestedSource
    );
    const sourceInput = document.getElementById('wander-frame-source');
    if (
        sourceInput &&
        Object.hasOwn(WANDER_FRAME_SOURCE_LABELS, requestedSource)
    ) {
        sourceInput.value = requestedSource;
    }

    const numericControls = [
        ['frame_order', 'wander-frame-order', 'val-wander-frame-order'],
        ['latent_colour', 'wander-latent-colour', 'val-wander-latent-colour'],
    ];
    numericControls.forEach(([key, inputId, displayId]) => {
        const value = Number(values[key]);
        if (!Number.isFinite(value) || activeControls.has(inputId)) return;
        const clamped = Math.max(0, Math.min(1, value));
        const input = document.getElementById(inputId);
        const display = document.getElementById(displayId);
        if (input) input.value = String(clamped);
        if (display) display.textContent = clamped.toFixed(2);
    });

    const status = document.getElementById('wander-render-status');
    if (!status) return;
    const requestedLabel = WANDER_FRAME_SOURCE_LABELS[requestedSource] || requestedSource;
    const effectiveLabel = WANDER_FRAME_SOURCE_LABELS[effectiveSource] || effectiveSource;
    if (!effectiveLabel) {
        status.textContent = 'Effective source unavailable';
        return;
    }

    const parts = [`Effective: ${effectiveLabel}`];
    if (requestedLabel && requestedSource !== effectiveSource) {
        parts.push(`requested ${requestedLabel}`);
    }
    if (typeof values.graph_available === 'boolean') {
        parts.push(values.graph_available ? 'graph ready' : 'graph unavailable');
    }
    status.textContent = parts.join(' · ');
}

// UI Control handlers
function setupControls() {
    const modeRandomBtn = document.getElementById('mode-random');
    const modeReorganizedBtn = document.getElementById('mode-reorganized');
    const modeManualBtn = document.getElementById('mode-manual');

    modeRandomBtn.addEventListener('click', () => {
        if (transportRunning) return;
        selectedNavigationMode = 'random';
        applyNavigationModeUI();
        sendTransportSetMode('random');
    });

    modeReorganizedBtn.addEventListener('click', () => {
        if (transportRunning) return;
        selectedNavigationMode = 'reorganized';
        applyNavigationModeUI();
        sendTransportSetMode('reorganized');
    });

    modeManualBtn.addEventListener('click', () => {
        if (transportRunning) return;
        selectedNavigationMode = 'manual';
        applyNavigationModeUI();
        sendTransportSetMode('manual');
    });

    document.getElementById('btn-start').addEventListener('click', () => {
        if (
            typeof pipelineCanStartTransport === 'function' &&
            !pipelineCanStartTransport()
        ) {
            return;
        }
        sendTransportAction('start');
    });

    document.getElementById('btn-stop').addEventListener('click', () => {
        sendTransportAction('stop');
    });

    // Random controls
    const randomControls = [
        'phrase_scale',
        'jump_rate',
        'timbre_lock',
        'drift',
        'repeat_avoid',
        'crossfile',
    ];

    randomControls.forEach(ctrl => {
        const input = document.getElementById(`random-${ctrl}`);
        const display = document.getElementById(`val-random-${ctrl}`);
        const inputId = `random-${ctrl}`;

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
                sendRandomControl(ctrl, value);
            });
        }
    });

    const wanderFrameSource = document.getElementById('wander-frame-source');
    if (wanderFrameSource) {
        wanderFrameSource.addEventListener('change', (e) => {
            sendWanderRenderControl('frame_source', e.target.value);
        });
    }

    [
        ['frame_order', 'wander-frame-order', 'val-wander-frame-order'],
        ['latent_colour', 'wander-latent-colour', 'val-wander-latent-colour'],
    ].forEach(([key, inputId, displayId]) => {
        const input = document.getElementById(inputId);
        const display = document.getElementById(displayId);
        if (!input) return;

        input.addEventListener('mousedown', () => activeControls.add(inputId));
        input.addEventListener('touchstart', () => activeControls.add(inputId));
        input.addEventListener('mouseup', () => setTimeout(() => activeControls.delete(inputId), 100));
        input.addEventListener('touchend', () => setTimeout(() => activeControls.delete(inputId), 100));
        input.addEventListener('mouseleave', () => setTimeout(() => activeControls.delete(inputId), 100));
        input.addEventListener('input', (e) => {
            const numeric = Number(e.target.value);
            if (!Number.isFinite(numeric)) return;
            const value = Math.max(0, Math.min(1, numeric));
            if (display) display.textContent = value.toFixed(2);
            sendWanderRenderControl(key, value);
        });
    });

    // Reorganized controls
    const reorganizedControls = [
        'morph_len',
        'jump_rate',
        'timbre_lock',
        'evolution',
        'novelty',
        'crossfile',
    ];
    reorganizedControls.forEach(ctrl => {
        const input = document.getElementById(`reorganized-${ctrl}`);
        const display = document.getElementById(`val-reorganized-${ctrl}`);
        const inputId = `reorganized-${ctrl}`;

        if (input) {
            input.addEventListener('mousedown', () => activeControls.add(inputId));
            input.addEventListener('touchstart', () => activeControls.add(inputId));
            input.addEventListener('mouseup', () => setTimeout(() => activeControls.delete(inputId), 100));
            input.addEventListener('touchend', () => setTimeout(() => activeControls.delete(inputId), 100));
            input.addEventListener('mouseleave', () => setTimeout(() => activeControls.delete(inputId), 100));

            input.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                if (display) display.textContent = value.toFixed(2);
                sendReorganizedControl(ctrl, value);
            });
        }
    });

    // Manual controls (X/Y/Z/W)
    for (let i = 0; i < manualFaders.length; i++) {
        const input = document.getElementById(`manual-${i}`);
        const display = document.getElementById(`val-manual-${i}`);
        const inputId = `manual-${i}`;

        if (input) {
            input.addEventListener('mousedown', () => activeControls.add(inputId));
            input.addEventListener('touchstart', () => activeControls.add(inputId));
            input.addEventListener('mouseup', () => setTimeout(() => activeControls.delete(inputId), 100));
            input.addEventListener('touchend', () => setTimeout(() => activeControls.delete(inputId), 100));
            input.addEventListener('mouseleave', () => setTimeout(() => activeControls.delete(inputId), 100));

            input.addEventListener('input', (e) => {
                if (i >= manualControlDim) {
                    return;
                }
                const value = parseFloat(e.target.value);
                const next = manualFaders.slice(0, manualControlDim);
                while (next.length < manualControlDim) {
                    next.push(0.5);
                }
                next[i] = value;
                manualFaders = next;
                if (display) display.textContent = value.toFixed(2);
                sendManualControls();
            });
        }
    }

    const manualWindowInput = document.getElementById('manual-window');
    const manualWindowDisplay = document.getElementById('val-manual-window');
    if (manualWindowInput) {
        const inputId = 'manual-window';
        manualWindowInput.addEventListener('mousedown', () => activeControls.add(inputId));
        manualWindowInput.addEventListener('touchstart', () => activeControls.add(inputId));
        manualWindowInput.addEventListener('mouseup', () => setTimeout(() => activeControls.delete(inputId), 100));
        manualWindowInput.addEventListener('touchend', () => setTimeout(() => activeControls.delete(inputId), 100));
        manualWindowInput.addEventListener('mouseleave', () => setTimeout(() => activeControls.delete(inputId), 100));

        const commitManualWindowSize = () => {
            const parsed = Math.round(parseFloat(manualWindowInput.value));
            if (!Number.isFinite(parsed)) return;
            const clamped = Math.max(
                manualDecodeWindowMin,
                Math.min(manualDecodeWindowMax, parsed)
            );
            manualDecodeWindow = clamped;
            manualWindowInput.value = String(clamped);
            if (manualWindowDisplay) manualWindowDisplay.textContent = String(clamped);
            if (clamped !== manualWindowLastSent) {
                manualWindowLastSent = clamped;
                sendManualWindowSize(clamped);
            }
        };

        manualWindowInput.addEventListener('input', (e) => {
            const value = Math.round(parseFloat(e.target.value));
            manualDecodeWindow = value;
            if (manualWindowDisplay) manualWindowDisplay.textContent = String(value);
        });
        manualWindowInput.addEventListener('change', commitManualWindowSize);
        manualWindowInput.addEventListener('mouseup', commitManualWindowSize);
        manualWindowInput.addEventListener('touchend', commitManualWindowSize);
    }

    const manualBufferInput = document.getElementById('manual-buffer-ratio');
    if (manualBufferInput) {
        const inputId = 'manual-buffer-ratio';
        const commitManualBufferRatio = () => {
            const parsed = parseFloat(manualBufferInput.value);
            if (!Number.isFinite(parsed)) return;
            const clamped = Math.max(0.10, Math.min(2.00, parsed));
            manualBufferRatio = clamped;
            manualBufferInput.value = clamped.toFixed(2);
            sendManualBufferRatio(clamped);
            setTimeout(() => activeControls.delete(inputId), 100);
        };
        manualBufferInput.addEventListener('focus', () => activeControls.add(inputId));
        manualBufferInput.addEventListener('change', commitManualBufferRatio);
        manualBufferInput.addEventListener('blur', commitManualBufferRatio);
        manualBufferInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                commitManualBufferRatio();
            }
        });
    }

    const manualDitherInput = document.getElementById('manual-dither');
    const manualDitherDisplay = document.getElementById('val-manual-dither');
    if (manualDitherInput) {
        const inputId = 'manual-dither';
        manualDitherInput.addEventListener('mousedown', () => activeControls.add(inputId));
        manualDitherInput.addEventListener('touchstart', () => activeControls.add(inputId));
        manualDitherInput.addEventListener('mouseup', () => setTimeout(() => activeControls.delete(inputId), 100));
        manualDitherInput.addEventListener('touchend', () => setTimeout(() => activeControls.delete(inputId), 100));
        manualDitherInput.addEventListener('mouseleave', () => setTimeout(() => activeControls.delete(inputId), 100));

        manualDitherInput.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            if (!Number.isFinite(value)) return;
            const clamped = Math.max(0, Math.min(0.20, value));
            if (manualDitherDisplay) manualDitherDisplay.textContent = clamped.toFixed(2);
            sendManualDither(clamped);
        });
    }

    const manualWanderInput = document.getElementById('manual-wander-k');
    const manualWanderDisplay = document.getElementById('val-manual-wander-k');
    if (manualWanderInput) {
        const inputId = 'manual-wander-k';
        manualWanderInput.addEventListener('mousedown', () => activeControls.add(inputId));
        manualWanderInput.addEventListener('touchstart', () => activeControls.add(inputId));
        manualWanderInput.addEventListener('mouseup', () => setTimeout(() => activeControls.delete(inputId), 100));
        manualWanderInput.addEventListener('touchend', () => setTimeout(() => activeControls.delete(inputId), 100));
        manualWanderInput.addEventListener('mouseleave', () => setTimeout(() => activeControls.delete(inputId), 100));

        manualWanderInput.addEventListener('input', (e) => {
            const value = Math.round(parseFloat(e.target.value));
            if (!Number.isFinite(value)) return;
            manualWanderK = Math.max(1, Math.min(64, value));
            if (manualWanderDisplay) manualWanderDisplay.textContent = String(manualWanderK);
            sendManualWander(manualWanderK);
        });
    }

    const cameraButtons = [
        { id: 'cam-left', dir: 'left' },
        { id: 'cam-right', dir: 'right' },
        { id: 'cam-over', dir: 'over' },
        { id: 'cam-under', dir: 'under' },
        { id: 'cam-forward', dir: 'forward' },
        { id: 'cam-backward', dir: 'backward' },
    ];
    for (const button of cameraButtons) {
        const el = document.getElementById(button.id);
        if (!el) continue;
        el.addEventListener('click', () => {
            setManualCameraToggle(button.dir, !manualCameraToggles[button.dir]);
        });
    }
    updateManualCameraButtonStates();

    const camSpeedInput = document.getElementById('cam-speed');
    const camSpeedDisplay = document.getElementById('val-cam-speed');
    if (camSpeedInput) {
        const inputId = 'cam-speed';
        camSpeedInput.addEventListener('mousedown', () => activeControls.add(inputId));
        camSpeedInput.addEventListener('touchstart', () => activeControls.add(inputId));
        camSpeedInput.addEventListener('mouseup', () => setTimeout(() => activeControls.delete(inputId), 100));
        camSpeedInput.addEventListener('touchend', () => setTimeout(() => activeControls.delete(inputId), 100));
        camSpeedInput.addEventListener('mouseleave', () => setTimeout(() => activeControls.delete(inputId), 100));

        camSpeedInput.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            manualCameraSpeed = value;
            if (camSpeedDisplay) camSpeedDisplay.textContent = value.toFixed(2);
        });
    }

    const manualColorAInput = document.getElementById('manual-color-a');
    const manualColorBInput = document.getElementById('manual-color-b');
    if (manualColorAInput) {
        const parsed = hexToRgbTriplet(manualColorAInput.value);
        if (parsed) manualColorA = parsed;
        manualColorAInput.addEventListener('input', (e) => {
            const next = hexToRgbTriplet(e.target.value);
            if (next) manualColorA = next;
        });
    }
    if (manualColorBInput) {
        const parsed = hexToRgbTriplet(manualColorBInput.value);
        if (parsed) manualColorB = parsed;
        manualColorBInput.addEventListener('input', (e) => {
            const next = hexToRgbTriplet(e.target.value);
            if (next) manualColorB = next;
        });
    }
    
    // Decoder controls
    const decoderControls = [
        { param: 'gain', format: 2 },
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

    // Reset buttons
    document.getElementById('btn-random-reset').addEventListener('click', () => {
        sendReset();
    });
    document.getElementById('btn-reorganized-reset').addEventListener('click', () => {
        sendReset();
    });
    
    // Reconnect button
    document.getElementById('btn-reconnect').addEventListener('click', () => {
        if (ws) {
            ws.close();
        }
        connectWebSocket();
    });

    // Exit button
    document.getElementById('btn-exit').addEventListener('click', () => {
        const confirmed = window.confirm('Stop the performer process?');
        if (confirmed) {
            sendExit();
        }
    });
}

function applyNavigationModeUI() {
    const shared3DPanel = document.getElementById('shared-3d-panel');
    const modeRandomBtn = document.getElementById('mode-random');
    const modeReorganizedBtn = document.getElementById('mode-reorganized');
    const modeManualBtn = document.getElementById('mode-manual');
    const randomPanel = document.getElementById('random-panel');
    const reorganizedPanel = document.getElementById('reorganized-panel');
    const manualPanel = document.getElementById('manual-panel');

    const isRandom = selectedNavigationMode === 'random';
    const isReorganized = selectedNavigationMode === 'reorganized';
    const isManual = selectedNavigationMode === 'manual';
    const useShared3DView = currentModeUsesShared3DView();

    modeRandomBtn.classList.toggle('active', isRandom);
    modeReorganizedBtn.classList.toggle('active', isReorganized);
    modeManualBtn.classList.toggle('active', isManual);

    shared3DPanel.classList.toggle('panel-hidden', !useShared3DView);
    randomPanel.classList.toggle('panel-hidden', !isRandom);
    reorganizedPanel.classList.toggle('panel-hidden', !isReorganized);
    manualPanel.classList.toggle('panel-hidden', !isManual);

    if (!useShared3DView) {
        manualPickDragging = false;
        resetManualCameraToggles();
    }

    modeRandomBtn.disabled = transportRunning;
    modeReorganizedBtn.disabled = transportRunning;
    modeManualBtn.disabled = transportRunning;
}

function sendRandomControl(name, value) {
    if (ws && wsConnected) {
        ws.send(JSON.stringify({
            type: 'random_control',
            controls: { [name]: value }
        }));
    }
}

function sendWanderRenderControl(name, value) {
    if (!ws || !wsConnected) return;

    let normalized = value;
    if (name === 'frame_source') {
        if (!Object.hasOwn(WANDER_FRAME_SOURCE_LABELS, value)) return;
    } else if (name === 'frame_order' || name === 'latent_colour') {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) return;
        normalized = Math.max(0, Math.min(1, numeric));
    } else {
        return;
    }

    ws.send(JSON.stringify({
        type: 'wander_render',
        controls: { [name]: normalized },
    }));
}

function sendReorganizedControl(name, value) {
    if (ws && wsConnected) {
        ws.send(JSON.stringify({
            type: 'reorganized_control',
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

function sendTransportSetMode(mode) {
    if (ws && wsConnected) {
        ws.send(JSON.stringify({
            type: 'transport',
            action: 'set_mode',
            mode,
        }));
    }
}

function sendTransportAction(action) {
    if (ws && wsConnected) {
        ws.send(JSON.stringify({
            type: 'transport',
            action,
        }));
    }
}

function sendManualControls() {
    if (ws && wsConnected) {
        const faders = manualFaders.slice(0, manualControlDim);
        while (faders.length < manualControlDim) {
            faders.push(0.5);
        }
        ws.send(JSON.stringify({
            type: 'manual_controls',
            faders,
        }));
    }
}

function sendManualWindowSize(size) {
    if (ws && wsConnected) {
        ws.send(JSON.stringify({
            type: 'manual_window',
            size: Math.round(size),
        }));
    }
}

function sendManualBufferRatio(ratio) {
    if (ws && wsConnected) {
        ws.send(JSON.stringify({
            type: 'manual_buffer',
            ratio: Number(ratio),
        }));
    }
}

function sendManualWander(k) {
    if (ws && wsConnected) {
        ws.send(JSON.stringify({
            type: 'manual_wander',
            k: Math.max(1, Math.min(64, Math.round(Number(k)))),
        }));
    }
}

function sendManualDither(amount) {
    if (ws && wsConnected) {
        ws.send(JSON.stringify({
            type: 'manual_dither',
            amount: Math.max(0, Math.min(0.20, Number(amount))),
        }));
    }
}

function sendExit() {
    if (ws && wsConnected) {
        ws.send(JSON.stringify({
            type: 'exit',
            reason: 'web_ui_button',
        }));
    }
}

// Handle mouse clicks on canvas
function mousePressed() {
    if (!currentModeUsesShared3DView()) {
        return;
    }
    if (mouseX < 0 || mouseX > width || mouseY < 0 || mouseY > height) {
        return;
    }
    manualPickDragging = true;
    const pointIdx = findClosestManualPointIndex(mouseX, mouseY);
    if (selectedNavigationMode === 'manual') {
        applyManualPointSelection(pointIdx);
    } else {
        sendCursorFor3DPoint(pointIdx);
    }
}

function mouseDragged() {
    if (!manualPickDragging || !currentModeUsesShared3DView()) {
        return;
    }
    if (mouseX < 0 || mouseX > width || mouseY < 0 || mouseY > height) {
        return;
    }
    const pointIdx = findClosestManualPointIndex(mouseX, mouseY);
    if (selectedNavigationMode === 'manual') {
        applyManualPointSelection(pointIdx);
    } else {
        sendCursorFor3DPoint(pointIdx);
    }
}

function mouseReleased() {
    manualPickDragging = false;
}

function mouseWheel(event) {
    // Keep wheel events available for page scroll/UI controls while using the 3D view.
    if (currentModeUsesShared3DView()) {
        return true;
    }
}

function hexToRgbTriplet(hex) {
    if (typeof hex !== 'string') return null;
    const trimmed = hex.trim();
    const match = /^#?([0-9a-fA-F]{6})$/.exec(trimmed);
    if (!match) return null;
    const value = match[1];
    return [
        parseInt(value.slice(0, 2), 16),
        parseInt(value.slice(2, 4), 16),
        parseInt(value.slice(4, 6), 16),
    ];
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
