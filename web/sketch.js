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
let manualCorpusPoints3D = [];
let manualCorpusColorValues = [];
let manualCorpusFileIds = [];
let totalCorpusPoints = 0;

// Navigation state
let currentPosition = [0.5, 0.5];
let trajectory = [];
let manualPosition3D = [0.5, 0.5, 0.5];
let manualTrajectory3D = [];
let currentIndex = 0;
let currentVelocity = 0;
let currentFileId = 0;
let selectedNavigationMode = 'policy';
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
    applyNavigationModeUI();

    frameRate(30);
}

function draw() {
    background(...COLORS.background);

    if (selectedNavigationMode === 'manual') {
        updateManualCameraMotion();
        drawManual3DScene();
    } else if (vizMode === 'scatter') {
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

function pickManualPointFromMouse(mx, my) {
    if (!manualCorpusPoints3D || manualCorpusPoints3D.length === 0) {
        return;
    }
    const ray = getManualRayFromScreen(mx, my);

    let bestIdx = 0;
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

    const selected = manualCorpusPoints3D[bestIdx];
    manualPosition3D = selected.slice(0, 3);
    const nextFaders = new Array(manualControlDim).fill(0.5);
    for (let i = 0; i < Math.min(3, manualControlDim); i++) {
        nextFaders[i] = constrain(selected[i], 0, 1);
    }
    if (manualControlDim >= 4) {
        const colorValue = manualCorpusColorValues[bestIdx];
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
            sendTransportSetMode(selectedNavigationMode);
            sendManualControls();
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
        manualCorpusPoints3D = data.manual_positions_3d || [];
        manualCorpusColorValues = data.manual_color_values || [];
        manualCorpusFileIds = data.manual_file_ids || corpusFileIds;
        totalCorpusPoints = data.total_points || 0;
        if (typeof data.navigation_mode === 'string') {
            selectedNavigationMode = data.navigation_mode;
            applyNavigationModeUI();
        }

        // Pre-compute and cache colors for all corpus points (performance optimization)
        cachedCorpusColors = corpusFileIds.map(fileId =>
            COLORS.files[fileId % COLORS.files.length]
        );

        console.log(
            `Received corpus: policy2d=${corpusPoints.length}, manual3d=${manualCorpusPoints3D.length}, manualColor=${manualCorpusColorValues.length}`
        );

    } else if (data.type === 'state') {
        const nav = data.navigation || {};
        const transport = data.transport || {};
        const manual = data.manual || {};

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
            currentVelocity = 0;
            currentFileId = manualCorpusFileIds[clampedIdx] || corpusFileIds[clampedIdx] || 0;
            manualTrajectory3D.push(manualPosition3D);
            if (manualTrajectory3D.length > trailLength) {
                manualTrajectory3D = manualTrajectory3D.slice(-trailLength);
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
            currentPosition = nav.position_2d || currentPosition;
            currentIndex = nav.index || 0;
            currentVelocity = nav.velocity || 0;
            currentFileId = nav.file_id || 0;
            
            if (nav.trajectory_2d && nav.trajectory_2d.length > 0) {
                trajectory = nav.trajectory_2d;
            }
        }

        if (selectedNavigationMode !== 'manual') {
            updateHeatmap();
        }
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
        const runStatus = transportRunning ? 'Running' : 'Idle';
        text.textContent = `Connected (${selectedNavigationMode}, ${runStatus})`;
    } else {
        dot.classList.remove('connected');
        text.textContent = 'Disconnected';
    }
}

function updateInfoDisplay(data) {
    if (selectedNavigationMode === 'manual') {
        document.getElementById('info-index').textContent = manualNearestIndex || 0;
        document.getElementById('info-velocity').textContent = Number(manualNearestDistance || 0).toFixed(2);
        document.getElementById('info-file').textContent = currentFileId || 0;
        return;
    }

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
    const modePolicyBtn = document.getElementById('mode-policy');
    const modeManualBtn = document.getElementById('mode-manual');

    modePolicyBtn.addEventListener('click', () => {
        if (transportRunning) return;
        selectedNavigationMode = 'policy';
        applyNavigationModeUI();
        sendTransportSetMode('policy');
    });

    modeManualBtn.addEventListener('click', () => {
        if (transportRunning) return;
        selectedNavigationMode = 'manual';
        applyNavigationModeUI();
        sendTransportSetMode('manual');
    });

    document.getElementById('btn-start').addEventListener('click', () => {
        sendTransportAction('start');
    });

    document.getElementById('btn-stop').addEventListener('click', () => {
        sendTransportAction('stop');
    });

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

    // Exit button
    document.getElementById('btn-exit').addEventListener('click', () => {
        const confirmed = window.confirm('Stop the performer process?');
        if (confirmed) {
            sendExit();
        }
    });
}

function applyNavigationModeUI() {
    const modePolicyBtn = document.getElementById('mode-policy');
    const modeManualBtn = document.getElementById('mode-manual');
    const policyPanel = document.getElementById('policy-panel');
    const manualPanel = document.getElementById('manual-panel');

    const isPolicy = selectedNavigationMode === 'policy';
    modePolicyBtn.classList.toggle('active', isPolicy);
    modeManualBtn.classList.toggle('active', !isPolicy);

    policyPanel.classList.toggle('panel-hidden', !isPolicy);
    manualPanel.classList.toggle('panel-hidden', isPolicy);

    if (isPolicy) {
        manualPickDragging = false;
        resetManualCameraToggles();
    }

    modePolicyBtn.disabled = transportRunning;
    modeManualBtn.disabled = transportRunning;
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
    if (selectedNavigationMode === 'manual') {
        if (mouseX < 0 || mouseX > width || mouseY < 0 || mouseY > height) {
            return;
        }
        manualPickDragging = true;
        pickManualPointFromMouse(mouseX, mouseY);
        return;
    }
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

function mouseDragged() {
    if (selectedNavigationMode !== 'manual' || !manualPickDragging) {
        return;
    }
    if (mouseX < 0 || mouseX > width || mouseY < 0 || mouseY > height) {
        return;
    }
    pickManualPointFromMouse(mouseX, mouseY);
}

function mouseReleased() {
    manualPickDragging = false;
}

function mouseWheel(event) {
    // Keep wheel events available for page scroll/UI controls in manual mode.
    if (selectedNavigationMode === 'manual') {
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
