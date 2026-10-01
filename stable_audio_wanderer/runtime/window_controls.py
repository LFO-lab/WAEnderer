"""Time-based window adaptation and continuous manual latent textures."""
from __future__ import annotations

import math
import numpy as np


class AdaptiveWindow:
    def __init__(self):
        self.position = None
        self.timestamp = None
        self.motion = 0.0
        self.changed_at = -math.inf

    def observe(self, position, now):
        position = np.asarray(position, dtype=np.float32)
        if self.timestamp is not None:
            dt = max(0.001, now - self.timestamp)
            speed = float(np.sqrt(np.mean((position - self.position) ** 2))) / dt
            # Fast attack, slower release; independent of decoder window cadence.
            tau = 0.08 if speed > self.motion else 0.6
            self.motion += (1 - math.exp(-dt / tau)) * (speed - self.motion)
        self.position = position.copy()
        self.timestamp = now

    def choose(self, current, windows, now):
        windows = sorted(windows)
        target = 2 ** (math.log2(windows[-1]) - min(1.0, self.motion / 2.0)
                       * math.log2(windows[-1] / windows[0]))
        if current not in windows:
            result = min(windows, key=lambda value: abs(value - target))
        elif target < current * 0.85 and now - self.changed_at >= 0.15:
            result = min(windows, key=lambda value: abs(value - target))
        elif target > current * 1.15 and now - self.changed_at >= 0.6:
            result = windows[min(windows.index(current) + 1, len(windows) - 1)]
        else:
            return current
        if result != current:
            self.changed_at = now
        return result


class ManualTexture:
    def __init__(self, latents, mean, std):
        self.latents = latents
        self.mean = mean
        self.std = np.maximum(std, 1e-6)
        self.current = None
        self.target = None
        self.remaining = 0
        self.anchor = None
        self.neighbors = None
        self.rng = np.random.default_rng(1729)

    def render(self, anchor, count, content, amount):
        centre = self.latents[anchor]
        if self.current is None:
            self.current = centre.copy()
        if anchor != self.anchor:
            self.anchor = anchor
            # Bound temporary memory for large corpora; only retain the closest 16.
            distance = np.empty(len(self.latents), dtype=np.float32)
            for start in range(0, len(self.latents), 4096):
                delta = (self.latents[start:start + 4096] - centre) / self.std
                distance[start:start + len(delta)] = np.mean(delta * delta, axis=1)
            count_nearest = min(16, len(distance))
            candidates = np.argpartition(distance, count_nearest - 1)[:count_nearest]
            self.neighbors = candidates[np.argsort(distance[candidates], kind="stable")]
            self.remaining = 0
        output = []
        for _ in range(count):
            if content == 'held' or amount == 0:
                self.target = centre
                self.remaining = 0
            elif self.remaining <= 0:
                neighbor = self.latents[int(self.rng.choice(self.neighbors))]
                self.target = centre + amount * (neighbor - centre)
                self.remaining = 12
            self.current = self.current + 0.25 * (self.target - self.current)
            self.remaining -= 1
            output.append(self.current.copy())
        return np.ascontiguousarray(output, dtype=np.float32)
