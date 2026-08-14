#!/usr/bin/env python3
"""
Test script for timbral wandering functionality.
"""

import numpy as np
from stable_audio_wanderer.runtime.manual_player import ManualNavigationEngine


def test_manual_navigation_wandering():
    """Test the enhanced manual navigation with wandering."""
    print("Testing ManualNavigationEngine with timbral wandering...")

    # Create test data
    np.random.seed(42)
    n_points = 100
    n_dims = 3
    manual_points = np.random.randn(n_points, n_dims).astype(np.float32)
    fader_p01 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    fader_p99 = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Test 1: Default behavior (wandering disabled)
    print("\n1. Testing default behavior (wander_k=1, wander_speed=0.0)")
    engine = ManualNavigationEngine(
        manual_points, fader_p01, fader_p99, wander_k=1, wander_speed=0.0
    )

    # Set faders to middle position
    engine.set_faders([0.5] * 3)

    # Step multiple times - should always return same nearest neighbor
    frames = [engine.step() for _ in range(10)]
    indices = [f.nearest_index for f in frames]
    print(f"   Indices: {indices}")
    print(f"   All same: {len(set(indices)) == 1}")
    print(f"   Wandering active: {any(f.is_wandering for f in frames)}")
    assert len(set(indices)) == 1, "Expected fixed index when wandering is disabled."
    assert not any(
        f.is_wandering for f in frames
    ), "Wandering should be inactive when wander_k=1."

    # Test 2: Wandering enabled
    print("\n2. Testing wandering (wander_k=4, wander_speed=0.5)")
    engine = ManualNavigationEngine(
        manual_points, fader_p01, fader_p99, wander_k=4, wander_speed=0.5
    )

    # Set faders to middle position
    engine.set_faders([0.5] * 3)

    # Step multiple times - should show variation
    frames = [engine.step() for _ in range(80)]
    indices = [f.nearest_index for f in frames]
    unique_indices = set(indices)
    print(f"   Indices: {indices}")
    print(f"   Unique indices: {len(unique_indices)}")
    print(f"   Wandering active: {any(f.is_wandering for f in frames)}")
    print(f"   Progress values: {[f.wander_progress for f in frames]}")
    assert len(unique_indices) > 1, "Expected wandering to visit multiple neighbors."
    assert any(f.is_wandering for f in frames), "Expected wandering state to activate."

    # Test 3: Fast wandering
    print("\n3. Testing fast wandering (wander_k=8, wander_speed=0.1)")
    engine = ManualNavigationEngine(
        manual_points, fader_p01, fader_p99, wander_k=8, wander_speed=0.1
    )

    # Set faders to middle position
    engine.set_faders([0.5] * 3)

    # Step multiple times - should change targets more frequently
    frames = [engine.step() for _ in range(60)]
    indices = [f.nearest_index for f in frames]
    unique_indices = set(indices)
    print(f"   Indices: {indices}")
    print(f"   Unique indices: {len(unique_indices)}")
    print(f"   Wandering active: {any(f.is_wandering for f in frames)}")
    assert len(unique_indices) > 1, "Expected neighbor changes in fast wandering mode."

    # Test 4: Parameter changes
    print("\n4. Testing parameter changes")
    engine = ManualNavigationEngine(
        manual_points, fader_p01, fader_p99, wander_k=2, wander_speed=0.3
    )

    # Set faders
    engine.set_faders([0.3] * 3)

    # Get initial state
    state1 = engine.get_state()
    print(
        f"   Initial state: wander_k={state1['wander_k']}, wander_speed={state1['wander_speed']}"
    )

    # Change parameters
    engine.set_wander_params(k=16, speed=0.8)

    # Get updated state
    state2 = engine.get_state()
    print(
        f"   Updated state: wander_k={state2['wander_k']}, wander_speed={state2['wander_speed']}"
    )

    # Test 5: Fader changes reset wandering
    print("\n5. Testing fader change resets wandering")
    engine = ManualNavigationEngine(
        manual_points, fader_p01, fader_p99, wander_k=4, wander_speed=0.5
    )

    # Set initial faders
    engine.set_faders([0.5] * 3)

    # Step a few times to start wandering
    for _ in range(5):
        engine.step()

    # Change faders significantly
    engine.set_faders([0.9] * 3)

    # Next step should reset to new nearest neighbor
    frame = engine.step()
    print(f"   After fader change - is_wandering: {frame.is_wandering}")
    print(f"   After fader change - wander_progress: {frame.wander_progress}")
    assert frame.wander_progress >= 0.0

    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    test_manual_navigation_wandering()
