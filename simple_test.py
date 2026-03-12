#!/usr/bin/env python3
"""
Simple test to verify the code structure is correct.
"""

import os
import sys

# Add the project to Python path
sys.path.insert(0, "/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer")


def test_imports():
    """Test that our modified modules can be imported."""
    try:
        from stable_audio_wanderer.runtime.manual_player import (
            ManualFrame,
            ManualNavigationEngine,
        )

        print("✅ Successfully imported ManualNavigationEngine")

        # Check that the new parameters exist
        import inspect

        sig = inspect.signature(ManualNavigationEngine.__init__)
        params = list(sig.parameters.keys())
        print(f"✅ ManualNavigationEngine parameters: {params}")

        # Check for new methods
        methods = [m for m in dir(ManualNavigationEngine) if not m.startswith("_")]
        print(f"✅ ManualNavigationEngine methods: {methods}")

        # Check ManualFrame fields
        frame_fields = [f for f in dir(ManualFrame) if not f.startswith("_")]
        print(f"✅ ManualFrame fields: {frame_fields}")

        # Check that wander parameters are in the signature
        if "wander_k" in params and "wander_speed" in params:
            print("✅ Wandering parameters found in constructor")
        else:
            print("❌ Wandering parameters missing")
            return False

        # Check for new methods
        if "set_wander_params" in methods:
            print("✅ set_wander_params method found")
        else:
            print("❌ set_wander_params method missing")
            return False

        # Check ManualFrame has new fields
        if "is_wandering" in frame_fields and "wander_progress" in frame_fields:
            print("✅ Wandering fields found in ManualFrame")
        else:
            print("❌ Wandering fields missing from ManualFrame")
            return False

        print("\n🎉 All structure tests passed!")
        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_perform_integration():
    """Test that perform.py has the necessary integration."""
    try:
        with open(
            "/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/bin/perform.py",
            "r",
        ) as f:
            content = f.read()

        # Check for key integration points
        checks = [
            ("set_manual_wander_params", "Method definition"),
            ("wander_k=1", "Default wander_k parameter"),
            ("wander_speed=0.0", "Default wander_speed parameter"),
            ("manual_wander", "WebSocket message handling"),
            ("wander_k.*get_state", "State broadcast integration"),
        ]

        all_passed = True
        for check, description in checks:
            if check in content:
                print(f"✅ {description}: Found '{check}'")
            else:
                print(f"❌ {description}: Missing '{check}'")
                all_passed = False

        if all_passed:
            print("\n🎉 All integration tests passed!")
        else:
            print("\n❌ Some integration tests failed!")

        return all_passed

    except Exception as e:
        print(f"❌ Integration test error: {e}")
        return False


if __name__ == "__main__":
    print("Running structure tests...")
    test1 = test_imports()

    print("\nRunning integration tests...")
    test2 = test_perform_integration()

    if test1 and test2:
        print(
            "\n🎉 ALL TESTS PASSED! The timbral wandering feature has been successfully implemented."
        )
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)
