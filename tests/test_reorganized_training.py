import numpy as np
import pytest

from stable_audio_wanderer.cli import train_policy


@pytest.mark.parametrize("ids", [[], [0], [0, 1, 2]])
def test_no_within_file_transitions_explains_remedy(ids):
    with pytest.raises(RuntimeError, match="at least two units in the same source file"):
        train_policy._build_reorganized_transition_samples(
            np.array(ids), np.arange(len(ids)), np.zeros((len(ids), 1)))


def test_two_units_preserve_observed_successor():
    samples = train_policy._build_reorganized_transition_samples(
        np.array([0, 0]), np.array([0, 5]), np.array([[0], [1]]))
    assert len(samples) == 1
    current, candidates, target = samples[0]
    assert current == 0
    assert candidates[target] == 1


def test_all_preflights_before_writing_manual_artifact(monkeypatch):
    monkeypatch.setattr(train_policy, "resolve_corpus_path", lambda _: "unused.npz")
    monkeypatch.setattr(train_policy, "load_corpus", lambda _: {})
    monkeypatch.setattr(train_policy, "_resolve_reorganized_artifact",
                        lambda *args: ({"unit_file_id": np.array([0])}, "embedded"))
    def unexpected_write(**kwargs):
        pytest.fail("Manual artifact must not be written before eligibility check")
    monkeypatch.setattr(train_policy, "_save_manual_navigation_artifact", unexpected_write)
    with pytest.raises(RuntimeError, match="1 units across 1 files"):
        train_policy.run_train("unused", navigation_mode="all")
