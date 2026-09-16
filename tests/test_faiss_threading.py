"""Exercise native FAISS retrieval in a subprocess to contain native crashes."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def test_large_index_search_from_separate_thread():
    pytest.importorskip("faiss")
    script = textwrap.dedent("""
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        from stable_audio_wanderer.runtime.player import LatentNavigationEngine
        import faiss

        rng = np.random.default_rng(42)
        embeddings = rng.normal(size=(44000, 64)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Isolate retrieval from model loading and corpus geometry generation.
        nav = LatentNavigationEngine.__new__(LatentNavigationEngine)
        nav.N = len(embeddings)
        nav._has_faiss = True
        faiss.omp_set_num_threads(1)
        nav.faiss_index = faiss.IndexFlatIP(64)
        nav.faiss_index.add(embeddings)

        def query():
            # A constructor-only limit must not be sufficient for this test.
            faiss.omp_set_num_threads(10)
            for i in range(20):
                indices, distances = nav._query_knn(
                    None, k=16, query_embedding=embeddings[i]
                )
                assert faiss.omp_get_max_threads() == 1
                assert indices[0] == i
                assert abs(distances[0]) < 1e-5
                assert np.all(np.isfinite(distances))

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(query).result()
    """)
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "OMP_NUM_THREADS": "10"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
