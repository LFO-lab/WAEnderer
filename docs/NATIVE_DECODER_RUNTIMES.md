# Native decoder runtimes and playback debugging

The server and native workers use the current Python interpreter by default. No virtual-environment name is required. A package installed in one virtual environment is unavailable to another interpreter. Adding a different environment's site-packages to `sys.path` would mix PyTorch and its dependencies, so each native decoder owns a persistent child process instead.

## Persistent paths

Optional overrides are loaded from `native_runtimes.json` in the platform's user configuration directory: `~/Library/Application Support/waenderer` on macOS, `$XDG_CONFIG_HOME/waenderer` (default `~/.config/waenderer`) on Linux, or `%APPDATA%/waenderer` on Windows. `WAENDERER_NATIVE_RUNTIMES` selects an explicit configuration file. Relative paths refer to that file's directory, not a source checkout or site-packages directory.

Start from [the generic example](native_runtimes.example.json). EAR model entries contain `weights`, `repo` and optional `config`. The optional `python` field is only needed when the user deliberately installs incompatible profiles in separate interpreters; it has no built-in environment names. Without it, workers use `sys.executable`. Discovery and Start check paths/dependencies. Web requests cannot choose an executable. Interpreter symlinks are preserved so virtual environments retain their dependency isolation.

Install model dependencies through the declared requirements profiles described in [the installation contract](INSTALLATION_PROFILES.md). This registry describes model assets and optional runtime overrides; it does not install dependencies or replace package requirements.

## Native process and MPS ownership

`native_runtime.py` reads paths and performs discovery/selection in the configured environment. `native_worker.py` loads one model, executes requests serially and returns float32 CPU PCM. `process_decoder.py` retains that worker across transport Stop/Start, serializes exchanges, validates PCM and closes the worker when the pipeline releases the decoder.

The active and candidate windows retain the production transport/OLA logic. A native decode request no longer shares a Metal command-buffer runtime with the navigation policy in the server. A native process abort becomes a transport error, with its assertion in the server log; it cannot directly abort the server. Worker silence has a 120-second timeout, after which the worker is terminated. PCM exchange time is included in reported decode duration.

A short EAR 44k MPS check of T8 and T30 requested concurrently, then T8 again, passed with finite stereo PCM (about 33, 107 and 30 ms including exchange). This checks isolated decoding, not the full physical navigation/audio session that originally crashed. Reproduce the window change manually after restarting the server.

## ONNX CPU settings

The artifact loader previously imposed one CPU thread and basic graph optimization on every model. Rack T8 took about 540 ms for an audio hop budget of 186 ms, explaining why sustained playback could not refill the buffer. Enabling all graph optimizations alone did not materially improve it. Eight intra-op threads reduced the short T8 measurement to about 150 ms; T16 took about 296 ms for a 372 ms hop budget. The four-thread comparison stayed within max error 1.83e-7 and RMSE 3.09e-8 of the former settings.

The artifact loader now uses up to eight intra-op threads, one inter-op thread, all graph optimizations and disabled idle spinning. Legacy packaged SAME-S keeps its existing loader. The transport also prepares one next latent window while decoding the current one. Each lane keeps at most one prefetched input; there is no extra inference backlog. This overlaps navigation planning with inference rather than adding their costs between hops. No audio-padding or buffer-size workaround hides underruns. The first ten-second production-navigation software run, with optimized ORT alone, still had 54 buffer underruns. With the bounded latent prefetch it passed a fresh ten-second run with 9.52 seconds of attributed PCM and zero underruns (T8 median 161 ms, p95 168 ms). These are local short measurements; sustained physical playback and contention still need manual checking.

## Retest

Restart the server using the same command as before, then refresh decoder availability.

1. Rack: ONNX CPU, T8 then T16. Listen for gaps and watch buffer/device underruns.
2. BurntMemory: confirm PyTorch CPU/MPS are offered through `.venv`, using the currently installed native profile.
3. EAR 44k MPS: play and change T8 → T30 → T8. If it fails, retain the worker assertion, transport error, mode and window. The Web connection should remain alive.

The focused worker tests cover interpreter symlinks, discovery from the correct environment, persistent reuse and containment of a killed worker. Existing artifact/pipeline/discovery checks continue to exercise selection and release.
