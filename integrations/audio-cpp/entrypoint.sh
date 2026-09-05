#!/usr/bin/env sh
set -eu

python3 /opt/audio-cpp-supervisor.py &
supervisor_pid="$!"
trap 'kill "$supervisor_pid" 2>/dev/null || true' EXIT INT TERM

exec python3 -m uvicorn server:app --host 0.0.0.0 --port 7860
