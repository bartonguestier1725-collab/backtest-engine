#!/usr/bin/env bash
# Blind LLM review of each case: the model sees bench/llm/prompt.txt and the
# strategy file only (not labels.json, not hires_lib.py). Run from the repo root.
# Model and flags used for bench/llm/*.sonnet.json (2026-11-22):
set -euo pipefail
for f in bench/cases/s*.py; do
  id=$(basename "$f" .py)
  { cat bench/llm/prompt.txt; echo '```python'; cat "$f"; echo '```'; } \
    | claude -p --model sonnet --tools "" --no-session-persistence > "bench/llm/$id.sonnet.json"
done
# s24 again, this time with the helper's source appended:
{ cat bench/llm/prompt.txt; echo '```python'; cat bench/cases/s24.py; echo '```';
  echo; echo 'The imported module hires_lib.py:'; echo '```python'; cat bench/cases/hires_lib.py; echo '```'; } \
  | claude -p --model sonnet --tools "" --no-session-persistence > bench/llm/s24_with_lib.sonnet.json
