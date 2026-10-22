# Invalid initial sweep

This directory is not a correctness result. `tools/evaluate.py` defaults to
`http://127.0.0.1:48702` but does not start a runner, so all 15 cases failed
with `ConnectionError`. The valid sweep is `003-rtx4060-full`.
