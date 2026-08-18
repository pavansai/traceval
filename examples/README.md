# Example trace

`feedback_form_live_trace.jsonl` is a real trace from a real run: `tasks/feedback_form`
via `LiveAgent` + `AnthropicProvider` (`claude-haiku-4-5`), scored by
`ModelGradedScorer` with a real judge (same model). Committed so the trace
schema (`docs/architecture.md`'s "Trace schema" section) is readable without
running anything yourself. `runs/` itself stays gitignored; this is the one
example worth keeping around.

Inspect it with `traceval`'s own tools rather than just reading the raw
JSONL:

```sh
uv run traceval report examples/
```

Or read a single field with any JSON tool, e.g. the judge's rationale:

```sh
tail -1 examples/feedback_form_live_trace.jsonl | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["scores"][0]["details"]["rationale"])'
```
