# Logging

Jarvis writes two logs from the same events:

| Sink | Level | Format | Purpose |
|------|-------|--------|---------|
| Console (stderr) | `system.log_level` (default `INFO`), `-v/--verbose` for `DEBUG` | `time level component turn message`, colored on a terminal | Follow what Jarvis is doing right now |
| `~/.jarvis/logs/jarvis.log` | always `DEBUG`, rotated at 10 MB x 5 | `date time.ms \| level \| logger \| turn \| message` | Diagnose after the fact; grep by turn |

Answers from `--text` go to stdout, logs go to stderr, so `jarvis --text "..." > out.txt`
captures only the reply. Set `NO_COLOR=1` to disable colors.

## Reading a turn

```
07:43:57 INFO  orchestrator      ready tools=34 browser_tools=15 desktop=cua memory=on voice=on confirm=off
07:43:57 INFO  orchestrator      waiting for wake word='hey_jarvis'
07:43:57 INFO  orchestrator      wake word detected
07:43:57 INFO  orchestrator      listening for command
07:43:57 INFO  orchestrator t1   heard text='open calculator'
07:43:57 INFO  agent        t1   route surface=desktop
07:43:57 INFO  agent        t1   tool call name=desktop_launch args={name='Calculator'}
07:43:57 INFO  agent        t1   tool ok name=desktop_launch ms=812 chars=96
07:43:57 INFO  orchestrator t1   reply text='Calculator is open.'
07:43:57 INFO  orchestrator t1   latency record=1.20s stt=0.40s memory=0.00s first_token=0.90s agent=2.90s speak=0.60s total=3.50s
```

Every line of one interaction carries the same turn tag (`t1`), including work that
finishes later (memory writes). To pull one turn out of the file:

```powershell
Select-String -Path ~\.jarvis\logs\jarvis.log -Pattern '\| t7 \|'
```

### The events you will see at INFO

| Event | Meaning |
|-------|---------|
| `route surface=... groups=... system_tools=N` | Which tool surface the request needs: `small_talk`, `chat`, `system`, `browser`, `desktop` |
| `tool call name=... args={...}` | The model asked for a tool (values shortened, secrets redacted) |
| `tool ok / rejected / error name=... ms=...` | Outcome. `rejected` = not offered for this request or invalid arguments; `error` = the tool raised |
| `tool denied ...` / `confirmation ... approved=...` | Permission gateway decisions and voice/text approvals |
| `latency record stt memory first_token agent speak total` | Where the time went (`agent` excludes speech; `total` = memory + agent + speak) |
| `barge-in playback interrupted` | The user talked over a reply |
| `turn cut short by barge-in; agent stopped` | The model stream was closed, so no further tool calls run; the history records the reply as interrupted |
| `transcript ignored reason=noise` | A Whisper hallucination or stutter was dropped |

Anything at WARN or ERROR is worth reading: a deadline was hit, a service failed to
start, or a tool was refused.

## Writing log messages

* A short lower-case event word first, then `key=value` fields: `tool ok name=x ms=12`.
* One line per event. No emoji, no banners, no Rich markup.
* Quote free text with `%r`, e.g. `heard text=%r`.
* INFO tells the story of a turn or a lifecycle change. Per-frame, per-token or
  per-iteration detail is DEBUG.
* Never log secrets or full file/typed contents. Use `_brief_args` (agent) as the model
  for shortening and redaction.
* Use `logger.error(..., exc_info=True)` only for unexpected failures; expected ones
  (timeouts, refusals) are a WARN line without a traceback.
* Get a logger with `get_logger(__name__)`. Do not call `setup_logging` from library
  code; only `jarvis/__main__.py` does.
