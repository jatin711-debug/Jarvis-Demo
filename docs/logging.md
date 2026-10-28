# Logging

Jarvis writes two logs from the same events:

| Sink | Level | Format | Purpose |
|------|-------|--------|---------|
| Console (stderr) | `system.log_level` (default `INFO`), `-v/--verbose` for `DEBUG` | `time level component turn message`, colored on a terminal | Follow what Jarvis is doing right now |
| `~/.jarvis/logs/jarvis.log` | always `DEBUG`, rotated at 10 MB with 5 backups | Text by default, optional JSON lines | Diagnose after the fact; filter by turn |

Answers from `--text` go to stdout, logs go to stderr, so `jarvis --text "..." > out.txt`
captures only the reply. Set `NO_COLOR=1` to disable colors.

Set `system.log_format: json` in the configuration for newline-delimited JSON
in the file. Each object includes a UTC timestamp, level, logger, process run ID,
turn tag, and message, plus exception or stack information when present. The
run ID distinguishes turn numbers across restarts; console formatting is unchanged.
Switching formats does not rewrite or rotate existing files, so an existing file
can contain both formats until it rotates. Archive or move old files first if a
collector requires a uniform format.
The rotating file handler is intended for one Jarvis process per log directory;
configure separate directories for concurrent processes to avoid rotation races.

## Reading a turn

```
07:43:57 INFO  orchestrator      ready tools=34 browser_tools=15 desktop=cua memory=on voice=on confirm=off
07:43:57 INFO  orchestrator      waiting for wake word='hey_jarvis'
07:43:57 INFO  orchestrator      wake word detected
07:43:57 INFO  orchestrator      listening for command
07:43:57 INFO  orchestrator t1   heard chars=15
07:43:57 INFO  agent        t1   route surface=desktop
07:43:57 INFO  agent        t1   tool call name=desktop_launch args_count=1
07:43:57 INFO  agent        t1   tool ok name=desktop_launch ms=812 chars=96
07:43:57 INFO  orchestrator t1   reply chars=19
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
| `tool call name=... args_count=N` | The model asked for a tool; argument values are omitted |
| `tool ok / rejected / error name=... ms=...` | Outcome. `rejected` = not offered for this request or invalid arguments; `error` = the tool raised |
| `tool denied ...` / `confirmation ... approved=...` | Permission gateway decisions and voice/text approvals |
| `latency record stt memory first_token agent speak total` | Where the time went (`agent` excludes speech; `total` = memory + agent + speak) |
| `barge-in playback interrupted` | The user talked over a reply |
| `turn cut short by barge-in; agent stopped` | The model stream was closed, so no further tool calls run; the history records the reply as interrupted |
| `transcript ignored reason=noise` | A Whisper hallucination or stutter was dropped |

Anything at WARN or ERROR is worth reading: a deadline was hit, a service failed to
start, or a tool was refused.

The voice loop contains device and engine crashes instead of exiting:

| Event | Meaning |
|-------|---------|
| `session failed error=...; retrying in Ns` / `voice loop failed ...` (ERROR) | A conversation session or the wake word stream crashed; Jarvis waits (1s, doubling to 10s, reset by a session that ends cleanly) and goes back to listening. The traceback is in the log file at DEBUG |
| `mic open failed` / `mic read failed` (WARN), `mic recovered` (INFO) | The microphone dropped (headset unplugged, driver reset) and is being reopened |
| `speech output failed` / `barge-in monitor failed` (WARN) | The speaker could not play a sentence, or the barge-in monitor could not open the mic so that sentence cannot be interrupted |
| `transcription failed` (ERROR) | Speech recognition raised; the attempt counts as silence |

Repeated device failures are rate-limited: the first one is a WARN with the error,
identical repeats drop to DEBUG (`... failed again`) until the component works again.

## Writing log messages

* A short lower-case event word first, then `key=value` fields: `tool ok name=x ms=12`.
* One line per event. No emoji, no banners, no Rich markup.
* Record counts and outcomes rather than request, reply, tool argument, or result text.
* INFO tells the story of a turn or a lifecycle change. Per-frame, per-token or
  per-iteration detail is DEBUG.
* Avoid logging secrets, file contents, command lines, and arbitrary user input.
* Use `logger.error(..., exc_info=True)` only for unexpected failures; expected ones
  (timeouts, refusals) are a WARN line without a traceback.
* Get a logger with `get_logger(__name__)`. Do not call `setup_logging` from library
  code; only `jarvis/__main__.py` does.

## Privacy and retention

Routine conversational events omit transcripts, replies, tool arguments, search
queries, and vision responses. This is not a guarantee that logs contain no
sensitive data: exceptions and tracebacks can include values or paths, third-party
libraries may emit their own messages, and the separately managed llama-server
log has its own behavior. Existing rotated logs are not sanitized or deleted by
this change. Protect the log directory, review logs before sharing, and apply a
retention policy appropriate to the deployment. Rotation limits file size, not age.
