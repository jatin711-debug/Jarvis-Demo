# Scenario benchmark

`python -m jarvis.bench` runs Jarvis's real agent against a set of scenarios and reports
what a person would care about: did it do the job, how long did they wait, how often did the
model call a tool wrongly, and how much memory did it take. Use it to compare models and
runtimes on Jarvis's own work (unit tests mock the model and cannot do that) and to catch
regressions after a change.

```powershell
python -m jarvis.bench                          # every scenario once, with your config
python -m jarvis.bench --repeat 5               # models are not deterministic: pass rate over 5 tries
python -m jarvis.bench --only write-file,follow-up
python -m jarvis.bench --config config.llamacpp.yaml --report 4b.json
python -m jarvis.bench --list
```

Run it with nothing else heavy running; the timings include whatever else the machine is doing.

## What a run does

- Starts Jarvis as `--text` does (model server included), makes one warm-up request, then runs
  each scenario in its own throwaway workspace, which is deleted afterwards.
- **Safe on any machine.** Security is forced to the standard profile confined to the workspace,
  whatever `config.local.yaml` allows; memory and desktop control are off. Confirmation prompts
  (a tool that changes something) are answered by the scenario, not a person.
- Drives the same streaming path as the voice loop, so "first" is the time to the first words.
- Records every tool call and how it ended: `ok`, `invalid` (arguments rejected), `denied`
  (outside the allowed folders), `cancelled` (the user said no), or `error`.

```
scenario             passed  median  first  invalid  tools
greeting             1/1     0.8s    0.2s   0        -
write-file           1/1     4.7s    4.2s   0        write_file
follow-up            1/1     5.4s    2.5s   0        write_file, read_file
...
Passed 13/13 runs (100%), median 2.5s, p95 8.5s
Tool calls: 16, rejected for bad arguments: 0
Peak memory: GPU in use 3.0 GB (+3.0 GB since start), Jarvis 0.1 GB, llama-server 2.9 GB
```

A failing run prints why (`never called read_file`, `hello.txt was not created`, `took 61.2s,
limit 45s`), what the model said, and which tools it used. `--report FILE` writes everything as
JSON. Exit code: 0 all passed, 3 some failed, 1 Jarvis could not start, 2 bad arguments or
scenario file.

GPU figures need `nvidia-smi` and cover the whole card, so close other GPU programs first.
`llama-server` memory is only measured when Jarvis started the server itself.

## Writing scenarios

The shipped scenarios are `jarvis/bench/scenarios.yaml`. To add your own, copy that file, add
yours, and pass `--scenarios mine.yaml`. Real things you say to Jarvis make the best scenarios.

```yaml
scenarios:
  - id: write-file
    prompt: "Create a file named hello.txt in the folder {workspace} that contains exactly: hi there"
    expect:
      tools_called: [write_file]
      files:
        "hello.txt": {contains: "hi there"}
      max_seconds: 45
```

| Key | Meaning |
|---|---|
| `id` | lowercase letters, digits, `-` and `_` |
| `prompt` or `turns` | one message, or several answered in order (the last reply is checked) |
| `files` | `{relative path: text}` created in the workspace first |
| `confirm` | `approve` (default) or `deny`: the answer to confirmation prompts |
| `timeout` | seconds before giving up on a model that never answers (default 120) |
| `expect.no_tools` | no tool may be called |
| `expect.tools_called` | each must be called at least once; `a\|b` accepts either |
| `expect.tools_not_called`, `max_tool_calls` | limits on what it may do |
| `expect.reply_matches` / `reply_excludes` | case-insensitive regular expressions on the final reply |
| `expect.files` | `{path: {exists: true/false, contains: text}}` checked after the run |
| `expect.max_seconds` | the whole scenario, all turns |

`{workspace}` in a prompt or a file is the workspace path. Unknown keys, paths that leave the
workspace, and bad regular expressions are rejected when the file loads, so a typo cannot
quietly switch a check off.

## What it has found

- A follow-up such as "Now read it back to me." matched no tool group, so the model was offered
  no tools and answered from memory, claiming it had read the file. Short follow-ups now keep
  the previous request's tools (`jarvis/agent/routing.py`).
- Phrases like "find the file" and "show me what it says" were taken as requests to look at the
  screen, which added a "screenshot could not be captured" note to the model's prompt. File
  requests no longer count as vision requests.
- The peak-memory line showed `llama-server` growing from 1 GB to 2.9 GB over 13 scenarios: its
  prompt cache defaults to 8 GB of host RAM. Capping it (`llm.server.cache_ram`, default 0)
  held it at 1.4 GB with the same reply times; see [llama.cpp](llamacpp.md#flags-that-matter).
