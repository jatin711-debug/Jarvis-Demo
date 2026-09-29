# Current permission behavior

Jarvis now checks built-in tool arguments at dispatch. The gateway runs before
confirmation or a handler, classifies the effect, and binds an approval to the
tool name and validated arguments with an action fingerprint. If a check fails
or the user refuses, the agent ends that turn without trying another mutation.
The text task CLI also checks its fixture output path or Calculator app scope
before acting. Starting or resuming that explicit CLI workflow is the approval
for its fixed sequence; it does not prompt for each button.

You can narrow a profile in YAML:

```yaml
security:
  confirm_dangerous_ops: true
  command_whitelist: true
  allowed_commands: [git, echo]
  allowed_directories: ['D:\Projects\MyProject']
  allowed_applications: [Calculator, Notepad]
  allowed_hosts: [example.com]
```

`allowed_directories` applies to reads, writes, copies, moves, deletes, and
screenshots. Resolved paths must fall under one of those directories; symlink
escapes are rejected when checked. An empty list denies all file targets.
`allowed_applications` and `allowed_hosts` are optional. Omit them (`null`) to
retain legacy unrestricted app/host selection; set an empty list to deny all.
Host names match exactly, not by suffix. Only absolute HTTP(S) URLs without
embedded credentials are accepted.

When confirmations are enabled, writes, destructive operations, commands,
desktop actions, and browser form actions require approval. A voice approval
must be exactly “yes” (apart from trailing punctuation); text mode also
requires `yes` in an interactive terminal. With confirmations disabled, scope
checks still apply, but approved-scope actions proceed without prompting.

Limitations are important:

- The shipped `config.yaml` is the standard profile (confirmations on, command
  whitelist on, file access under the home directory). A full-access profile
  (no confirmations, no whitelist, both drive roots) is a deliberate opt-in; see
  "Profiles" below. It is intentionally broad and should be narrowed before
  unattended or multi-user use.
- The shell command tool remains powerful. Its whitelist rejects shell
  compositions and command interpreters, but it is not a process sandbox.
  Catastrophic patterns are blocked even when the whitelist is disabled.
- Global typing, keystrokes, screen-coordinate clicks and title-substring
  window actions were removed because they act without an exact target. Desktop
  control goes through Cua's observed pid/window/snapshot tools; with
  `desktop.backend: legacy`, Jarvis can open applications but cannot drive them.
- With `allowed_hosts` set, active-page browser actions are currently refused
  until the URL/tab is passed to the gateway. Direct URL navigation is checked
  before dispatch, but redirects and DNS changes are not yet constrained.
- Filesystem path checks are not an OS sandbox and cannot rule out a race if
  another process changes a symlink between checking and use. Direct calls to
  a handler outside the tool registry also bypass the dispatch gateway.

## Profiles

`config.yaml` in the repository is the **standard profile**: writes, commands,
desktop actions and browser form actions ask for an exact "yes"; shell commands
must start with an allowed executable; file tools stay under `~`. Catastrophic
commands are blocked in every profile.

Machine-specific choices go in `config.local.yaml`, which is untracked and is
loaded automatically ahead of `config.yaml`. It can start from the standard
profile with `extends:` and override only what differs (mappings merge, lists
are replaced):

```yaml
extends: config.yaml
voice:
  input_device: 2
security:
  allowed_directories: ['~', 'D:\Projects']
```

A **full-access profile** for a personal machine sets `confirm_dangerous_ops:
false`, `command_whitelist: false`, empty `allowed_commands`/`blocked_commands`
and adds the drive roots to `allowed_directories`. Argument validation, scope
checks and the catastrophic-command block still run. Use `jarvis --config PATH`
or `JARVIS_CONFIG` to select a specific file.
