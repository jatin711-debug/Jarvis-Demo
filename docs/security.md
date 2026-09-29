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

- `config.yaml` currently disables confirmations, disables the command
  whitelist, and permits both drive roots. That is intentionally broad and
  should be narrowed before unattended or multi-user use.
- The shell command tool remains powerful. Its whitelist rejects shell
  compositions and command interpreters, but it is not a process sandbox.
  Catastrophic patterns are blocked even when the whitelist is disabled.
- Legacy global typing, keystrokes, and title-substring window actions are
  refused by the gateway because they lack an exact target; use Cua's observed
  pid/window/snapshot controls. The task CLI's scoped workflows do not use
  these legacy actions.
- With `allowed_hosts` set, active-page browser actions are currently refused
  until the URL/tab is passed to the gateway. Direct URL navigation is checked
  before dispatch, but redirects and DNS changes are not yet constrained.
- Filesystem path checks are not an OS sandbox and cannot rule out a race if
  another process changes a symlink between checking and use. Direct calls to
  a handler outside the tool registry also bypass the dispatch gateway.
