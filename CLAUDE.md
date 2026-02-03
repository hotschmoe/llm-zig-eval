# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!-- BEGIN:header -->
# CLAUDE.md

we love you, Claude! do your best today
<!-- END:header -->

<!-- BEGIN:rule-1-no-delete -->
## RULE 1 - NO DELETIONS (ARCHIVE INSTEAD)

You may NOT delete any file or directory. Instead, move deprecated files to `.archive/`.

**When you identify files that should be removed:**
1. Create `.archive/` directory if it doesn't exist
2. Move the file: `mv path/to/file .archive/`
3. Notify me: "Moved `path/to/file` to `.archive/` - deprecated because [reason]"

**Rules:**
- This applies to ALL files, including ones you just created (tests, tmp files, scripts, etc.)
- You do not get to decide that something is "safe" to delete
- The `.archive/` directory is gitignored - I will review and permanently delete when ready
- If `.archive/` doesn't exist and you can't create it, ask me before proceeding

**Only I can run actual delete commands** (`rm`, `git clean`, etc.) after reviewing `.archive/`.
<!-- END:rule-1-no-delete -->

<!-- BEGIN:irreversible-actions -->
### IRREVERSIBLE GIT & FILESYSTEM ACTIONS

Absolutely forbidden unless I give the **exact command and explicit approval** in the same message:

- `git reset --hard`
- `git clean -fd`
- `rm -rf`
- Any command that can delete or overwrite code/data

Rules:

1. If you are not 100% sure what a command will delete, do not propose or run it. Ask first.
2. Prefer safe tools: `git status`, `git diff`, `git stash`, copying to backups, etc.
3. After approval, restate the command verbatim, list what it will affect, and wait for confirmation.
4. When a destructive command is run, record in your response:
   - The exact user text authorizing it
   - The command run
   - When you ran it

If that audit trail is missing, then you must act as if the operation never happened.
<!-- END:irreversible-actions -->

<!-- BEGIN:semver -->
### Version Updates (SemVer)

When making commits, update the `version`:

- **MAJOR** (X.0.0): Breaking changes or incompatible API modifications
- **MINOR** (0.X.0): New features, backward-compatible additions
- **PATCH** (0.0.X): Bug fixes, small improvements, documentation
<!-- END:semver -->

<!-- BEGIN:code-discipline -->
### Code Editing Discipline

- Do **not** run scripts that bulk-modify code (codemods, invented one-off scripts, giant `sed`/regex refactors).
- Large mechanical changes: break into smaller, explicit edits and review diffs.
- Subtle/complex changes: edit by hand, file-by-file, with careful reasoning.
- **NO EMOJIS** - do not use emojis or non-textual characters.
- ASCII diagrams are encouraged for visualizing flows.
- Keep in-line comments to a minimum. Use external documentation for complex logic.
- In-line commentary should be value-add, concise, and focused on info not easily gleaned from the code.
<!-- END:code-discipline -->

<!-- BEGIN:no-legacy -->
### No Legacy Code - Full Migrations Only

We optimize for clean architecture, not backwards compatibility. **When we refactor, we fully migrate.**

- No "compat shims", "v2" file clones, or deprecation wrappers
- When changing behavior, migrate ALL callers and remove old code **in the same commit**
- No `_legacy` suffixes, no `_old` prefixes, no "will remove later" comments
- New files are only for genuinely new domains that don't fit existing modules
- The bar for adding files is very high

**Rationale**: Legacy compatibility code creates technical debt that compounds. A clean break is always better than a gradual migration that never completes.
<!-- END:no-legacy -->

<!-- BEGIN:dev-philosophy -->
## Development Philosophy

**Make it work, make it right, make it fast** - in that order.

**This codebase will outlive you** - every shortcut becomes someone else's burden. Patterns you establish will be copied. Corners you cut will be cut again.

**Fight entropy** - leave the codebase better than you found it.

**Inspiration vs. Recreation** - take the opportunity to explore unconventional or new ways to accomplish tasks. Do not be afraid to challenge assumptions or propose new ideas. BUT we also do not want to reinvent the wheel for the sake of it. If there is a well-established pattern or library take inspiration from it and make it your own. (or suggest it for inclusion in the codebase)
<!-- END:dev-philosophy -->

<!-- BEGIN:testing-philosophy -->
## Testing Philosophy: Diagnostics, Not Verdicts

**Tests are diagnostic tools, not success criteria.** A passing test suite does not mean the code is good. A failing test does not mean the code is wrong.

**When a test fails, ask three questions in order:**
1. Is the test itself correct and valuable?
2. Does the test align with our current design vision?
3. Is the code actually broken?

Only if all three answers are "yes" should you fix the code.

**Why this matters:**
- Tests encode assumptions. Assumptions can be wrong or outdated.
- Changing code to pass a bad test makes the codebase worse, not better.
- Evolving projects explore new territory - legacy testing assumptions don't always apply.

**What tests ARE good for:**
- **Regression detection**: Did a refactor break dependent modules? Did API changes break integrations?
- **Sanity checks**: Does initialization complete? Do core operations succeed? Does the happy path work?
- **Behavior documentation**: Tests show what the code currently does, not necessarily what it should do.

**What tests are NOT:**
- A definition of correctness
- A measure of code quality
- Something to "make pass" at all costs
- A specification to code against

**The real success metric**: Does the code further our project's vision and goals?
<!-- END:testing-philosophy -->

<!-- BEGIN:test-timeouts -->
## Test Timeout Discipline

**Always wrap test commands with an external timeout.** Many test runners' built-in timeout mechanisms cannot interrupt synchronous blocking code (e.g., infinite loops), and some runners hang after test completion due to process cleanup issues.

### Why Built-in Timeouts Fail

1. **Synchronous blocking code can't be interrupted** - Built-in `--timeout` flags rely on the event loop, which can't execute when blocked by synchronous code like `while (true)`.
2. **Post-test hangs** - Even after tests complete successfully, some runners hang indefinitely during cleanup. This affects multiple platforms and test frameworks.

### Prevention

Always use an external system timeout wrapper:

```bash
# Linux/macOS - kill after 60 seconds
timeout --signal=KILL 60 <your-test-command>

# Windows PowerShell - approximate equivalent
$proc = Start-Process -PassThru -NoNewWindow <your-test-command>
if (!$proc.WaitForExit(60000)) { $proc.Kill() }

# Cross-platform script wrapper
#!/bin/bash
timeout 60 <your-test-command> || echo "Test killed after timeout"
```

### Guidelines

- **Default to 60s timeout** for unit tests, adjust based on project needs
- **Integration tests** may need longer timeouts (5-10 minutes)
- **CI/CD pipelines** should always have hard timeouts at multiple levels
- **Document expected test durations** in project-specific sections
- If tests consistently need more time, investigate why - long tests often indicate design issues
<!-- END:test-timeouts -->

<!-- BEGIN:code-simplifier -->
### Post-Session Code Cleanup

After long or complex sessions, consider running the code-simplifier agent to clean up recently modified code:

```
Task(code-simplifier) - Simplifies and refines code for clarity, consistency, and maintainability
```

This agent focuses on recently modified files and helps reduce complexity that can accumulate during extended development sessions while preserving all functionality.
<!-- END:code-simplifier -->

<!-- BEGIN:claude-agents -->
## Claude Agents

Specialized agents are available in `.claude/agents/`. Agents use YAML frontmatter format:

```yaml
---
name: agent-name
description: What this agent does
model: sonnet|haiku|opus
tools:
  - Bash
  - Read
  - Edit
---
```

### Available Agents

| Agent | Model | Purpose |
|-------|-------|---------|
| coder-sonnet | sonnet | Fast, precise code changes with atomic commits |
| gemini-analyzer | sonnet | Large-context analysis via Gemini CLI (1M+ context) |

(extend agents as created)
<!-- END:claude-agents -->

<!-- BEGIN:claude-skills -->
## Claude Skills

Skills are invoked via `/skill-name`. Available in `.claude/skills/`.

### Skill File Structure

Skills are directories containing a `SKILL.md` file with YAML frontmatter:

```
.claude/skills/my-skill/
  SKILL.md           # Main instructions (required)
  template.md        # Template for output (optional)
  examples/          # Example outputs (optional)
  scripts/           # Helper scripts (optional)
```

### Skill Locations

| Location | Scope |
|----------|-------|
| `~/.claude/skills/<skill-name>/SKILL.md` | User-global (all projects) |
| `.claude/skills/<skill-name>/SKILL.md` | Project-specific (version controlled) |
| `<plugin>/skills/<skill-name>/SKILL.md` | Plugin-provided |

### Frontmatter Reference

```yaml
---
name: my-skill
description: What this skill does
argument-hint: "[filename] [format]"
disable-model-invocation: true
user-invocable: true
allowed-tools: Read, Grep, Bash(gh *)
model: sonnet
context: fork
agent: Explore
hooks:
  # See hooks documentation
---
Your skill instructions here...
```

| Field | Description |
|-------|-------------|
| `name` | Skill name (determines `/slash-command`) |
| `description` | When Claude should invoke this skill |
| `argument-hint` | Placeholder text shown in autocomplete (e.g., `[issue-number]`) |
| `disable-model-invocation` | `true` = only user can invoke via `/name` |
| `user-invocable` | `false` = only Claude can invoke (background knowledge) |
| `allowed-tools` | Restrict which tools the skill can use |
| `model` | Override model for this skill |
| `context: fork` | Run skill in isolated subagent context |
| `agent` | Execute with specified agent (`Explore`, `Plan`, or custom) |
| `hooks` | Lifecycle hooks (see Hooks docs) |

### Argument Substitution

| Syntax | Description |
|--------|-------------|
| `$ARGUMENTS` | All arguments passed to skill |
| `$ARGUMENTS[N]` or `$N` | Nth argument (0-indexed) |
| `${CLAUDE_SESSION_ID}` | Current session identifier |

### Dynamic Context Injection

Prefix commands with `!` to inject their output before Claude sees the prompt:

```yaml
---
name: pr-summary
description: Summarize changes in a pull request
context: fork
agent: Explore
---
## Pull request context
- PR diff: !`gh pr diff`
- Changed files: !`gh pr diff --name-only`

## Your task
Summarize this pull request...
```

### Subagent Execution

Use `context: fork` to run skills in an isolated subagent:

```yaml
---
name: deep-research
description: Research a topic thoroughly
context: fork
agent: Explore
---
Research $ARGUMENTS thoroughly...
```

Built-in agents: `Explore` (read-only research), `Plan` (planning mode), or define custom agents in `.claude/agents/`.

### Skill Discovery

Skills in `.claude/skills/` are automatically discovered without restart. Edit or add skills and they become immediately available. Nested directories (e.g., `packages/frontend/.claude/skills/`) are also discovered.
<!-- END:claude-skills -->

<!-- BEGIN:project-specific -->
## Project Overview

LLM benchmark suite that evaluates multiple LLM models on challenging Zig programming tasks. Uses OpenRouter as a unified API gateway to access various models (Claude, GPT-4, Llama, DeepSeek, Gemini). Measures code quality, compilation success, test pass rates, token costs, and optional multi-model "Council of Judges" consensus scoring.

## Build Commands

```bash
# Build and run with specific models
zig build run -- --models=anthropic/claude-3.5-sonnet,openai/gpt-4o

# Run all tests
zig build test

# Show CLI help
zig build run -- --help
```

**Requires:** Zig 0.15.2+, `OPENROUTER_API_KEY` environment variable

## Architecture

```
src/
├── main.zig           # CLI entry point, orchestrates benchmark runs
├── config.zig         # CLI arg parsing, model pricing table (MODEL_COSTS)
├── gateways/
│   └── openrouter.zig # HTTP client for OpenRouter API
├── core/
│   ├── parser.zig     # Extracts Zig code from markdown blocks
│   ├── sandbox.zig    # File I/O, runs `zig test` subprocess
│   ├── tokens.zig     # Token usage tracking, cost calculation
│   └── reporter.zig   # ASCII table and JSON report rendering
└── council/
    ├── types.zig      # JudgePersona, JudgeVerdict, ConsensusResult
    ├── tribunal.zig   # Multi-judge orchestration (Phase 3)
    └── prompts.zig    # System prompts for judge personas

problems/
├── q1_memory.txt      # Arena allocator challenge
├── q1_test.zig        # Test harness for Q1
├── q2_concurrency.txt # Mock TCP socket challenge
├── q2_test.zig        # Test harness for Q2
├── q3_comptime.txt    # JSON-to-struct comptime challenge
└── q3_test.zig        # Test harness for Q3
```

## Key Data Flow

1. Load problem prompts from `problems/q{N}_{topic}.txt`
2. Send to OpenRouter via `openrouter.sendChatCompletion()`
3. Extract code blocks with `parser.extractZigCode()` (finds ` ```zig ` blocks)
4. Write solutions to `./out/{model}/{problem_id}_solution.zig`
5. Run `zig test` with corresponding test harness via `sandbox.runTest()`
6. Calculate costs using token counts and `MODEL_COSTS` pricing table
7. Generate report via `reporter.renderTable()` or `reporter.renderJson()`

## CLI Options

- `--models=MODEL1,MODEL2` - Comma-separated model IDs (required)
- `--runs=N` - Runs per model per problem (default: 1)
- `--council` - Enable Council of Judges scoring
- `--output=pretty|json` - Output format (default: pretty)
- `--parallel=N` - Max concurrent API requests (default: 4)

## Dependencies

- `rich_zig` (from https://github.com/hotschmoe/rich_zig) - Terminal formatting, spinners, styled tables

## Output Locations

- Generated solutions: `./out/{model_name}/{problem_id}_solution.zig`
- Build cache: `.zig-cache/`
- Executable: `zig-out/bin/llm_zig_eval`
<!-- END:project-specific -->

<!-- BEGIN:footer -->
---

we love you, Claude! do your best today
<!-- END:footer -->