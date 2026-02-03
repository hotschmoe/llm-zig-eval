# llm-zig-eval

**Find which LLM writes the best Zig code.**

A comprehensive benchmark suite that evaluates LLM models on challenging Zig programming tasks - testing memory management, concurrency, and comptime metaprogramming.

## Goal

Determine which model produces the highest-quality Zig code, with cost as a secondary consideration. If Model B achieves 90% of Model A's performance at half the price, we want to know.

## Features

- **OpenRouter Integration** - Single API gateway to test models from OpenAI, Anthropic, Meta, DeepSeek, and more
- **The Gauntlet** - 3 hard problems testing core Zig competencies
- **Parallel Execution** - Benchmark multiple models concurrently with configurable parallelism
- **Automated Evaluation** - Compile, test, and measure each solution
- **Council of Judges** - Multi-model consensus scoring to eliminate bias
- **Cost Tracking** - Per-request token usage and dollar costs
- **Rich Terminal UI** - Styled panels, progress indicators, and formatted tables via [rich_zig](https://github.com/hotschmoe/rich_zig)

## Installation

### Requirements

- Zig 0.15.2+
- OpenRouter API key ([get one here](https://openrouter.ai/keys))

### Build

```bash
git clone https://github.com/hotschmoe/llm-zig-eval
cd llm-zig-eval
zig build
```

### Environment Setup

**PowerShell:**
```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
```

**Bash/Zsh:**
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

**Or create a `.env` file:**
```
OPENROUTER_API_KEY=sk-or-v1-...
```

## CLI Reference

### Usage

```bash
zig build run -- [OPTIONS]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--models=MODEL1,MODEL2` | (required) | Comma-separated list of model IDs to benchmark |
| `--runs=N` | 1 | Number of runs per model per problem |
| `--parallel=N` | 4 | Max concurrent model benchmarks |
| `--council` | off | Enable Council of Judges consensus scoring |
| `--output=FORMAT` | pretty | Output format: `pretty` or `json` |
| `--help`, `-h` | - | Show help message |

### Examples

```bash
# Single model benchmark
zig build run -- --models=openai/gpt-4o-mini

# Multiple models in parallel
zig build run -- --models=anthropic/claude-3.5-sonnet,openai/gpt-4o --parallel=2

# Full benchmark with council scoring and multiple runs
zig build run -- --models=anthropic/claude-3.5-sonnet,openai/gpt-4o,meta-llama/llama-3-70b-instruct --council --runs=3

# JSON output for CI/CD integration
zig build run -- --models=openai/gpt-4o-mini --output=json

# Sequential execution (disable parallelism)
zig build run -- --models=anthropic/claude-3.5-sonnet,openai/gpt-4o --parallel=1
```

## The Gauntlet (Benchmark Problems)

| # | Problem | Tests |
|---|---------|-------|
| 1 | **Arena Allocator** | Memory layout, alignment, manual allocation |
| 2 | **Mock TCP Socket** | Threading, async patterns, error handling |
| 3 | **JSON-to-Struct** | Comptime reflection, `@typeInfo`, parsing |

## Output

```
MODEL                   | TIME   | SCORE | COST    | RATING
------------------------+--------+-------+---------+--------
anthropic/claude-3.5    | 4.2s   | 3/3   | $0.0042 | S (9.5)
openai/gpt-4o           | 2.8s   | 3/3   | $0.0038 | A (8.8)
meta/llama-3-70b        | 5.1s   | 1/3   | $0.0005 | C (6.0)
```

## Dependencies

- [rich_zig](https://github.com/hotschmoe/rich_zig) - Terminal formatting, progress indicators, and styled output

## Why OpenRouter?

Wide model access with normalized API responses. We can benchmark everything from GPT-4o to open-source Llama models through one client. If an open-source model makes a compelling case, we'll invest in local hardware.

## License

MIT
