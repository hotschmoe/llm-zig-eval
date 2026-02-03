# llm-zig-eval

**Find which LLM writes the best Zig code.**

A comprehensive benchmark suite that evaluates LLM models on challenging Zig programming tasks—testing memory management, concurrency, and comptime metaprogramming.

## Goal

Determine which model produces the highest-quality Zig code, with cost as a secondary consideration. If Model B achieves 90% of Model A's performance at half the price, we want to know.

## Features

- **OpenRouter Integration** — Single API gateway to test models from OpenAI, Anthropic, Meta, DeepSeek, and more
- **The Gauntlet** — 3 hard problems testing core Zig competencies
- **Automated Evaluation** — Compile, test, and measure each solution
- **Council of Judges** — Multi-model consensus scoring to eliminate bias
- **Cost Tracking** — Per-request token usage and dollar costs
- **Rich Terminal UI** — Live progress bars, spinners, and formatted tables via [rich_zig](https://github.com/hotschmoe/rich_zig)

## Dependencies

- [rich_zig](https://github.com/hotschmoe/rich_zig) — Terminal formatting, progress indicators, and styled output

## Quick Start

```bash
# Set your API key
$env:OPENROUTER_API_KEY = "sk-or-v1-..."

# Run the benchmark
zig build run -- --models=anthropic/claude-3.5-sonnet,openai/gpt-4o
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

## Why OpenRouter?

Wide model access with normalized API responses. We can benchmark everything from GPT-4o to open-source Llama models through one client. If an open-source model makes a compelling case, we'll invest in local hardware.

## License

MIT
