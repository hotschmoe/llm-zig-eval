This is a smart architectural shift. Using **OpenRouter** as the unified gateway simplifies your HTTP client logic significantly (you only write one API client that speaks "OpenAI-flavor" JSON) while giving you access to nearly every top-tier model (Claude 3.5, GPT-4o, Llama 3, etc.) to benchmark against each other.

It also solves the **Cost/Token tracking** problem, as OpenRouter normalizes the `usage` fields in the JSON response.

Here is the updated ASCII architecture, file structure, and the "Gauntlet" (Benchmark Questions).

### 1. ASCII Diagram: The OpenRouter Architecture

```text
+-----------------------------------------------------------------------+
|                       USER CLI (swe-zig-bench)                        |
| $ zig-bench --models=anthropic/claude-3-opus,google/gemini-pro-1.5    |
+-----------------------------------------------------------------------+
           |
           v
+-----------------------------+      +----------------------------------+
|      MAIN ORCHESTRATOR      |      |   BENCHMARK REGISTRY (Files)     |
|   (src/main.zig)            | <--- |   3 Hard Problems (.txt)         |
|   - Thread Pool Manager     |      |   Test Harnesses (.zig)          |
+-----------------------------+      +----------------------------------+
           |
           | (HTTP POST /api/v1/chat/completions)
           v
+-----------------------------+
|    OPENROUTER GATEWAY       |
|   (src/gateways/router.zig) |
|   - Headers: HTTP-Referer   |
|   - Auth: Bearer $OR_KEY    |
+-----------------------------+
           |
           v
    ( The Internet )
           |
+-------------------------------------------------------+
|                 OPENROUTER.AI API                     |
|  [ Routes request to Anthropic, OpenAI, Meta, etc. ]  |
|  [ Returns JSON with "usage": { "total_tokens": N } ] |
+-------------------------------------------------------+
           |
           | (JSON Response + Token Counts)
           v
+-----------------------------+      +----------------------------------+
|    RESULT PARSER & WRITER   | ---> |   WORKSPACE / SANDBOX            |
|   (src/core/parser.zig)     |      |   ./out/claude-3-opus/q1.zig     |
|   - Extracts code block     |      +----------------------------------+
|   - Captures Cost/Tokens    |                |
+-----------------------------+                | (System Call)
           |                                   v
           |                         +----------------------------------+
           |                         |   ZIG COMPILER & TEST RUNNER     |
           |                         |   $ zig test q1_harness.zig      |
           |                         +----------------------------------+
           v                                   |
+-----------------------------+                | (Pass/Fail + Duration)
|      REPORT GENERATOR       | <--------------+
|   - Calcs $ based on Model  |
|   - Generates Table         |
+-----------------------------+
```

---

### 2. File Structure

We structure this so that the `client` is agnostic to the model, relying on OpenRouter's standardization.

```text
swe-zig-bench/
├── build.zig                 
├── build.zig.zon             # Dependencies (e.g., a zig JSON parser if not using std)
├── .env                      # OPENROUTER_API_KEY=sk-or-v1-...
├── problems/                 # The "Gauntlet"
│   ├── q1_memory.txt         # Prompt for Q1
│   ├── q1_test.zig           # Test Harness for Q1
│   ├── q2_concurrency.txt
│   ├── q2_test.zig
│   ├── q3_comptime.txt
│   └── q3_test.zig
├── src/
│   ├── main.zig              # Entry point
│   ├── config.zig            # Loads .env and Model Cost Table (if OR doesn't send $)
│   ├── gateways/
│   │   └── openrouter.zig    # Handles HTTP Req, sends "model" param dynamically
│   ├── core/
│   │   ├── sandbox.zig       # Writes files, runs `zig test` child process
│   │   ├── tokens.zig        # Parses the `usage` JSON object
│   │   └── reporter.zig      # ASCII Table renderer
│   └── council/
│       └── judge.zig         # (Stretch) Sends code BACK to OpenRouter for rating
└── zig-cache/                # (Ignored) Build artifacts
```

---

### 3. Implementation Logic: Intercepting Costs

OpenRouter responses follow the standard OpenAI format. Your `openrouter.zig` needs to parse this JSON structure:

```json
{
  "id": "gen-123...",
  "model": "anthropic/claude-3-opus",
  "choices": [ ... ],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 340,
    "total_tokens": 490
  }
}
```

**Note on Cost:** OpenRouter does not always send the *dollar amount* in the response headers. You usually have two options in your `config.zig`:
1.  **Hardcoded Rates:** A hashmap of `model_id` -> `cost_per_m_tokens`.
2.  **Generation Stats:** OpenRouter has an API endpoint `/api/v1/generation?id=...` that returns the exact cost, but that requires a second API call.
    *   *Recommendation:* Start with a lookup table in `config.zig` for the top 10 models you test. It's faster.

---

### 4. Benchmark Examples (The "Gauntlet")

These questions are designed to be "Single File Compliant"—meaning the LLM creates a file that imports `std` and your test harness imports the LLM's file.

#### Problem 1: The "Arena Allocator from Scratch"
**Focus:** Memory, Pointers, Manual Layout
**Prompt:**
> "Write a Zig struct named `MiniArena` in a file `solution.zig`.
> 1. It must manage a fixed-size buffer of 1024 bytes internally (stack or static).
> 2. Implement `fn alloc(self: *MiniArena, size: usize) ![]u8`.
> 3. It must return a slice to the internal buffer, advancing an internal pointer.
> 4. Return `error.OutOfMemory` if the buffer is full.
> 5. Implement `fn reset(self: *MiniArena) void` to rewind the pointer.
> 6. Ensure proper alignment (8-byte alignment) for the allocations."

**Evaluation:**
The `q1_test.zig` harness will import your solution, alloc 3 ints, check pointer addresses to ensure 8-byte alignment, then `reset()`, and alloc again to ensure memory reuse.

#### Problem 2: The "Async TCP Handshake" (Simulated)
**Focus:** Concurrency, Event Loops (Simulated), Error Handling
**Prompt:**
> "Write a Zig struct named `MockSocket` in `solution.zig`.
> 1. Implement a function `fn connect(address: []const u8) !void`.
> 2. This function must simulate a network delay using `std.time.sleep`.
> 3. If the address is 'bad_host', return `error.ConnectionRefused`.
> 4. Use `std.Thread` to spawn the connection logic so it doesn't block the main thread.
> 5. Return a 'handle' or struct that allows the caller to check `isConnected() bool`."

**Evaluation:**
The harness spawns 5 `MockSocket` connections. It asserts that the total runtime is roughly equal to *one* sleep duration (proving they ran in parallel), not the sum of all sleeps.

#### Problem 3: The "JSON Parser Mixin"
**Focus:** Comptime, Reflection, String Parsing
**Prompt:**
> "Write a generic function `fn jsonToStruct(comptime T: type, json: []const u8) !T` in `solution.zig`.
> 1. The function must accept a string like `{\"x\": 10, \"y\": 20}`.
> 2. It must use `@typeInfo` to iterate over the fields of `T` at compile time.
> 3. It must parse the simple JSON string and populate the struct.
> 4. LIMITATION: Do not use `std.json`. Write a naive parser that only handles top-level keys matching the struct field names.
> 5. Support `u8`, `u32`, and `[]const u8` (strings) as field types."

**Evaluation:**
The harness defines a struct `Point { x: u32, y: u32 }`, passes it and a string to your function, and asserts the result. This tests if the model understands Zig's powerful `comptime` reflection.

---

### 5. The Output Report

Once `swe-zig-bench` finishes, it parses the JSON usage logs and the `zig test` outputs to render this:

```text
BENCHMARK REPORT
================================================================================
MODEL                   | TIME   | SCORE | COST    | MEM SAFETY | LOC | RATING
--------------------------------------------------------------------------------
anthropic/claude-3.5    | 4.2s   | 3/3   | $0.0042 | PASS       | 145 | S (9.5)
openai/gpt-4o           | 2.8s   | 3/3   | $0.0038 | PASS       | 120 | A (8.8)
meta/llama-3-70b        | 5.1s   | 1/3   | $0.0005 | FAIL       | 160 | C (6.0)
--------------------------------------------------------------------------------
* Rating provided by "The Council" (anthropic/claude-3-opus judge)
```