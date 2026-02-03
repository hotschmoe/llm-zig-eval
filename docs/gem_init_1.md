Here is a comprehensive blueprint for **`swe-zig-bench`**.

### 1. ASCII Diagram: System Architecture

This diagram illustrates the flow of data from the CLI trigger to the final report.

```text
+-----------------------------------------------------------------------+
|                         USER (CLI ARGUMENTS)                          |
|  $ swe-zig-bench --models=gpt4,claude3 --runs=3 --eval-council=true   |
+-----------------------------------------------------------------------+
            |
            v
+-----------------------------+       +---------------------------------+
|   MAIN ORCHESTRATOR (ZIG)   | ----> |   PROMPT MANAGER (FileSystem)   |
|   (src/main.zig)            |       |   (Loads 3 Hard problems)       |
+-----------------------------+       +---------------------------------+
            |
            | (Parallel Requests)
            v
+-----------------------------+       +---------------------------------+
|      LLM CLIENT LAYER       | <---> |  EXTERNAL LLM APIs              |
|   (src/llm_client.zig)      |       |  (OpenAI, Anthropic, DeepSeek)  |
+-----------------------------+       +---------------------------------+
            |
            | (Returns Raw Code)
            v
+-----------------------------+       +---------------------------------+
|     FILE WRITER & SANITIZER | ----> |  WORKSPACE DIR                  |
|   (Extracts code blocks)    |       |  ./zig-cache/gen/{model}/{q1}/  |
+-----------------------------+       +---------------------------------+
            |
            v
+-----------------------------+
|      EVALUATION ENGINE      |
|     (src/evaluator.zig)     |
+-----------------------------+
| 1. COMPILATION (zig build)  |--[Fail?]--> Record Compile Error
| 2. TEST RUN (zig test)      |--[Fail?]--> Record Test Fail
| 3. STATS (LOC / Timer)      |--[Pass?]--> Record Success
+-----------------------------+
            |
            | (If Stretch Goal Active: Send Code back to LLM)
            v
+-----------------------------+       +---------------------------------+
|     THE "COUNCIL" JUDGE     | <---> |  LLM API (e.g., GPT-4o)         |
|   (src/council.zig)         |       |  (Prompts for Code Review)      |
+-----------------------------+       +---------------------------------+
            |
            v
+-----------------------------------------------------------------------+
|                             FINAL REPORT                              |
| --------------------------------------------------------------------- |
| Model   | Q1 (Time) | Q2 (Time) | Q3 (Time) | Cost  | LOC | Council   |
| --------------------------------------------------------------------- |
| Claude3 | PASS (2s) | FAIL      | PASS (5s) | $0.04 | 120 | 9/10 (S)  |
| GPT-4o  | PASS (1s) | PASS (2s) | PASS (3s) | $0.05 | 110 | 8/10 (A)  |
+-----------------------------------------------------------------------+
```

---

### 2. File Structure

This structure leverages Zig's build system to treat generated solutions as individual compile targets.

```text
swe-zig-bench/
├── build.zig                 # Main build script (orchestrates the bench tool)
├── build.zig.zon             # Dependencies (http client, env loader)
├── .env                      # API Keys (OPENAI_KEY, ANTHROPIC_KEY)
├── prompts/                  # The System Prompts & Question descriptions
│   ├── system_prompt.txt     # "You are an expert Zig programmer..."
│   ├── q1_allocator.txt
│   ├── q2_concurrency.txt
│   └── q3_comptime.txt
├── src/
│   ├── main.zig              # Entry point: parses args, starts threads
│   ├── config.zig            # Handles API keys and Model definitions
│   ├── llm/
│   │   ├── client.zig        # HTTP implementation (std.http or generic)
│   │   └── providers.zig     # Structs for OpenAI/Anthropic/etc response parsing
│   ├── core/
│   │   ├── generator.zig     # Sends prompts, receives code, writes .zig files
│   │   ├── runner.zig        # Spawns `zig test`, captures stderr/stdout, measures time
│   │   └── council.zig       # (Stretch) Sends solution to LLM for qualitative rating
│   └── util/
│       ├── tokens.zig        # Estimator for token costs
│       └── reporter.zig      # Formats the final ASCII table/JSON output
└── tests/
    └── harnesses/            # Pre-written tests the AI must pass
        ├── q1_test.zig       # Imports generated code and runs assertions
        ├── q2_test.zig
        └── q3_test.zig
```

---

### 3. Benchmark Questions (The "Gauntlet")

To properly benchmark Zig proficiency, we need questions that touch on **Allocators**, **Concurrency**, and **Comptime**.

The prompt sent to the LLM will provide the "Interface Specification" and the "Test Harness" they must satisfy.

#### Question 1: Memory Safety & Manual Management
**Title:** "The Zero-Copy Parser"
**Objective:** Test usage of `std.mem`, `std.ArrayList`, and proper Allocator lifecycle management (defer release).

**The Prompt:**
> Create a Zig struct named `LogParser`.
> 1. It must accept an `std.mem.Allocator` in `init`.
> 2. It must have a method `parse(self: *LogParser, raw_log: []const u8) !ParsedEntry`.
> 3. `ParsedEntry` must utilize a `HashMap` to count occurrences of specific keywords found in the log line.
> 4. **Constraint:** You must minimize memory copying. Use slices (`[]const u8`) referencing the original input string where possible, but ensure the HashMap manages its keys correctly.
> 5. Implement `deinit` to clean up all memory strictly. No leaks allowed (our test runner uses `GeneralPurposeAllocator` with `.detect_leaks = true`).

#### Question 2: Concurrency & Thread Safety
**Title:** "The Thread-Safe Job Queue"
**Objective:** Test usage of `std.Thread`, `std.Thread.Mutex`, and `std.Thread.Condition`.

**The Prompt:**
> Implement a generic thread-safe queue named `JobQueue(T: type)`.
> 1. The struct must be generic over type `T`.
> 2. Implement `push(item: T) void` and `pop() T` (blocking wait if empty).
> 3. Implement `tryPop() ?T` (non-blocking).
> 4. Use `std.Thread.Mutex` to ensure safety and `std.Thread.Condition` to handle the blocking `pop`.
> 5. The solution must compile and pass a test where 10 producer threads push integers and 10 consumer threads pop them simultaneously without deadlocking or race conditions.

#### Question 3: Metaprogramming (Comptime)
**Title:** "The Compile-Time Schema Validator"
**Objective:** Test Zig's `comptime`, `@typeInfo`, and reflection capabilities.

**The Prompt:**
> Write a function `validateStruct(comptime T: type, data: T) !void`.
> 1. This function must inspect the struct `T` at compile time.
> 2. It must check for a custom declaration (field mixin) or naming convention.
>    - If a field is named `email`, the function must validate at runtime that the string contains an '@'.
>    - If a field is of type `u8` and named `age`, it must ensure the value is > 0 and < 150.
> 3. If validation fails, return a specific error from an error set `ValidationError`.
> 4. The iteration over the struct fields must happen at **comptime** using `std.meta` or `@typeInfo`, generating the validation code for the specific type `T` efficiently.

---

### 4. Implementation Details for the "Council" (Stretch Goal)

To achieve the qualitative rating, your `src/council.zig` should construct a prompt like this to send to a high-end model (e.g., GPT-4o or Claude 3.5 Sonnet) after the code is generated:

**System Prompt for Council:**
> "You are the High Council of Zig. You evaluate code based on 4 criteria:
> 1. **Idiomatic Zig:** (Usage of defer, error handling, strict types).
> 2. **Memory Safety:** (Correct allocator usage, no potential double-frees).
> 3. **Performance:** (Avoidance of unnecessary allocations).
> 4. **Readability:** (Variable naming, logic flow).
>
> Input Code: [INSERT_GENERATED_CODE]
>
> Output a JSON object: { "score": 8.5, "critique": "Good use of defer, but used an ArrayList where a fixed buffer would suffice." }"

### 5. Calculated Metrics Logic

When the `swe-zig-bench` reporter runs, it calculates:

1.  **Time:** `std.time.nanoTimestamp()` before and after the LLM request.
2.  **Solutions:** The number of tests passed via `zig test`.
3.  **LOC:** Read the generated file, strip comments/whitespace, count lines.
4.  **Cost:**
    *   *Input Tokens:* Length of Prompt / ~4 chars.
    *   *Output Tokens:* Length of Solution / ~4 chars.
    *   *Math:* `(In * Price_In) + (Out * Price_Out)` (Store pricing in `config.zig`).