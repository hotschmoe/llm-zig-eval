//! llm-zig-eval main entry point
//! CLI orchestrator for benchmarking LLMs on Zig programming tasks.

const std = @import("std");
const rich = @import("rich_zig");
const lib = @import("llm_zig_eval");

const Config = lib.Config;
const Client = lib.Client;
const Message = lib.Message;
const Report = lib.Report;
const ModelResult = lib.ModelResult;
const ProblemResult = lib.ProblemResult;
const Sandbox = lib.Sandbox;
const SandboxResult = lib.SandboxResult;
const TokenUsage = lib.TokenUsage;
const PROBLEMS = lib.PROBLEMS;
const Tribunal = lib.Tribunal;
const ConsensusResult = lib.ConsensusResult;

const parser = lib.parser;
const sandbox = lib.sandbox;
const tokens = lib.tokens;
const config = lib.config;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{
        .safety = true,
    }){};
    defer {
        const check = gpa.deinit();
        if (check == .leak) {
            std.debug.print("WARNING: Memory leaks detected!\n", .{});
        }
    }
    const allocator = gpa.allocator();

    // Parse CLI arguments
    var cfg = config.parseArgs(allocator) catch |err| {
        switch (err) {
            error.HelpRequested => return,
            error.MissingApiKey => {
                std.debug.print("\nHint: Set OPENROUTER_API_KEY environment variable\n", .{});
                std.debug.print("  PowerShell: $env:OPENROUTER_API_KEY = \"sk-or-v1-...\"\n", .{});
                std.debug.print("  Bash: export OPENROUTER_API_KEY=\"sk-or-v1-...\"\n", .{});
                return err;
            },
            error.MissingModels => {
                std.debug.print("\nHint: Specify models to benchmark\n", .{});
                std.debug.print("  llm-zig-eval --models=anthropic/claude-3.5-sonnet,openai/gpt-4o\n", .{});
                return err;
            },
            else => return err,
        }
    };
    defer cfg.deinit();

    // Initialize rich console
    var console = rich.Console.init(allocator);
    defer console.deinit();

    // Print banner
    try printBanner(&console);

    // Initialize OpenRouter client
    var client = Client.init(allocator, cfg.api_key);
    defer client.deinit();

    // Initialize sandbox
    var sbx = Sandbox.init(allocator, "out");

    // Initialize report
    var report = Report.init(allocator);
    defer report.deinit();

    // Initialize tribunal for council judging (if enabled)
    var tribunal = Tribunal.init(allocator, &client);

    // Run benchmark for each model
    for (cfg.models) |model_id| {
        std.debug.print("\n🔄 Benchmarking: {s}\n", .{model_id});

        const model_result = try runModelBenchmark(
            allocator,
            &client,
            &sbx,
            &tribunal,
            model_id,
            cfg.runs,
            cfg.council,
        );

        try report.addResult(model_result);
    }

    // Render report
    switch (cfg.output_format) {
        .pretty => try report.renderTable(&console),
        .json => {
            const stdout_file = std.fs.File.stdout();
            var buf: [4096]u8 = undefined;
            var stdout = stdout_file.writer(&buf);
            try report.renderJson(&stdout.interface);
            try stdout.interface.flush();
        },
    }
}

/// Tracks a passed solution for council judging
const PassedSolution = struct {
    prompt: []const u8,
    code: []const u8,
};

fn runModelBenchmark(
    allocator: std.mem.Allocator,
    client: *Client,
    sbx: *Sandbox,
    tribunal: *Tribunal,
    model_id: []const u8,
    runs: u32,
    enable_council: bool,
) !ModelResult {
    var problem_results: std.ArrayList(ProblemResult) = .empty;
    errdefer problem_results.deinit(allocator);

    // Track passed solutions for council judging
    var passed_solutions: std.ArrayList(PassedSolution) = .empty;
    defer {
        for (passed_solutions.items) |sol| {
            allocator.free(sol.prompt);
            allocator.free(sol.code);
        }
        passed_solutions.deinit(allocator);
    }

    var total_usage = TokenUsage.init();
    var total_time_ms: i64 = 0;
    var passed: u32 = 0;

    // Create model output directory
    const model_dir = try sbx.createModelDir(model_id);
    defer allocator.free(model_dir);

    // Run each problem
    for (PROBLEMS) |problem| {
        std.debug.print("  ├─ {s}... ", .{problem.name});

        // Load problem prompt
        const prompt = try sandbox.loadProblemPrompt(allocator, problem);
        errdefer allocator.free(prompt);

        // Best result across runs
        var best_status: SandboxResult.Status = .compile_error;
        var best_loc: usize = 0;
        var best_code: ?[]const u8 = null;
        var problem_time_ms: i64 = 0;
        var retries_used: u32 = 0;

        const MAX_RETRIES: u32 = 4;
        const system_prompt =
            \\You are an expert Zig 0.15 programmer. Provide only the requested code in a single ```zig code block. No explanations outside the code.
            \\
            \\CRITICAL Zig 0.15 API Notes:
            \\- All exported types/functions must be `pub`
            \\- Use `std.Thread.sleep(ns)` not `std.time.sleep()`
            \\- Use `@typeInfo(T).@"struct".fields` not `.Struct.fields`
            \\- ArrayList uses `.empty` init: `var list: std.ArrayList(u8) = .empty;`
            \\- ArrayList methods take allocator: `list.append(allocator, item)`
        ;

        for (0..runs) |_| {
            // Build conversation for retry loop
            var conversation: std.ArrayList(Message) = .empty;
            defer conversation.deinit(allocator);

            // Track allocated strings for cleanup
            var allocated_msgs: std.ArrayList([]const u8) = .empty;
            defer {
                for (allocated_msgs.items) |msg| allocator.free(msg);
                allocated_msgs.deinit(allocator);
            }

            // Initial messages
            try conversation.append(allocator, .{ .role = .system, .content = system_prompt });
            try conversation.append(allocator, .{ .role = .user, .content = prompt });

            var retry: u32 = 0;
            while (retry < MAX_RETRIES) : (retry += 1) {
                var response = try client.sendChatCompletion(model_id, conversation.items);
                defer response.deinit(allocator);

                problem_time_ms += response.response_time_ms;
                total_usage.add(.{
                    .prompt_tokens = response.usage.prompt_tokens,
                    .completion_tokens = response.usage.completion_tokens,
                    .total_tokens = response.usage.total_tokens,
                });

                // Extract code
                const code = try parser.extractZigCode(allocator, response.content) orelse {
                    std.debug.print("(no code) ", .{});
                    break; // Can't retry without code
                };

                // Write to sandbox
                const solution_path = try sbx.writeSolution(model_dir, problem.id, code);
                defer allocator.free(solution_path);

                // Run tests
                var test_result = try sbx.runTest(solution_path, problem.test_path);
                defer test_result.deinit();

                // Check result
                if (test_result.status == .pass) {
                    // Success!
                    if (best_code) |old_code| allocator.free(old_code);
                    best_status = .pass;
                    best_loc = parser.countLoc(code);
                    best_code = code;
                    retries_used = retry;
                    break;
                }

                // Track best result even if not pass
                if (@intFromEnum(test_result.status) < @intFromEnum(best_status)) {
                    if (best_code) |old_code| allocator.free(old_code);
                    best_status = test_result.status;
                    best_loc = parser.countLoc(code);
                    best_code = code;
                } else {
                    allocator.free(code);
                }

                // If compile/test error and we have retries left, send error back to LLM
                if (retry + 1 < MAX_RETRIES and test_result.stderr.len > 0) {
                    std.debug.print("(retry {d}) ", .{retry + 1});

                    // Add assistant's code to conversation
                    const assistant_msg = try allocator.dupe(u8, response.content);
                    try allocated_msgs.append(allocator, assistant_msg);
                    try conversation.append(allocator, .{ .role = .assistant, .content = assistant_msg });

                    // Add error feedback
                    const error_limit = @min(test_result.stderr.len, 500);
                    const error_msg = try std.fmt.allocPrint(allocator,
                        \\Compilation failed with error:
                        \\```
                        \\{s}
                        \\```
                        \\Please fix the code and provide the corrected version in a ```zig code block.
                    , .{test_result.stderr[0..error_limit]});
                    try allocated_msgs.append(allocator, error_msg);
                    try conversation.append(allocator, .{ .role = .user, .content = error_msg });
                }
            }
        }

        // Record problem result
        const status_str = switch (best_status) {
            .pass => "✓",
            .compile_error => "✗ compile",
            .test_error => "✗ test",
            .timeout => "✗ timeout",
        };
        std.debug.print("{s}\n", .{status_str});

        if (best_status == .pass) {
            passed += 1;
            // Transfer ownership of prompt and code to council for later judging
            if (enable_council) {
                if (best_code) |code| {
                    try passed_solutions.append(allocator, .{
                        .prompt = prompt,
                        .code = code,
                    });
                    best_code = null; // Ownership transferred to passed_solutions
                }
            }
        }
        total_time_ms += problem_time_ms;

        // Clean up if not transferred to council
        if (best_code) |code| allocator.free(code);
        if (!enable_council or best_status != .pass) allocator.free(prompt);

        try problem_results.append(allocator, .{
            .problem_id = problem.id,
            .problem_name = problem.name,
            .status = best_status,
            .response_time_ms = problem_time_ms,
            .loc = best_loc,
            .retries = retries_used,
        });
    }

    // Calculate cost
    const cost = tokens.calculateCost(model_id, total_usage);

    // Council judging (only if enabled and we have passed solutions)
    var rating: ?[]const u8 = null;
    if (enable_council and passed_solutions.items.len > 0) {
        std.debug.print("  └─ Council judging...\n", .{});
        var total_score: f32 = 0;
        var judged_count: u32 = 0;

        for (passed_solutions.items) |sol| {
            var consensus = tribunal.convene(sol.prompt, sol.code) catch |err| {
                std.debug.print("    ⚠ Council error: {}\n", .{err});
                continue;
            };
            defer consensus.deinit();
            total_score += consensus.average_score;
            judged_count += 1;
        }

        if (judged_count > 0) {
            const avg_score = total_score / @as(f32, @floatFromInt(judged_count));
            const rating_enum = ConsensusResult.Rating.fromScore(avg_score);
            rating = try std.fmt.allocPrint(allocator, "{s} ({d:.1})", .{ rating_enum.toString(), avg_score });
        }
    }

    return ModelResult{
        .model_id = model_id,
        .problems = try problem_results.toOwnedSlice(allocator),
        .total_time_ms = total_time_ms,
        .score = passed,
        .total_problems = @intCast(PROBLEMS.len),
        .usage = total_usage,
        .cost = cost,
        .rating = rating,
    };
}

fn printBanner(console: *rich.Console) !void {
    const banner_text =
        \\LLM  ZIG  EVAL
        \\
        \\Find which LLM writes the best Zig code.
    ;
    const panel = rich.Panel.fromText(console.allocator, banner_text)
        .withTitle("Benchmark Suite")
        .withWidth(50)
        .rounded();
    try console.printRenderable(panel);
}

test "main module tests" {
    // Basic sanity tests
    const allocator = std.testing.allocator;
    _ = allocator;
}
