//! llm-zig-eval main entry point
//! CLI orchestrator for benchmarking LLMs on Zig programming tasks.

const std = @import("std");
const rich = @import("rich_zig");
const lib = @import("llm_zig_eval");

const Config = lib.Config;
const Client = lib.Client;
const Message = lib.Message;
const Report = lib.Report;
const ThreadSafeReport = lib.ThreadSafeReport;
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

const SYSTEM_PROMPT =
    \\You are an expert Zig 0.15 programmer. Provide only the requested code in a single ```zig code block. No explanations outside the code.
    \\
    \\CRITICAL Zig 0.15 API Notes:
    \\- All exported types/functions must be `pub`
    \\- Use `std.Thread.sleep(ns)` not `std.time.sleep()`
    \\- Use `@typeInfo(T).@"struct".fields` not `.Struct.fields`
    \\- ArrayList uses `.empty` init: `var list: std.ArrayList(u8) = .empty;`
    \\- ArrayList methods take allocator: `list.append(allocator, item)`
;

const MAX_RETRIES: u32 = 4;

/// Task context for parallel benchmark execution
const BenchmarkTask = struct {
    model_id: []const u8,
    api_key: []const u8,
    runs: u32,
    enable_council: bool,
    allocator: std.mem.Allocator,
    result: ?ModelResult = null,
    err: ?anyerror = null,
    progress: *ProgressState,
};

/// Shared progress state for UI updates
const ProgressState = struct {
    mutex: std.Thread.Mutex = .{},
    model_status: std.StringHashMapUnmanaged(ModelProgress),

    const ModelProgress = struct {
        current_problem: []const u8 = "",
        problems_done: u32 = 0,
        total_problems: u32 = 0,
        status: Status = .pending,

        const Status = enum { pending, running, done, failed };
    };

    fn init() ProgressState {
        return .{ .model_status = .empty };
    }

    fn deinit(self: *ProgressState, allocator: std.mem.Allocator) void {
        self.model_status.deinit(allocator);
    }

    fn update(self: *ProgressState, allocator: std.mem.Allocator, model_id: []const u8, problem_name: []const u8, done: u32, total: u32) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const gop = self.model_status.getOrPut(allocator, model_id) catch return;
        gop.value_ptr.* = .{
            .current_problem = problem_name,
            .problems_done = done,
            .total_problems = total,
            .status = .running,
        };
    }

    fn markDone(self: *ProgressState, allocator: std.mem.Allocator, model_id: []const u8, success: bool) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const gop = self.model_status.getOrPut(allocator, model_id) catch return;
        gop.value_ptr.status = if (success) .done else .failed;
    }
};

/// Tracks a passed solution for council judging
const PassedSolution = struct {
    prompt: []const u8,
    code: []const u8,
};

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

    // Initialize report
    var report = Report.init(allocator);
    defer report.deinit();

    // Initialize thread-safe report wrapper
    var ts_report = ThreadSafeReport.init(&report);

    // Initialize progress state
    var progress = ProgressState.init();
    defer progress.deinit(allocator);

    // Determine parallelism (min of parallel setting and model count)
    const num_models = cfg.models.len;
    const effective_parallel = @min(cfg.parallel, @as(u32, @intCast(num_models)));

    // Print execution info
    try console.print("\n");
    var info_buf: [128]u8 = undefined;
    const info_msg = std.fmt.bufPrint(&info_buf, "Benchmarking {d} model(s) with parallelism={d}\n", .{ num_models, effective_parallel }) catch "Benchmarking...\n";
    try console.print(info_msg);

    const use_parallel = effective_parallel > 1 and num_models > 1;
    if (use_parallel) {
        try runParallelBenchmarks(allocator, &cfg, &ts_report, &progress, &console);
    } else {
        try runSequentialBenchmarks(allocator, &cfg, &report, &console);
    }

    // Render final report
    try console.print("\n");
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

/// Run benchmarks in parallel using thread pool
fn runParallelBenchmarks(
    allocator: std.mem.Allocator,
    cfg: *Config,
    ts_report: *ThreadSafeReport,
    progress: *ProgressState,
    console: *rich.Console,
) !void {
    const num_models = cfg.models.len;

    // Allocate task array
    var tasks = try allocator.alloc(BenchmarkTask, num_models);
    defer allocator.free(tasks);

    // Initialize tasks
    for (cfg.models, 0..) |model_id, i| {
        tasks[i] = .{
            .model_id = model_id,
            .api_key = cfg.api_key,
            .runs = cfg.runs,
            .enable_council = cfg.council,
            .allocator = allocator,
            .progress = progress,
        };
    }

    // Initialize thread pool
    var pool: std.Thread.Pool = undefined;
    try pool.init(.{
        .allocator = allocator,
        .n_jobs = @min(cfg.parallel, @as(u32, @intCast(num_models))),
    });
    defer pool.deinit();

    // Spawn tasks
    var wg: std.Thread.WaitGroup = .{};
    for (tasks) |*task| {
        pool.spawnWg(&wg, runBenchmarkTaskWrapper, .{task});
    }

    // Wait for all tasks to complete
    wg.wait();

    // Collect results
    var success_count: u32 = 0;
    var fail_count: u32 = 0;

    for (tasks) |*task| {
        if (task.result) |result| {
            try ts_report.addResult(result);
            success_count += 1;
            try printResultPanel(console, task.model_id, result.score, result.total_problems);
        } else if (task.err) |err| {
            fail_count += 1;
            std.debug.print("Error benchmarking {s}: {}\n", .{ task.model_id, err });
        }
    }

    var summary_buf: [64]u8 = undefined;
    const summary = std.fmt.bufPrint(&summary_buf, "\nCompleted: {d} succeeded, {d} failed\n", .{ success_count, fail_count }) catch "\n";
    try console.print(summary);
}

/// Wrapper function for thread pool execution
fn runBenchmarkTaskWrapper(task: *BenchmarkTask) void {
    // Create per-thread HTTP client (std.http.Client is NOT thread-safe)
    var client = Client.init(task.allocator, task.api_key);
    defer client.deinit();

    // Create per-thread sandbox
    var sbx = Sandbox.init(task.allocator, "out");

    // Create per-thread tribunal (if council enabled)
    var tribunal = Tribunal.init(task.allocator, &client);

    task.result = runModelBenchmarkWithProgress(
        task.allocator,
        &client,
        &sbx,
        &tribunal,
        task.model_id,
        task.runs,
        task.enable_council,
        task.progress,
    ) catch |err| {
        task.err = err;
        task.progress.markDone(task.allocator, task.model_id, false);
        return;
    };

    task.progress.markDone(task.allocator, task.model_id, true);
}

/// Run benchmarks sequentially (original behavior)
fn runSequentialBenchmarks(
    allocator: std.mem.Allocator,
    cfg: *Config,
    report: *Report,
    console: *rich.Console,
) !void {
    // Initialize OpenRouter client
    var client = Client.init(allocator, cfg.api_key);
    defer client.deinit();

    // Initialize sandbox
    var sbx = Sandbox.init(allocator, "out");

    // Initialize tribunal for council judging (if enabled)
    var tribunal = Tribunal.init(allocator, &client);

    // Run benchmark for each model
    for (cfg.models) |model_id| {
        try console.print("\n");
        var model_buf: [128]u8 = undefined;
        const model_msg = std.fmt.bufPrint(&model_buf, "[bold]Benchmarking:[/] {s}\n", .{model_id}) catch "Benchmarking...\n";
        try console.print(model_msg);

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
        try printResultPanel(console, model_id, model_result.score, model_result.total_problems);
    }
}

/// Run model benchmark with progress updates (for parallel execution)
fn runModelBenchmarkWithProgress(
    allocator: std.mem.Allocator,
    client: *Client,
    sbx: *Sandbox,
    tribunal: *Tribunal,
    model_id: []const u8,
    runs: u32,
    enable_council: bool,
    progress: *ProgressState,
) !ModelResult {
    return runModelBenchmarkCore(allocator, client, sbx, tribunal, model_id, runs, enable_council, progress, false);
}

/// Run model benchmark with console output (for sequential execution)
fn runModelBenchmark(
    allocator: std.mem.Allocator,
    client: *Client,
    sbx: *Sandbox,
    tribunal: *Tribunal,
    model_id: []const u8,
    runs: u32,
    enable_council: bool,
) !ModelResult {
    return runModelBenchmarkCore(allocator, client, sbx, tribunal, model_id, runs, enable_council, null, true);
}

/// Core benchmark logic for a single model
fn runModelBenchmarkCore(
    allocator: std.mem.Allocator,
    client: *Client,
    sbx: *Sandbox,
    tribunal: *Tribunal,
    model_id: []const u8,
    runs: u32,
    enable_council: bool,
    progress: ?*ProgressState,
    verbose: bool,
) !ModelResult {
    var problem_results: std.ArrayList(ProblemResult) = .empty;
    errdefer problem_results.deinit(allocator);

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

    const model_dir = try sbx.createModelDir(model_id);
    defer allocator.free(model_dir);

    const total_problems: u32 = @intCast(PROBLEMS.len);

    for (PROBLEMS, 0..) |problem, idx| {
        if (progress) |p| {
            p.update(allocator, model_id, problem.name, @intCast(idx), total_problems);
        }
        if (verbose) {
            std.debug.print("  |-- {s}... ", .{problem.name});
        }

        const prompt = try sandbox.loadProblemPrompt(allocator, problem);
        errdefer allocator.free(prompt);

        var best_status: SandboxResult.Status = .compile_error;
        var best_loc: usize = 0;
        var best_code: ?[]const u8 = null;
        var problem_time_ms: i64 = 0;
        var retries_used: u32 = 0;

        for (0..runs) |_| {
            var conversation: std.ArrayList(Message) = .empty;
            defer conversation.deinit(allocator);

            var allocated_msgs: std.ArrayList([]const u8) = .empty;
            defer {
                for (allocated_msgs.items) |msg| allocator.free(msg);
                allocated_msgs.deinit(allocator);
            }

            try conversation.append(allocator, .{ .role = .system, .content = SYSTEM_PROMPT });
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

                const code = try parser.extractZigCode(allocator, response.content) orelse {
                    if (verbose) std.debug.print("(no code) ", .{});
                    break;
                };

                const solution_path = try sbx.writeSolution(model_dir, problem.id, code);
                defer allocator.free(solution_path);

                var test_result = try sbx.runTest(solution_path, problem.test_path);
                defer test_result.deinit();

                if (test_result.status == .pass) {
                    if (best_code) |old_code| allocator.free(old_code);
                    best_status = .pass;
                    best_loc = parser.countLoc(code);
                    best_code = code;
                    retries_used = retry;
                    break;
                }

                if (@intFromEnum(test_result.status) < @intFromEnum(best_status)) {
                    if (best_code) |old_code| allocator.free(old_code);
                    best_status = test_result.status;
                    best_loc = parser.countLoc(code);
                    best_code = code;
                } else {
                    allocator.free(code);
                }

                if (retry + 1 < MAX_RETRIES and test_result.stderr.len > 0) {
                    if (verbose) std.debug.print("(retry {d}) ", .{retry + 1});

                    const assistant_msg = try allocator.dupe(u8, response.content);
                    try allocated_msgs.append(allocator, assistant_msg);
                    try conversation.append(allocator, .{ .role = .assistant, .content = assistant_msg });

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

        if (verbose) {
            const status_str = switch (best_status) {
                .pass => "[PASS]",
                .compile_error => "[FAIL compile]",
                .test_error => "[FAIL test]",
                .timeout => "[FAIL timeout]",
            };
            std.debug.print("{s}\n", .{status_str});
        }

        if (best_status == .pass) {
            passed += 1;
            if (enable_council) {
                if (best_code) |code| {
                    try passed_solutions.append(allocator, .{
                        .prompt = prompt,
                        .code = code,
                    });
                    best_code = null;
                }
            }
        }
        total_time_ms += problem_time_ms;

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

    if (progress) |p| {
        p.update(allocator, model_id, "done", total_problems, total_problems);
    }

    const cost = tokens.calculateCost(model_id, total_usage);

    var rating: ?[]const u8 = null;
    if (enable_council and passed_solutions.items.len > 0) {
        if (verbose) std.debug.print("  `-- Council judging...\n", .{});

        var total_score: f32 = 0;
        var judged_count: u32 = 0;

        for (passed_solutions.items) |sol| {
            var consensus = tribunal.convene(sol.prompt, sol.code) catch |err| {
                if (verbose) std.debug.print("    ! Council error: {}\n", .{err});
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
        .total_problems = total_problems,
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
        .double();
    try console.printRenderable(panel);
}

fn printResultPanel(console: *rich.Console, model_id: []const u8, score: u32, total: u32) !void {
    var panel_buf: [256]u8 = undefined;
    const panel_msg = std.fmt.bufPrint(&panel_buf, "{s}: {d}/{d} passed", .{
        model_id,
        score,
        total,
    }) catch "Completed";

    const panel = if (score == total)
        rich.Panel.success(console.allocator, panel_msg)
    else if (score > 0)
        rich.Panel.warning(console.allocator, panel_msg)
    else
        rich.Panel.err(console.allocator, panel_msg);

    try console.printRenderable(panel);
}

test "main module tests" {
    // Basic sanity tests
    const allocator = std.testing.allocator;
    _ = allocator;
}
