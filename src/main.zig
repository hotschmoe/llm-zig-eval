//! llm-zig-eval main entry point
//! CLI orchestrator for benchmarking LLMs on Zig programming tasks.

const std = @import("std");
const rich = @import("rich_zig");
const lib = @import("llm_zig_eval");
const model_selector = @import("tui/model_selector.zig");
const benchmark_runner = @import("tui/benchmark_runner.zig");

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

pub const ProgressCallback = benchmark_runner.ProgressCallback;
pub const LogLevel = benchmark_runner.LogLevel;

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

const BenchmarkTask = struct {
    model_id: []const u8,
    api_key: []const u8,
    runs: u32,
    enable_council: bool,
    council_models: ?[]const []const u8,
    allocator: std.mem.Allocator,
    result: ?ModelResult = null,
    err: ?anyerror = null,
};

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
                std.debug.print("  llm-zig-eval --models-file=models.txt\n", .{});
                std.debug.print("  llm-zig-eval --select  (interactive selector)\n", .{});
                return err;
            },
            error.NoModelsSpecified => {
                const has_council = hasCouncilFlag();
                return launchModelSelector(allocator, has_council);
            },
            else => return err,
        }
    };
    defer cfg.deinit();

    try runBenchmarkSuite(allocator, &cfg);
}

fn runBenchmarkSuite(allocator: std.mem.Allocator, cfg: *Config) !void {
    switch (cfg.output_format) {
        .pretty => try benchmark_runner.launchBenchmarkRunner(allocator, cfg),
        .json => try runBenchmarkSuiteClassic(allocator, cfg),
    }
}

fn runBenchmarkSuiteClassic(allocator: std.mem.Allocator, cfg: *Config) !void {
    var console = rich.Console.init(allocator);
    defer console.deinit();

    try printBanner(&console);

    var report = Report.init(allocator);
    defer report.deinit();

    var ts_report = ThreadSafeReport.init(&report);

    const num_models = cfg.models.len;
    const effective_parallel = @min(cfg.parallel, @as(u32, @intCast(num_models)));

    try console.print("\n");
    var info_buf: [128]u8 = undefined;
    const info_msg = std.fmt.bufPrint(&info_buf, "Benchmarking {d} model(s) with parallelism={d}\n", .{ num_models, effective_parallel }) catch "Benchmarking...\n";
    try console.print(info_msg);

    const use_parallel = effective_parallel > 1 and num_models > 1;
    if (use_parallel) {
        try runParallelBenchmarks(allocator, cfg, &ts_report, &console);
    } else {
        try runSequentialBenchmarks(allocator, cfg, &report, &console);
    }

    try console.print("\n");
    const stdout_file = std.fs.File.stdout();
    var buf: [4096]u8 = undefined;
    var stdout = stdout_file.writer(&buf);
    try report.renderJson(&stdout.interface);
    try stdout.interface.flush();
}

fn runParallelBenchmarks(
    allocator: std.mem.Allocator,
    cfg: *Config,
    ts_report: *ThreadSafeReport,
    console: *rich.Console,
) !void {
    const num_models = cfg.models.len;

    var tasks = try allocator.alloc(BenchmarkTask, num_models);
    defer allocator.free(tasks);

    for (cfg.models, 0..) |model_id, i| {
        tasks[i] = .{
            .model_id = model_id,
            .api_key = cfg.api_key,
            .runs = cfg.runs,
            .enable_council = cfg.council,
            .council_models = cfg.council_models,
            .allocator = allocator,
        };
    }

    var pool: std.Thread.Pool = undefined;
    try pool.init(.{
        .allocator = allocator,
        .n_jobs = @min(cfg.parallel, @as(u32, @intCast(num_models))),
    });
    defer pool.deinit();

    var wg: std.Thread.WaitGroup = .{};
    for (tasks) |*task| {
        pool.spawnWg(&wg, runBenchmarkTaskWrapper, .{task});
    }

    wg.wait();

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

fn runBenchmarkTaskWrapper(task: *BenchmarkTask) void {
    var client = Client.init(task.allocator, task.api_key);
    defer client.deinit();

    var sbx = Sandbox.init(task.allocator, "out");

    var tribunal = if (task.council_models) |cm|
        Tribunal.initWithModels(task.allocator, &client, cm) catch {
            task.err = error.OutOfMemory;
            return;
        }
    else
        Tribunal.init(task.allocator, &client);
    defer tribunal.deinit();

    const cb = ProgressCallback{ .stdout = .{ .verbose = false } };

    task.result = runModelBenchmarkCore(
        task.allocator,
        &client,
        &sbx,
        &tribunal,
        task.model_id,
        task.runs,
        task.enable_council,
        cb,
    ) catch |err| {
        task.err = err;
        return;
    };
}

fn runSequentialBenchmarks(
    allocator: std.mem.Allocator,
    cfg: *Config,
    report: *Report,
    console: *rich.Console,
) !void {
    var client = Client.init(allocator, cfg.api_key);
    defer client.deinit();

    var sbx = Sandbox.init(allocator, "out");

    var tribunal = if (cfg.council_models) |cm|
        try Tribunal.initWithModels(allocator, &client, cm)
    else
        Tribunal.init(allocator, &client);
    defer tribunal.deinit();

    const cb = ProgressCallback{ .stdout = .{ .verbose = true } };

    for (cfg.models) |model_id| {
        try console.print("\n");
        var model_buf: [128]u8 = undefined;
        const model_msg = std.fmt.bufPrint(&model_buf, "[bold]Benchmarking:[/] {s}\n", .{model_id}) catch "Benchmarking...\n";
        try console.print(model_msg);

        const model_result = try runModelBenchmarkCore(
            allocator,
            &client,
            &sbx,
            &tribunal,
            model_id,
            cfg.runs,
            cfg.council,
            cb,
        );

        try report.addResult(model_result);
        try printResultPanel(console, model_id, model_result.score, model_result.total_problems);
    }
}

pub fn runModelBenchmarkCore(
    allocator: std.mem.Allocator,
    client: *Client,
    sbx: *Sandbox,
    tribunal: *Tribunal,
    model_id: []const u8,
    runs: u32,
    enable_council: bool,
    cb: ProgressCallback,
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

    cb.onLog(model_id, "Starting benchmark", .info);

    for (PROBLEMS, 0..) |problem, idx| {
        cb.onProgress(model_id, problem.name, @intCast(idx), total_problems);

        var log_buf: [128]u8 = undefined;
        const start_msg = std.fmt.bufPrint(&log_buf, "{s} ...", .{problem.name}) catch problem.name;
        cb.onLog(model_id, start_msg, .info);

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
                var response = client.sendChatCompletion(model_id, conversation.items) catch |err| {
                    var err_buf: [128]u8 = undefined;
                    const err_msg = std.fmt.bufPrint(&err_buf, "API error: {}", .{err}) catch "API error";
                    cb.onLog(model_id, err_msg, .err);
                    return err;
                };
                defer response.deinit(allocator);

                problem_time_ms += response.response_time_ms;
                total_usage.add(.{
                    .prompt_tokens = response.usage.prompt_tokens,
                    .completion_tokens = response.usage.completion_tokens,
                    .total_tokens = response.usage.total_tokens,
                });

                const code = try parser.extractZigCode(allocator, response.content) orelse {
                    cb.onLog(model_id, "No code block in response", .warn);
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

                    var pass_buf: [128]u8 = undefined;
                    const pass_msg = if (retry > 0)
                        std.fmt.bufPrint(&pass_buf, "{s} PASS (retry {d})", .{ problem.name, retry }) catch "PASS"
                    else
                        std.fmt.bufPrint(&pass_buf, "{s} PASS", .{problem.name}) catch "PASS";
                    cb.onLog(model_id, pass_msg, .success);
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
                    var retry_buf: [128]u8 = undefined;
                    const status_tag = switch (test_result.status) {
                        .compile_error => "compile error",
                        .test_error => "test error",
                        .timeout => "timeout",
                        .pass => "pass",
                    };
                    const retry_msg = std.fmt.bufPrint(&retry_buf, "{s} {s}, retry {d}", .{
                        problem.name,
                        status_tag,
                        retry + 1,
                    }) catch "retrying";
                    cb.onLog(model_id, retry_msg, .warn);

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

        if (best_status != .pass) {
            var fail_buf: [128]u8 = undefined;
            const fail_tag = switch (best_status) {
                .compile_error => "FAIL (compile)",
                .test_error => "FAIL (test)",
                .timeout => "FAIL (timeout)",
                .pass => unreachable,
            };
            const fail_msg = std.fmt.bufPrint(&fail_buf, "{s} {s}", .{ problem.name, fail_tag }) catch "FAIL";
            cb.onLog(model_id, fail_msg, .err);
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

    cb.onProgress(model_id, "done", total_problems, total_problems);

    const cost = tokens.calculateCost(model_id, total_usage);

    var rating: ?[]const u8 = null;
    if (enable_council and passed_solutions.items.len > 0) {
        cb.onLog(model_id, "Council judging...", .info);

        var total_score: f32 = 0;
        var judged_count: u32 = 0;

        for (passed_solutions.items) |sol| {
            var consensus = tribunal.convene(sol.prompt, sol.code) catch |err| {
                var judge_err_buf: [128]u8 = undefined;
                const judge_err = std.fmt.bufPrint(&judge_err_buf, "Council error: {}", .{err}) catch "Council error";
                cb.onLog(model_id, judge_err, .warn);
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

            var rating_buf: [128]u8 = undefined;
            const rating_msg = std.fmt.bufPrint(&rating_buf, "Council rating: {s} ({d:.1})", .{ rating_enum.toString(), avg_score }) catch "Council rated";
            cb.onLog(model_id, rating_msg, .success);
        }
    }

    cb.onLog(model_id, "Complete", .success);
    cb.onDone(model_id, true);

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
        \\
        \\  L L M   |   Z I G   |   E V A L
        \\
        \\  Find which LLM writes the best Zig code.
        \\
    ;
    const panel = rich.Panel.fromText(console.allocator, banner_text)
        .withTitle("Benchmark Suite")
        .withSubtitle("v0.5.0")
        .withTitleAlignment(.center)
        .withSubtitleAlignment(.center)
        .withWidth(48)
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

    const base = blk: {
        if (score == total) break :blk rich.Panel.success(console.allocator, panel_msg);
        if (score > 0) break :blk rich.Panel.warning(console.allocator, panel_msg);
        break :blk rich.Panel.err(console.allocator, panel_msg);
    };
    const panel = base.withWidth(@min(panel_msg.len + 6, 52));

    try console.printRenderable(panel);
}

fn hasCouncilFlag() bool {
    var args = std.process.args();
    _ = args.skip();
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--council")) return true;
    }
    return false;
}

fn launchModelSelector(allocator: std.mem.Allocator, council_enabled: bool) !void {
    const api_key = try config.loadApiKey(allocator);
    defer allocator.free(api_key);

    const result = model_selector.runSelector(allocator, api_key, council_enabled) catch |err| {
        std.debug.print("Model selector failed: {}\n", .{err});
        return err;
    };

    const sel = result orelse {
        std.debug.print("No models selected. Exiting.\n", .{});
        return;
    };
    defer {
        for (sel.benchmark_models) |m| allocator.free(m);
        allocator.free(sel.benchmark_models);
        if (sel.council_models) |cm| {
            for (cm) |m| allocator.free(m);
            allocator.free(cm);
        }
    }

    if (sel.benchmark_models.len == 0) {
        std.debug.print("No models selected. Exiting.\n", .{});
        return;
    }

    const enable_council = council_enabled or sel.council_models != null;

    std.debug.print("Selected {d} model(s)", .{sel.benchmark_models.len});
    if (sel.council_models) |cm| {
        std.debug.print(", {d} council judge(s)", .{cm.len});
    } else if (enable_council) {
        std.debug.print(" (council: default judges)", .{});
    }
    std.debug.print(". Running benchmark...\n", .{});

    var cfg = Config{
        .models = sel.benchmark_models,
        .runs = 1,
        .council = enable_council,
        .output_format = .pretty,
        .parallel = 4,
        .council_models = sel.council_models,
        .api_key = api_key,
        .allocator = allocator,
    };

    try runBenchmarkSuite(allocator, &cfg);
}

test "main module compiles" {
    std.testing.refAllDecls(@This());
}
