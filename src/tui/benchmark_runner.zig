//! TUI Benchmark Runner
//! Real-time progress display for benchmark execution using zithril.
//! Shows per-model progress bars, a scrollable system log, and transitions
//! to a results view on completion.

const std = @import("std");
const zithril = @import("zithril");
const lib = @import("llm_zig_eval");

const Report = lib.Report;
const ThreadSafeReport = lib.ThreadSafeReport;
const ModelResult = lib.ModelResult;
const Config = lib.Config;
const Client = lib.Client;
const Sandbox = lib.Sandbox;
const Tribunal = lib.Tribunal;

const App = zithril.App;
const Event = zithril.Event;
const Action = zithril.Action;
const Frame = zithril.Frame;
const Rect = zithril.Rect;
const Style = zithril.Style;
const Block = zithril.Block;
const BorderType = zithril.BorderType;
const ScrollState = zithril.ScrollState;
const ScrollableList = zithril.ScrollableList;
const Constraint = zithril.Constraint;
const Direction = zithril.Direction;
const Key = zithril.Key;
const KeyCode = zithril.KeyCode;
const Gauge = zithril.Gauge;
const Buffer = zithril.Buffer;

// ---------------------------------------------------------------------------
// Log / Progress types
// ---------------------------------------------------------------------------

pub const LogLevel = enum {
    info,
    warn,
    err,
    success,
};

pub const LogEntry = struct {
    timestamp_ms: i64,
    model_id: []const u8, // borrowed, lives as long as Config
    message: []const u8, // owned by channel allocator
    level: LogLevel,
};

pub const ModelProgress = struct {
    problems_done: u32 = 0,
    total_problems: u32 = 0,
    current_problem: []const u8 = "",
    status: Status = .pending,

    pub const Status = enum { pending, running, done, failed };
};

// ---------------------------------------------------------------------------
// BenchmarkChannel -- thread-safe bridge between workers and TUI
// ---------------------------------------------------------------------------

pub const BenchmarkChannel = struct {
    mutex: std.Thread.Mutex = .{},
    allocator: std.mem.Allocator,
    log_queue: std.ArrayList(LogEntry),
    progress_map: std.StringHashMapUnmanaged(ModelProgress),

    pub fn init(allocator: std.mem.Allocator) BenchmarkChannel {
        return .{
            .allocator = allocator,
            .log_queue = .empty,
            .progress_map = .empty,
        };
    }

    pub fn deinit(self: *BenchmarkChannel) void {
        for (self.log_queue.items) |entry| {
            self.allocator.free(entry.message);
        }
        self.log_queue.deinit(self.allocator);
        self.progress_map.deinit(self.allocator);
    }

    pub fn postLog(self: *BenchmarkChannel, model_id: []const u8, message: []const u8, level: LogLevel) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const owned_msg = self.allocator.dupe(u8, message) catch return;
        const now = std.time.milliTimestamp();
        self.log_queue.append(self.allocator, .{
            .timestamp_ms = now,
            .model_id = model_id,
            .message = owned_msg,
            .level = level,
        }) catch {
            self.allocator.free(owned_msg);
        };
    }

    pub fn updateProgress(self: *BenchmarkChannel, model_id: []const u8, problem_name: []const u8, done: u32, total: u32) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const gop = self.progress_map.getOrPut(self.allocator, model_id) catch return;
        gop.value_ptr.* = .{
            .problems_done = done,
            .total_problems = total,
            .current_problem = problem_name,
            .status = .running,
        };
    }

    pub fn markModelDone(self: *BenchmarkChannel, model_id: []const u8, success: bool) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const gop = self.progress_map.getOrPut(self.allocator, model_id) catch return;
        gop.value_ptr.status = if (success) .done else .failed;
    }

    /// Drain queued log entries into the caller's list. Caller owns the messages.
    pub fn drainLogs(self: *BenchmarkChannel, dest: *std.ArrayList(LogEntry), dest_allocator: std.mem.Allocator) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.log_queue.items) |entry| {
            dest.append(dest_allocator, entry) catch {
                self.allocator.free(entry.message);
            };
        }
        // Clear without freeing messages (ownership transferred)
        self.log_queue.clearRetainingCapacity();
    }

    /// Snapshot current progress for all models.
    pub fn getProgressSnapshot(self: *BenchmarkChannel, dest: *std.StringHashMapUnmanaged(ModelProgress), allocator: std.mem.Allocator) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var it = self.progress_map.iterator();
        while (it.next()) |kv| {
            const gop = dest.getOrPut(allocator, kv.key_ptr.*) catch continue;
            gop.value_ptr.* = kv.value_ptr.*;
        }
    }
};

// ---------------------------------------------------------------------------
// ProgressCallback -- abstraction for stdout vs channel progress reporting
// ---------------------------------------------------------------------------

pub const ProgressCallback = union(enum) {
    stdout: StdoutProgress,
    channel: *BenchmarkChannel,

    pub const StdoutProgress = struct {
        verbose: bool,
    };

    pub fn onProgress(self: ProgressCallback, model_id: []const u8, problem_name: []const u8, done: u32, total: u32) void {
        switch (self) {
            .channel => |ch| ch.updateProgress(model_id, problem_name, done, total),
            .stdout => {},
        }
    }

    pub fn onLog(self: ProgressCallback, model_id: []const u8, message: []const u8, level: LogLevel) void {
        switch (self) {
            .channel => |ch| ch.postLog(model_id, message, level),
            .stdout => |s| {
                if (s.verbose) {
                    switch (level) {
                        .err => std.debug.print("  ! {s}: {s}\n", .{ model_id, message }),
                        .warn => std.debug.print("  ? {s}: {s}\n", .{ model_id, message }),
                        .success => std.debug.print("  {s}: {s}\n", .{ model_id, message }),
                        .info => std.debug.print("  {s}: {s}\n", .{ model_id, message }),
                    }
                }
            },
        }
    }

    pub fn onDone(self: ProgressCallback, model_id: []const u8, success: bool) void {
        switch (self) {
            .channel => |ch| ch.markModelDone(model_id, success),
            .stdout => {},
        }
    }
};

// ---------------------------------------------------------------------------
// RunnerState -- TUI state
// ---------------------------------------------------------------------------

const MAX_LOG_ENTRIES: usize = 500;
const SPINNER_CHARS = [_][]const u8{ "|", "/", "-", "\\" };

pub const ViewMode = enum { progress, results };

pub const RunnerState = struct {
    allocator: std.mem.Allocator,
    channel: *BenchmarkChannel,
    model_ids: []const []const u8,
    view_mode: ViewMode = .progress,

    // Log display
    log_entries: std.ArrayList(LogEntry),
    log_display: std.ArrayList([]const u8), // formatted strings for ScrollableList
    log_scroll: ScrollState,
    log_auto_scroll: bool = true,

    // Progress snapshot (updated each tick)
    progress_snap: std.StringHashMapUnmanaged(ModelProgress),

    // Results
    report: *Report,
    result_lines: std.ArrayList([]const u8),
    result_scroll: ScrollState,
    results_built: bool = false,

    // Animation / timing
    spinner_frame: u8 = 0,
    start_time: i64,
    completed_models: u32 = 0,
    total_models: u32,
    all_done: bool = false,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        channel: *BenchmarkChannel,
        model_ids: []const []const u8,
        report: *Report,
    ) Self {
        return .{
            .allocator = allocator,
            .channel = channel,
            .model_ids = model_ids,
            .log_entries = .empty,
            .log_display = .empty,
            .log_scroll = ScrollState.init(0),
            .progress_snap = .empty,
            .report = report,
            .result_lines = .empty,
            .result_scroll = ScrollState.init(0),
            .start_time = std.time.milliTimestamp(),
            .total_models = @intCast(model_ids.len),
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.log_entries.items) |entry| {
            self.allocator.free(entry.message);
        }
        self.log_entries.deinit(self.allocator);
        for (self.log_display.items) |s| {
            self.allocator.free(s);
        }
        self.log_display.deinit(self.allocator);
        self.progress_snap.deinit(self.allocator);
        for (self.result_lines.items) |s| {
            self.allocator.free(s);
        }
        self.result_lines.deinit(self.allocator);
    }

    /// Pull new logs from channel, cap at MAX_LOG_ENTRIES
    fn drainAndFormat(self: *Self) void {
        self.channel.drainLogs(&self.log_entries, self.allocator);

        // Trim old entries if over cap
        while (self.log_entries.items.len > MAX_LOG_ENTRIES) {
            const removed = self.log_entries.orderedRemove(0);
            self.allocator.free(removed.message);
            // Also remove corresponding display string
            if (self.log_display.items.len > 0) {
                const ds = self.log_display.orderedRemove(0);
                self.allocator.free(ds);
            }
        }

        // Format any new entries that don't have display strings yet
        while (self.log_display.items.len < self.log_entries.items.len) {
            const entry = self.log_entries.items[self.log_display.items.len];
            const formatted = self.formatLogEntry(entry) catch break;
            self.log_display.append(self.allocator, formatted) catch {
                self.allocator.free(formatted);
                break;
            };
        }

        self.log_scroll.setTotal(self.log_display.items.len);
        if (self.log_auto_scroll and self.log_display.items.len > 0) {
            self.log_scroll.scrollToEnd();
        }
    }

    fn formatLogEntry(self: *Self, entry: LogEntry) ![]const u8 {
        // Format timestamp as HH:MM:SS
        const epoch_secs: u64 = @intCast(@divFloor(entry.timestamp_ms, 1000));
        const day_secs = epoch_secs % 86400;
        const hours = day_secs / 3600;
        const minutes = (day_secs % 3600) / 60;
        const seconds = day_secs % 60;

        const level_prefix: []const u8 = switch (entry.level) {
            .info => " ",
            .warn => "?",
            .err => "!",
            .success => "+",
        };

        // Truncate model_id to last component for brevity
        const short_model = shortModelName(entry.model_id);

        return std.fmt.allocPrint(self.allocator, "{d:0>2}:{d:0>2}:{d:0>2} {s} {s}: {s}", .{
            hours,
            minutes,
            seconds,
            level_prefix,
            short_model,
            entry.message,
        });
    }

    fn refreshProgressSnapshot(self: *Self) void {
        self.channel.getProgressSnapshot(&self.progress_snap, self.allocator);

        // Count completed models
        var done: u32 = 0;
        var it = self.progress_snap.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.status == .done or kv.value_ptr.status == .failed) {
                done += 1;
            }
        }
        self.completed_models = done;

        if (done >= self.total_models and self.total_models > 0 and !self.all_done) {
            self.all_done = true;
        }
    }
};

fn shortModelName(model_id: []const u8) []const u8 {
    if (std.mem.lastIndexOfScalar(u8, model_id, '/')) |idx| {
        return model_id[idx + 1 ..];
    }
    return model_id;
}

// ---------------------------------------------------------------------------
// TUI update / view
// ---------------------------------------------------------------------------

fn update(state: *RunnerState, event: Event) Action {
    switch (event) {
        .tick => {
            state.drainAndFormat();
            state.refreshProgressSnapshot();
            state.spinner_frame +%= 1;
            return Action.none_action;
        },
        .key => |key| {
            if (key.action == .release) return Action.none_action;

            switch (key.code) {
                .char => |c| {
                    if (key.modifiers.ctrl and c == 'c') return Action.quit_action;
                    if (c == 'q') return Action.quit_action;
                    if (c == 'r' and state.all_done) {
                        if (state.view_mode == .progress) {
                            if (!state.results_built) {
                                state.result_lines = buildResultLines(state.allocator, state.report);
                                state.result_scroll.setTotal(state.result_lines.items.len);
                                state.results_built = true;
                            }
                            state.view_mode = .results;
                        } else {
                            state.view_mode = .progress;
                        }
                    }
                },
                .escape => return Action.quit_action,
                .up => {
                    if (state.view_mode == .progress) {
                        state.log_auto_scroll = false;
                        state.log_scroll.scrollUp();
                    } else {
                        state.result_scroll.scrollUp();
                    }
                },
                .down => {
                    if (state.view_mode == .progress) {
                        state.log_scroll.scrollDown();
                        // Re-enable auto scroll if at bottom
                        if (state.log_scroll.atEnd()) {
                            state.log_auto_scroll = true;
                        }
                    } else {
                        state.result_scroll.scrollDown();
                    }
                },
                .page_up => {
                    if (state.view_mode == .progress) {
                        state.log_auto_scroll = false;
                        state.log_scroll.pageUp();
                    } else {
                        state.result_scroll.pageUp();
                    }
                },
                .page_down => {
                    if (state.view_mode == .progress) {
                        state.log_scroll.pageDown();
                        if (state.log_scroll.atEnd()) {
                            state.log_auto_scroll = true;
                        }
                    } else {
                        state.result_scroll.pageDown();
                    }
                },
                else => {},
            }
        },
        else => {},
    }
    return Action.none_action;
}

fn view(state: *RunnerState, frame: *Frame(App(RunnerState).DefaultMaxWidgets)) void {
    const area = frame.size();

    switch (state.view_mode) {
        .progress => renderProgressView(state, frame, area),
        .results => renderResultsView(state, frame, area),
    }
}

fn renderProgressView(state: *RunnerState, frame: *Frame(App(RunnerState).DefaultMaxWidgets), area: Rect) void {
    const model_count: u16 = @intCast(state.model_ids.len);
    // header(3) + progress panel(models + 2 for border) + log(flex) + status(1)
    const progress_height: u16 = model_count + 2;

    const chunks = frame.layout(area, Direction.vertical, &.{
        Constraint.len(3), // header
        Constraint.len(progress_height), // progress panel
        Constraint.flexible(1), // system log
        Constraint.len(1), // status bar
    });

    renderHeader(frame, chunks.get(0));
    renderProgressPanel(state, frame, chunks.get(1));
    renderLogPanel(state, frame, chunks.get(2));
    renderStatusBar(state, frame, chunks.get(3));
}

fn renderResultsView(state: *RunnerState, frame: *Frame(App(RunnerState).DefaultMaxWidgets), area: Rect) void {
    const chunks = frame.layout(area, Direction.vertical, &.{
        Constraint.len(3), // header
        Constraint.flexible(1), // results
        Constraint.len(1), // status bar
    });

    renderHeader(frame, chunks.get(0));
    renderResultsPanel(state, frame, chunks.get(1));
    renderResultsStatusBar(state, frame, chunks.get(2));
}

fn renderHeader(frame: *Frame(App(RunnerState).DefaultMaxWidgets), area: Rect) void {
    const header_block = Block{
        .title = " LLM | ZIG | EVAL v0.5.0 ",
        .title_alignment = .center,
        .border = BorderType.double,
        .border_style = Style.init().fg(.cyan).bold(),
    };
    frame.render(header_block, area);
}

fn renderProgressPanel(state: *RunnerState, frame: *Frame(App(RunnerState).DefaultMaxWidgets), area: Rect) void {
    const panel_block = Block{
        .title = " Progress ",
        .border = BorderType.rounded,
        .border_style = Style.init().fg(.white),
    };
    frame.render(panel_block, area);

    const inner = panel_block.inner(area);
    if (inner.height == 0 or inner.width == 0) return;

    const buf = frame.buffer;

    for (state.model_ids, 0..) |model_id, i| {
        const row: u16 = inner.y + @as(u16, @intCast(i));
        if (row >= inner.y + inner.height) break;

        const progress = state.progress_snap.get(model_id) orelse ModelProgress{};

        // Model name (left portion, up to 28 chars)
        const name = shortModelName(model_id);
        const name_width: u16 = @min(28, inner.width / 2);
        const name_style = switch (progress.status) {
            .done => Style.init().fg(.green),
            .failed => Style.init().fg(.red),
            .running => Style.init().fg(.yellow),
            .pending => Style.init().fg(.bright_black),
        };
        buf.setString(inner.x, row, name, name_style);

        // Gauge (right portion)
        const gauge_x = inner.x + name_width + 1;
        if (gauge_x >= inner.x + inner.width) continue;
        const gauge_width = inner.width - name_width - 1;

        // Count label
        var count_buf: [16]u8 = undefined;
        const count_str = std.fmt.bufPrint(&count_buf, " {d}/{d}", .{
            progress.problems_done,
            progress.total_problems,
        }) catch "";

        const gauge_style = switch (progress.status) {
            .done => Style.init().bg(.green),
            .failed => Style.init().bg(.red),
            .running => Style.init().bg(.blue),
            .pending => Style.init().bg(.bright_black),
        };

        const gauge = Gauge{
            .ratio = if (progress.total_problems > 0)
                @as(f32, @floatFromInt(progress.problems_done)) / @as(f32, @floatFromInt(progress.total_problems))
            else
                0.0,
            .label = count_str,
            .gauge_style = gauge_style,
        };
        const gauge_rect = Rect.init(gauge_x, row, gauge_width, 1);
        frame.render(gauge, gauge_rect);
    }
}

fn renderLogPanel(state: *RunnerState, frame: *Frame(App(RunnerState).DefaultMaxWidgets), area: Rect) void {
    const log_block = Block{
        .title = " System Log ",
        .border = BorderType.rounded,
        .border_style = Style.init().fg(.bright_black),
    };
    frame.render(log_block, area);

    const inner = log_block.inner(area);
    state.log_scroll.setViewport(inner.height);

    const list = ScrollableList{
        .items = state.log_display.items,
        .scroll = &state.log_scroll,
        .style = Style.init().fg(.white),
        .highlight_style = Style.empty, // no highlight -- just a log
        .highlight_symbol = "",
        .show_scrollbar = true,
    };
    frame.render(list, inner);
}

fn renderStatusBar(state: *RunnerState, frame: *Frame(App(RunnerState).DefaultMaxWidgets), area: Rect) void {
    const buf = frame.buffer;
    const elapsed_ms = std.time.milliTimestamp() - state.start_time;
    const elapsed_s = @divFloor(elapsed_ms, 1000);

    const spinner_idx = state.spinner_frame % SPINNER_CHARS.len;
    const spinner = SPINNER_CHARS[spinner_idx];

    var status_buf: [128]u8 = undefined;
    const status_msg = if (state.all_done)
        std.fmt.bufPrint(&status_buf, " Done! {d}s  r:results q:quit", .{elapsed_s}) catch " Done!"
    else
        std.fmt.bufPrint(&status_buf, " [{s}] Running... {d}s  ({d}/{d} models)  q:quit", .{
            spinner,
            elapsed_s,
            state.completed_models,
            state.total_models,
        }) catch " Running...";

    buf.setString(area.x, area.y, status_msg, Style.init().bg(.black).fg(.white).bold());
}

fn renderResultsPanel(state: *RunnerState, frame: *Frame(App(RunnerState).DefaultMaxWidgets), area: Rect) void {
    const results_block = Block{
        .title = " Benchmark Results ",
        .border = BorderType.rounded,
        .border_style = Style.init().fg(.cyan),
    };
    frame.render(results_block, area);

    const inner = results_block.inner(area);
    state.result_scroll.setViewport(inner.height);

    const list = ScrollableList{
        .items = state.result_lines.items,
        .scroll = &state.result_scroll,
        .style = Style.init().fg(.white),
        .highlight_style = Style.empty,
        .highlight_symbol = "",
        .show_scrollbar = true,
    };
    frame.render(list, inner);
}

fn renderResultsStatusBar(state: *RunnerState, frame: *Frame(App(RunnerState).DefaultMaxWidgets), area: Rect) void {
    const buf = frame.buffer;
    const elapsed_ms = std.time.milliTimestamp() - state.start_time;
    const elapsed_s = @divFloor(elapsed_ms, 1000);

    var status_buf: [128]u8 = undefined;
    const status_msg = std.fmt.bufPrint(&status_buf, " Complete! {d}s total  r:progress q:quit", .{elapsed_s}) catch " Complete!";
    buf.setString(area.x, area.y, status_msg, Style.init().bg(.black).fg(.white).bold());
}

// ---------------------------------------------------------------------------
// Results formatting
// ---------------------------------------------------------------------------

fn buildResultLines(allocator: std.mem.Allocator, report: *Report) std.ArrayList([]const u8) {
    var lines: std.ArrayList([]const u8) = .empty;

    // Sort results by score descending, then cost ascending
    std.mem.sort(ModelResult, report.results.items, {}, struct {
        fn lessThan(_: void, a: ModelResult, b: ModelResult) bool {
            if (a.score != b.score) return a.score > b.score;
            return a.cost < b.cost;
        }
    }.lessThan);

    // Header
    const header = std.fmt.allocPrint(allocator, "  {s:<30} {s:>8} {s:>8} {s:>10} {s:>6} {s:>14}", .{
        "MODEL", "TIME", "SCORE", "COST", "LOC", "RATING",
    }) catch return lines;
    lines.append(allocator, header) catch {};

    const sep = std.fmt.allocPrint(allocator, "  {s}", .{"-" ** 80}) catch return lines;
    lines.append(allocator, sep) catch {};

    for (report.results.items) |result| {
        const name_len = @min(result.model_id.len, 30);
        const model_name = result.model_id[0..name_len];

        const time_s = @as(f64, @floatFromInt(result.total_time_ms)) / 1000.0;

        var total_loc: usize = 0;
        for (result.problems) |prob| {
            total_loc += prob.loc;
        }

        const rating = result.rating orelse "N/A";

        const line = std.fmt.allocPrint(allocator, "  {s:<30} {d:>6.1}s {d:>4}/{d:<3} ${d:>8.4} {d:>6} {s:>14}", .{
            model_name,
            time_s,
            result.score,
            result.total_problems,
            result.cost,
            total_loc,
            rating,
        }) catch continue;
        lines.append(allocator, line) catch {
            allocator.free(line);
            continue;
        };

        // Problem breakdown
        for (result.problems) |prob| {
            const status_str = switch (prob.status) {
                .pass => "[pass]",
                .compile_error => "[compile err]",
                .test_error => "[test err]",
                .timeout => "[timeout]",
            };
            const prob_line = if (prob.retries > 0)
                std.fmt.allocPrint(allocator, "    {s} {s} (retries:{d})", .{ prob.problem_name, status_str, prob.retries }) catch continue
            else
                std.fmt.allocPrint(allocator, "    {s} {s}", .{ prob.problem_name, status_str }) catch continue;
            lines.append(allocator, prob_line) catch {
                allocator.free(prob_line);
            };
        }

        // Blank separator between models
        const blank = std.fmt.allocPrint(allocator, "", .{}) catch continue;
        lines.append(allocator, blank) catch {
            allocator.free(blank);
        };
    }

    return lines;
}

// ---------------------------------------------------------------------------
// Worker task for TUI mode
// ---------------------------------------------------------------------------

const main_module = @import("../main.zig");

pub const BenchmarkTaskTui = struct {
    model_id: []const u8,
    api_key: []const u8,
    runs: u32,
    enable_council: bool,
    council_models: ?[]const []const u8,
    allocator: std.mem.Allocator,
    channel: *BenchmarkChannel,
    ts_report: *ThreadSafeReport,
    result: ?ModelResult = null,
    err: ?anyerror = null,
};

pub fn runBenchmarkTaskTui(task: *BenchmarkTaskTui) void {
    var client = Client.init(task.allocator, task.api_key);
    defer client.deinit();

    var sbx = Sandbox.init(task.allocator, "out");

    var tribunal = if (task.council_models) |cm|
        Tribunal.initWithModels(task.allocator, &client, cm) catch {
            task.err = error.OutOfMemory;
            task.channel.markModelDone(task.model_id, false);
            return;
        }
    else
        Tribunal.init(task.allocator, &client);
    defer tribunal.deinit();

    const cb = ProgressCallback{ .channel = task.channel };

    task.result = main_module.runModelBenchmarkCore(
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
        task.channel.postLog(task.model_id, "Benchmark failed", .err);
        task.channel.markModelDone(task.model_id, false);
        return;
    };

    if (task.result) |result| {
        task.ts_report.addResult(result) catch {};
    }
    task.channel.markModelDone(task.model_id, true);
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn launchBenchmarkRunner(
    allocator: std.mem.Allocator,
    cfg: *Config,
) !void {
    var channel = BenchmarkChannel.init(allocator);
    defer channel.deinit();

    var report = Report.init(allocator);
    defer report.deinit();

    var ts_report = ThreadSafeReport.init(&report);

    const num_models = cfg.models.len;

    // Create tasks
    var tasks = try allocator.alloc(BenchmarkTaskTui, num_models);
    defer allocator.free(tasks);

    for (cfg.models, 0..) |model_id, i| {
        tasks[i] = .{
            .model_id = model_id,
            .api_key = cfg.api_key,
            .runs = cfg.runs,
            .enable_council = cfg.council,
            .council_models = cfg.council_models,
            .allocator = allocator,
            .channel = &channel,
            .ts_report = &ts_report,
        };
    }

    // Post initial log
    channel.postLog("system", "Starting benchmark...", .info);
    for (cfg.models) |model_id| {
        var msg_buf: [128]u8 = undefined;
        const msg = std.fmt.bufPrint(&msg_buf, "Queued for benchmark", .{}) catch "Queued";
        channel.postLog(model_id, msg, .info);
    }

    // Init thread pool
    var pool: std.Thread.Pool = undefined;
    try pool.init(.{
        .allocator = allocator,
        .n_jobs = @min(cfg.parallel, @as(u32, @intCast(num_models))),
    });
    defer pool.deinit();

    // Spawn workers
    var wg: std.Thread.WaitGroup = .{};
    for (tasks) |*task| {
        pool.spawnWg(&wg, runBenchmarkTaskTui, .{task});
    }

    // Create TUI state and run
    var state = RunnerState.init(allocator, &channel, cfg.models, &report);
    defer state.deinit();

    var app = App(RunnerState).init(.{
        .state = &state,
        .update = update,
        .view = view,
        .tick_rate_ms = 250,
        .alternate_screen = true,
    });

    app.run(allocator) catch |err| {
        std.debug.print("TUI error: {}\n", .{err});
    };

    // Wait for workers after TUI exits
    wg.wait();

    // Build result lines for results view (if user quit during progress,
    // we still want to show a summary on stdout)
    if (report.results.items.len > 0) {
        // If user quit before seeing results, print summary to stdout
        if (state.view_mode == .progress) {
            var console = @import("rich_zig").Console.init(allocator);
            defer console.deinit();

            console.print("\n") catch {};
            report.renderTable(&console) catch {};
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "shortModelName" {
    try std.testing.expectEqualStrings("claude-3.5-sonnet", shortModelName("anthropic/claude-3.5-sonnet"));
    try std.testing.expectEqualStrings("gpt-4o", shortModelName("openai/gpt-4o"));
    try std.testing.expectEqualStrings("local-model", shortModelName("local-model"));
}

test "BenchmarkChannel init/deinit" {
    const allocator = std.testing.allocator;
    var ch = BenchmarkChannel.init(allocator);
    defer ch.deinit();

    ch.postLog("test-model", "hello", .info);
    ch.updateProgress("test-model", "q1", 1, 3);
    ch.markModelDone("test-model", true);

    var dest: std.ArrayList(LogEntry) = .empty;
    defer {
        for (dest.items) |e| allocator.free(e.message);
        dest.deinit(allocator);
    }
    ch.drainLogs(&dest, allocator);
    try std.testing.expectEqual(@as(usize, 1), dest.items.len);
}
