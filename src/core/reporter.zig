//! Report generation module
//! Formats benchmark results as styled tables or JSON.

const std = @import("std");
const rich = @import("rich_zig");
const tokens = @import("tokens.zig");
const sandbox = @import("sandbox.zig");

/// Result for a single model on a single problem
pub const ProblemResult = struct {
    problem_id: []const u8,
    problem_name: []const u8,
    status: sandbox.SandboxResult.Status,
    response_time_ms: i64,
    loc: usize,
    retries: u32 = 0, // Number of error retries used
};

/// Aggregated result for a model across all problems
pub const ModelResult = struct {
    model_id: []const u8,
    problems: []ProblemResult,
    total_time_ms: i64,
    score: u32, // Number of passed problems
    total_problems: u32,
    usage: tokens.TokenUsage,
    cost: f64,
    rating: ?[]const u8, // Council rating if available

    pub fn deinit(self: *ModelResult, allocator: std.mem.Allocator) void {
        allocator.free(self.problems);
        if (self.rating) |r| allocator.free(r);
    }
};

/// Thread-safe wrapper for Report allowing concurrent addResult() calls
pub const ThreadSafeReport = struct {
    report: *Report,
    mutex: std.Thread.Mutex = .{},

    pub fn init(report: *Report) ThreadSafeReport {
        return .{ .report = report };
    }

    pub fn addResult(self: *ThreadSafeReport, result: ModelResult) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.report.addResult(result);
    }
};

/// Full benchmark report
pub const Report = struct {
    results: std.ArrayList(ModelResult),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Report {
        return .{
            .results = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Report) void {
        for (self.results.items) |*result| {
            result.deinit(self.allocator);
        }
        self.results.deinit(self.allocator);
    }

    pub fn addResult(self: *Report, result: ModelResult) !void {
        try self.results.append(self.allocator, result);
    }

    /// Render report as a styled table using rich_zig
    pub fn renderTable(self: *Report, console: *rich.Console) !void {
        // Sort results by score (descending), then by cost (ascending)
        std.mem.sort(ModelResult, self.results.items, {}, struct {
            fn lessThan(_: void, a: ModelResult, b: ModelResult) bool {
                if (a.score != b.score) return a.score > b.score;
                return a.cost < b.cost;
            }
        }.lessThan);

        // Create table with columns
        var table = rich.Table.init(self.allocator);
        defer table.deinit();

        // Track allocated strings for cleanup
        var allocated_strings: std.ArrayList([]const u8) = .empty;
        defer {
            for (allocated_strings.items) |s| self.allocator.free(s);
            allocated_strings.deinit(self.allocator);
        }

        _ = table.withTitle("BENCHMARK REPORT")
            .withBoxStyle(.rounded)
            .addColumn("MODEL")
            .addColumn("TIME")
            .addColumn("SCORE")
            .addColumn("COST")
            .addColumn("LOC")
            .addColumn("RATING");

        // Add rows
        for (self.results.items) |result| {
            // Truncate model name if too long
            const name_len = @min(result.model_id.len, 30);
            const model_name = result.model_id[0..name_len];

            // Format time - allocate to avoid buffer reuse issues
            const time_s = @as(f64, @floatFromInt(result.total_time_ms)) / 1000.0;
            const time_str = try std.fmt.allocPrint(self.allocator, "{d:.1}s", .{time_s});
            try allocated_strings.append(self.allocator, time_str);

            // Calculate total LOC
            var total_loc: usize = 0;
            for (result.problems) |prob| {
                total_loc += prob.loc;
            }

            // Format score - allocate
            const score_str = try std.fmt.allocPrint(self.allocator, "{d}/{d}", .{ result.score, result.total_problems });
            try allocated_strings.append(self.allocator, score_str);

            // Format cost - allocate
            const cost_str = try std.fmt.allocPrint(self.allocator, "${d:.4}", .{result.cost});
            try allocated_strings.append(self.allocator, cost_str);

            // Format LOC - allocate
            const loc_str = try std.fmt.allocPrint(self.allocator, "{d}", .{total_loc});
            try allocated_strings.append(self.allocator, loc_str);

            // Rating
            const rating = result.rating orelse "N/A";

            try table.addRow(&[_][]const u8{ model_name, time_str, score_str, cost_str, loc_str, rating });

            // Add problem breakdown as sub-rows
            for (result.problems) |prob| {
                const status_icon = switch (prob.status) {
                    .pass => "[pass]",
                    .compile_error => "[compile err]",
                    .test_error => "[test err]",
                    .timeout => "[timeout]",
                };
                const prob_str = if (prob.retries > 0)
                    try std.fmt.allocPrint(self.allocator, "  {s} {s} (retries:{d})", .{ prob.problem_name, status_icon, prob.retries })
                else
                    try std.fmt.allocPrint(self.allocator, "  {s} {s}", .{ prob.problem_name, status_icon });
                try allocated_strings.append(self.allocator, prob_str);
                try table.addRow(&[_][]const u8{ prob_str, "", "", "", "", "" });
            }
        }

        try console.printRenderable(table);
        try console.print("\nLegend: SCORE = passed problems, LOC = lines of code, RATING = council score");
    }

    /// Render report as JSON
    pub fn renderJson(self: *Report, writer: anytype) !void {
        try writer.writeAll("{\"results\":[");

        for (self.results.items, 0..) |result, i| {
            if (i > 0) try writer.writeAll(",");

            try writer.print("{{\"model\":\"{s}\",\"time_ms\":{d},\"score\":{d},\"total_problems\":{d},\"cost\":{d:.6},\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d}}}", .{
                result.model_id,
                result.total_time_ms,
                result.score,
                result.total_problems,
                result.cost,
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
                result.usage.total_tokens,
            });

            if (result.rating) |rating| {
                try writer.print(",\"rating\":\"{s}\"", .{rating});
            }

            try writer.writeAll(",\"problems\":[");
            for (result.problems, 0..) |prob, j| {
                if (j > 0) try writer.writeAll(",");
                try writer.print("{{\"id\":\"{s}\",\"name\":\"{s}\",\"status\":\"{s}\",\"time_ms\":{d},\"loc\":{d}}}", .{
                    prob.problem_id,
                    prob.problem_name,
                    @tagName(prob.status),
                    prob.response_time_ms,
                    prob.loc,
                });
            }
            try writer.writeAll("]}");
        }

        try writer.writeAll("]}\n");
    }
};

// Tests
test "report initialization" {
    const allocator = std.testing.allocator;
    var report = Report.init(allocator);
    defer report.deinit();

    try std.testing.expectEqual(@as(usize, 0), report.results.items.len);
}

