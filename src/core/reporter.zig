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

            // Format time
            const time_s = @as(f64, @floatFromInt(result.total_time_ms)) / 1000.0;
            var time_buf: [16]u8 = undefined;
            const time_str = std.fmt.bufPrint(&time_buf, "{d:.1}s", .{time_s}) catch "?";

            // Calculate total LOC
            var total_loc: usize = 0;
            for (result.problems) |prob| {
                total_loc += prob.loc;
            }

            // Format score
            var score_buf: [8]u8 = undefined;
            const score_str = std.fmt.bufPrint(&score_buf, "{d}/{d}", .{ result.score, result.total_problems }) catch "?";

            // Format cost
            var cost_buf: [12]u8 = undefined;
            const cost_str = formatCostBuf(result.cost, &cost_buf);

            // Format LOC
            var loc_buf: [8]u8 = undefined;
            const loc_str = std.fmt.bufPrint(&loc_buf, "{d}", .{total_loc}) catch "?";

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
                var prob_buf: [48]u8 = undefined;
                const prob_str = if (prob.retries > 0)
                    std.fmt.bufPrint(&prob_buf, "  {s} {s} (retries:{d})", .{ prob.problem_name, status_icon, prob.retries }) catch "?"
                else
                    std.fmt.bufPrint(&prob_buf, "  {s} {s}", .{ prob.problem_name, status_icon }) catch "?";
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

fn formatCostBuf(cost: f64, buf: []u8) []const u8 {
    const result = std.fmt.bufPrint(buf, "${d:.4}", .{cost}) catch "$???";
    return result;
}

// Tests
test "report initialization" {
    const allocator = std.testing.allocator;
    var report = Report.init(allocator);
    defer report.deinit();

    try std.testing.expectEqual(@as(usize, 0), report.results.items.len);
}

test "format cost buffer" {
    var buf: [10]u8 = undefined;

    const small = formatCostBuf(0.0042, &buf);
    try std.testing.expect(std.mem.indexOf(u8, small, "$0.0042") != null);
}
