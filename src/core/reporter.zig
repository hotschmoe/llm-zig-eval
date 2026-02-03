//! Report generation module
//! Formats benchmark results as styled tables or JSON.

const std = @import("std");
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

    /// Render report as a styled table
    pub fn renderTable(self: *Report, writer: anytype) !void {
        // Header
        try writer.writeAll("\n");
        try writer.writeAll("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        try writer.writeAll("║                           BENCHMARK REPORT                                   ║\n");
        try writer.writeAll("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        try writer.writeAll("║ MODEL                         │ TIME    │ SCORE │ COST     │ LOC  │ RATING  ║\n");
        try writer.writeAll("╠═══════════════════════════════╪═════════╪═══════╪══════════╪══════╪═════════╣\n");

        // Sort results by score (descending), then by cost (ascending)
        std.mem.sort(ModelResult, self.results.items, {}, struct {
            fn lessThan(_: void, a: ModelResult, b: ModelResult) bool {
                if (a.score != b.score) return a.score > b.score;
                return a.cost < b.cost;
            }
        }.lessThan);

        // Rows
        for (self.results.items) |result| {
            // Truncate model name if too long
            var model_name: [30]u8 = [_]u8{' '} ** 30;
            const name_len = @min(result.model_id.len, 30);
            @memcpy(model_name[0..name_len], result.model_id[0..name_len]);

            // Format time
            const time_s = @as(f64, @floatFromInt(result.total_time_ms)) / 1000.0;

            // Calculate total LOC
            var total_loc: usize = 0;
            for (result.problems) |prob| {
                total_loc += prob.loc;
            }

            // Format cost
            var cost_buf: [10]u8 = undefined;
            const cost_str = formatCostBuf(result.cost, &cost_buf);

            // Rating
            const rating = result.rating orelse "N/A";

            try writer.print("║ {s} │ {d:>5.1}s │ {d}/{d}   │ {s:<8} │ {d:>4} │ {s:<7} ║\n", .{
                model_name,
                time_s,
                result.score,
                result.total_problems,
                cost_str,
                total_loc,
                rating,
            });

            // Problem breakdown
            for (result.problems) |prob| {
                const status_icon = switch (prob.status) {
                    .pass => "✓",
                    .compile_error => "✗ compile",
                    .test_error => "✗ test",
                    .timeout => "⏱ timeout",
                };
                const retry_str = if (prob.retries > 0) blk: {
                    var buf: [16]u8 = undefined;
                    break :blk std.fmt.bufPrint(&buf, "(retries:{d})", .{prob.retries}) catch "";
                } else "";
                try writer.print("║   └─ {s:<20} {s:<10} {s:<12}                     ║\n", .{
                    prob.problem_name,
                    status_icon,
                    retry_str,
                });
            }
        }

        try writer.writeAll("╚══════════════════════════════════════════════════════════════════════════════╝\n");

        // Legend
        try writer.writeAll("\n");
        try writer.writeAll("Legend: SCORE = passed problems, LOC = lines of code, RATING = council score\n");
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
