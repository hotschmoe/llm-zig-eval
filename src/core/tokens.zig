//! Token usage and cost calculation module

const std = @import("std");
const config = @import("../config.zig");

/// Token usage from an API response
pub const TokenUsage = struct {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,

    pub fn init() TokenUsage {
        return .{
            .prompt_tokens = 0,
            .completion_tokens = 0,
            .total_tokens = 0,
        };
    }

    /// Add usage from another response
    pub fn add(self: *TokenUsage, other: TokenUsage) void {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
    }
};

/// Calculate cost in dollars for a given model and token usage
pub fn calculateCost(model_id: []const u8, usage: TokenUsage) f64 {
    const model_cost = config.getModelCost(model_id) orelse {
        // Unknown model - use a default estimate
        return calculateCostWithRates(usage, 1.0, 2.0);
    };

    return calculateCostWithRates(usage, model_cost.input_cost, model_cost.output_cost);
}

/// Calculate cost with explicit rates ($ per million tokens)
pub fn calculateCostWithRates(usage: TokenUsage, input_rate: f64, output_rate: f64) f64 {
    const input_cost = @as(f64, @floatFromInt(usage.prompt_tokens)) * input_rate / 1_000_000.0;
    const output_cost = @as(f64, @floatFromInt(usage.completion_tokens)) * output_rate / 1_000_000.0;
    return input_cost + output_cost;
}

/// Format cost as a dollar string
pub fn formatCost(allocator: std.mem.Allocator, cost: f64) ![]const u8 {
    if (cost < 0.01) {
        return try std.fmt.allocPrint(allocator, "${d:.4}", .{cost});
    } else {
        return try std.fmt.allocPrint(allocator, "${d:.2}", .{cost});
    }
}

// Tests
test "calculate cost for known model" {
    const usage = TokenUsage{
        .prompt_tokens = 1000,
        .completion_tokens = 500,
        .total_tokens = 1500,
    };

    const cost = calculateCost("anthropic/claude-3.5-sonnet", usage);

    // Input: 1000 tokens * $3.0/M = $0.003
    // Output: 500 tokens * $15.0/M = $0.0075
    // Total: $0.0105
    try std.testing.expectApproxEqAbs(@as(f64, 0.0105), cost, 0.0001);
}

test "calculate cost for unknown model" {
    const usage = TokenUsage{
        .prompt_tokens = 1000,
        .completion_tokens = 1000,
        .total_tokens = 2000,
    };

    const cost = calculateCost("unknown/model", usage);

    // Should use default rates (1.0, 2.0)
    // Input: 1000 * $1.0/M = $0.001
    // Output: 1000 * $2.0/M = $0.002
    // Total: $0.003
    try std.testing.expectApproxEqAbs(@as(f64, 0.003), cost, 0.0001);
}

test "usage addition" {
    var usage1 = TokenUsage{
        .prompt_tokens = 100,
        .completion_tokens = 50,
        .total_tokens = 150,
    };

    const usage2 = TokenUsage{
        .prompt_tokens = 200,
        .completion_tokens = 100,
        .total_tokens = 300,
    };

    usage1.add(usage2);

    try std.testing.expectEqual(@as(u32, 300), usage1.prompt_tokens);
    try std.testing.expectEqual(@as(u32, 150), usage1.completion_tokens);
    try std.testing.expectEqual(@as(u32, 450), usage1.total_tokens);
}

test "format cost" {
    const allocator = std.testing.allocator;

    const small = try formatCost(allocator, 0.00123);
    defer allocator.free(small);
    try std.testing.expect(std.mem.indexOf(u8, small, "$0.0012") != null);

    const large = try formatCost(allocator, 1.234);
    defer allocator.free(large);
    try std.testing.expect(std.mem.indexOf(u8, large, "$1.23") != null);
}
