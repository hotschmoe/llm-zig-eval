//! Council of Judges type definitions
//! Types for multi-model consensus scoring.

const std = @import("std");
const prompts = @import("prompts.zig");

/// Judge persona configuration
pub const JudgePersona = struct {
    name: []const u8,
    model_id: []const u8,
    focus: []const u8,
    system_prompt: []const u8,
};

/// Default judge personas
pub const DEFAULT_JUDGES = [_]JudgePersona{
    .{
        .name = "Pedant",
        .model_id = "anthropic/claude-3.5-sonnet",
        .focus = "Safety, defer, strict types",
        .system_prompt = prompts.PEDANT_PROMPT,
    },
    .{
        .name = "Architect",
        .model_id = "openai/gpt-4o",
        .focus = "Readability, structure, logic",
        .system_prompt = prompts.ARCHITECT_PROMPT,
    },
    .{
        .name = "Hacker",
        .model_id = "deepseek/deepseek-coder",
        .focus = "Performance, cleverness, brevity",
        .system_prompt = prompts.HACKER_PROMPT,
    },
};

/// Verdict from a single judge
pub const JudgeVerdict = struct {
    judge_name: []const u8,
    score: f32, // 0.0 - 10.0
    rationale: []const u8,
    safety_pass: bool,
    correctness_pass: bool,
    zig_zen_score: f32, // Idiomatic Zig usage

    pub fn deinit(self: *JudgeVerdict, allocator: std.mem.Allocator) void {
        allocator.free(self.rationale);
    }
};

/// Consensus result from all judges
pub const ConsensusResult = struct {
    verdicts: []JudgeVerdict,
    average_score: f32,
    rating: Rating,
    allocator: std.mem.Allocator,

    pub const Rating = enum {
        S, // 9.0 - 10.0: Exceptional
        A, // 8.0 - 8.9: Excellent
        B, // 7.0 - 7.9: Good
        C, // 6.0 - 6.9: Acceptable
        D, // 5.0 - 5.9: Poor
        F, // 0.0 - 4.9: Fail

        pub fn fromScore(score: f32) Rating {
            if (score >= 9.0) return .S;
            if (score >= 8.0) return .A;
            if (score >= 7.0) return .B;
            if (score >= 6.0) return .C;
            if (score >= 5.0) return .D;
            return .F;
        }

        pub fn toString(self: Rating) []const u8 {
            return switch (self) {
                .S => "S",
                .A => "A",
                .B => "B",
                .C => "C",
                .D => "D",
                .F => "F",
            };
        }
    };

    pub fn deinit(self: *ConsensusResult) void {
        for (self.verdicts) |*v| {
            v.deinit(self.allocator);
        }
        self.allocator.free(self.verdicts);
    }

    /// Format as string like "A (8.5)"
    pub fn format(self: *const ConsensusResult, allocator: std.mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(allocator, "{s} ({d:.1})", .{
            self.rating.toString(),
            self.average_score,
        });
    }
};

/// Grading criteria
pub const GradingCriteria = struct {
    safety_weight: f32 = 0.4, // Memory safety, defer usage
    correctness_weight: f32 = 0.4, // Actually solves the problem
    zig_zen_weight: f32 = 0.2, // Idiomatic Zig style
};

// Tests
test "rating from score" {
    try std.testing.expectEqual(ConsensusResult.Rating.S, ConsensusResult.Rating.fromScore(9.5));
    try std.testing.expectEqual(ConsensusResult.Rating.A, ConsensusResult.Rating.fromScore(8.5));
    try std.testing.expectEqual(ConsensusResult.Rating.B, ConsensusResult.Rating.fromScore(7.5));
    try std.testing.expectEqual(ConsensusResult.Rating.C, ConsensusResult.Rating.fromScore(6.5));
    try std.testing.expectEqual(ConsensusResult.Rating.D, ConsensusResult.Rating.fromScore(5.5));
    try std.testing.expectEqual(ConsensusResult.Rating.F, ConsensusResult.Rating.fromScore(4.0));
}

test "rating toString" {
    try std.testing.expectEqualStrings("S", ConsensusResult.Rating.S.toString());
    try std.testing.expectEqualStrings("F", ConsensusResult.Rating.F.toString());
}
