//! Council tribunal orchestration
//! Coordinates multiple judge models for consensus scoring.
//!
//! NOTE: This module is a placeholder for Phase 3.
//! The council feature adds significant complexity and API cost,
//! and is only activated when --council flag is passed.

const std = @import("std");
const types = @import("types.zig");
const openrouter = @import("../gateways/openrouter.zig");

pub const Tribunal = struct {
    allocator: std.mem.Allocator,
    client: *openrouter.Client,
    judges: []const types.JudgePersona,

    pub fn init(
        allocator: std.mem.Allocator,
        client: *openrouter.Client,
    ) Tribunal {
        return .{
            .allocator = allocator,
            .client = client,
            .judges = &types.DEFAULT_JUDGES,
        };
    }

    /// Convene the council to judge a solution
    /// Returns a consensus rating based on multiple model evaluations
    pub fn convene(
        self: *Tribunal,
        problem_description: []const u8,
        solution_code: []const u8,
    ) !types.ConsensusResult {
        var verdicts: std.ArrayList(types.JudgeVerdict) = .empty;
        errdefer {
            for (verdicts.items) |*v| v.deinit(self.allocator);
            verdicts.deinit(self.allocator);
        }

        // Phase 1: Blind evaluation - each judge scores independently
        for (self.judges) |judge| {
            const verdict = try self.getJudgeVerdict(judge, problem_description, solution_code, null);
            try verdicts.append(self.allocator, verdict);
        }

        // Phase 2: Cross-pollination (optional, adds cost)
        // TODO: Implement in future - let judges see each other's rationale

        // Calculate consensus
        var total_score: f32 = 0;
        for (verdicts.items) |v| {
            total_score += v.score;
        }
        const average_score = total_score / @as(f32, @floatFromInt(verdicts.items.len));

        return types.ConsensusResult{
            .verdicts = try verdicts.toOwnedSlice(self.allocator),
            .average_score = average_score,
            .rating = types.ConsensusResult.Rating.fromScore(average_score),
            .allocator = self.allocator,
        };
    }

    fn getJudgeVerdict(
        self: *Tribunal,
        judge: types.JudgePersona,
        problem_description: []const u8,
        solution_code: []const u8,
        _: ?[]const types.JudgeVerdict, // other_verdicts for phase 2
    ) !types.JudgeVerdict {
        // Build prompt for the judge
        const user_prompt = try std.fmt.allocPrint(self.allocator,
            \\## Problem
            \\{s}
            \\
            \\## Candidate Solution
            \\```zig
            \\{s}
            \\```
            \\
            \\Please evaluate this solution on a scale of 0-10, providing:
            \\1. Overall score (0-10)
            \\2. Safety assessment (PASS/FAIL) - memory safety, proper error handling
            \\3. Correctness assessment (PASS/FAIL) - solves the stated problem
            \\4. Zig-Zen score (0-10) - idiomatic Zig usage
            \\5. Brief rationale (2-3 sentences)
            \\
            \\Format your response as:
            \\SCORE: X.X
            \\SAFETY: PASS/FAIL
            \\CORRECTNESS: PASS/FAIL
            \\ZIG_ZEN: X.X
            \\RATIONALE: Your explanation here
        , .{ problem_description, solution_code });
        defer self.allocator.free(user_prompt);

        const messages = [_]openrouter.Message{
            .{ .role = .system, .content = judge.system_prompt },
            .{ .role = .user, .content = user_prompt },
        };

        var response = try self.client.sendChatCompletion(judge.model_id, &messages);
        defer response.deinit(self.allocator);

        // Parse the response
        return try self.parseVerdictResponse(judge.name, response.content);
    }

    fn parseVerdictResponse(self: *Tribunal, judge_name: []const u8, response: []const u8) !types.JudgeVerdict {
        var score: f32 = 5.0;
        var safety_pass = false;
        var correctness_pass = false;
        var zig_zen_score: f32 = 5.0;
        var rationale: []const u8 = "Unable to parse judge response";

        var lines = std.mem.splitScalar(u8, response, '\n');
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");

            if (std.mem.startsWith(u8, trimmed, "SCORE:")) {
                const score_str = std.mem.trim(u8, trimmed["SCORE:".len..], " ");
                score = std.fmt.parseFloat(f32, score_str) catch 5.0;
            } else if (std.mem.startsWith(u8, trimmed, "SAFETY:")) {
                const val = std.mem.trim(u8, trimmed["SAFETY:".len..], " ");
                safety_pass = std.mem.eql(u8, val, "PASS");
            } else if (std.mem.startsWith(u8, trimmed, "CORRECTNESS:")) {
                const val = std.mem.trim(u8, trimmed["CORRECTNESS:".len..], " ");
                correctness_pass = std.mem.eql(u8, val, "PASS");
            } else if (std.mem.startsWith(u8, trimmed, "ZIG_ZEN:")) {
                const score_str = std.mem.trim(u8, trimmed["ZIG_ZEN:".len..], " ");
                zig_zen_score = std.fmt.parseFloat(f32, score_str) catch 5.0;
            } else if (std.mem.startsWith(u8, trimmed, "RATIONALE:")) {
                rationale = std.mem.trim(u8, trimmed["RATIONALE:".len..], " ");
            }
        }

        return types.JudgeVerdict{
            .judge_name = judge_name,
            .score = score,
            .rationale = try self.allocator.dupe(u8, rationale),
            .safety_pass = safety_pass,
            .correctness_pass = correctness_pass,
            .zig_zen_score = zig_zen_score,
        };
    }
};

// Tests
test "tribunal init" {
    // Basic initialization test
    const allocator = std.testing.allocator;
    var client = openrouter.Client.init(allocator, "test-key");
    defer client.deinit();

    const tribunal = Tribunal.init(allocator, &client);
    _ = tribunal;
}
