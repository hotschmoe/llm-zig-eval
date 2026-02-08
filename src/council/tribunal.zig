//! Council tribunal orchestration
//! Coordinates multiple judge models for consensus scoring.
//!
//! Phase 1: Blind evaluation - each judge scores independently.
//! Phase 2: Cross-pollination - judges revise after seeing others' rationale.

const std = @import("std");
const types = @import("types.zig");
const openrouter = @import("../gateways/openrouter.zig");

pub const Tribunal = struct {
    allocator: std.mem.Allocator,
    client: *openrouter.Client,
    judges: []const types.JudgePersona,
    custom_judges: ?[]types.JudgePersona = null,

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

    /// Create a Tribunal with custom model IDs paired with cycling personas.
    /// Personas cycle: Pedant, Architect, Hacker, Pedant, ...
    pub fn initWithModels(
        allocator: std.mem.Allocator,
        client: *openrouter.Client,
        model_ids: []const []const u8,
    ) !Tribunal {
        const custom = try allocator.alloc(types.JudgePersona, model_ids.len);
        for (model_ids, 0..) |model_id, i| {
            const persona = types.JUDGE_PERSONAS[i % types.JUDGE_PERSONAS.len];
            custom[i] = .{
                .name = persona.name,
                .model_id = model_id,
                .focus = persona.focus,
                .system_prompt = persona.system_prompt,
            };
        }
        return .{
            .allocator = allocator,
            .client = client,
            .judges = custom,
            .custom_judges = custom,
        };
    }

    pub fn deinit(self: *Tribunal) void {
        if (self.custom_judges) |cj| {
            self.allocator.free(cj);
        }
    }

    /// Convene the council to judge a solution.
    /// Phase 1: Each judge evaluates independently (blind).
    /// Phase 2: Judges see all Phase 1 verdicts and may revise scores.
    /// Returns consensus rating from Phase 2 (revised) scores.
    pub fn convene(
        self: *Tribunal,
        problem_description: []const u8,
        solution_code: []const u8,
    ) !types.ConsensusResult {
        // Phase 1: Blind evaluation - each judge scores independently
        var phase1: std.ArrayList(types.JudgeVerdict) = .empty;
        defer {
            for (phase1.items) |*v| v.deinit(self.allocator);
            phase1.deinit(self.allocator);
        }

        for (self.judges) |judge| {
            const verdict = try self.getJudgeVerdict(judge, problem_description, solution_code, null);
            try phase1.append(self.allocator, verdict);
        }

        // Phase 2: Cross-pollination - judges revise after seeing others' rationale
        var phase2: std.ArrayList(types.JudgeVerdict) = .empty;
        errdefer {
            for (phase2.items) |*v| v.deinit(self.allocator);
            phase2.deinit(self.allocator);
        }

        for (self.judges) |judge| {
            const verdict = try self.getJudgeVerdict(judge, problem_description, solution_code, phase1.items);
            try phase2.append(self.allocator, verdict);
        }

        // Calculate consensus from Phase 2 (revised) verdicts
        var total_score: f32 = 0;
        for (phase2.items) |v| {
            total_score += v.score;
        }
        const average_score = total_score / @as(f32, @floatFromInt(phase2.items.len));

        return types.ConsensusResult{
            .verdicts = try phase2.toOwnedSlice(self.allocator),
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
        phase1_verdicts: ?[]const types.JudgeVerdict,
    ) !types.JudgeVerdict {
        const user_prompt = if (phase1_verdicts) |verdicts|
            try self.buildCrossPollinationPrompt(judge.name, problem_description, solution_code, verdicts)
        else
            try std.fmt.allocPrint(self.allocator,
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

        return try self.parseVerdictResponse(judge.name, response.content);
    }

    /// Build Phase 2 prompt with all Phase 1 verdicts for cross-pollination.
    fn buildCrossPollinationPrompt(
        self: *Tribunal,
        judge_name: []const u8,
        problem_description: []const u8,
        solution_code: []const u8,
        verdicts: []const types.JudgeVerdict,
    ) ![]const u8 {
        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(self.allocator);

        const header = try std.fmt.allocPrint(self.allocator,
            \\## Problem
            \\{s}
            \\
            \\## Candidate Solution
            \\```zig
            \\{s}
            \\```
            \\
            \\## Phase 1 Council Evaluations
            \\The following are the blind evaluations from all judges.
            \\
        , .{ problem_description, solution_code });
        defer self.allocator.free(header);
        try buf.appendSlice(self.allocator, header);

        for (verdicts) |v| {
            const is_self = std.mem.eql(u8, v.judge_name, judge_name);
            const suffix: []const u8 = if (is_self) " (your Phase 1 evaluation)" else "";
            const safety: []const u8 = if (v.safety_pass) "PASS" else "FAIL";
            const correctness: []const u8 = if (v.correctness_pass) "PASS" else "FAIL";
            const entry = try std.fmt.allocPrint(self.allocator,
                \\### {s}{s}
                \\Score: {d:.1} | Safety: {s} | Correctness: {s} | Zig-Zen: {d:.1}
                \\Rationale: {s}
                \\
            , .{ v.judge_name, suffix, v.score, safety, correctness, v.zig_zen_score, v.rationale });
            defer self.allocator.free(entry);
            try buf.appendSlice(self.allocator, entry);
        }

        try buf.appendSlice(self.allocator,
            \\## Cross-Pollination Review
            \\Consider the other judges' perspectives and rationale.
            \\You may revise your scores if their reasoning reveals issues you
            \\missed, or stand firm with justification if you disagree.
            \\
            \\Provide your revised evaluation:
            \\SCORE: X.X
            \\SAFETY: PASS/FAIL
            \\CORRECTNESS: PASS/FAIL
            \\ZIG_ZEN: X.X
            \\RATIONALE: Your explanation here
        );

        return try buf.toOwnedSlice(self.allocator);
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

test "tribunal init" {
    const allocator = std.testing.allocator;
    var client = openrouter.Client.init(allocator, "test-key");
    defer client.deinit();

    var t = Tribunal.init(allocator, &client);
    defer t.deinit();
    try std.testing.expectEqual(@as(usize, 3), t.judges.len);
}

test "tribunal initWithModels" {
    const allocator = std.testing.allocator;
    var client = openrouter.Client.init(allocator, "test-key");
    defer client.deinit();

    const models = &[_][]const u8{ "model/a", "model/b" };
    var t = try Tribunal.initWithModels(allocator, &client, models);
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 2), t.judges.len);
    try std.testing.expectEqualStrings("Pedant", t.judges[0].name);
    try std.testing.expectEqualStrings("model/a", t.judges[0].model_id);
    try std.testing.expectEqualStrings("Architect", t.judges[1].name);
    try std.testing.expectEqualStrings("model/b", t.judges[1].model_id);
}
