//! llm-zig-eval library root
//! Exports all modules for library usage.

const std = @import("std");

// Core modules
pub const config = @import("config.zig");
pub const parser = @import("core/parser.zig");
pub const sandbox = @import("core/sandbox.zig");
pub const tokens = @import("core/tokens.zig");
pub const reporter = @import("core/reporter.zig");

// Gateway modules
pub const openrouter = @import("gateways/openrouter.zig");

// Council modules
pub const council_types = @import("council/types.zig");
pub const tribunal = @import("council/tribunal.zig");
pub const prompts = @import("council/prompts.zig");

// Re-export commonly used types
pub const Config = config.Config;
pub const ModelCost = config.ModelCost;
pub const Client = openrouter.Client;
pub const Message = openrouter.Message;
pub const ChatResponse = openrouter.ChatResponse;
pub const TokenUsage = tokens.TokenUsage;
pub const Report = reporter.Report;
pub const ModelResult = reporter.ModelResult;
pub const ProblemResult = reporter.ProblemResult;
pub const Sandbox = sandbox.Sandbox;
pub const SandboxResult = sandbox.SandboxResult;
pub const Problem = sandbox.Problem;
pub const PROBLEMS = sandbox.PROBLEMS;
pub const Tribunal = tribunal.Tribunal;
pub const ConsensusResult = council_types.ConsensusResult;

// Tests
test "module imports" {
    // Verify all imports work
    _ = config;
    _ = parser;
    _ = sandbox;
    _ = tokens;
    _ = reporter;
    _ = openrouter;
    _ = council_types;
    _ = tribunal;
    _ = prompts;
}

test "all module tests" {
    std.testing.refAllDecls(@This());
}
