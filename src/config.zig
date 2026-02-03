//! Configuration module for llm-zig-eval
//! Handles environment variables, CLI arguments, and model cost tables.

const std = @import("std");

/// Model pricing information ($ per million tokens)
pub const ModelCost = struct {
    model_id: []const u8,
    input_cost: f64, // $ per 1M input tokens
    output_cost: f64, // $ per 1M output tokens
};

/// Known model costs (as of 2024)
/// Source: https://openrouter.ai/docs#models
pub const MODEL_COSTS = [_]ModelCost{
    // Anthropic
    .{ .model_id = "anthropic/claude-3.5-sonnet", .input_cost = 3.0, .output_cost = 15.0 },
    .{ .model_id = "anthropic/claude-3-opus", .input_cost = 15.0, .output_cost = 75.0 },
    .{ .model_id = "anthropic/claude-3-sonnet", .input_cost = 3.0, .output_cost = 15.0 },
    .{ .model_id = "anthropic/claude-3-haiku", .input_cost = 0.25, .output_cost = 1.25 },
    // OpenAI
    .{ .model_id = "openai/gpt-4o", .input_cost = 2.5, .output_cost = 10.0 },
    .{ .model_id = "openai/gpt-4o-mini", .input_cost = 0.15, .output_cost = 0.6 },
    .{ .model_id = "openai/gpt-4-turbo", .input_cost = 10.0, .output_cost = 30.0 },
    // Google
    .{ .model_id = "google/gemini-1.5-pro", .input_cost = 2.5, .output_cost = 7.5 },
    .{ .model_id = "google/gemini-1.5-flash", .input_cost = 0.075, .output_cost = 0.3 },
    // Meta
    .{ .model_id = "meta-llama/llama-3.1-70b-instruct", .input_cost = 0.52, .output_cost = 0.75 },
    .{ .model_id = "meta-llama/llama-3.1-8b-instruct", .input_cost = 0.06, .output_cost = 0.06 },
    // DeepSeek
    .{ .model_id = "deepseek/deepseek-chat", .input_cost = 0.14, .output_cost = 0.28 },
};

/// Look up cost for a model, returns null if unknown
pub fn getModelCost(model_id: []const u8) ?ModelCost {
    for (MODEL_COSTS) |cost| {
        if (std.mem.eql(u8, cost.model_id, model_id)) {
            return cost;
        }
    }
    return null;
}

/// CLI configuration
pub const Config = struct {
    /// List of model IDs to benchmark
    models: []const []const u8,
    /// Number of runs per model per problem
    runs: u32 = 1,
    /// Enable Council of Judges scoring
    council: bool = false,
    /// Output format
    output_format: OutputFormat = .pretty,
    /// Maximum concurrent API requests
    parallel: u32 = 4,
    /// OpenRouter API key
    api_key: []const u8,
    /// Allocator for dynamic allocations
    allocator: std.mem.Allocator,

    pub const OutputFormat = enum {
        pretty,
        json,
    };

    pub fn deinit(self: *Config) void {
        for (self.models) |model| {
            self.allocator.free(model);
        }
        self.allocator.free(self.models);
        self.allocator.free(self.api_key);
    }
};

/// Parse CLI arguments into Config
pub fn parseArgs(allocator: std.mem.Allocator) !Config {
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    // Skip program name
    _ = args.skip();

    var models_str: ?[]const u8 = null;
    var runs: u32 = 1;
    var council = false;
    var output_format = Config.OutputFormat.pretty;
    var parallel: u32 = 4;

    while (args.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "--models=")) {
            models_str = arg["--models=".len..];
        } else if (std.mem.startsWith(u8, arg, "--runs=")) {
            const num_str = arg["--runs=".len..];
            runs = try std.fmt.parseInt(u32, num_str, 10);
        } else if (std.mem.eql(u8, arg, "--council")) {
            council = true;
        } else if (std.mem.startsWith(u8, arg, "--output=")) {
            const fmt_str = arg["--output=".len..];
            if (std.mem.eql(u8, fmt_str, "json")) {
                output_format = .json;
            } else if (std.mem.eql(u8, fmt_str, "pretty")) {
                output_format = .pretty;
            } else {
                return error.InvalidOutputFormat;
            }
        } else if (std.mem.startsWith(u8, arg, "--parallel=")) {
            const num_str = arg["--parallel=".len..];
            parallel = try std.fmt.parseInt(u32, num_str, 10);
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return error.HelpRequested;
        }
    }

    // Get API key from environment or .env file
    const api_key = blk: {
        // First try environment variable
        if (std.process.getEnvVarOwned(allocator, "OPENROUTER_API_KEY")) |key| {
            break :blk key;
        } else |err| {
            if (err != error.EnvironmentVariableNotFound) {
                return err;
            }
        }
        // Fall back to .env file
        if (loadEnvFile(allocator, "OPENROUTER_API_KEY") catch null) |key| {
            std.debug.print("Loaded API key from .env file\n", .{});
            break :blk key;
        }
        std.debug.print("Error: OPENROUTER_API_KEY not found in environment or .env file\n", .{});
        return error.MissingApiKey;
    };
    errdefer allocator.free(api_key);

    // Parse models list
    if (models_str == null) {
        std.debug.print("Error: --models argument is required\n", .{});
        return error.MissingModels;
    }

    var model_list: std.ArrayList([]const u8) = .empty;
    errdefer model_list.deinit(allocator);

    var it = std.mem.splitScalar(u8, models_str.?, ',');
    while (it.next()) |model| {
        const trimmed = std.mem.trim(u8, model, " ");
        if (trimmed.len > 0) {
            // Duplicate to own the memory (args will be freed after parseArgs returns)
            const owned = try allocator.dupe(u8, trimmed);
            try model_list.append(allocator, owned);
        }
    }

    if (model_list.items.len == 0) {
        return error.MissingModels;
    }

    return Config{
        .models = try model_list.toOwnedSlice(allocator),
        .runs = runs,
        .council = council,
        .output_format = output_format,
        .parallel = parallel,
        .api_key = api_key,
        .allocator = allocator,
    };
}

/// Load a value from .env file
fn loadEnvFile(allocator: std.mem.Allocator, key: []const u8) !?[]const u8 {
    const file = std.fs.cwd().openFile(".env", .{}) catch |err| {
        if (err == error.FileNotFound) return null;
        return err;
    };
    defer file.close();

    var buf: [4096]u8 = undefined;
    const bytes_read = file.readAll(&buf) catch return null;
    const content = buf[0..bytes_read];

    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0 or trimmed[0] == '#') continue;

        if (std.mem.indexOf(u8, trimmed, "=")) |eq_pos| {
            const line_key = std.mem.trim(u8, trimmed[0..eq_pos], " \t");
            if (std.mem.eql(u8, line_key, key)) {
                const value = std.mem.trim(u8, trimmed[eq_pos + 1 ..], " \t\"'");
                return try allocator.dupe(u8, value);
            }
        }
    }
    return null;
}

fn printUsage() void {
    const usage =
        \\llm-zig-eval - Benchmark LLMs on Zig programming tasks
        \\
        \\Usage: llm-zig-eval [options]
        \\
        \\Options:
        \\  --models=MODEL1,MODEL2    Comma-separated list of model IDs (required)
        \\  --runs=N                  Number of runs per model per problem (default: 1)
        \\  --council                 Enable Council of Judges scoring
        \\  --output=FORMAT           Output format: pretty, json (default: pretty)
        \\  --parallel=N              Max concurrent API requests (default: 4)
        \\  --help, -h                Show this help message
        \\
        \\Environment:
        \\  OPENROUTER_API_KEY        Your OpenRouter API key (required)
        \\
        \\Examples:
        \\  llm-zig-eval --models=anthropic/claude-3.5-sonnet,openai/gpt-4o
        \\  llm-zig-eval --models=anthropic/claude-3-haiku --runs=3 --council
        \\
    ;
    std.debug.print("{s}", .{usage});
}

pub const ConfigError = error{
    InvalidOutputFormat,
    HelpRequested,
    MissingApiKey,
    MissingModels,
};

// Tests
test "getModelCost finds known model" {
    const cost = getModelCost("anthropic/claude-3.5-sonnet");
    try std.testing.expect(cost != null);
    try std.testing.expectEqual(@as(f64, 3.0), cost.?.input_cost);
    try std.testing.expectEqual(@as(f64, 15.0), cost.?.output_cost);
}

test "getModelCost returns null for unknown model" {
    const cost = getModelCost("unknown/model");
    try std.testing.expect(cost == null);
}
