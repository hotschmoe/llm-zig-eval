//! OpenRouter API gateway
//! HTTP client for sending chat completion requests to OpenRouter.
//!
//! Thread Safety: std.http.Client is NOT thread-safe. When using parallel
//! execution, each thread must create its own Client instance.

const std = @import("std");
const json = std.json;
const tokens = @import("../core/tokens.zig");

pub const OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions";

/// A message in the chat conversation
pub const Message = struct {
    role: Role,
    content: []const u8,

    pub const Role = enum {
        system,
        user,
        assistant,
    };
};

/// Re-export TokenUsage from tokens module
pub const TokenUsage = tokens.TokenUsage;

/// Response from OpenRouter API
pub const ChatResponse = struct {
    content: []const u8,
    usage: TokenUsage,
    model: []const u8,
    response_time_ms: i64,

    pub fn deinit(self: *ChatResponse, allocator: std.mem.Allocator) void {
        allocator.free(self.content);
        allocator.free(self.model);
    }
};

/// Error types from OpenRouter
pub const ApiError = error{
    HttpError,
    JsonParseError,
    ApiRateLimited,
    ApiUnauthorized,
    ApiServerError,
    NoContent,
    InvalidResponse,
};

/// OpenRouter client
pub const Client = struct {
    allocator: std.mem.Allocator,
    api_key: []const u8,
    http_client: std.http.Client,

    pub fn init(allocator: std.mem.Allocator, api_key: []const u8) Client {
        return .{
            .allocator = allocator,
            .api_key = api_key,
            .http_client = std.http.Client{ .allocator = allocator },
        };
    }

    pub fn deinit(self: *Client) void {
        self.http_client.deinit();
    }

    /// Send a chat completion request
    pub fn sendChatCompletion(
        self: *Client,
        model: []const u8,
        messages: []const Message,
    ) !ChatResponse {
        const start_time = std.time.milliTimestamp();

        // Build request body
        const body = try self.buildRequestBody(model, messages);
        defer self.allocator.free(body);

        // Make HTTP request
        const response = try self.makeRequest(body);
        defer self.allocator.free(response);

        const end_time = std.time.milliTimestamp();

        // Parse response
        return try self.parseResponse(response, end_time - start_time);
    }

    fn buildRequestBody(self: *Client, model: []const u8, messages: []const Message) ![]u8 {
        var body: std.ArrayList(u8) = .empty;
        errdefer body.deinit(self.allocator);

        try body.appendSlice(self.allocator, "{\"model\":\"");
        try body.appendSlice(self.allocator, model);
        try body.appendSlice(self.allocator, "\",\"messages\":[");

        for (messages, 0..) |msg, i| {
            if (i > 0) try body.appendSlice(self.allocator, ",");

            try body.appendSlice(self.allocator, "{\"role\":\"");
            try body.appendSlice(self.allocator, switch (msg.role) {
                .system => "system",
                .user => "user",
                .assistant => "assistant",
            });
            try body.appendSlice(self.allocator, "\",\"content\":");
            // JSON-escape the content string
            try body.append(self.allocator, '"');
            for (msg.content) |c| {
                switch (c) {
                    '"' => try body.appendSlice(self.allocator, "\\\""),
                    '\\' => try body.appendSlice(self.allocator, "\\\\"),
                    '\n' => try body.appendSlice(self.allocator, "\\n"),
                    '\r' => try body.appendSlice(self.allocator, "\\r"),
                    '\t' => try body.appendSlice(self.allocator, "\\t"),
                    else => try body.append(self.allocator, c),
                }
            }
            try body.append(self.allocator, '"');
            try body.appendSlice(self.allocator, "}");
        }

        try body.appendSlice(self.allocator, "]}");

        return try body.toOwnedSlice(self.allocator);
    }

    fn makeRequest(self: *Client, body: []const u8) ![]u8 {
        // Build authorization header
        const auth_header = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.api_key});
        defer self.allocator.free(auth_header);

        // Prepare response body buffer using Zig 0.15 Writer.Allocating
        var response_writer: std.Io.Writer.Allocating = .init(self.allocator);
        errdefer response_writer.deinit();

        // Use fetch API - Zig 0.15 style
        const result = try self.http_client.fetch(.{
            .location = .{ .url = OPENROUTER_API_URL },
            .method = .POST,
            .extra_headers = &.{
                .{ .name = "Content-Type", .value = "application/json" },
                .{ .name = "Authorization", .value = auth_header },
                .{ .name = "HTTP-Referer", .value = "https://github.com/hotschmoe/llm-zig-eval" },
                .{ .name = "X-Title", .value = "llm-zig-eval" },
            },
            .payload = body,
            .response_writer = &response_writer.writer,
        });

        // Check status code
        if (result.status != .ok) {
            std.debug.print("HTTP Error: {d} ({s})\n", .{ @intFromEnum(result.status), @tagName(result.status) });
            // Print response body for debugging
            const body_slice = response_writer.written();
            if (body_slice.len > 0) {
                std.debug.print("Response: {s}\n", .{body_slice[0..@min(body_slice.len, 500)]});
            }
            return switch (result.status) {
                .unauthorized => ApiError.ApiUnauthorized,
                .too_many_requests => ApiError.ApiRateLimited,
                else => if (@intFromEnum(result.status) >= 500) ApiError.ApiServerError else ApiError.HttpError,
            };
        }

        return try response_writer.toOwnedSlice();
    }

    fn parseResponse(self: *Client, response: []const u8, response_time_ms: i64) !ChatResponse {
        const ParsedResponse = struct {
            choices: ?[]const struct {
                message: ?struct {
                    content: ?[]const u8 = null,
                } = null,
            } = null,
            usage: ?struct {
                prompt_tokens: ?u32 = null,
                completion_tokens: ?u32 = null,
                total_tokens: ?u32 = null,
            } = null,
            model: ?[]const u8 = null,
        };

        const parsed = json.parseFromSlice(ParsedResponse, self.allocator, response, .{
            .ignore_unknown_fields = true,
        }) catch |err| {
            std.debug.print("JSON parse error: {}\n", .{err});
            std.debug.print("Response: {s}\n", .{response});
            return ApiError.JsonParseError;
        };
        defer parsed.deinit();

        const value = parsed.value;

        // Extract content
        const content = blk: {
            if (value.choices) |choices| {
                if (choices.len > 0) {
                    if (choices[0].message) |msg| {
                        if (msg.content) |c| {
                            break :blk try self.allocator.dupe(u8, c);
                        }
                    }
                }
            }
            return ApiError.NoContent;
        };
        errdefer self.allocator.free(content);

        // Extract usage
        const usage = TokenUsage{
            .prompt_tokens = if (value.usage) |u| u.prompt_tokens orelse 0 else 0,
            .completion_tokens = if (value.usage) |u| u.completion_tokens orelse 0 else 0,
            .total_tokens = if (value.usage) |u| u.total_tokens orelse 0 else 0,
        };

        // Extract model
        const model = try self.allocator.dupe(u8, value.model orelse "unknown");
        errdefer self.allocator.free(model);

        return ChatResponse{
            .content = content,
            .usage = usage,
            .model = model,
            .response_time_ms = response_time_ms,
        };
    }
};

// Tests
test "Message role serialization" {
    try std.testing.expectEqual(Message.Role.system, Message.Role.system);
    try std.testing.expectEqual(Message.Role.user, Message.Role.user);
    try std.testing.expectEqual(Message.Role.assistant, Message.Role.assistant);
}
