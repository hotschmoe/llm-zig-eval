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
    RequestTimeout,
};

/// OpenRouter client
pub const Client = struct {
    allocator: std.mem.Allocator,
    api_key: []const u8,
    http_client: std.http.Client,
    request_timeout_s: isize = 120,

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
                    0x08 => try body.appendSlice(self.allocator, "\\b"),
                    0x0C => try body.appendSlice(self.allocator, "\\f"),
                    else => {
                        if (c < 0x20) {
                            try body.appendSlice(self.allocator, "\\u00");
                            const hex = "0123456789abcdef";
                            try body.append(self.allocator, hex[c >> 4]);
                            try body.append(self.allocator, hex[c & 0x0F]);
                        } else {
                            try body.append(self.allocator, c);
                        }
                    },
                }
            }
            try body.append(self.allocator, '"');
            try body.appendSlice(self.allocator, "}");
        }

        try body.appendSlice(self.allocator, "]}");

        return try body.toOwnedSlice(self.allocator);
    }

    fn makeRequest(self: *Client, body: []const u8) ![]u8 {
        const auth_header = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.api_key});
        defer self.allocator.free(auth_header);

        const uri = try std.Uri.parse(OPENROUTER_API_URL);

        var req = try self.http_client.request(.POST, uri, .{
            .redirect_behavior = .unhandled,
            .extra_headers = &.{
                .{ .name = "Content-Type", .value = "application/json" },
                .{ .name = "Authorization", .value = auth_header },
                .{ .name = "HTTP-Referer", .value = "https://github.com/hotschmoe/llm-zig-eval" },
                .{ .name = "X-Title", .value = "llm-zig-eval" },
            },
        });
        defer req.deinit();

        if (req.connection) |conn| {
            applySocketTimeout(conn.stream_reader.getStream().handle, self.request_timeout_s);
        }

        req.transfer_encoding = .{ .content_length = body.len };
        var body_writer = try req.sendBodyUnflushed(&.{});
        try body_writer.writer.writeAll(body);
        try body_writer.end();
        try req.connection.?.flush();

        var response = req.receiveHead(&.{}) catch |err| {
            if (isTimeoutError(err)) return ApiError.RequestTimeout;
            return err;
        };

        var response_buf: std.Io.Writer.Allocating = .init(self.allocator);
        errdefer response_buf.deinit();

        var transfer_buffer: [64]u8 = undefined;
        var decompress: std.http.Decompress = undefined;
        const decompress_buffer: []u8 = switch (response.head.content_encoding) {
            .identity => &.{},
            .zstd => try self.allocator.alloc(u8, std.compress.zstd.default_window_len),
            .deflate, .gzip => try self.allocator.alloc(u8, std.compress.flate.max_window_len),
            .compress => return ApiError.HttpError,
        };
        defer if (decompress_buffer.len > 0) self.allocator.free(decompress_buffer);

        const reader = response.readerDecompressing(&transfer_buffer, &decompress, decompress_buffer);
        _ = reader.streamRemaining(&response_buf.writer) catch |err| {
            if (response.bodyErr()) |_| return ApiError.RequestTimeout;
            return err;
        };

        if (response.head.status != .ok) {
            std.debug.print("HTTP Error: {d} ({s})\n", .{ @intFromEnum(response.head.status), @tagName(response.head.status) });
            const resp_slice = response_buf.written();
            if (resp_slice.len > 0) {
                std.debug.print("Response: {s}\n", .{resp_slice[0..@min(resp_slice.len, 500)]});
            }
            return switch (response.head.status) {
                .unauthorized => ApiError.ApiUnauthorized,
                .too_many_requests => ApiError.ApiRateLimited,
                else => if (@intFromEnum(response.head.status) >= 500) ApiError.ApiServerError else ApiError.HttpError,
            };
        }

        return try response_buf.toOwnedSlice();
    }

    fn applySocketTimeout(fd: std.net.Stream.Handle, timeout_s: isize) void {
        const timeout: std.posix.timeval = .{ .sec = timeout_s, .usec = 0 };
        const timeout_bytes = std.mem.asBytes(&timeout);
        std.posix.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, timeout_bytes) catch {};
        std.posix.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.SNDTIMEO, timeout_bytes) catch {};
    }

    fn isTimeoutError(err: anytype) bool {
        return switch (err) {
            error.ReadFailed, error.ConnectionTimedOut => true,
            else => false,
        };
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
