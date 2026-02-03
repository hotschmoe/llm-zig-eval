//! Response parser module
//! Extracts Zig code blocks from LLM responses.

const std = @import("std");

/// Extract Zig code from a markdown code block
/// Looks for ```zig ... ``` patterns
pub fn extractZigCode(allocator: std.mem.Allocator, response: []const u8) !?[]const u8 {
    // Look for ```zig code blocks
    const zig_start = "```zig";
    const code_start = "```";
    const code_end = "```";

    // First try to find ```zig
    var start_idx: ?usize = null;
    var search_pos: usize = 0;

    // Look for ```zig block
    if (std.mem.indexOf(u8, response, zig_start)) |idx| {
        // Find the newline after ```zig
        if (std.mem.indexOfPos(u8, response, idx + zig_start.len, "\n")) |newline_idx| {
            start_idx = newline_idx + 1;
            search_pos = newline_idx + 1;
        }
    }

    // If no ```zig found, try plain ``` (first one)
    if (start_idx == null) {
        if (std.mem.indexOf(u8, response, code_start)) |idx| {
            if (std.mem.indexOfPos(u8, response, idx + code_start.len, "\n")) |newline_idx| {
                start_idx = newline_idx + 1;
                search_pos = newline_idx + 1;
            }
        }
    }

    if (start_idx == null) {
        return null;
    }

    // Find the closing ```
    if (std.mem.indexOfPos(u8, response, search_pos, code_end)) |end_idx| {
        // Find the last newline before ```
        var actual_end = end_idx;
        if (end_idx > 0 and response[end_idx - 1] == '\n') {
            actual_end = end_idx - 1;
        }
        // Handle Windows line endings
        if (actual_end > 0 and response[actual_end - 1] == '\r') {
            actual_end -= 1;
        }

        const code = response[start_idx.?..actual_end];

        // Return a copy
        return try allocator.dupe(u8, code);
    }

    return null;
}

/// Count lines of code (excluding empty lines and comments)
pub fn countLoc(code: []const u8) usize {
    var count: usize = 0;
    var lines = std.mem.splitScalar(u8, code, '\n');

    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");

        // Skip empty lines
        if (trimmed.len == 0) continue;

        // Skip comment-only lines
        if (std.mem.startsWith(u8, trimmed, "//")) continue;

        count += 1;
    }

    return count;
}

// Tests
test "extract zig code from markdown" {
    const allocator = std.testing.allocator;

    const response =
        \\Here's the solution:
        \\
        \\```zig
        \\const std = @import("std");
        \\
        \\pub fn main() void {
        \\    std.debug.print("Hello\n", .{});
        \\}
        \\```
        \\
        \\That should work!
    ;

    const code = try extractZigCode(allocator, response);
    defer if (code) |c| allocator.free(c);

    try std.testing.expect(code != null);
    try std.testing.expect(std.mem.indexOf(u8, code.?, "const std") != null);
    try std.testing.expect(std.mem.indexOf(u8, code.?, "pub fn main") != null);
}

test "extract from plain code block" {
    const allocator = std.testing.allocator;

    const response =
        \\```
        \\const x = 42;
        \\```
    ;

    const code = try extractZigCode(allocator, response);
    defer if (code) |c| allocator.free(c);

    try std.testing.expect(code != null);
    try std.testing.expect(std.mem.indexOf(u8, code.?, "const x = 42") != null);
}

test "returns null when no code block" {
    const allocator = std.testing.allocator;

    const response = "This response has no code blocks.";

    const code = try extractZigCode(allocator, response);

    try std.testing.expect(code == null);
}

test "count LOC" {
    const code =
        \\const std = @import("std");
        \\
        \\// This is a comment
        \\pub fn main() void {
        \\    // Another comment
        \\    std.debug.print("Hello\n", .{});
        \\}
        \\
    ;

    const loc = countLoc(code);
    try std.testing.expectEqual(@as(usize, 4), loc); // import, pub fn, print, closing brace
}
