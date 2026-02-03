//! Test harness for Problem 1: Arena Allocator
//! This file imports the LLM-generated solution and validates it.

const std = @import("std");
const testing = std.testing;

// Import the solution - this path will be set up by the sandbox
const solution = @import("solution.zig");
const MiniArena = solution.MiniArena;

test "basic allocation works" {
    var arena = MiniArena.init();

    const slice = try arena.alloc(100);
    try testing.expectEqual(@as(usize, 100), slice.len);
}

test "multiple allocations work" {
    var arena = MiniArena.init();

    const slice1 = try arena.alloc(100);
    const slice2 = try arena.alloc(200);
    const slice3 = try arena.alloc(50);

    try testing.expectEqual(@as(usize, 100), slice1.len);
    try testing.expectEqual(@as(usize, 200), slice2.len);
    try testing.expectEqual(@as(usize, 50), slice3.len);

    // Slices should not overlap
    const ptr1_end = @intFromPtr(slice1.ptr) + slice1.len;
    const ptr2_start = @intFromPtr(slice2.ptr);
    try testing.expect(ptr1_end <= ptr2_start);
}

test "allocations are 8-byte aligned" {
    var arena = MiniArena.init();

    // Allocate odd sizes to test alignment
    const slice1 = try arena.alloc(7);
    const slice2 = try arena.alloc(13);
    const slice3 = try arena.alloc(1);

    // All pointers should be 8-byte aligned
    try testing.expectEqual(@as(usize, 0), @intFromPtr(slice1.ptr) % 8);
    try testing.expectEqual(@as(usize, 0), @intFromPtr(slice2.ptr) % 8);
    try testing.expectEqual(@as(usize, 0), @intFromPtr(slice3.ptr) % 8);
}

test "OutOfMemory when buffer full" {
    var arena = MiniArena.init();

    // Allocate most of the buffer
    _ = try arena.alloc(900);

    // This should fail - not enough space
    const result = arena.alloc(200);
    try testing.expectError(error.OutOfMemory, result);
}

test "reset allows reuse" {
    var arena = MiniArena.init();

    // Fill up the arena
    _ = try arena.alloc(500);
    _ = try arena.alloc(500);

    // This should fail
    try testing.expectError(error.OutOfMemory, arena.alloc(100));

    // Reset and try again
    arena.reset();

    // Now allocation should work
    const slice = try arena.alloc(1000);
    try testing.expectEqual(@as(usize, 1000), slice.len);
}

test "can allocate entire buffer after reset" {
    var arena = MiniArena.init();

    _ = try arena.alloc(100);
    arena.reset();

    // Should be able to allocate close to full buffer
    // Account for alignment - asking for 1016 bytes (1024 - 8 for alignment slack)
    const slice = try arena.alloc(1016);
    try testing.expectEqual(@as(usize, 1016), slice.len);
}

test "allocated memory is writable" {
    var arena = MiniArena.init();

    const slice = try arena.alloc(10);

    // Write to the memory
    for (slice, 0..) |*byte, i| {
        byte.* = @truncate(i);
    }

    // Read it back
    for (slice, 0..) |byte, i| {
        try testing.expectEqual(@as(u8, @truncate(i)), byte);
    }
}

test "zero-size allocation" {
    var arena = MiniArena.init();

    // Zero-size allocation should succeed
    const slice = try arena.alloc(0);
    try testing.expectEqual(@as(usize, 0), slice.len);
}
