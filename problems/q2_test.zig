//! Test harness for Problem 2: Mock TCP Socket
//! This file imports the LLM-generated solution and validates it.

const std = @import("std");
const testing = std.testing;
const time = std.time;

// Import the solution
const solution = @import("solution.zig");
const MockSocket = solution.MockSocket;

test "basic connection works" {
    var sock = MockSocket.init();
    defer sock.deinit();

    try sock.connect("localhost");
    sock.waitForConnection();

    try testing.expect(sock.isConnected());
}

test "bad_host returns ConnectionRefused" {
    var sock = MockSocket.init();
    defer sock.deinit();

    const result = sock.connect("bad_host");
    try testing.expectError(error.ConnectionRefused, result);
    try testing.expect(!sock.isConnected());
}

test "initially not connected" {
    var sock = MockSocket.init();
    defer sock.deinit();

    try testing.expect(!sock.isConnected());
}

test "parallel connections complete in ~100ms not 500ms" {
    const NUM_SOCKETS = 5;
    var sockets: [NUM_SOCKETS]MockSocket = undefined;

    for (&sockets) |*sock| {
        sock.* = MockSocket.init();
    }
    defer {
        for (&sockets) |*sock| {
            sock.deinit();
        }
    }

    const start = time.nanoTimestamp();

    // Start all connections in parallel
    for (&sockets) |*sock| {
        try sock.connect("good_host");
    }

    // Wait for all to complete
    for (&sockets) |*sock| {
        sock.waitForConnection();
    }

    const elapsed_ns = time.nanoTimestamp() - start;
    const elapsed_ms = @divFloor(elapsed_ns, 1_000_000);

    // Should complete in ~100-200ms, not 500ms (5 × 100ms)
    // Allow generous margin for test timing variance
    try testing.expect(elapsed_ms < 400);

    // All should be connected
    for (&sockets) |*sock| {
        try testing.expect(sock.isConnected());
    }
}

test "mixed good and bad hosts" {
    var good_sock = MockSocket.init();
    defer good_sock.deinit();

    var bad_sock = MockSocket.init();
    defer bad_sock.deinit();

    // Good host should connect
    try good_sock.connect("good_host");
    good_sock.waitForConnection();
    try testing.expect(good_sock.isConnected());

    // Bad host should fail
    try testing.expectError(error.ConnectionRefused, bad_sock.connect("bad_host"));
    try testing.expect(!bad_sock.isConnected());
}

test "connect state is thread-safe" {
    var sock = MockSocket.init();
    defer sock.deinit();

    // Start connection
    try sock.connect("localhost");

    // Poll isConnected while connection is in progress
    // This should not crash or race
    var checks: usize = 0;
    while (!sock.isConnected() and checks < 100) : (checks += 1) {
        std.Thread.sleep(10_000_000); // 10ms
    }

    sock.waitForConnection();
    try testing.expect(sock.isConnected());
}
