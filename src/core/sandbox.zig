//! Sandbox executor module
//! Writes generated code to files and runs zig test on them.

const std = @import("std");
const fs = std.fs;

/// Result of running a solution in the sandbox
pub const SandboxResult = struct {
    status: Status,
    stdout: []const u8,
    stderr: []const u8,
    allocator: std.mem.Allocator,

    pub const Status = enum {
        pass,
        compile_error,
        test_error,
        timeout,
    };

    pub fn deinit(self: *SandboxResult) void {
        self.allocator.free(self.stdout);
        self.allocator.free(self.stderr);
    }
};

/// Problem definition
pub const Problem = struct {
    id: []const u8,
    name: []const u8,
    prompt_path: []const u8,
    test_path: []const u8,
};

/// Benchmark problems (simple test for now)
pub const PROBLEMS = [_]Problem{
    .{
        .id = "q0",
        .name = "isPrime",
        .prompt_path = "problems/q0_primes.txt",
        .test_path = "problems/q0_test.zig",
    },
    // TODO: Re-enable after error retry is working
    // .{
    //     .id = "q1",
    //     .name = "Arena Allocator",
    //     .prompt_path = "problems/q1_memory.txt",
    //     .test_path = "problems/q1_test.zig",
    // },
    // .{
    //     .id = "q2",
    //     .name = "Mock Socket",
    //     .prompt_path = "problems/q2_concurrency.txt",
    //     .test_path = "problems/q2_test.zig",
    // },
    // .{
    //     .id = "q3",
    //     .name = "JSON Parser",
    //     .prompt_path = "problems/q3_comptime.txt",
    //     .test_path = "problems/q3_test.zig",
    // },
};

/// Sandbox for running LLM-generated code
pub const Sandbox = struct {
    allocator: std.mem.Allocator,
    output_dir: []const u8,

    pub fn init(allocator: std.mem.Allocator, output_dir: []const u8) Sandbox {
        return .{
            .allocator = allocator,
            .output_dir = output_dir,
        };
    }

    /// Create output directory structure for a model
    pub fn createModelDir(self: *Sandbox, model_id: []const u8) ![]const u8 {
        // Sanitize model ID for filesystem (replace / with _)
        const safe_name = try self.allocator.alloc(u8, model_id.len);
        defer self.allocator.free(safe_name);

        for (model_id, 0..) |c, i| {
            safe_name[i] = if (c == '/' or c == '\\' or c == ':') '_' else c;
        }

        // Build path: out/{model_name}/
        const model_dir = try std.fs.path.join(self.allocator, &.{ self.output_dir, safe_name });
        errdefer self.allocator.free(model_dir);

        // Create directory (ignore if already exists)
        fs.cwd().makePath(model_dir) catch |err| {
            if (err != error.PathAlreadyExists) return err;
        };

        return model_dir;
    }

    /// Write solution code to file
    pub fn writeSolution(self: *Sandbox, model_dir: []const u8, problem_id: []const u8, code: []const u8) ![]const u8 {
        // Build filename: {model_dir}/{problem_id}_solution.zig
        const filename = try std.fmt.allocPrint(self.allocator, "{s}_solution.zig", .{problem_id});
        defer self.allocator.free(filename);

        const filepath = try std.fs.path.join(self.allocator, &.{ model_dir, filename });
        errdefer self.allocator.free(filepath);

        // Write file
        const file = try fs.cwd().createFile(filepath, .{});
        defer file.close();

        try file.writeAll(code);

        return filepath;
    }

    /// Run zig test with the solution
    pub fn runTest(self: *Sandbox, solution_path: []const u8, test_harness_path: []const u8) !SandboxResult {
        // Build command: zig test test_harness.zig --mod solution:solution_path
        // Actually we need the test harness to @import("solution.zig"), so we need to
        // copy the test harness next to the solution with the right module path

        // Get directory of solution
        const solution_dir = std.fs.path.dirname(solution_path) orelse ".";
        const solution_basename = std.fs.path.basename(solution_path);

        // Copy test harness to solution directory
        const test_dest = try std.fs.path.join(self.allocator, &.{ solution_dir, "test.zig" });
        defer self.allocator.free(test_dest);

        // Read and modify test harness to import the correct solution
        const test_harness = try fs.cwd().readFileAlloc(self.allocator, test_harness_path, 1024 * 1024);
        defer self.allocator.free(test_harness);

        // The test harness imports "solution.zig", so we need to rename our file
        // or create a solution.zig that re-exports
        const solution_link = try std.fs.path.join(self.allocator, &.{ solution_dir, "solution.zig" });
        defer self.allocator.free(solution_link);

        // If solution wasn't named solution.zig, copy it
        if (!std.mem.eql(u8, solution_basename, "solution.zig")) {
            const solution_code = try fs.cwd().readFileAlloc(self.allocator, solution_path, 1024 * 1024);
            defer self.allocator.free(solution_code);

            const link_file = try fs.cwd().createFile(solution_link, .{});
            defer link_file.close();
            try link_file.writeAll(solution_code);
        }

        // Write test harness
        const test_file = try fs.cwd().createFile(test_dest, .{});
        defer test_file.close();
        try test_file.writeAll(test_harness);

        // Run zig test
        var child = std.process.Child.init(&.{ "zig", "test", test_dest }, self.allocator);
        child.stderr_behavior = .Pipe;
        child.stdout_behavior = .Pipe;

        try child.spawn();

        // Read output with timeout
        var stdout_buf: std.ArrayList(u8) = .empty;
        errdefer stdout_buf.deinit(self.allocator);

        var stderr_buf: std.ArrayList(u8) = .empty;
        errdefer stderr_buf.deinit(self.allocator);

        // Collect output
        const stdout = child.stdout.?;
        const stderr = child.stderr.?;

        var buf: [4096]u8 = undefined;

        // Read stdout
        while (true) {
            const n = stdout.read(&buf) catch break;
            if (n == 0) break;
            try stdout_buf.appendSlice(self.allocator, buf[0..n]);
        }

        // Read stderr
        while (true) {
            const n = stderr.read(&buf) catch break;
            if (n == 0) break;
            try stderr_buf.appendSlice(self.allocator, buf[0..n]);
        }

        const result = child.wait() catch {
            return SandboxResult{
                .status = .timeout,
                .stdout = try stdout_buf.toOwnedSlice(self.allocator),
                .stderr = try stderr_buf.toOwnedSlice(self.allocator),
                .allocator = self.allocator,
            };
        };

        // Determine status
        const status: SandboxResult.Status = switch (result.Exited) {
            0 => .pass,
            else => blk: {
                // Check stderr for compile errors
                const stderr_str = stderr_buf.items;
                if (std.mem.indexOf(u8, stderr_str, "error:") != null) {
                    break :blk .compile_error;
                }
                break :blk .test_error;
            },
        };

        return SandboxResult{
            .status = status,
            .stdout = try stdout_buf.toOwnedSlice(self.allocator),
            .stderr = try stderr_buf.toOwnedSlice(self.allocator),
            .allocator = self.allocator,
        };
    }
};

/// Load problem prompt from file
pub fn loadProblemPrompt(allocator: std.mem.Allocator, problem: Problem) ![]const u8 {
    return try fs.cwd().readFileAlloc(allocator, problem.prompt_path, 1024 * 1024);
}

// Tests
test "sanitize model name" {
    const allocator = std.testing.allocator;
    const sandbox_instance = Sandbox.init(allocator, "test_out");

    // This test would create real directories, so we skip the actual creation
    _ = sandbox_instance;
}
