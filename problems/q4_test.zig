const std = @import("std");
const testing = std.testing;
const solution = @import("solution.zig");

const RingBuffer = solution.RingBuffer;

test "basic push and pop" {
    var rb = RingBuffer(u32, 4).init();
    try rb.push(10);
    try rb.push(20);
    try testing.expectEqual(@as(u32, 10), rb.pop().?);
    try testing.expectEqual(@as(u32, 20), rb.pop().?);
    try testing.expect(rb.pop() == null);
}

test "FIFO ordering preserved" {
    var rb = RingBuffer(u32, 8).init();
    for (0..5) |i| {
        try rb.push(@as(u32, @intCast(i + 1)));
    }
    for (0..5) |i| {
        try testing.expectEqual(@as(u32, @intCast(i + 1)), rb.pop().?);
    }
}

test "overflow on full buffer" {
    var rb = RingBuffer(u8, 3).init();
    try rb.push(1);
    try rb.push(2);
    try rb.push(3);
    try testing.expect(rb.isFull());
    try testing.expectError(error.Overflow, rb.push(4));
}

test "pop and peek on empty buffer return null" {
    var rb = RingBuffer(i32, 4).init();
    try testing.expect(rb.pop() == null);
    try testing.expect(rb.peek() == null);
}

test "peek does not consume" {
    var rb = RingBuffer(u32, 4).init();
    try rb.push(42);
    try testing.expectEqual(@as(u32, 42), rb.peek().?);
    try testing.expectEqual(@as(u32, 42), rb.peek().?);
    try testing.expectEqual(@as(usize, 1), rb.len());
    try testing.expectEqual(@as(u32, 42), rb.pop().?);
    try testing.expect(rb.isEmpty());
}

test "wrap-around maintains FIFO order" {
    var rb = RingBuffer(u32, 4).init();
    try rb.push(1);
    try rb.push(2);
    try rb.push(3);
    try rb.push(4);
    // drain two from front
    try testing.expectEqual(@as(u32, 1), rb.pop().?);
    try testing.expectEqual(@as(u32, 2), rb.pop().?);
    // push two more (these wrap around the backing array)
    try rb.push(5);
    try rb.push(6);
    try testing.expect(rb.isFull());
    // verify order: 3, 4, 5, 6
    try testing.expectEqual(@as(u32, 3), rb.pop().?);
    try testing.expectEqual(@as(u32, 4), rb.pop().?);
    try testing.expectEqual(@as(u32, 5), rb.pop().?);
    try testing.expectEqual(@as(u32, 6), rb.pop().?);
    try testing.expect(rb.isEmpty());
}

test "len isEmpty isFull state transitions" {
    var rb = RingBuffer(u32, 3).init();
    try testing.expect(rb.isEmpty());
    try testing.expect(!rb.isFull());
    try testing.expectEqual(@as(usize, 0), rb.len());

    try rb.push(1);
    try testing.expect(!rb.isEmpty());
    try testing.expect(!rb.isFull());
    try testing.expectEqual(@as(usize, 1), rb.len());

    try rb.push(2);
    try rb.push(3);
    try testing.expect(rb.isFull());
    try testing.expectEqual(@as(usize, 3), rb.len());

    _ = rb.pop();
    try testing.expect(!rb.isFull());
    try testing.expectEqual(@as(usize, 2), rb.len());
}

test "reset clears buffer" {
    var rb = RingBuffer(u32, 4).init();
    try rb.push(1);
    try rb.push(2);
    try rb.push(3);
    rb.reset();
    try testing.expect(rb.isEmpty());
    try testing.expectEqual(@as(usize, 0), rb.len());
    try testing.expect(rb.pop() == null);
    // can push again after reset
    try rb.push(99);
    try testing.expectEqual(@as(u32, 99), rb.pop().?);
}

test "iterator traverses front to back" {
    var rb = RingBuffer(u32, 8).init();
    try rb.push(10);
    try rb.push(20);
    try rb.push(30);

    var it = rb.iterator();
    try testing.expectEqual(@as(u32, 10), it.next().?);
    try testing.expectEqual(@as(u32, 20), it.next().?);
    try testing.expectEqual(@as(u32, 30), it.next().?);
    try testing.expect(it.next() == null);

    // iterator did not consume items
    try testing.expectEqual(@as(usize, 3), rb.len());
    try testing.expectEqual(@as(u32, 10), rb.pop().?);
}

test "iterator handles wrap-around" {
    var rb = RingBuffer(u32, 4).init();
    try rb.push(1);
    try rb.push(2);
    try rb.push(3);
    try rb.push(4);
    _ = rb.pop();
    _ = rb.pop();
    try rb.push(5);
    try rb.push(6);
    // elements in order: 3, 4, 5, 6 (wrapping around backing array)
    var it = rb.iterator();
    try testing.expectEqual(@as(u32, 3), it.next().?);
    try testing.expectEqual(@as(u32, 4), it.next().?);
    try testing.expectEqual(@as(u32, 5), it.next().?);
    try testing.expectEqual(@as(u32, 6), it.next().?);
    try testing.expect(it.next() == null);
}

test "slices contiguous case" {
    var rb = RingBuffer(u32, 4).init();
    try rb.push(1);
    try rb.push(2);
    try rb.push(3);

    const s = rb.slices();
    try testing.expectEqual(@as(usize, 3), s.first.len);
    try testing.expectEqual(@as(usize, 0), s.second.len);
    try testing.expectEqual(@as(u32, 1), s.first[0]);
    try testing.expectEqual(@as(u32, 2), s.first[1]);
    try testing.expectEqual(@as(u32, 3), s.first[2]);
}

test "slices with wrap-around" {
    var rb = RingBuffer(u32, 4).init();
    try rb.push(1);
    try rb.push(2);
    try rb.push(3);
    try rb.push(4);
    _ = rb.pop();
    _ = rb.pop();
    try rb.push(5);
    try rb.push(6);
    // backing array: [5, 6, 3, 4], head=2, count=4
    // logical order: 3, 4, 5, 6
    const s = rb.slices();
    // first slice: from head to end of array = [3, 4]
    try testing.expectEqual(@as(usize, 2), s.first.len);
    try testing.expectEqual(@as(u32, 3), s.first[0]);
    try testing.expectEqual(@as(u32, 4), s.first[1]);
    // second slice: from start of array to tail = [5, 6]
    try testing.expectEqual(@as(usize, 2), s.second.len);
    try testing.expectEqual(@as(u32, 5), s.second[0]);
    try testing.expectEqual(@as(u32, 6), s.second[1]);
}

test "works with struct element type" {
    const Vec2 = struct { x: f32, y: f32 };
    var rb = RingBuffer(Vec2, 3).init();
    try rb.push(.{ .x = 1.0, .y = 2.0 });
    try rb.push(.{ .x = 3.0, .y = 4.0 });

    const v = rb.pop().?;
    try testing.expectEqual(@as(f32, 1.0), v.x);
    try testing.expectEqual(@as(f32, 2.0), v.y);

    try testing.expectEqual(@as(usize, 1), rb.len());
    const v2 = rb.peek().?;
    try testing.expectEqual(@as(f32, 3.0), v2.x);
    try testing.expectEqual(@as(f32, 4.0), v2.y);
}

test "capacity of 1 edge case" {
    var rb = RingBuffer(u8, 1).init();
    try testing.expect(rb.isEmpty());

    try rb.push(42);
    try testing.expect(rb.isFull());
    try testing.expectError(error.Overflow, rb.push(99));
    try testing.expectEqual(@as(u8, 42), rb.pop().?);
    try testing.expect(rb.isEmpty());

    // re-use after drain
    try rb.push(7);
    try testing.expectEqual(@as(u8, 7), rb.peek().?);

    var it = rb.iterator();
    try testing.expectEqual(@as(u8, 7), it.next().?);
    try testing.expect(it.next() == null);
}
