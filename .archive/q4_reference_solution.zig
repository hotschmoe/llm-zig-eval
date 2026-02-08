pub fn RingBuffer(comptime T: type, comptime capacity: usize) type {
    return struct {
        const Self = @This();

        buf: [capacity]T = undefined,
        head: usize = 0,
        count: usize = 0,

        pub fn init() Self {
            return .{};
        }

        pub fn push(self: *Self, item: T) error{Overflow}!void {
            if (self.count == capacity) return error.Overflow;
            self.buf[(self.head + self.count) % capacity] = item;
            self.count += 1;
        }

        pub fn pop(self: *Self) ?T {
            if (self.count == 0) return null;
            const item = self.buf[self.head];
            self.head = (self.head + 1) % capacity;
            self.count -= 1;
            return item;
        }

        pub fn peek(self: *const Self) ?T {
            if (self.count == 0) return null;
            return self.buf[self.head];
        }

        pub fn len(self: *const Self) usize {
            return self.count;
        }

        pub fn isFull(self: *const Self) bool {
            return self.count == capacity;
        }

        pub fn isEmpty(self: *const Self) bool {
            return self.count == 0;
        }

        pub fn reset(self: *Self) void {
            self.head = 0;
            self.count = 0;
        }

        pub const Iterator = struct {
            buf: *const [capacity]T,
            index: usize,
            remaining: usize,

            pub fn next(self: *Iterator) ?T {
                if (self.remaining == 0) return null;
                const item = self.buf[self.index];
                self.index = (self.index + 1) % capacity;
                self.remaining -= 1;
                return item;
            }
        };

        pub fn iterator(self: *const Self) Iterator {
            return .{
                .buf = &self.buf,
                .index = self.head,
                .remaining = self.count,
            };
        }

        pub fn slices(self: *const Self) struct { first: []const T, second: []const T } {
            if (self.count == 0) {
                return .{
                    .first = self.buf[0..0],
                    .second = self.buf[0..0],
                };
            }
            const tail = (self.head + self.count) % capacity;
            if (tail > self.head) {
                return .{
                    .first = self.buf[self.head..tail],
                    .second = self.buf[0..0],
                };
            } else {
                return .{
                    .first = self.buf[self.head..capacity],
                    .second = self.buf[0..tail],
                };
            }
        }
    };
}
