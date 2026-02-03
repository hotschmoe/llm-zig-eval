//! Test harness for Problem 3: JSON-to-Struct
//! This file imports the LLM-generated solution and validates it.

const std = @import("std");
const testing = std.testing;

// Import the solution
const solution = @import("solution.zig");
const jsonToStruct = solution.jsonToStruct;

const Point = struct {
    x: u32,
    y: u32,
};

const Person = struct {
    name: []const u8,
    age: u32,
};

const MixedTypes = struct {
    count: u64,
    label: []const u8,
    small: u8,
};

test "parse simple Point struct" {
    const json = "{\"x\": 10, \"y\": 20}";
    const result = try jsonToStruct(Point, json);

    try testing.expectEqual(@as(u32, 10), result.x);
    try testing.expectEqual(@as(u32, 20), result.y);
}

test "parse Point with different order" {
    const json = "{\"y\": 99, \"x\": 42}";
    const result = try jsonToStruct(Point, json);

    try testing.expectEqual(@as(u32, 42), result.x);
    try testing.expectEqual(@as(u32, 99), result.y);
}

test "parse Person with string" {
    const json = "{\"name\": \"Alice\", \"age\": 30}";
    const result = try jsonToStruct(Person, json);

    try testing.expectEqualStrings("Alice", result.name);
    try testing.expectEqual(@as(u32, 30), result.age);
}

test "parse with extra whitespace" {
    const json =
        \\{
        \\  "x"  :  100  ,
        \\  "y"  :  200
        \\}
    ;
    const result = try jsonToStruct(Point, json);

    try testing.expectEqual(@as(u32, 100), result.x);
    try testing.expectEqual(@as(u32, 200), result.y);
}

test "parse mixed types" {
    const json = "{\"count\": 9999999, \"label\": \"test\", \"small\": 255}";
    const result = try jsonToStruct(MixedTypes, json);

    try testing.expectEqual(@as(u64, 9999999), result.count);
    try testing.expectEqualStrings("test", result.label);
    try testing.expectEqual(@as(u8, 255), result.small);
}

test "parse with no whitespace" {
    const json = "{\"x\":1,\"y\":2}";
    const result = try jsonToStruct(Point, json);

    try testing.expectEqual(@as(u32, 1), result.x);
    try testing.expectEqual(@as(u32, 2), result.y);
}

test "empty string value" {
    const StrOnly = struct {
        text: []const u8,
    };

    const json = "{\"text\": \"\"}";
    const result = try jsonToStruct(StrOnly, json);

    try testing.expectEqualStrings("", result.text);
}

test "zero values" {
    const json = "{\"x\": 0, \"y\": 0}";
    const result = try jsonToStruct(Point, json);

    try testing.expectEqual(@as(u32, 0), result.x);
    try testing.expectEqual(@as(u32, 0), result.y);
}

test "invalid JSON returns error" {
    const json = "not valid json";
    const result = jsonToStruct(Point, json);
    try testing.expectError(error.InvalidJson, result);
}

test "missing field returns error" {
    const json = "{\"x\": 10}"; // missing y
    const result = jsonToStruct(Point, json);
    try testing.expectError(error.MissingField, result);
}
