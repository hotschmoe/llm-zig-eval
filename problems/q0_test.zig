//! Test harness for Problem 0: isPrime function
const std = @import("std");
const testing = std.testing;
const solution = @import("solution.zig");

test "2 is prime" {
    try testing.expect(solution.isPrime(2));
}

test "3 is prime" {
    try testing.expect(solution.isPrime(3));
}

test "4 is not prime" {
    try testing.expect(!solution.isPrime(4));
}

test "17 is prime" {
    try testing.expect(solution.isPrime(17));
}

test "1 is not prime" {
    try testing.expect(!solution.isPrime(1));
}

test "0 is not prime" {
    try testing.expect(!solution.isPrime(0));
}
