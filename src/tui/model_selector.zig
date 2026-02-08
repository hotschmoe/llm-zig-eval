//! Interactive TUI model selector using zithril
//! Fetches available models from OpenRouter API and allows interactive selection
//!
//! This module is only available to the executable (not the library) since
//! it depends on zithril for TUI functionality.

const std = @import("std");
const zithril = @import("zithril");

pub const SelectorError = error{
    FetchModelsFailed,
    TuiError,
    ApiRequestFailed,
    JsonParseError,
    InvalidApiResponse,
};

const App = zithril.App;
const Event = zithril.Event;
const Action = zithril.Action;
const Frame = zithril.Frame;
const Rect = zithril.Rect;
const Style = zithril.Style;
const Block = zithril.Block;
const BorderType = zithril.BorderType;
const ScrollState = zithril.ScrollState;
const ScrollableList = zithril.ScrollableList;
const TextInput = zithril.TextInput;
const TextInputState = zithril.TextInputState;
const Constraint = zithril.Constraint;
const Direction = zithril.Direction;
const Key = zithril.Key;
const KeyCode = zithril.KeyCode;

/// Active selection mode in the TUI
pub const SelectionMode = enum {
    benchmark,
    council,
};

/// Result returned from the model selector
pub const SelectorResult = struct {
    benchmark_models: []const []const u8,
    council_models: ?[]const []const u8,
};

/// Model information fetched from OpenRouter API
pub const ModelInfo = struct {
    id: []const u8,
    name: []const u8,
    context_length: u32,
    prompt_price: f64, // $ per 1M tokens
    completion_price: f64, // $ per 1M tokens
    selected: bool = false,
    council_selected: bool = false,

    /// Format price for display (e.g. "$3" or "$0.15")
    pub fn formatPrice(price: f64, buf: []u8) []const u8 {
        if (price >= 1.0) {
            return std.fmt.bufPrint(buf, "${d:.0}", .{price}) catch "$?";
        } else if (price >= 0.01) {
            return std.fmt.bufPrint(buf, "${d:.2}", .{price}) catch "$?";
        } else {
            return std.fmt.bufPrint(buf, "${d:.3}", .{price}) catch "$?";
        }
    }

    /// Format context length for display (e.g. "128k" or "1M")
    pub fn formatContext(ctx: u32, buf: []u8) []const u8 {
        if (ctx >= 1_000_000) {
            return std.fmt.bufPrint(buf, "{d}M", .{ctx / 1_000_000}) catch "?";
        } else {
            return std.fmt.bufPrint(buf, "{d}k", .{ctx / 1000}) catch "?";
        }
    }
};

/// State for the model selector TUI
pub const SelectorState = struct {
    allocator: std.mem.Allocator,
    models: []ModelInfo,
    display_items: std.ArrayList([]const u8),
    filtered_indices: std.ArrayList(usize),
    filter_buffer: [256]u8 = undefined,
    filter_state: TextInputState,
    cursor: usize = 0,
    scroll: ScrollState,
    submitted: bool = false,
    quit_without_select: bool = false,
    mode: SelectionMode = .benchmark,
    council_enabled: bool = false,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, models: []ModelInfo) !Self {
        var self = Self{
            .allocator = allocator,
            .models = models,
            .display_items = .empty,
            .filtered_indices = .empty,
            .filter_state = undefined,
            .scroll = ScrollState.init(models.len),
        };

        // Build initial display items and indices (all models)
        for (models, 0..) |model, i| {
            const display = try self.formatDisplayItem(model);
            try self.display_items.append(allocator, display);
            try self.filtered_indices.append(allocator, i);
        }

        return self;
    }

    /// Must be called after init, once the struct is in its final memory location.
    /// TextInputState holds a slice into filter_buffer, so this cannot be done
    /// inside init() where the struct is on a temporary stack frame.
    pub fn initFilterState(self: *Self) void {
        self.filter_state = TextInputState.init(&self.filter_buffer);
    }

    pub fn deinit(self: *Self) void {
        for (self.display_items.items) |item| {
            self.allocator.free(item);
        }
        self.display_items.deinit(self.allocator);
        self.filtered_indices.deinit(self.allocator);
    }

    fn formatDisplayItem(self: *Self, model: ModelInfo) ![]const u8 {
        var price_buf: [16]u8 = undefined;
        var ctx_buf: [16]u8 = undefined;
        const price_str = ModelInfo.formatPrice(model.prompt_price, &price_buf);
        const ctx_str = ModelInfo.formatContext(model.context_length, &ctx_buf);

        const checkbox: []const u8 = if (isSelectedForMode(model, self.mode)) "[x]" else "[ ]";

        return try std.fmt.allocPrint(self.allocator, "{s} {s} {s} {s}", .{
            checkbox,
            model.id,
            price_str,
            ctx_str,
        });
    }

    pub fn rebuildDisplayItems(self: *Self) !void {
        // Clear existing
        for (self.display_items.items) |item| {
            self.allocator.free(item);
        }
        self.display_items.clearRetainingCapacity();
        self.filtered_indices.clearRetainingCapacity();

        const filter_text = self.filter_state.text();

        for (self.models, 0..) |model, i| {
            // Apply filter - case-insensitive search in model id and name
            if (filter_text.len > 0 and
                !containsIgnoreCase(model.id, filter_text) and
                !containsIgnoreCase(model.name, filter_text))
            {
                continue;
            }

            const display = try self.formatDisplayItem(model);
            try self.display_items.append(self.allocator, display);
            try self.filtered_indices.append(self.allocator, i);
        }

        // Reset cursor if out of bounds
        if (self.cursor >= self.filtered_indices.items.len) {
            self.cursor = if (self.filtered_indices.items.len > 0) self.filtered_indices.items.len - 1 else 0;
        }

        // Update scroll state
        self.scroll.setTotal(self.filtered_indices.items.len);
    }

    pub fn toggleSelection(self: *Self) void {
        if (self.filtered_indices.items.len == 0) return;
        const model_idx = self.filtered_indices.items[self.cursor];
        switch (self.mode) {
            .benchmark => self.models[model_idx].selected = !self.models[model_idx].selected,
            .council => self.models[model_idx].council_selected = !self.models[model_idx].council_selected,
        }

        // Update display item
        if (self.cursor < self.display_items.items.len) {
            self.allocator.free(self.display_items.items[self.cursor]);
            self.display_items.items[self.cursor] = self.formatDisplayItem(self.models[model_idx]) catch return;
        }
    }

    pub fn getSelectedModels(self: Self) ![]const []const u8 {
        return self.collectModelIds(.benchmark);
    }

    pub fn selectedCount(self: Self) usize {
        return self.countByMode(.benchmark);
    }

    pub fn getCouncilModels(self: Self) ![]const []const u8 {
        return self.collectModelIds(.council);
    }

    pub fn councilCount(self: Self) usize {
        return self.countByMode(.council);
    }

    fn isSelectedForMode(model: ModelInfo, mode: SelectionMode) bool {
        return switch (mode) {
            .benchmark => model.selected,
            .council => model.council_selected,
        };
    }

    fn collectModelIds(self: Self, mode: SelectionMode) ![]const []const u8 {
        var result: std.ArrayList([]const u8) = .empty;
        for (self.models) |model| {
            if (isSelectedForMode(model, mode)) {
                try result.append(self.allocator, try self.allocator.dupe(u8, model.id));
            }
        }
        return try result.toOwnedSlice(self.allocator);
    }

    fn countByMode(self: Self, mode: SelectionMode) usize {
        var count: usize = 0;
        for (self.models) |model| {
            if (isSelectedForMode(model, mode)) count += 1;
        }
        return count;
    }
};

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    outer: while (i + needle.len <= haystack.len) : (i += 1) {
        for (needle, 0..) |nc, j| {
            const hc = haystack[i + j];
            if (std.ascii.toLower(hc) != std.ascii.toLower(nc)) {
                continue :outer;
            }
        }
        return true;
    }
    return false;
}

/// Update function for the TUI app
fn update(state: *SelectorState, event: Event) Action {
    const key = switch (event) {
        .key => |k| k,
        else => return Action.none_action,
    };

    if (key.action == .release) return Action.none_action;

    switch (key.code) {
        .escape => {
            state.quit_without_select = true;
            return Action.quit_action;
        },
        .enter => {
            if (state.selectedCount() > 0) {
                state.submitted = true;
                return Action.quit_action;
            }
        },
        .tab => {
            if (state.council_enabled) {
                state.mode = switch (state.mode) {
                    .benchmark => .council,
                    .council => .benchmark,
                };
                state.rebuildDisplayItems() catch {};
            }
        },
        .char => |c| {
            if (key.modifiers.ctrl) {
                if (c == 'c') {
                    state.quit_without_select = true;
                    return Action.quit_action;
                }
                // Ctrl+H is backspace in some terminals
                if (c == 'h' or c == 8) {
                    handleFilterKey(state, .{ .code = .backspace });
                }
                return Action.none_action;
            }
            // Reject non-ASCII and non-printable characters
            if (c > 0x7f) return Action.none_action;
            const byte: u8 = @intCast(c);
            if (!std.ascii.isPrint(byte)) return Action.none_action;

            if (c == ' ') {
                state.toggleSelection();
            } else {
                handleFilterKey(state, key);
            }
        },
        .up => {
            if (state.cursor > 0) {
                state.cursor -= 1;
                state.scroll.ensureVisible(state.cursor);
            }
        },
        .down => {
            if (state.cursor + 1 < state.filtered_indices.items.len) {
                state.cursor += 1;
                state.scroll.ensureVisible(state.cursor);
            }
        },
        .page_up => {
            const page = @min(state.scroll.viewport, state.cursor);
            state.cursor -|= page;
            state.scroll.ensureVisible(state.cursor);
        },
        .page_down => {
            const page = state.scroll.viewport;
            state.cursor = @min(state.cursor + page, state.filtered_indices.items.len -| 1);
            state.scroll.ensureVisible(state.cursor);
        },
        .home => {
            state.cursor = 0;
            state.scroll.ensureVisible(0);
        },
        .end => {
            if (state.filtered_indices.items.len > 0) {
                state.cursor = state.filtered_indices.items.len - 1;
                state.scroll.ensureVisible(state.cursor);
            }
        },
        .backspace, .delete => handleFilterKey(state, key),
        else => {},
    }
    return Action.none_action;
}

fn handleFilterKey(state: *SelectorState, key: Key) void {
    if (state.filter_state.handleKey(key)) {
        state.rebuildDisplayItems() catch {};
    }
}

/// View function for the TUI app
fn view(state: *SelectorState, frame: *Frame(App(SelectorState).DefaultMaxWidgets)) void {
    const area = frame.size();

    // Main layout: title, filter, list, status
    const chunks = frame.layout(area, Direction.vertical, &.{
        Constraint.len(3), // Filter input with border
        Constraint.flexible(1), // Model list
        Constraint.len(1), // Status bar
    });

    // Filter input box
    const filter_block = Block{
        .title = "Filter",
        .border = BorderType.rounded,
        .border_style = Style.init().fg(.cyan),
    };
    frame.render(filter_block, chunks.get(0));

    const filter_inner = filter_block.inner(chunks.get(0));
    const filter_widget = TextInput{
        .state = &state.filter_state,
        .placeholder = "Type to filter models...",
        .placeholder_style = Style.init().dim(),
        .focused = true,
    };
    frame.render(filter_widget, filter_inner);

    // Model list with scroll - title and highlight color vary by mode
    const is_council_mode = state.mode == .council;
    const mode_title: []const u8 = if (is_council_mode) "Models [Council Judges]" else "Models [Benchmark]";

    const list_block = Block{
        .title = mode_title,
        .border = BorderType.rounded,
        .border_style = if (is_council_mode) Style.init().fg(.magenta) else Style.init().fg(.white),
    };
    frame.render(list_block, chunks.get(1));

    const list_inner = list_block.inner(chunks.get(1));
    state.scroll.setViewport(list_inner.height);

    const list = ScrollableList{
        .items = state.display_items.items,
        .scroll = &state.scroll,
        .selected = state.cursor,
        .highlight_style = if (is_council_mode) Style.init().bg(.magenta).fg(.white).bold() else Style.init().bg(.blue).fg(.white).bold(),
        .highlight_symbol = "> ",
        .show_scrollbar = true,
    };
    frame.render(list, list_inner);

    // Status bar
    var status_buf: [128]u8 = undefined;
    const status_msg = if (state.council_enabled)
        std.fmt.bufPrint(&status_buf, " Bench: {d} | Council: {d} | tab:mode space:toggle enter:run esc:quit", .{
            state.selectedCount(),
            state.councilCount(),
        }) catch " tab:mode space:toggle enter:run esc:quit"
    else
        std.fmt.bufPrint(&status_buf, " Selected: {d} | space:toggle enter:run esc:quit", .{
            state.selectedCount(),
        }) catch " space:toggle enter:run esc:quit";

    const status_text = zithril.widgets.Text{
        .content = status_msg,
        .style = Style.init().bg(.black).fg(.white),
    };
    frame.render(status_text, chunks.get(2));
}

/// Fetch available models from OpenRouter API
pub fn fetchModels(allocator: std.mem.Allocator, api_key: []const u8) ![]ModelInfo {
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    // Build authorization header value
    const auth_header = try std.fmt.allocPrint(allocator, "Bearer {s}", .{api_key});
    defer allocator.free(auth_header);

    // Prepare response body buffer using Zig 0.15 Writer.Allocating
    var response_writer: std.Io.Writer.Allocating = .init(allocator);
    errdefer response_writer.deinit();

    const result = client.fetch(.{
        .location = .{ .url = "https://openrouter.ai/api/v1/models" },
        .extra_headers = &.{
            .{ .name = "Authorization", .value = auth_header },
        },
        .response_writer = &response_writer.writer,
    }) catch {
        return error.ApiRequestFailed;
    };

    if (result.status != .ok) {
        return error.ApiRequestFailed;
    }

    const body = try response_writer.toOwnedSlice();
    defer allocator.free(body);

    // Parse JSON
    return parseModelsResponse(allocator, body);
}

/// Parse a price value from JSON (can be string, float, or integer), returns $ per million tokens
fn parsePriceValue(value: ?std.json.Value) f64 {
    const v = value orelse return 0;
    return switch (v) {
        .string => |s| (std.fmt.parseFloat(f64, s) catch 0) * 1_000_000,
        .float => |f| f * 1_000_000,
        .integer => |i| @as(f64, @floatFromInt(i)) * 1_000_000,
        else => 0,
    };
}

fn parseModelsResponse(allocator: std.mem.Allocator, json_body: []const u8) ![]ModelInfo {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json_body, .{}) catch {
        return error.JsonParseError;
    };
    defer parsed.deinit();

    const root = parsed.value;
    const data = root.object.get("data") orelse return error.InvalidApiResponse;

    var models: std.ArrayList(ModelInfo) = .empty;
    errdefer {
        for (models.items) |m| {
            allocator.free(m.id);
            allocator.free(m.name);
        }
        models.deinit(allocator);
    }

    for (data.array.items) |item| {
        const obj = item.object;

        const id_val = obj.get("id") orelse continue;
        const id = switch (id_val) {
            .string => |s| s,
            else => continue,
        };

        const name_val = obj.get("name") orelse id_val;
        const name = switch (name_val) {
            .string => |s| s,
            else => id,
        };

        // Get context length
        const ctx_val = obj.get("context_length") orelse continue;
        const context_length: u32 = switch (ctx_val) {
            .integer => |i| @intCast(@max(0, i)),
            else => 0,
        };

        // Get pricing (convert to $ per million tokens)
        const pricing = obj.get("pricing");
        const prompt_price = if (pricing) |p| parsePriceValue(p.object.get("prompt")) else 0;
        const completion_price = if (pricing) |p| parsePriceValue(p.object.get("completion")) else 0;

        try models.append(allocator, .{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .context_length = context_length,
            .prompt_price = prompt_price,
            .completion_price = completion_price,
        });
    }

    return try models.toOwnedSlice(allocator);
}

/// Run the interactive model selector.
/// If council_enabled is true, Tab toggles between benchmark/council selection modes.
/// Returns SelectorResult or null if user quit without selecting.
pub fn runSelector(allocator: std.mem.Allocator, api_key: []const u8, council_enabled: bool) !?SelectorResult {
    std.debug.print("Fetching models from OpenRouter...\n", .{});
    const models = fetchModels(allocator, api_key) catch |err| {
        std.debug.print("Failed to fetch models: {}\n", .{err});
        return error.FetchModelsFailed;
    };
    defer {
        for (models) |m| {
            allocator.free(m.id);
            allocator.free(m.name);
        }
        allocator.free(models);
    }

    if (models.len == 0) {
        std.debug.print("No models available from API\n", .{});
        return null;
    }

    std.debug.print("Found {d} models. Launching selector...\n", .{models.len});

    var state = try SelectorState.init(allocator, models);
    defer state.deinit();
    state.initFilterState();
    state.council_enabled = council_enabled;

    var app = App(SelectorState).init(.{
        .state = &state,
        .update = update,
        .view = view,
        .alternate_screen = true,
    });

    app.run(allocator) catch |err| {
        std.debug.print("TUI error: {}\n", .{err});
        return error.TuiError;
    };

    if (state.quit_without_select) {
        return null;
    }

    if (state.submitted) {
        const bench = try state.getSelectedModels();
        const council = if (council_enabled and state.councilCount() > 0)
            try state.getCouncilModels()
        else
            null;
        return SelectorResult{
            .benchmark_models = bench,
            .council_models = council,
        };
    }

    return null;
}

test "containsIgnoreCase" {
    try std.testing.expect(containsIgnoreCase("Hello World", "world"));
    try std.testing.expect(containsIgnoreCase("anthropic/claude-3.5-sonnet", "claude"));
    try std.testing.expect(containsIgnoreCase("anthropic/claude-3.5-sonnet", "CLAUDE"));
    try std.testing.expect(!containsIgnoreCase("hello", "worldx"));
}

test "ModelInfo.formatPrice" {
    var buf: [16]u8 = undefined;

    try std.testing.expectEqualStrings("$3", ModelInfo.formatPrice(3.0, &buf));
    try std.testing.expectEqualStrings("$15", ModelInfo.formatPrice(15.0, &buf));
    try std.testing.expectEqualStrings("$0.25", ModelInfo.formatPrice(0.25, &buf));
    try std.testing.expectEqualStrings("$0.08", ModelInfo.formatPrice(0.075, &buf));
}

test "ModelInfo.formatContext" {
    var buf: [16]u8 = undefined;

    try std.testing.expectEqualStrings("128k", ModelInfo.formatContext(128000, &buf));
    try std.testing.expectEqualStrings("200k", ModelInfo.formatContext(200000, &buf));
    try std.testing.expectEqualStrings("1M", ModelInfo.formatContext(1000000, &buf));
}
