##
## EnginesTab — Tab 3: cognitive engine dispatch grid.
##
## 29 engine tiles in a GridContainer.
## Green border = ran this turn | gray = skipped | dim = not in dispatch table
## Click a tile → shows result JSON in an inspector panel below the grid.
##
class_name EnginesTab
extends VBoxContainer

# Engine number → short name (from constants.py ENGINE_IDS)
const ENGINE_NAMES := {
	1:  "Contradiction",  2:  "LogicTrap",     3:  "Paradox",
	4:  "Fallacy",        5:  "Bias",           6:  "HeuristicBias",
	7:  "MemCompress",    8:  "SemanticFacets", 9:  "AtomSpace",
	10: "PLN",            11: "InputRelevance", 12: "LogicBrain",
	13: "Creative",       14: "Decision",       15: "DataAnalysis",
	16: "Recursive",      17: "ContextLearn",   18: "EntityExtract",
	19: "PatternID",      20: "PatternComp",    21: "Reflective",
	22: "ReflectID",      23: "IntentionMap",   24: "Relevance",
	25: "MetaLearn",      26: "RetroAlign",     27: "NeuroChem",
	28: "EmotionDetect",  29: "Homeostatic",
}

var _tiles       : Dictionary = {}   # engine_num → Button
var _inspector   : RichTextLabel
var _last_results: Dictionary = {}   # engine_num → result dict

# ---------------------------------------------------------------------------

func _ready() -> void:
	add_theme_constant_override("separation", 6)
	_build_ui()


func _build_ui() -> void:
	# Turn selector label
	var turn_lbl := Label.new()
	turn_lbl.name = "TurnLabel"
	turn_lbl.text = "No data yet"
	turn_lbl.add_theme_font_size_override("font_size", 10)
	turn_lbl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	add_child(turn_lbl)

	# Engine grid
	var scroll := ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	scroll.custom_minimum_size    = Vector2(0, 180)
	add_child(scroll)

	var grid := GridContainer.new()
	grid.columns = 3
	grid.size_flags_horizontal = SIZE_EXPAND_FILL
	grid.add_theme_constant_override("h_separation", 3)
	grid.add_theme_constant_override("v_separation", 3)
	scroll.add_child(grid)

	for num in ENGINE_NAMES:
		var btn := Button.new()
		btn.text              = "E%d\n%s" % [num, ENGINE_NAMES[num]]
		btn.custom_minimum_size = Vector2(0, 36)
		btn.size_flags_horizontal = SIZE_EXPAND_FILL
		btn.focus_mode        = Control.FOCUS_NONE
		btn.add_theme_font_size_override("font_size", 9)
		btn.pressed.connect(_on_tile_pressed.bind(num))
		grid.add_child(btn)
		_tiles[num] = btn

	# Inspector
	var sep := HSeparator.new()
	add_child(sep)

	var inspector_scroll := ScrollContainer.new()
	inspector_scroll.size_flags_vertical = SIZE_EXPAND_FILL
	inspector_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	add_child(inspector_scroll)

	_inspector = RichTextLabel.new()
	_inspector.bbcode_enabled       = true
	_inspector.fit_content          = true
	_inspector.scroll_active        = false
	_inspector.size_flags_horizontal = SIZE_EXPAND_FILL
	_inspector.add_theme_font_size_override("font_size", 10)
	_inspector.text = "[color=#555555]Click an engine tile to inspect its result.[/color]"
	inspector_scroll.add_child(_inspector)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

func refresh(turn_index: int, dispatch: Dictionary) -> void:
	var ran     : Array = dispatch.get("engines_run",     [])
	var skipped : Array = dispatch.get("engines_skipped", [])
	_last_results        = dispatch.get("engine_results", {})

	var tl := get_node_or_null("TurnLabel")
	if tl:
		tl.text = "Turn %d" % turn_index

	for num in _tiles:
		var btn : Button = _tiles[num]
		_style_tile(btn, num in ran, num in skipped)


func _style_tile(btn: Button, ran: bool, skipped: bool) -> void:
	var style_normal := StyleBoxFlat.new()
	style_normal.set_corner_radius_all(4)

	if ran:
		style_normal.bg_color           = Color(0.10, 0.20, 0.13)
		style_normal.border_color       = Color(0.20, 0.65, 0.30)
		style_normal.border_width_left  = 1
		style_normal.border_width_right = 1
		style_normal.border_width_top   = 1
		style_normal.border_width_bottom = 1
		btn.add_theme_color_override("font_color", Color(0.75, 0.95, 0.80))
	elif skipped:
		style_normal.bg_color = Color(0.10, 0.10, 0.12)
		btn.add_theme_color_override("font_color", Color(0.35, 0.35, 0.38))
	else:
		style_normal.bg_color = Color(0.08, 0.08, 0.10)
		btn.add_theme_color_override("font_color", Color(0.28, 0.28, 0.30))

	btn.add_theme_stylebox_override("normal",   style_normal)
	btn.add_theme_stylebox_override("hover",    style_normal)
	btn.add_theme_stylebox_override("pressed",  style_normal)


func _on_tile_pressed(num: int) -> void:
	if not _last_results.has(str(num)) and not _last_results.has(num):
		_inspector.text = "[color=#555555]Engine E%d — no result data this turn.[/color]" % num
		return

	var result = _last_results.get(str(num), _last_results.get(num, {}))
	_inspector.clear()
	_inspector.append_text("[color=#8888aa][b]E%d — %s[/b][/color]\n\n" % [num, ENGINE_NAMES.get(num, "?")])

	# Structured display for specific engines
	if num == 8 and result is Dictionary:
		_render_e8_facets(result)
	elif num == 23 and result is Dictionary:
		_render_e23_intent(result)
	elif num == 28 and result is Dictionary:
		_render_e28_emotions(result)
	else:
		_inspector.append_text("[color=#667788]%s[/color]" % JSON.stringify(result, "  "))


func _render_e8_facets(result: Dictionary) -> void:
	var facets : Array = result.get("facets", result.get("semantic_facets", []))
	if facets.is_empty():
		_inspector.append_text("[color=#667788]%s[/color]" % JSON.stringify(result, "  "))
		return
	_inspector.append_text("[color=#6688aa]Semantic Facets:[/color]\n")
	for f in facets:
		if f is Dictionary:
			var label : String = str(f.get("label", f.get("facet", "?")))
			var weight : float = float(f.get("weight", f.get("relevance", 0.0)))
			_inspector.append_text("  [color=#88aacc]%s[/color]  [color=#557788]%.2f[/color]\n" % [label, weight])
		else:
			_inspector.append_text("  [color=#88aacc]%s[/color]\n" % str(f))


func _render_e23_intent(result: Dictionary) -> void:
	var intent : String = str(result.get("primary_intent", result.get("intent", "—")))
	var conf : float = float(result.get("confidence", 0.0))
	_inspector.append_text("[color=#ccaa55]Intent:[/color] [color=#eeddaa]%s[/color]  " % intent)
	_inspector.append_text("[color=#888866]conf: %.2f[/color]\n\n" % conf)
	var secondary : Array = result.get("secondary_intents", [])
	if not secondary.is_empty():
		_inspector.append_text("[color=#888866]Secondary:[/color]\n")
		for s in secondary:
			_inspector.append_text("  [color=#aaaaaa]%s[/color]\n" % str(s))
	# Fallback to full JSON for remaining keys
	var shown_keys := ["primary_intent", "intent", "confidence", "secondary_intents"]
	var remaining := {}
	for k in result:
		if k not in shown_keys:
			remaining[k] = result[k]
	if not remaining.is_empty():
		_inspector.append_text("\n[color=#667788]%s[/color]" % JSON.stringify(remaining, "  "))


func _render_e28_emotions(result: Dictionary) -> void:
	var emotions : Dictionary = result.get("emotions", result.get("emotion_scores", {}))
	if emotions.is_empty():
		_inspector.append_text("[color=#667788]%s[/color]" % JSON.stringify(result, "  "))
		return
	_inspector.append_text("[color=#aa6688]Emotion Scores (28-dim):[/color]\n")
	# Sort by value descending
	var sorted_keys : Array = emotions.keys()
	sorted_keys.sort_custom(func(a, b): return float(emotions[b]) < float(emotions[a]))
	for k in sorted_keys:
		var val : float = float(emotions[k])
		var bar_len : int = int(val * 20.0)
		var bar_str : String = "|".repeat(maxi(bar_len, 0))
		var color := "#aa6688" if val > 0.3 else "#556677"
		_inspector.append_text("  [color=%s]%-18s %s %.3f[/color]\n" % [color, str(k), bar_str, val])
