##
## RewardSystemPanel — Phase5 reward domain scores + meta directive.
##
extends VBoxContainer

var _mode_lbl    : Label
var _urgency_bar : ProgressBar
var _domains_box : VBoxContainer
var _directive_box : HBoxContainer
var _nt_grid     : GridContainer
var _weight_sliders : Dictionary = {}  # domain_key → HSlider

func _ready() -> void:
	add_theme_constant_override("separation", 6)
	add_child(_make_header())
	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical    = SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	add_child(scroll)
	var inner := VBoxContainer.new()
	inner.size_flags_horizontal = SIZE_EXPAND_FILL
	inner.add_theme_constant_override("separation", 10)
	scroll.add_child(inner)
	_build_body(inner)
	ZADOSClient.dev_data_received.connect(_on_dev_data)
	ZADOSClient.turn_complete.connect(func(_r): refresh())


func refresh() -> void:
	ZADOSClient.get_dev("reward")


func _on_dev_data(key: String, data: Dictionary) -> void:
	if key == "reward/override_weights/result" or key == "reward/reset_weights/result":
		refresh()
		return
	if key != "reward":
		return
	if data.get("status", "") == "no_result_yet":
		return
	_populate(data)


func _build_body(parent: VBoxContainer) -> void:
	# Mode + urgency
	var top := HBoxContainer.new()
	top.add_theme_constant_override("separation", 12)
	parent.add_child(top)
	var mode_section := VBoxContainer.new()
	mode_section.size_flags_horizontal = SIZE_EXPAND_FILL
	top.add_child(mode_section)
	var mode_title := Label.new()
	mode_title.text = "Selected Mode"
	mode_title.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	mode_title.add_theme_font_size_override("font_size", 10)
	mode_section.add_child(mode_title)
	_mode_lbl = Label.new()
	_mode_lbl.text = "—"
	_mode_lbl.add_theme_color_override("font_color", Color(0.85, 0.88, 0.92))
	mode_section.add_child(_mode_lbl)

	var urgency_section := VBoxContainer.new()
	urgency_section.size_flags_horizontal = SIZE_EXPAND_FILL
	top.add_child(urgency_section)
	var urgency_title := Label.new()
	urgency_title.text = "Urgency Risk"
	urgency_title.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	urgency_title.add_theme_font_size_override("font_size", 10)
	urgency_section.add_child(urgency_title)
	_urgency_bar = ProgressBar.new()
	_urgency_bar.min_value = 0.0
	_urgency_bar.max_value = 1.0
	_urgency_bar.value     = 0.0
	_urgency_bar.custom_minimum_size = Vector2(0, 14)
	urgency_section.add_child(_urgency_bar)

	# Directive flags
	var dir_title := Label.new()
	dir_title.text = "Meta Directive"
	dir_title.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	dir_title.add_theme_font_size_override("font_size", 11)
	parent.add_child(dir_title)
	_directive_box = HBoxContainer.new()
	_directive_box.add_theme_constant_override("separation", 8)
	parent.add_child(_directive_box)

	# Domain scores
	var domain_title := Label.new()
	domain_title.text = "Domain Scores"
	domain_title.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	domain_title.add_theme_font_size_override("font_size", 11)
	parent.add_child(domain_title)
	_domains_box = VBoxContainer.new()
	_domains_box.add_theme_constant_override("separation", 6)
	parent.add_child(_domains_box)

	# Weight override section
	var weight_title := Label.new()
	weight_title.text = "Domain Weight Override"
	weight_title.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	weight_title.add_theme_font_size_override("font_size", 11)
	parent.add_child(weight_title)
	var weight_box := VBoxContainer.new()
	weight_box.add_theme_constant_override("separation", 4)
	parent.add_child(weight_box)
	for domain_key in ["logic_weight", "ethics_weight", "innovation_weight", "attunement_weight"]:
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 8)
		weight_box.add_child(row)
		var lbl := Label.new()
		lbl.text = domain_key.replace("_weight", "").capitalize()
		lbl.custom_minimum_size = Vector2(90, 0)
		lbl.add_theme_color_override("font_color", Color(0.75, 0.78, 0.82))
		lbl.add_theme_font_size_override("font_size", 11)
		row.add_child(lbl)
		var slider := HSlider.new()
		slider.min_value = 0.0
		slider.max_value = 1.0
		slider.step = 0.05
		slider.value = 0.25
		slider.size_flags_horizontal = SIZE_EXPAND_FILL
		slider.custom_minimum_size = Vector2(0, 20)
		row.add_child(slider)
		var val_lbl := Label.new()
		val_lbl.text = "0.25"
		val_lbl.custom_minimum_size = Vector2(34, 0)
		val_lbl.add_theme_color_override("font_color", Color(0.55, 0.58, 0.62))
		val_lbl.add_theme_font_size_override("font_size", 10)
		row.add_child(val_lbl)
		slider.value_changed.connect(func(v): val_lbl.text = "%.2f" % v)
		_weight_sliders[domain_key] = slider
	var btn_row := HBoxContainer.new()
	btn_row.add_theme_constant_override("separation", 8)
	weight_box.add_child(btn_row)
	var override_btn := Button.new()
	override_btn.text = "Set as Override"
	override_btn.focus_mode = Control.FOCUS_NONE
	override_btn.add_theme_font_size_override("font_size", 11)
	override_btn.pressed.connect(_apply_weight_override)
	btn_row.add_child(override_btn)
	var reset_btn := Button.new()
	reset_btn.text = "Reset to Static"
	reset_btn.flat = true
	reset_btn.focus_mode = Control.FOCUS_NONE
	reset_btn.add_theme_font_size_override("font_size", 11)
	reset_btn.pressed.connect(_reset_weights)
	btn_row.add_child(reset_btn)

	# NT signals
	var nt_title := Label.new()
	nt_title.text = "NT Signals"
	nt_title.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	nt_title.add_theme_font_size_override("font_size", 11)
	parent.add_child(nt_title)
	_nt_grid = GridContainer.new()
	_nt_grid.columns = 2
	_nt_grid.add_theme_constant_override("h_separation", 12)
	_nt_grid.add_theme_constant_override("v_separation", 2)
	parent.add_child(_nt_grid)


func _populate(d: Dictionary) -> void:
	_mode_lbl.text = str(d.get("selected_mode", "—"))
	_urgency_bar.value = float(d.get("urgency_risk", 0.0))

	# Directive flags
	for child in _directive_box.get_children():
		child.queue_free()
	var meta : Dictionary = d.get("meta_directive", {})
	var allow   : bool = meta.get("allow_output", true)
	var abstain : bool = meta.get("abstain", false)
	var suppress: bool = meta.get("suppress", false)
	_directive_box.add_child(_flag_badge(
		"Allow" if allow else "Suppressed",
		Color(0.2, 0.7, 0.4) if allow else Color(0.8, 0.2, 0.2)))
	if abstain:
		_directive_box.add_child(_flag_badge("Abstain", Color(0.8, 0.6, 0.1)))
	if suppress:
		_directive_box.add_child(_flag_badge("Suppress", Color(0.8, 0.2, 0.2)))

	# Domain score bars
	for child in _domains_box.get_children():
		child.queue_free()
	var domains : Dictionary = d.get("domains", {})
	var domain_colors := {
		"innovation":       Color(0.4, 0.7, 1.0),
		"logic":            Color(0.5, 0.9, 0.5),
		"human_attunement": Color(1.0, 0.7, 0.4),
		"ethics":           Color(0.8, 0.5, 0.9),
	}
	for domain in domains:
		var info : Dictionary = domains[domain]
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 8)
		_domains_box.add_child(row)
		var lbl := Label.new()
		lbl.text = str(domain).capitalize().replace("_", " ")
		lbl.custom_minimum_size = Vector2(130, 0)
		lbl.add_theme_color_override("font_color",
			domain_colors.get(str(domain).to_lower(), Color(0.75, 0.78, 0.82)))
		lbl.add_theme_font_size_override("font_size", 11)
		row.add_child(lbl)
		var bar := ProgressBar.new()
		bar.min_value = 0.0
		bar.max_value = 1.0
		bar.value     = float(info.get("general_score", 0.0))
		bar.size_flags_horizontal = SIZE_EXPAND_FILL
		bar.custom_minimum_size = Vector2(0, 14)
		row.add_child(bar)
		var score_lbl := Label.new()
		score_lbl.text = "%.2f" % float(info.get("general_score", 0.0))
		score_lbl.add_theme_color_override("font_color", Color(0.55, 0.58, 0.62))
		score_lbl.add_theme_font_size_override("font_size", 10)
		score_lbl.custom_minimum_size = Vector2(34, 0)
		row.add_child(score_lbl)

	# NT signals
	for child in _nt_grid.get_children():
		child.queue_free()
	var nt : Dictionary = d.get("nt_signals", {})
	for k in nt:
		var kl := Label.new()
		kl.text = str(k) + ":"
		kl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
		kl.add_theme_font_size_override("font_size", 10)
		kl.custom_minimum_size = Vector2(80, 0)
		_nt_grid.add_child(kl)
		var vl := Label.new()
		vl.text = str(nt[k])
		vl.add_theme_color_override("font_color", Color(0.75, 0.78, 0.82))
		vl.add_theme_font_size_override("font_size", 10)
		vl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		_nt_grid.add_child(vl)


func _flag_badge(text: String, color: Color) -> PanelContainer:
	var panel := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = color.darkened(0.55)
	style.border_color = color
	style.border_width_bottom = 1; style.border_width_top = 1
	style.border_width_left   = 1; style.border_width_right = 1
	style.corner_radius_top_left     = 4; style.corner_radius_top_right    = 4
	style.corner_radius_bottom_left  = 4; style.corner_radius_bottom_right = 4
	style.content_margin_left = 8; style.content_margin_right  = 8
	style.content_margin_top  = 3; style.content_margin_bottom = 3
	panel.add_theme_stylebox_override("panel", style)
	var lbl := Label.new()
	lbl.text = text
	lbl.add_theme_color_override("font_color", color)
	lbl.add_theme_font_size_override("font_size", 10)
	panel.add_child(lbl)
	return panel


func _apply_weight_override() -> void:
	var body := {}
	for k in _weight_sliders:
		body[k] = _weight_sliders[k].value
	ZADOSClient.post_dev("reward/override_weights", body)


func _reset_weights() -> void:
	ZADOSClient.post_dev("reward/reset_weights", {})
	for k in _weight_sliders:
		_weight_sliders[k].value = 0.25


func _make_header() -> HBoxContainer:
	var hbox := HBoxContainer.new()
	var lbl  := Label.new()
	lbl.text = "Reward System"
	lbl.size_flags_horizontal = SIZE_EXPAND_FILL
	lbl.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	hbox.add_child(lbl)
	var btn := Button.new()
	btn.text = "↺ Refresh"
	btn.flat = true
	btn.focus_mode = Control.FOCUS_NONE
	btn.add_theme_font_size_override("font_size", 11)
	btn.pressed.connect(refresh)
	hbox.add_child(btn)
	return hbox
