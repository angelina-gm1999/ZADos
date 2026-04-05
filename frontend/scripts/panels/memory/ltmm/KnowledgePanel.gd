##
## KnowledgePanel — LTMM Knowledge: Lessons | Notebook | Academic Buffer | Library
##
## Addendum B.2.9: notebook "New Entry" inline editor.
## Addendum B.2.10: academic buffer resolve modal with source_question_id link.
## Addendum B.2.4: library upload FileDialog lifecycle with progress toast.
##
extends VBoxContainer

const _ConfirmDialog = preload("res://scripts/components/ConfirmationDialog.gd")
const _Toast = preload("res://scripts/components/Toast.gd")
const _ErrorDisplay = preload("res://scripts/components/ErrorDisplay.gd")

var _tabs          : TabContainer
var _lessons_list  : VBoxContainer
var _notebook_list : VBoxContainer
var _buffer_list   : VBoxContainer
var _maps_list     : VBoxContainer
var _library_list  : VBoxContainer
var _library_tab   : VBoxContainer
var _file_dialog   : FileDialog

# Notebook inline editor state
var _notebook_editor   : VBoxContainer = null
var _notebook_input    : TextEdit = null
var _notebook_subject  : LineEdit = null

func _ready() -> void:
	add_theme_constant_override("separation", 0)
	_build_ui()
	ZADOSClient.memory_data_received.connect(_on_data)
	ZADOSClient.memory_post_result.connect(_on_post_result)
	ZADOSClient.map_data_received.connect(_on_map_data)
	ZADOSClient.request_failed.connect(_on_request_failed)


func _build_ui() -> void:
	_tabs = TabContainer.new()
	_tabs.size_flags_vertical = SIZE_EXPAND_FILL
	_tabs.add_theme_constant_override("side_margin", 0)
	add_child(_tabs)
	_lessons_list  = _make_list_tab("Lessons")
	_notebook_list = _make_notebook_tab()
	_buffer_list   = _make_list_tab("Academic Buffer")
	_maps_list     = _make_list_tab("Knowledge Maps")
	_library_tab   = _make_library_tab()


func _make_list_tab(tab_name: String) -> VBoxContainer:
	var tab_wrap := VBoxContainer.new()
	tab_wrap.name = tab_name
	tab_wrap.add_theme_constant_override("separation", 4)

	var hdr := HBoxContainer.new()
	var lbl := Label.new()
	lbl.text = tab_name
	lbl.size_flags_horizontal = SIZE_EXPAND_FILL
	lbl.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	hdr.add_child(lbl)
	var btn := Button.new()
	btn.text = "↺ Refresh"
	btn.flat = true
	btn.focus_mode = Control.FOCUS_NONE
	btn.add_theme_font_size_override("font_size", 11)
	btn.pressed.connect(func(): _refresh_tab(tab_name))
	hdr.add_child(btn)
	tab_wrap.add_child(hdr)

	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical    = SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	tab_wrap.add_child(scroll)

	var list := VBoxContainer.new()
	list.size_flags_horizontal = SIZE_EXPAND_FILL
	list.add_theme_constant_override("separation", 6)
	scroll.add_child(list)

	_tabs.add_child(tab_wrap)
	return list


## Notebook tab with "New Entry" inline editor (B.2.9)
func _make_notebook_tab() -> VBoxContainer:
	var tab_wrap := VBoxContainer.new()
	tab_wrap.name = "Notebook"
	tab_wrap.add_theme_constant_override("separation", 4)

	var hdr := HBoxContainer.new()
	var lbl := Label.new()
	lbl.text = "Notebook"
	lbl.size_flags_horizontal = SIZE_EXPAND_FILL
	lbl.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	hdr.add_child(lbl)

	var new_btn := Button.new()
	new_btn.text = "+ New Entry"
	new_btn.flat = false
	new_btn.focus_mode = Control.FOCUS_NONE
	new_btn.add_theme_font_size_override("font_size", 11)
	new_btn.pressed.connect(_toggle_notebook_editor)
	hdr.add_child(new_btn)

	var refresh_btn := Button.new()
	refresh_btn.text = "↺ Refresh"
	refresh_btn.flat = true
	refresh_btn.focus_mode = Control.FOCUS_NONE
	refresh_btn.add_theme_font_size_override("font_size", 11)
	refresh_btn.pressed.connect(func(): _refresh_tab("Notebook"))
	hdr.add_child(refresh_btn)
	tab_wrap.add_child(hdr)

	# Inline editor (initially hidden)
	_notebook_editor = VBoxContainer.new()
	_notebook_editor.visible = false
	_notebook_editor.add_theme_constant_override("separation", 4)

	var editor_bg := PanelContainer.new()
	var editor_style := StyleBoxFlat.new()
	editor_style.bg_color = Color(0.08, 0.08, 0.11)
	editor_style.set_corner_radius_all(5)
	editor_style.content_margin_left = 10; editor_style.content_margin_right = 10
	editor_style.content_margin_top = 8; editor_style.content_margin_bottom = 8
	editor_bg.add_theme_stylebox_override("panel", editor_style)

	var editor_inner := VBoxContainer.new()
	editor_inner.add_theme_constant_override("separation", 6)
	editor_bg.add_child(editor_inner)

	var subj_row := HBoxContainer.new()
	var subj_lbl := Label.new()
	subj_lbl.text = "Subject:"
	subj_lbl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	subj_lbl.add_theme_font_size_override("font_size", 10)
	subj_row.add_child(subj_lbl)
	_notebook_subject = LineEdit.new()
	_notebook_subject.placeholder_text = "e.g. philosophy, math, identity..."
	_notebook_subject.size_flags_horizontal = SIZE_EXPAND_FILL
	_notebook_subject.add_theme_font_size_override("font_size", 11)
	subj_row.add_child(_notebook_subject)
	editor_inner.add_child(subj_row)

	_notebook_input = TextEdit.new()
	_notebook_input.placeholder_text = "Write your notebook entry here..."
	_notebook_input.custom_minimum_size = Vector2(0, 80)
	_notebook_input.size_flags_horizontal = SIZE_EXPAND_FILL
	_notebook_input.add_theme_font_size_override("font_size", 11)
	editor_inner.add_child(_notebook_input)

	var action_row := HBoxContainer.new()
	action_row.add_theme_constant_override("separation", 8)
	var save_btn := Button.new()
	save_btn.text = "Save Entry"
	save_btn.focus_mode = Control.FOCUS_NONE
	save_btn.add_theme_font_size_override("font_size", 11)
	save_btn.pressed.connect(_save_notebook_entry)
	action_row.add_child(save_btn)
	var cancel_btn := Button.new()
	cancel_btn.text = "Cancel"
	cancel_btn.flat = true
	cancel_btn.focus_mode = Control.FOCUS_NONE
	cancel_btn.add_theme_font_size_override("font_size", 11)
	cancel_btn.pressed.connect(func(): _notebook_editor.visible = false)
	action_row.add_child(cancel_btn)
	editor_inner.add_child(action_row)

	_notebook_editor.add_child(editor_bg)
	tab_wrap.add_child(_notebook_editor)

	# Scroll list
	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	tab_wrap.add_child(scroll)

	var list := VBoxContainer.new()
	list.size_flags_horizontal = SIZE_EXPAND_FILL
	list.add_theme_constant_override("separation", 6)
	scroll.add_child(list)

	_tabs.add_child(tab_wrap)
	return list


func _toggle_notebook_editor() -> void:
	_notebook_editor.visible = not _notebook_editor.visible
	if _notebook_editor.visible:
		_notebook_input.text = ""
		_notebook_subject.text = ""
		_notebook_input.grab_focus()


func _save_notebook_entry() -> void:
	var content := _notebook_input.text.strip_edges()
	if content.length() < 5:
		_show_toast("Entry must be at least 5 characters.", _Toast.Level.WARNING)
		return
	var subject := _notebook_subject.text.strip_edges()
	ZADOSClient.post_memory("ltmm/knowledge/notebook", {
		"content": content,
		"subject_category": subject if not subject.is_empty() else "general",
		"source_mode": "dev_interface",
	})
	_notebook_editor.visible = false
	_notebook_input.text = ""
	_notebook_subject.text = ""


func _refresh_tab(tab_name: String) -> void:
	match tab_name:
		"Lessons":         ZADOSClient.get_memory("ltmm/knowledge/lessons")
		"Notebook":        ZADOSClient.get_memory("ltmm/knowledge/notebook")
		"Academic Buffer": ZADOSClient.get_memory("ltmm/knowledge/academic_buffer")
		"Knowledge Maps":  ZADOSClient.get_map("knowledge_maps")
		"Library":         ZADOSClient.get_memory("ltmm/knowledge/library")


func refresh() -> void:
	ZADOSClient.get_memory("ltmm/knowledge/lessons")
	ZADOSClient.get_memory("ltmm/knowledge/notebook")
	ZADOSClient.get_memory("ltmm/knowledge/academic_buffer")
	ZADOSClient.get_map("knowledge_maps")
	ZADOSClient.get_memory("ltmm/knowledge/library")


func _on_data(key: String, data: Dictionary) -> void:
	match key:
		"ltmm/knowledge/lessons":
			_populate(_lessons_list, data, _make_lesson_card)
		"ltmm/knowledge/notebook":
			_populate(_notebook_list, data, _make_note_card)
		"ltmm/knowledge/academic_buffer":
			_populate(_buffer_list, data, _make_buffer_card)
		"ltmm/knowledge/library":
			_populate(_library_list, data, _make_library_card)


func _on_request_failed(path: String, error: Dictionary) -> void:
	if "/memory/ltmm/knowledge" not in path:
		return
	var msg := "HTTP %s — %s" % [
		str(error.get("http_code", "?")),
		str(error.get("body", "Connection failed")).left(120)]
	var target_list : VBoxContainer = null
	var section_name := "Knowledge"
	if "lessons" in path:
		target_list = _lessons_list; section_name = "Lessons"
	elif "notebook" in path:
		target_list = _notebook_list; section_name = "Notebook"
	elif "academic_buffer" in path:
		target_list = _buffer_list; section_name = "Academic Buffer"
	elif "library" in path:
		target_list = _library_list; section_name = "Library"
	if target_list:
		for child in target_list.get_children():
			child.queue_free()
		var err := _ErrorDisplay.new()
		err.show_error(section_name, msg)
		err.retry_pressed.connect(refresh)
		target_list.add_child(err)


func _on_post_result(key: String, _data: Dictionary) -> void:
	if key == "ltmm/knowledge/notebook":
		_show_toast("Notebook entry saved", _Toast.Level.SUCCESS)
		_refresh_tab("Notebook")
	elif "academic_buffer" in key and "resolve" in key:
		_show_toast("Academic question resolved", _Toast.Level.SUCCESS)
		_refresh_tab("Academic Buffer")
	elif "library/import" in key:
		_show_toast("Library import started", _Toast.Level.SUCCESS)
		# Poll for completion after a delay
		await get_tree().create_timer(2.0).timeout
		_refresh_tab("Library")


func _on_map_data(key: String, data: Dictionary) -> void:
	if key != "knowledge_maps":
		return
	_populate(_maps_list, data, _make_map_card)


func _populate(list: VBoxContainer, d: Dictionary, fn: Callable) -> void:
	for child in list.get_children():
		child.queue_free()
	var items : Array = d.get("items", [])
	if items.is_empty():
		var lbl := Label.new()
		lbl.text = "Empty."
		lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		list.add_child(lbl)
		return
	for item in items:
		list.add_child(fn.call(item))


func _make_lesson_card(l: Dictionary) -> PanelContainer:
	var fields := {
		"Subject":     l.get("subject_category", "—"),
		"Source Mode": l.get("source_mode",      "—"),
		"Confidence":  "%.2f" % float(l.get("confidence", 0.0)),
		"Status":      l.get("validation_status","—"),
		"Reinforced":  str(l.get("reinforcement_count", 0)) + "×",
	}
	return _card(l.get("content",""), fields, l.get("created_at",""), "")


func _make_note_card(n: Dictionary) -> PanelContainer:
	var fields := {
		"Subject": n.get("subject_category","—"),
		"Mode":    n.get("source_mode","—"),
	}
	return _card(n.get("content",""), fields, n.get("timestamp",""), "")


## Academic buffer card with resolve modal (B.2.10)
func _make_buffer_card(e: Dictionary) -> PanelContainer:
	var is_dream : bool = e.get("dream_candidate", false)
	var resolved  : bool = e.get("resolved", false)
	var fields := {
		"Subject":    e.get("subject_category","—"),
		"Source":     e.get("source_engine",  "—"),
		"Stagnation": str(e.get("stagnation_cycles", 0)) + " cycles",
		"Status":     "✓ Resolved" if resolved else ("⚑ Dream Candidate" if is_dream else "Pending"),
	}
	# Show source_question_id link if present
	var source_q_id : String = e.get("source_question_id", "")
	if not source_q_id.is_empty():
		fields["Linked Question"] = source_q_id.left(12) + "…"

	var card := _card(e.get("concept_formulation",""), fields, e.get("timestamp",""),
		e.get("blocking_reason",""))

	if not resolved:
		var footer := HBoxContainer.new()
		footer.add_theme_constant_override("separation", 8)
		card.get_child(0).add_child(footer)

		var entry_id : String = e.get("entry_id","")

		# B.2.10 — Resolve with required note modal
		var resolve_btn := Button.new()
		resolve_btn.text = "Mark Resolved"
		resolve_btn.flat = false
		resolve_btn.focus_mode = Control.FOCUS_NONE
		resolve_btn.add_theme_font_size_override("font_size", 11)
		resolve_btn.pressed.connect(func(): _show_resolve_modal(entry_id, source_q_id))
		footer.add_child(resolve_btn)

		# View linked unsolved question
		if not source_q_id.is_empty():
			var link_btn := Button.new()
			link_btn.text = "View Source Question"
			link_btn.flat = true
			link_btn.focus_mode = Control.FOCUS_NONE
			link_btn.add_theme_font_size_override("font_size", 11)
			link_btn.add_theme_color_override("font_color", Color(0.55, 0.65, 0.85))
			link_btn.tooltip_text = "Source: %s" % source_q_id
			link_btn.pressed.connect(func():
				_show_toast("Source question: %s" % source_q_id, _Toast.Level.INFO))
			footer.add_child(link_btn)

	return card


## Resolve modal with required note for academic buffer (B.2.10)
func _show_resolve_modal(entry_id: String, source_question_id: String) -> void:
	var dlg := _ConfirmDialog.new()
	var prompt := "How was this academic question resolved?"
	if not source_question_id.is_empty():
		prompt += "\n\nLinked to unsolved question: %s" % source_question_id.left(20)
	dlg.show_dialog_with_input(
		prompt,
		"Provide a resolution note describing how this was addressed.",
		"Resolve", "Cancel",
		"Resolution note...", 10)
	var main := get_tree().get_root().get_node_or_null("Main/ModalContainer")
	if main:
		main.add_child(dlg)
	else:
		add_child(dlg)
	dlg.result_with_text.connect(func(confirmed: bool, text: String):
		if confirmed and text.length() >= 10:
			var body := {"note": text}
			if not source_question_id.is_empty():
				body["source_question_id"] = source_question_id
			ZADOSClient.post_memory(
				"ltmm/knowledge/academic_buffer/" + entry_id + "/resolve", body)
	)


func _make_map_card(m: Dictionary) -> PanelContainer:
	var fields := {
		"Subject":  m.get("subject_category", "—"),
		"Nodes":    str(m.get("node_count", 0)),
		"Edges":    str(m.get("edge_count", 0)),
		"Updated":  (m.get("last_updated", "") as String).left(19),
	}
	var card := _card(m.get("title", "Untitled Map"), fields, "", m.get("description", ""))
	var open_btn := Button.new()
	open_btn.text = "Open in Map Editor"
	open_btn.flat = false
	open_btn.focus_mode = Control.FOCUS_NONE
	open_btn.add_theme_font_size_override("font_size", 11)
	var _map_id : String = m.get("map_id", "")
	open_btn.pressed.connect(func():
		var main := get_tree().get_root().get_node_or_null("Main")
		if main and main.has_method("_switch_to"):
			main._switch_to("map"))
	card.get_child(0).add_child(open_btn)
	return card


# ---------------------------------------------------------------------------
# Library tab — custom layout with import buttons + progress (B.2.4)
# ---------------------------------------------------------------------------

func _make_library_tab() -> VBoxContainer:
	var tab_wrap := VBoxContainer.new()
	tab_wrap.name = "Library"
	tab_wrap.add_theme_constant_override("separation", 4)

	# Header row with title + buttons
	var hdr := HBoxContainer.new()
	hdr.add_theme_constant_override("separation", 8)
	var lbl := Label.new()
	lbl.text = "Library"
	lbl.size_flags_horizontal = SIZE_EXPAND_FILL
	lbl.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	hdr.add_child(lbl)

	var import_btn := Button.new()
	import_btn.text = "Import .txt File"
	import_btn.flat = false
	import_btn.focus_mode = Control.FOCUS_NONE
	import_btn.add_theme_font_size_override("font_size", 11)
	import_btn.pressed.connect(_on_import_pressed)
	hdr.add_child(import_btn)

	var refresh_btn := Button.new()
	refresh_btn.text = "↺ Refresh"
	refresh_btn.flat = true
	refresh_btn.focus_mode = Control.FOCUS_NONE
	refresh_btn.add_theme_font_size_override("font_size", 11)
	refresh_btn.pressed.connect(func(): _refresh_tab("Library"))
	hdr.add_child(refresh_btn)
	tab_wrap.add_child(hdr)

	# Scroll + list
	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	tab_wrap.add_child(scroll)

	_library_list = VBoxContainer.new()
	_library_list.size_flags_horizontal = SIZE_EXPAND_FILL
	_library_list.add_theme_constant_override("separation", 6)
	scroll.add_child(_library_list)

	_tabs.add_child(tab_wrap)

	# Prepare the file dialog (lazy, added to tree on first use)
	_file_dialog = FileDialog.new()
	_file_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	_file_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_file_dialog.filters = PackedStringArray(["*.txt ; Text Files"])
	_file_dialog.title = "Import Text File into Library"
	_file_dialog.size = Vector2i(700, 450)
	_file_dialog.file_selected.connect(_on_file_selected)

	return tab_wrap


func _on_import_pressed() -> void:
	if _file_dialog.get_parent() == null:
		add_child(_file_dialog)
	_file_dialog.popup_centered()


## B.2.4 — library upload with progress toast
func _on_file_selected(path: String) -> void:
	# Derive a title from the filename
	var basename := path.get_file().get_basename()
	_show_toast("Importing '%s'..." % basename, _Toast.Level.INFO)
	ZADOSClient.post_memory("ltmm/knowledge/library/import", {
		"file_path": path,
		"title": basename,
		"source_type": "book",
		"domain": "",
		"tags": [],
		"strategy": "auto",
	})
	# Post result handler will toast success and refresh


func _make_library_card(e: Dictionary) -> PanelContainer:
	var tags_arr : Array = e.get("tags", [])
	var tag_str := ""
	for t in tags_arr:
		if not str(t).begins_with("group:") and not str(t).begins_with("chunk:"):
			if not tag_str.is_empty():
				tag_str += ", "
			tag_str += str(t)

	var fields := {
		"Type":   e.get("source_type", "—"),
		"Domain": e.get("domain", "—") if not (e.get("domain", "") as String).is_empty() else "—",
		"Tags":   tag_str if not tag_str.is_empty() else "—",
	}
	return _card(e.get("title", "Untitled"), fields, e.get("timestamp", ""),
		(e.get("content", "") as String).left(200))


func _card(title: String, fields: Dictionary, timestamp: String, subtitle: String) -> PanelContainer:
	var panel := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.10, 0.10, 0.13)
	style.corner_radius_top_left = 5; style.corner_radius_top_right = 5
	style.corner_radius_bottom_left = 5; style.corner_radius_bottom_right = 5
	style.content_margin_left = 10; style.content_margin_right = 10
	style.content_margin_top = 8; style.content_margin_bottom = 8
	panel.add_theme_stylebox_override("panel", style)
	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 4)
	panel.add_child(vbox)

	if not timestamp.is_empty():
		var ts := Label.new()
		ts.text = (timestamp as String).left(19)
		ts.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		ts.add_theme_font_size_override("font_size", 10)
		vbox.add_child(ts)

	if not title.is_empty():
		var tl := Label.new()
		tl.text = (title as String).left(300)
		tl.add_theme_color_override("font_color", Color(0.85, 0.88, 0.92))
		tl.add_theme_font_size_override("font_size", 11)
		tl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		vbox.add_child(tl)

	if not subtitle.is_empty():
		var sl := Label.new()
		sl.text = subtitle
		sl.add_theme_color_override("font_color", Color(0.65, 0.55, 0.45))
		sl.add_theme_font_size_override("font_size", 10)
		sl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		vbox.add_child(sl)

	var grid := GridContainer.new()
	grid.columns = 2
	grid.add_theme_constant_override("h_separation", 12)
	grid.add_theme_constant_override("v_separation", 2)
	for k in fields:
		var kl := Label.new()
		kl.text = str(k) + ":"
		kl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
		kl.add_theme_font_size_override("font_size", 10)
		kl.custom_minimum_size = Vector2(80, 0)
		grid.add_child(kl)
		var vl := Label.new()
		vl.text = str(fields[k])
		vl.add_theme_color_override("font_color", Color(0.75, 0.78, 0.82))
		vl.add_theme_font_size_override("font_size", 10)
		vl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		grid.add_child(vl)
	vbox.add_child(grid)
	return panel


func _show_toast(text: String, level: int) -> void:
	var tc = get_tree().get_root().get_node_or_null("Main/ToastContainer")
	if tc:
		tc.show_toast(text, level)
