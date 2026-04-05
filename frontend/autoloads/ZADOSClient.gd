##
## ZADOSClient — autoload singleton.
##
## All HTTP and WebSocket communication with the ZADOS bridge server lives here.
## Every workspace connects to these signals instead of making direct HTTP calls.
##
## Addendum A.2: full behavioural spec — state properties, generation guard,
## reconnection strategy, pending result buffering, workspace staleness.
##
extends Node

# ---------------------------------------------------------------------------
# Signals  (expanded per addendum A.2)
# ---------------------------------------------------------------------------

# Connection lifecycle
signal connection_ready()
signal connection_lost()
signal connection_restored()

# Session
signal session_opened(data: Dictionary)
signal session_state_received(state: Dictionary)
signal mode_changed(old_mode: String, new_mode: String)

# Turn processing
signal generation_started()
signal generation_cancelled()
signal turn_phase_updated(phase: int, data: Dictionary)
signal turn_token(phase: int, text: String)
signal turn_complete(result: Dictionary)
signal turn_error(error: Dictionary)

# Metrics
signal metrics_updated(metrics: Dictionary)

# Memory
signal memory_data_received(key: String, data: Dictionary)
signal memory_post_result(key: String, data: Dictionary)

# Mode
signal session_mode_set(mode: String)

# Homework / Reflective (streaming)
signal homework_phase_updated(phase: int, status: String, data: Dictionary)
signal homework_complete(result: Dictionary)
signal homework_error(phase: int, error: String)
signal reflective_complete(result: Dictionary)

# Dev
signal dev_data_received(key: String, data: Dictionary)

# Sleep
signal sleep_activated(sleep_type: String)
signal sleep_exited()
signal sleep_triggered(result: Dictionary)
signal rem_phase_updated(phase: int, status: String, data: Dictionary)
signal rem_complete(result: Dictionary)
signal dream_complete(result: Dictionary)
signal sleep_state_received(data: Dictionary)

# Map / AtomSpace
signal atom_delta(changed_atoms: Array)
signal map_data_received(key: String, data: Dictionary)

# Error reporting — panels can show inline ErrorDisplay on failure
signal request_failed(path: String, error: Dictionary)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

const BASE_URL := "http://localhost:8000"
const WS_URL   := "ws://localhost:8000"

const PROCESS_TIMEOUT  := 30.0   # seconds for /process
const DEFAULT_TIMEOUT  := 10.0   # seconds for other endpoints
const GEN_TIMEOUT      := 120.0  # total generation timeout
const GEN_INACTIVITY   := 30.0   # seconds of no WS events before timeout
const METRICS_POLL_SEC := 5.0    # polling interval for /metrics

# ---------------------------------------------------------------------------
# State  (addendum A.2)
# ---------------------------------------------------------------------------

var session_id: String = ""
var is_ready: bool = false
var is_generating: bool = false
var ws_connected: bool = false
var last_error: Dictionary = {}
var current_mode: String = "Normal"
var generation_target_workspace: String = ""
var current_phase: int = 0

## Temporary text to pre-fill into the conversation input bar on next workspace load.
var prefill_text: String = ""

## Buffered result when generation completes while user is on a different workspace.
var _pending_result: Dictionary = {}

## Workspace staleness flags — set true on turn_complete, cleared on workspace visit.
var _workspace_stale: Dictionary = {
	"memory_stmm": false, "memory_mtmm": false, "memory_ltmm": false,
	"learning": false, "dev": false, "map": false,
}

## Which workspace is currently active (set by Main on switch).
var active_workspace: String = "conversation"

## Metrics polling
var _metrics_timer: Timer = null
var _metrics_polling: bool = false

# WebSocket state
var _ws: WebSocketPeer = null
var _ws_active: bool = false
var _ws_pending: String = ""
var _ws_sent: bool = false
var _ws_last_event_time: float = 0.0
var _gen_start_time: float = 0.0

# Persistent WebSocket (for reconnection)
var _persistent_ws: WebSocketPeer = null
var _persistent_ws_connected: bool = false
var _reconnect_timer: Timer = null
var _reconnect_delay: float = 1.0
var _reconnect_max_delay: float = 30.0

# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

func _ready() -> void:
	# Metrics polling timer
	_metrics_timer = Timer.new()
	_metrics_timer.wait_time = METRICS_POLL_SEC
	_metrics_timer.timeout.connect(_poll_metrics)
	add_child(_metrics_timer)

	# Reconnect timer
	_reconnect_timer = Timer.new()
	_reconnect_timer.one_shot = true
	_reconnect_timer.timeout.connect(_attempt_reconnect)
	add_child(_reconnect_timer)


# ---------------------------------------------------------------------------
# Startup sequence  (addendum A.1)
# ---------------------------------------------------------------------------

## Step 2: health check with retries.
func check_health(retries: int = 3, backoff: float = 2.0) -> Dictionary:
	for attempt in retries:
		var data := await _http_get("/health")
		if not data.is_empty():
			return data
		if attempt < retries - 1:
			await get_tree().create_timer(backoff).timeout
	return {}


## Step 4: open session.
func open_session() -> void:
	var data: Dictionary = await _http_post("/session/open", {})
	if data.is_empty():
		return
	session_id = data.get("session_id", "")
	current_mode = data.get("initial_mode", "Normal")
	session_opened.emit(data)


## Step 7: poll bootstrap status.
func poll_bootstrap_status() -> Dictionary:
	return await _http_get("/session/bootstrap_status")


## Step 6 finalisation: mark ready, start metrics, establish WS.
func finalise_startup() -> void:
	is_ready = true
	_start_metrics_polling()
	connection_ready.emit()


# ---------------------------------------------------------------------------
# Session API
# ---------------------------------------------------------------------------

func set_briefing(briefing: String) -> void:
	await _http_post("/session/briefing", {"briefing": briefing})


func get_session_state() -> void:
	var data := await _http_get("/session/state")
	session_state_received.emit(data)


func get_metrics() -> void:
	var data := await _http_get("/metrics")
	if not data.is_empty():
		metrics_updated.emit(data)


func set_session_mode(new_mode: String) -> void:
	var old := current_mode
	var result := await _http_post("/session/set_mode", {"mode": new_mode})
	if not result.is_empty():
		current_mode = result.get("active_mode", new_mode)
		session_mode_set.emit(current_mode)
		if old != current_mode:
			mode_changed.emit(old, current_mode)
	else:
		# Revert on failure — callers should listen for session_mode_set
		last_error = {"action": "set_mode", "reason": "request failed"}


## Fetch any memory sub-path and emit memory_data_received(key, data).
func get_memory(key: String) -> void:
	var data := await _http_get("/memory/" + key)
	memory_data_received.emit(key, data)


## POST to a memory sub-path (e.g. resolve endpoints).
func post_memory(key: String, body: Dictionary) -> void:
	var data := await _http_post("/memory/" + key, body)
	memory_post_result.emit(key, data)


## Trigger HomeworkPipeline; emits homework_complete when done.
func run_homework() -> void:
	var result := await _http_post("/homework", {})
	homework_complete.emit(result)


## Trigger ReflectivePipeline; emits reflective_complete when done.
func run_reflective() -> void:
	var result := await _http_post("/reflective", {})
	reflective_complete.emit(result)


## Fetch any dev sub-path and emit dev_data_received(key, data).
func get_dev(key: String) -> void:
	var data := await _http_get("/dev/" + key)
	if not data.is_empty():
		dev_data_received.emit(key, data)


## POST to a dev sub-path and emit dev_data_received(key+"/result", data).
func post_dev(key: String, body: Dictionary) -> void:
	var data := await _http_post("/dev/" + key, body)
	if not data.is_empty():
		dev_data_received.emit(key + "/result", data)


## Manually trigger a sleep cycle; emits sleep_triggered when done.
func trigger_sleep() -> void:
	var result := await _http_post("/dev/sleep/trigger", {})
	sleep_triggered.emit(result)


## Activate a specific sleep mode (rem/dream/triage).
func activate_sleep(mode: String) -> void:
	var result := await _http_post("/dev/sleep/activate", {"mode": mode})
	if not result.is_empty():
		sleep_activated.emit(mode)


## Exit sleep mode.
func exit_sleep() -> void:
	var result := await _http_post("/dev/sleep/exit", {})
	if not result.is_empty():
		sleep_exited.emit()


## Run REM pipeline specifically; emits rem_complete when done.
func run_rem() -> void:
	var result := await _http_post("/dev/sleep/rem", {})
	rem_complete.emit(result)


## Run Dream pipeline specifically; emits dream_complete when done.
func run_dream() -> void:
	var result := await _http_post("/dev/sleep/dream", {})
	dream_complete.emit(result)


## Fetch current sleep state (NT snapshot, dream candidates, MTMM summary).
func get_sleep_state() -> void:
	var data := await _http_get("/dev/sleep/state")
	if not data.is_empty():
		sleep_state_received.emit(data)


## Fetch any map sub-path and emit map_data_received(key, data).
func get_map(key: String) -> void:
	var data := await _http_get("/map/" + key)
	if not data.is_empty():
		map_data_received.emit(key, data)


## POST to a map sub-path and emit map_data_received(key+"/result", data).
func post_map(key: String, body: Dictionary) -> void:
	var data := await _http_post("/map/" + key, body)
	if not data.is_empty():
		map_data_received.emit(key + "/result", data)


## Cancel a running pipeline (homework/reflective).
func cancel_pipeline() -> void:
	await _http_post("/dev/pipeline/cancel", {})


# ---------------------------------------------------------------------------
# Processing  (addendum A.2 generation guard)
# ---------------------------------------------------------------------------

## Send a message for processing. Returns immediately if already generating.
func send_message(text: String, workspace: String = "conversation") -> void:
	if is_generating:
		return
	is_generating = true
	generation_target_workspace = workspace
	current_phase = 0
	_pending_result = {}
	_gen_start_time = Time.get_ticks_msec() / 1000.0
	_ws_last_event_time = _gen_start_time
	generation_started.emit()
	stream_turn(text)


## Synchronous turn — fires turn_complete when the full result arrives.
func process_turn(text: String) -> void:
	var result := await _http_post("/process", {"text": text})
	_finish_generation(result)


## Streaming turn — fires turn_phase_updated and turn_token as data arrives,
## then fires turn_complete when the "complete" message lands.
func stream_turn(text: String) -> void:
	if _ws_active:
		push_warning("ZADOSClient: stream already in progress, ignoring.")
		return
	_ws = WebSocketPeer.new()
	var err := _ws.connect_to_url(WS_URL + "/stream/process")
	if err != OK:
		push_error("ZADOSClient: WebSocket connect failed (err %d)" % err)
		_ws = null
		_finish_generation_error("ws_connect_failed", 0)
		return
	_ws_active  = true
	_ws_sent    = false
	_ws_pending = JSON.stringify({"text": text})


## Cancel the current generation.
func cancel_generation() -> void:
	if not is_generating:
		return
	if _ws != null and _ws_active:
		# Send cancel event over WebSocket
		_ws.send_text(JSON.stringify({"type": "cancel"}))
		_ws.close()
		_ws = null
		_ws_active = false
	is_generating = false
	generation_cancelled.emit()


## Poll the WebSocket every frame while a stream is active.
func _process(delta: float) -> void:
	if not _ws_active or _ws == null:
		return

	_ws.poll()

	match _ws.get_ready_state():
		WebSocketPeer.STATE_OPEN:
			# Send the request message once the connection is open.
			if not _ws_sent:
				_ws.send_text(_ws_pending)
				_ws_sent = true

			# Drain all available packets.
			while _ws.get_available_packet_count() > 0:
				var raw := _ws.get_packet().get_string_from_utf8()
				var msg  = JSON.parse_string(raw)
				if msg is Dictionary:
					_handle_ws_message(msg)

			# Generation inactivity timeout
			if is_generating:
				var now := Time.get_ticks_msec() / 1000.0
				if now - _ws_last_event_time > GEN_INACTIVITY:
					_finish_generation_error("timeout", current_phase)
				elif now - _gen_start_time > GEN_TIMEOUT:
					_finish_generation_error("timeout", current_phase)

		WebSocketPeer.STATE_CLOSED:
			if is_generating:
				_finish_generation_error("ws_disconnect", current_phase)
			_ws_active = false
			_ws        = null


func _handle_ws_message(msg: Dictionary) -> void:
	_ws_last_event_time = Time.get_ticks_msec() / 1000.0

	match msg.get("type", ""):
		# --- Conversation turn ---
		"phase_start":
			current_phase = msg.get("phase", 0)
		"phase_complete":
			current_phase = msg.get("phase", 0)
			turn_phase_updated.emit(msg.get("phase", 0), msg.get("data", {}))
		"token":
			turn_token.emit(msg.get("phase", 0), msg.get("text", ""))
		"complete", "turn_complete":
			_finish_generation(msg.get("result", {}))
			_ws_active = false
		"turn_error":
			_finish_generation_error(msg.get("error", "unknown"), msg.get("phase", 0))
			_ws_active = false

		# --- Homework ---
		"homework_phase":
			homework_phase_updated.emit(
				msg.get("phase", 0),
				msg.get("status", ""),
				msg.get("data", {}))
		"homework_complete":
			homework_complete.emit(msg.get("summary", {}))
		"homework_error":
			homework_error.emit(msg.get("phase", 0), msg.get("error", ""))

		# --- REM ---
		"rem_phase":
			rem_phase_updated.emit(
				msg.get("phase", 0),
				msg.get("status", ""),
				msg.get("data", {}))
		"rem_complete":
			rem_complete.emit(msg.get("summary", {}))
			_mark_stale(["memory_ltmm", "map"])

		# --- Dream ---
		"dream_candidate_start", "dream_recombination", "dream_scene_shift":
			# Forward raw to any connected handler
			turn_phase_updated.emit(-1, msg)
		"dream_complete":
			dream_complete.emit(msg.get("summary", {}))
			_mark_stale(["memory_ltmm", "map"])

		# --- System ---
		"error":
			push_error("ZADOSClient WS: " + str(msg.get("message", "")))
			if is_generating:
				_finish_generation_error(msg.get("message", "unknown"), current_phase)
			_ws_active = false
		"cancel_ack":
			pass   # already handled by cancel_generation()


func _finish_generation(result: Dictionary) -> void:
	is_generating = false
	current_phase = 0

	# Mark all workspaces stale
	for key in _workspace_stale:
		_workspace_stale[key] = true

	# If user is on the target workspace, emit immediately.
	# Otherwise buffer the result for when they return.
	if active_workspace == generation_target_workspace:
		turn_complete.emit(result)
	else:
		_pending_result = result
		turn_complete.emit(result)   # still emit so StatusStrip etc. update

	generation_target_workspace = ""


func _finish_generation_error(reason: String, phase: int) -> void:
	is_generating = false
	current_phase = 0
	_ws_active = false
	if _ws != null:
		_ws.close()
		_ws = null
	last_error = {"reason": reason, "phase": phase}
	turn_error.emit(last_error)
	generation_target_workspace = ""


## Consume the pending result (called by ConversationWorkspace on re-entry).
func consume_pending_result() -> Dictionary:
	var result := _pending_result
	_pending_result = {}
	return result


# ---------------------------------------------------------------------------
# Workspace staleness  (addendum A.6)
# ---------------------------------------------------------------------------

func is_stale(workspace_key: String) -> bool:
	return _workspace_stale.get(workspace_key, false)


func clear_stale(workspace_key: String) -> void:
	_workspace_stale[workspace_key] = false


func _mark_stale(keys: Array) -> void:
	for key in keys:
		if _workspace_stale.has(key):
			_workspace_stale[key] = true


# ---------------------------------------------------------------------------
# Metrics polling control  (addendum A.2)
# ---------------------------------------------------------------------------

func _start_metrics_polling() -> void:
	if not _metrics_polling:
		_metrics_polling = true
		_metrics_timer.start()


func _stop_metrics_polling() -> void:
	if _metrics_polling:
		_metrics_polling = false
		_metrics_timer.stop()


func _poll_metrics() -> void:
	# Only poll when on a workspace that displays neurochem data.
	if active_workspace in ["conversation", "dev"]:
		get_metrics()
	elif active_workspace == "map":
		# Only if Live Link is on — callers set this.
		pass


## Called by Main when workspace changes.
func set_active_workspace(key: String) -> void:
	active_workspace = key
	# Restart or stop metrics polling based on workspace.
	if key in ["conversation", "dev"]:
		_start_metrics_polling()
	else:
		_stop_metrics_polling()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

func _http_post(path: String, body: Dictionary) -> Dictionary:
	return await _http_request(HTTPClient.METHOD_POST, path, JSON.stringify(body))


func _http_get(path: String) -> Dictionary:
	return await _http_request(HTTPClient.METHOD_GET, path, "")


func _http_request(method: int, path: String, body: String) -> Dictionary:
	var http := HTTPRequest.new()
	# Timeout: longer for /process, shorter for everything else.
	if "/process" in path:
		http.timeout = PROCESS_TIMEOUT
	else:
		http.timeout = DEFAULT_TIMEOUT
	add_child(http)

	var headers := PackedStringArray(["Content-Type: application/json"])
	if not session_id.is_empty():
		headers.append("X-Session-Id: " + session_id)
	var err := http.request(BASE_URL + path, headers, method, body)

	if err != OK:
		push_error("ZADOSClient: request error %d on %s" % [err, path])
		http.queue_free()
		return {}

	var response: Array = await http.request_completed
	http.queue_free()

	var result_code: int = response[0]
	var http_code: int   = response[1]
	var body_str: String = response[3].get_string_from_utf8()

	if result_code != HTTPRequest.RESULT_SUCCESS:
		push_error("ZADOSClient: connection failed (result=%d) for %s" % [result_code, path])
		last_error = {"path": path, "result_code": result_code, "reason": "connection_failed"}
		connection_lost.emit()
		return {}

	if http_code < 200 or http_code >= 300:
		push_error("ZADOSClient: HTTP %d for %s — body: %s" % [http_code, path, body_str.left(200)])
		last_error = {"path": path, "http_code": http_code, "body": body_str.left(500)}
		request_failed.emit(path, last_error)
		return {}

	if body_str.is_empty():
		return {}

	var parsed = JSON.parse_string(body_str)

	if parsed == null:
		push_error("ZADOSClient: JSON parse failed for %s — body: %s" % [path, body_str.left(200)])
		return {}

	return parsed if parsed is Dictionary else {}


# ---------------------------------------------------------------------------
# Reconnection  (addendum A.2)
# ---------------------------------------------------------------------------

func _attempt_reconnect() -> void:
	# Placeholder for persistent WS reconnection.
	# The current architecture uses per-turn WS connections.
	# When switching to a persistent WS, this will implement exponential backoff.
	pass
