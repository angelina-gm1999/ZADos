##
## ZADOSClient — autoload singleton.
##
## All HTTP and WebSocket communication with the ZADOS bridge server lives here.
## Every workspace connects to these signals instead of making direct HTTP calls.
##
## Usage:
##   ZADOSClient.open_session()
##   ZADOSClient.process_turn("Hello!")
##   ZADOSClient.stream_turn("Hello!")     # fires phase/token signals as data arrives
##
extends Node

# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

## Fired once after POST /session/open completes.
signal session_opened(data: Dictionary)

## Fired after GET /session/state completes.
signal session_state_received(state: Dictionary)

## Fired after a full turn completes (both sync and streaming paths).
signal turn_complete(result: Dictionary)

## Fired during a streaming turn as each phase finishes.
## phase: 1–6  |  data: phase-specific dict
signal turn_phase_updated(phase: int, data: Dictionary)

## Fired during a streaming turn as text tokens arrive for phases 4 and 6.
signal turn_token(phase: int, text: String)

## Fired after GET /metrics completes.
signal metrics_updated(metrics: Dictionary)

## Fired after any GET /memory/... or POST /memory/... completes.
## key matches the path after "/memory/" (e.g. "stmm", "mtmm/packets").
signal memory_data_received(key: String, data: Dictionary)

## Fired after POST /session/set_mode completes.
signal session_mode_set(mode: String)

## Fired when POST /homework completes.
signal homework_complete(result: Dictionary)

## Fired when POST /reflective completes.
signal reflective_complete(result: Dictionary)

## Fired after any GET /dev/... or POST /dev/... completes.
## key matches the path after "/dev/" (e.g. "reward", "pipeline").
signal dev_data_received(key: String, data: Dictionary)

## Fired when POST /dev/sleep/trigger completes.
signal sleep_triggered(result: Dictionary)

## Fired when POST /dev/sleep/rem completes.
signal rem_complete(result: Dictionary)

## Fired when POST /dev/sleep/dream completes.
signal dream_complete(result: Dictionary)

## Fired after GET /dev/sleep/state completes.
signal sleep_state_received(data: Dictionary)

## Fired after any GET /map/... completes.
signal map_data_received(key: String, data: Dictionary)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

const BASE_URL := "http://localhost:8000"
const WS_URL   := "ws://localhost:8000"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

var session_id: String = ""

## Temporary text to pre-fill into the conversation input bar on next workspace load.
var prefill_text: String = ""

var _ws: WebSocketPeer = null
var _ws_active: bool   = false
var _ws_pending: String = ""
var _ws_sent: bool      = false

# ---------------------------------------------------------------------------
# Session API
# ---------------------------------------------------------------------------

func open_session() -> void:
	var data: Dictionary = await _http_post("/session/open", {})
	if data.is_empty():
		return
	session_id = data.get("session_id", "")
	session_opened.emit(data)


func set_briefing(briefing: String) -> void:
	await _http_post("/session/briefing", {"briefing": briefing})


func get_session_state() -> void:
	var data := await _http_get("/session/state")
	session_state_received.emit(data)


func get_metrics() -> void:
	var data := await _http_get("/metrics")
	if not data.is_empty():
		metrics_updated.emit(data)


func set_session_mode(mode: String) -> void:
	var result := await _http_post("/session/set_mode", {"mode": mode})
	if not result.is_empty():
		session_mode_set.emit(result.get("active_mode", mode))


## Fetch any memory sub-path and emit memory_data_received(key, data).
func get_memory(key: String) -> void:
	var data := await _http_get("/memory/" + key)
	if not data.is_empty():
		memory_data_received.emit(key, data)


## POST to a memory sub-path (e.g. resolve endpoints).
func post_memory(key: String, body: Dictionary) -> void:
	var data := await _http_post("/memory/" + key, body)
	if not data.is_empty():
		memory_data_received.emit(key + "/result", data)


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

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

## Synchronous turn — fires turn_complete when the full result arrives.
func process_turn(text: String) -> void:
	var result := await _http_post("/process", {"text": text})
	turn_complete.emit(result)


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
		return
	_ws_active  = true
	_ws_sent    = false
	_ws_pending = JSON.stringify({"text": text})


## Poll the WebSocket every frame while a stream is active.
func _process(_delta: float) -> void:
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

		WebSocketPeer.STATE_CLOSED:
			_ws_active = false
			_ws        = null


func _handle_ws_message(msg: Dictionary) -> void:
	match msg.get("type", ""):
		"phase_complete":
			turn_phase_updated.emit(msg.get("phase", 0), msg.get("data", {}))
		"token":
			turn_token.emit(msg.get("phase", 0), msg.get("text", ""))
		"complete":
			turn_complete.emit(msg.get("result", {}))
			_ws_active = false
		"error":
			push_error("ZADOSClient WS: " + str(msg.get("message", "")))
			_ws_active = false

# ---------------------------------------------------------------------------
# HTTP helpers  (prefixed _http_ to avoid collision with built-in _get/_set)
# ---------------------------------------------------------------------------

func _http_post(path: String, body: Dictionary) -> Dictionary:
	return await _http_request(HTTPClient.METHOD_POST, path, JSON.stringify(body))


func _http_get(path: String) -> Dictionary:
	return await _http_request(HTTPClient.METHOD_GET, path, "")


func _http_request(method: int, path: String, body: String) -> Dictionary:
	var http := HTTPRequest.new()
	add_child(http)

	var headers := PackedStringArray(["Content-Type: application/json"])
	var err      := http.request(BASE_URL + path, headers, method, body)

	if err != OK:
		push_error("ZADOSClient: request error %d on %s" % [err, path])
		http.queue_free()
		return {}

	# await returns [result_code, response_code, headers, body_bytes]
	var response: Array = await http.request_completed
	http.queue_free()

	var body_str: String = response[3].get_string_from_utf8()
	var parsed           = JSON.parse_string(body_str)

	if parsed == null:
		push_error("ZADOSClient: JSON parse failed for %s — body: %s" % [path, body_str.left(200)])
		return {}

	return parsed if parsed is Dictionary else {}
