from __future__ import annotations

from datetime import datetime, timedelta, timezone

from flask import Blueprint, flash, jsonify, redirect, render_template, request, url_for
from sqlalchemy import func

from hybrid.ai_comfort_service import predict_live_comfort
from hybrid.ai_config import is_room_ai_enabled, load_ai_config, set_room_ai_enabled, update_ai_config
from hybrid.controller import evaluate_room_control
from hybrid_dashboard.ble_runtime import get_runtime_status, start_runtime, stop_runtime
from hybrid.hospital_initializer import (
    DEFAULT_ROOM_MAP,
    add_sensor_to_nordic_device,
    ensure_simulated_devices,
    initialize_hybrid_hospital,
    register_nordic_device,
    update_nordic_sensor,
    update_nordic_device_mac,
)
from hybrid.state_store import get_room_state, get_zone_state
from persistence.database import session_scope
from persistence.models import ComfortPreference, Data, Device, HybridZoneState, Medication, Room, RoomAssignment, Visit


hybrid_bp = Blueprint("hybrid_bp", __name__, template_folder="templates")
SENSOR_TYPE_OPTIONS = (
    ("temperature", "Temperature", "C"),
    ("humidity", "Humidity", "%"),
    ("co2", "CO2", "ppm"),
    ("light", "Light", "lux"),
    ("sound", "Sound", "dB"),
)
UNIT_BY_SENSOR_TYPE = {value: unit for value, _, unit in SENSOR_TYPE_OPTIONS}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_form_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.strip())
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        parsed = int(raw)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _save_comfort_preference(room_id: int, patient_id: int | None, redirect_endpoint: str):
    submitted_main = (request.form.get("temperature_main") or "").strip()
    submitted_toilet = (request.form.get("temperature_toilet") or "").strip()
    with session_scope() as session:
        latest_preference = (
            session.query(ComfortPreference)
            .filter(ComfortPreference.room_id == room_id)
            .order_by(ComfortPreference.timestamp.desc())
            .first()
        )

        def _carry_float(field_name: str, latest_value: float | None) -> float | None:
            raw = request.form.get(field_name)
            if raw is None:
                return latest_value
            raw = raw.strip()
            if raw == "":
                return latest_value
            return float(raw)

        temperature_main = _carry_float(
            "temperature_main",
            latest_preference.temperature_main if latest_preference else None,
        )
        if temperature_main is None:
            flash("Main temperature is required at least once before partial updates can be used.")
            return redirect(url_for(redirect_endpoint, room_id=room_id))

        preference = ComfortPreference(
            room_id=room_id,
            patient_id=patient_id,
            temperature_main=temperature_main,
            temperature_toilet=_carry_float(
                "temperature_toilet",
                latest_preference.temperature_toilet if latest_preference else None,
            ),
            light_intensity=_carry_float(
                "light_intensity",
                latest_preference.light_intensity if latest_preference else None,
            ),
            sound_level=_carry_float(
                "sound_level",
                latest_preference.sound_level if latest_preference else None,
            ),
            airflow=(
                True
                if request.form.get("airflow_choice") == "on"
                else False
                if request.form.get("airflow_choice") == "off"
                else (latest_preference.airflow if latest_preference else False)
            ),
            source="manual",
            timestamp=_utc_now(),
        )
        session.add(preference)
        session.flush()
        if submitted_main:
            evaluate_room_control(session, room_id, preference, location="main")
        if submitted_toilet:
            evaluate_room_control(session, room_id, preference, location="toilet")
        if not submitted_main and not submitted_toilet:
            evaluate_room_control(session, room_id, preference, location="main")
    flash("Comfort preference saved.")
    return redirect(url_for(redirect_endpoint, room_id=room_id))


def _save_ai_comfort_preference(room_id: int, patient_id: int | None, redirect_endpoint: str):
    with session_scope() as session:
        prediction, metadata = predict_live_comfort(session, room_id, patient_id)
        preference = ComfortPreference(
            room_id=room_id,
            patient_id=metadata["patient_id"],
            temperature_main=prediction["temperature_main"],
            temperature_toilet=prediction["temperature_toilet"],
            light_intensity=prediction["light_intensity"],
            sound_level=prediction["sound_level"],
            airflow=bool(prediction["airflow"]),
            source="ai",
            timestamp=_utc_now(),
        )
        session.add(preference)
        session.flush()
        evaluate_room_control(session, room_id, preference, location="main")
        evaluate_room_control(session, room_id, preference, location="toilet")
    flash(f"AI comfort applied using {metadata['source']} / {metadata['model_type']} weights.")
    return redirect(url_for(redirect_endpoint, room_id=room_id))


def _active_assignment_for_room(session, room_id: int) -> RoomAssignment | None:
    now = _utc_now()
    return (
        session.query(RoomAssignment)
        .filter(
            RoomAssignment.room_id == room_id,
            RoomAssignment.start_time <= now,
            func.coalesce(RoomAssignment.end_time, now + timedelta(days=3650)) >= now,
        )
        .order_by(RoomAssignment.start_time.desc())
        .first()
    )


def _format_sensor_value(sensor_type: str | None, value: float | None) -> str | None:
    if value is None:
        return None
    if sensor_type == "temperature":
        return f"{value:.1f}"
    if isinstance(value, float) and value.is_integer():
        return f"{value:.0f}"
    return str(value)


def _room_device_status(session, device: Device) -> dict[str, str | None]:
    runtime = get_runtime_status(device.device_id)
    latest_data = None
    if device.sensors:
        latest_data = (
            session.query(Data)
            .filter(Data.sensor_id.in_([sensor.sensor_id for sensor in device.sensors]))
            .order_by(Data.timestamp.desc())
            .first()
        )
    if latest_data is None:
        activity_state = "saved"
        seen = None
    else:
        seen = _as_utc(latest_data.timestamp)
        recent_threshold = _utc_now() - timedelta(minutes=5)
        activity_state = "active" if seen and seen >= recent_threshold else "seen"
    return {
        "status": runtime["state"] if runtime["state"] != "saved" else activity_state,
        "last_seen": runtime["last_seen"] or (seen.isoformat() if seen else None),
        "last_error": runtime["last_error"],
        "last_connected_at": runtime["last_seen"] if runtime["connected"] else None,
    }


def _latest_sensor_rows(session, room_id: int) -> list[dict]:
    rows: list[dict] = []
    zone_rows = {
        row.location: row
        for row in session.query(HybridZoneState).filter(HybridZoneState.room_id == room_id).all()
    }
    devices = session.query(Device).filter(Device.room_id == room_id).all()
    for device in devices:
        for sensor in device.sensors:
            latest = (
                session.query(Data)
                .filter(Data.sensor_id == sensor.sensor_id)
                .order_by(Data.timestamp.desc(), Data.data_id.desc())
                .first()
            )
            value = latest.value if latest else None
            if sensor.sensor_type == "temperature":
                zone_row = zone_rows.get(device.location or "main")
                if zone_row and zone_row.virtual_temp is not None:
                    value = zone_row.virtual_temp
                else:
                    zone_state = get_zone_state(room_id, device.location)
                    if zone_state.virtual_temp is not None:
                        value = zone_state.virtual_temp
            rows.append(
                {
                    "device_id": device.device_id,
                    "location": device.location,
                    "sensor_type": sensor.sensor_type,
                    "uuid": sensor.uuid,
                    "value": value,
                    "value_display": _format_sensor_value(sensor.sensor_type, value),
                    "timestamp": latest.timestamp if latest else None,
                    "unit": sensor.unit,
                }
            )
    return rows


def _actuator_rows(session, room_id: int) -> list[dict]:
    ensure_simulated_devices(room_id)
    room_state = get_room_state(room_id)
    devices = (
        session.query(Device)
        .filter(Device.room_id == room_id, Device.device_type.in_(["light", "ventilation", "speaker", "toilet_light", "toilet_heater"]))
        .order_by(Device.location.asc(), Device.device_type.asc())
        .all()
    )
    latest_pref = (
        session.query(ComfortPreference)
        .filter(ComfortPreference.room_id == room_id)
        .order_by(ComfortPreference.timestamp.desc())
        .first()
    )
    rows = []
    for device in devices:
        state = "--"
        if device.device_type == "ventilation":
            if room_state.main.hvac_mode == "airflow" or (latest_pref and latest_pref.airflow):
                state = "on"
            else:
                state = "off"
            rows.append(
                {
                    "device_id": device.device_id,
                    "device_type": device.device_type,
                    "display_type": "airflow",
                    "location": device.location,
                    "state": state,
                }
            )
            rows.append(
                {
                    "device_id": device.device_id,
                    "device_type": "hvac",
                    "display_type": "hvac",
                    "location": device.location,
                    "state": (
                        "heating"
                        if room_state.main.hvac_mode == "heat"
                        else "cooling"
                        if room_state.main.hvac_mode == "cool"
                        else "off"
                    ),
                }
            )
            continue
        elif device.device_type == "speaker":
            level = (
                latest_pref.sound_level
                if latest_pref and latest_pref.sound_level is not None
                else device.speaker.level if device.speaker else 0.0
            )
            state = f"level {level}"
        elif device.device_type == "toilet_light":
            state = "on" if device.toilet_light and device.toilet_light.state else "off"
        elif device.device_type == "toilet_heater":
            state = "on" if room_state.toilet.hvac_mode == "heat" else "off"
            if device.toilet_heater is not None:
                device.toilet_heater.state = room_state.toilet.hvac_mode == "heat"
                device.toilet_heater.timestamp = _utc_now().replace(tzinfo=None)
        elif device.device_type == "light":
            state = f"target {latest_pref.light_intensity}" if latest_pref and latest_pref.light_intensity is not None else "off"
        rows.append(
            {
                "device_id": device.device_id,
                "device_type": device.device_type,
                "display_type": device.device_type.replace("_", " "),
                "location": device.location,
                "state": state,
            }
        )
    return rows


def _room_card(session, room: Room) -> dict:
    assignment = _active_assignment_for_room(session, room.room_id)
    patient = assignment.patient if assignment else None
    sensor_rows = _latest_sensor_rows(session, room.room_id)
    recent_threshold = _utc_now() - timedelta(minutes=5)
    active_count = sum(1 for row in sensor_rows if _as_utc(row["timestamp"]) and _as_utc(row["timestamp"]) >= recent_threshold)
    return {
        "room_id": room.room_id,
        "room_number": room.room_number,
        "patient_id": patient.patient_id if patient else None,
        "patient_name": patient.name if patient else "Unassigned",
        "sensor_count": len(sensor_rows),
        "active_sensor_count": active_count,
    }


@hybrid_bp.route("/")
def landing():
    with session_scope() as session:
        rooms = session.query(Room).filter(Room.room_id.in_(list(DEFAULT_ROOM_MAP.values()))).order_by(Room.room_id.asc()).all()
        room_cards = [_room_card(session, room) for room in rooms]
    return render_template("landing.html", rooms=room_cards)


@hybrid_bp.route("/admin")
def admin_home():
    ai_config = load_ai_config()
    with session_scope() as session:
        rooms = session.query(Room).filter(Room.room_id.in_(list(DEFAULT_ROOM_MAP.values()))).order_by(Room.room_id.asc()).all()
        room_cards = [_room_card(session, room) for room in rooms]
    return render_template("admin_home.html", rooms=room_cards, ai_config=ai_config)


@hybrid_bp.route("/admin/ai-config", methods=["POST"])
def admin_update_ai_config():
    current_config = load_ai_config()
    selected_source = str(request.form.get("selected_source") or "k_hours").strip()
    k_hours_model_type = str(request.form.get("k_hours_model_type") or "mlp").strip()
    event_based_model_type = "mlp"
    k_hours_weights_path = str(request.form.get("k_hours_weights_path") or "").strip() or current_config["k_hours_weights_path"]
    event_based_weights_path = (
        str(request.form.get("event_based_weights_path") or "").strip() or current_config["event_based_weights_path"]
    )
    update_ai_config(
        selected_source=selected_source,
        k_hours_model_type=k_hours_model_type,
        event_based_model_type=event_based_model_type,
        k_hours_weights_path=k_hours_weights_path,
        event_based_weights_path=event_based_weights_path,
    )
    flash(f"AI configuration updated. Active source: {selected_source} ({k_hours_model_type if selected_source == 'k_hours' else 'mlp'}).")
    return redirect(url_for("hybrid_bp.admin_home"))


@hybrid_bp.route("/admin/initialize", methods=["POST"])
def admin_initialize():
    summary = initialize_hybrid_hospital()
    flash(
        f"Initialized hybrid hospital: rooms={summary.rooms}, patients={summary.patients}, admissions={summary.admissions}, "
        f"assignments={summary.assignments}, medications={summary.medications}, visits={summary.visits}, "
        f"simulated_devices={summary.simulated_devices}"
    )
    return redirect(url_for("hybrid_bp.admin_home"))


@hybrid_bp.route("/admin/rooms/<int:room_id>")
def admin_room(room_id: int):
    ai_config = load_ai_config()
    with session_scope() as session:
        room = session.get(Room, room_id)
        if room is None:
            return "Room not found", 404
        ensure_simulated_devices(room_id)
        assignment = _active_assignment_for_room(session, room_id)
        patient = assignment.patient if assignment else None
        devices = (
            session.query(Device)
            .filter(Device.room_id == room_id)
            .order_by(Device.device_type.asc(), Device.location.asc())
            .all()
        )
        device_rows = []
        for device in devices:
            status = _room_device_status(session, device)
            device_rows.append(
                {
                    "device_id": device.device_id,
                    "device_type": device.device_type,
                    "location": device.location,
                    "mac_address": device.mac_address,
                    "sensors": [
                        {
                            "sensor_id": sensor.sensor_id,
                            "sensor_type": sensor.sensor_type,
                            "uuid": sensor.uuid,
                            "unit": sensor.unit,
                            "default_unit": UNIT_BY_SENSOR_TYPE.get(sensor.sensor_type, ""),
                        }
                        for sensor in device.sensors
                    ],
                    "status": status["status"],
                    "last_seen": status["last_seen"],
                    "last_error": status["last_error"],
                    "last_connected_at": status["last_connected_at"],
                }
            )
        medications = (
            session.query(Medication)
            .filter(Medication.patient_id == patient.patient_id)
            .order_by(Medication.medication_time.desc())
            .all()
            if patient
            else []
        )
        visits = (
            session.query(Visit)
            .filter(Visit.patient_id == patient.patient_id)
            .order_by(Visit.visit_time.desc())
            .all()
            if patient
            else []
        )
        latest_comfort = (
            session.query(ComfortPreference)
            .filter(ComfortPreference.room_id == room_id)
            .order_by(ComfortPreference.timestamp.desc())
            .first()
        )
        sensor_rows = _latest_sensor_rows(session, room_id)
        actuator_rows = _actuator_rows(session, room_id)
        return render_template(
            "admin_room.html",
            room=room,
            patient=patient,
            assignment=assignment,
            devices=device_rows,
            sensor_rows=sensor_rows,
            actuator_rows=actuator_rows,
            medications=medications,
            visits=visits,
            latest_comfort=latest_comfort,
            ai_config=ai_config,
            sensor_type_options=SENSOR_TYPE_OPTIONS,
            unit_by_sensor_type=UNIT_BY_SENSOR_TYPE,
        )


@hybrid_bp.route("/admin/rooms/<int:room_id>/devices", methods=["POST"])
def add_room_device(room_id: int):
    location = request.form.get("location", "main").strip().lower()
    mac_address = request.form.get("mac_address", "").strip()
    if location not in {"main", "toilet"} or not mac_address:
        flash("Location and MAC address are required.")
        return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))
    try:
        register_nordic_device(room_id=room_id, location=location, mac_address=mac_address)
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))
    flash(f"Registered Nordic Thingy:52 device for room {room_id} ({location}). You can connect it from the device list.")
    return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))


@hybrid_bp.route("/admin/devices/<int:device_id>/mac", methods=["POST"])
def update_device_mac(device_id: int):
    room_id = int(request.form["room_id"])
    mac_address = request.form.get("mac_address", "").strip()
    if not mac_address:
        flash("MAC address is required.")
        return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))
    try:
        update_nordic_device_mac(device_id=device_id, mac_address=mac_address)
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))
    stop_runtime(device_id)
    flash("Nordic MAC address updated.")
    return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))


@hybrid_bp.route("/admin/devices/<int:device_id>/connect", methods=["POST"])
def connect_device(device_id: int):
    room_id = int(request.form["room_id"])
    try:
        start_runtime(device_id)
    except Exception as exc:
        flash(f"Bluetooth runtime failed to start: {exc}")
        return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))
    flash("Bluetooth runtime started. Status will update live as notifications arrive.")
    return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))


@hybrid_bp.route("/admin/devices/<int:device_id>/disconnect", methods=["POST"])
def disconnect_device(device_id: int):
    room_id = int(request.form["room_id"])
    stop_runtime(device_id)
    flash("Device runtime stopped.")
    return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))


@hybrid_bp.route("/admin/devices/<int:device_id>/sensors", methods=["POST"])
def add_device_sensor(device_id: int):
    room_id = int(request.form["room_id"])
    sensor_type = request.form.get("sensor_type", "").strip().lower()
    uuid = request.form.get("uuid", "").strip()
    unit = request.form.get("unit", "").strip() or UNIT_BY_SENSOR_TYPE.get(sensor_type)
    valid_sensor_types = {value for value, _, _ in SENSOR_TYPE_OPTIONS}
    if sensor_type not in valid_sensor_types or not uuid:
        flash("Sensor type and UUID are required.")
        return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))
    try:
        add_sensor_to_nordic_device(device_id=device_id, sensor_type=sensor_type, uuid=uuid, unit=unit)
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))
    flash(f"Added {sensor_type} sensor UUID to Nordic device.")
    return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))


@hybrid_bp.route("/admin/sensors/<int:sensor_id>", methods=["POST"])
def update_device_sensor(sensor_id: int):
    room_id = int(request.form["room_id"])
    sensor_type = request.form.get("sensor_type", "").strip().lower()
    uuid = request.form.get("uuid", "").strip()
    unit = request.form.get("unit", "").strip() or UNIT_BY_SENSOR_TYPE.get(sensor_type)
    valid_sensor_types = {value for value, _, _ in SENSOR_TYPE_OPTIONS}
    if sensor_type not in valid_sensor_types or not uuid:
        flash("Sensor type and UUID are required.")
        return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))
    try:
        update_nordic_sensor(sensor_id=sensor_id, sensor_type=sensor_type, uuid=uuid, unit=unit)
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))
    flash("Sensor UUID updated.")
    return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))


@hybrid_bp.route("/admin/patients/<int:patient_id>/visits", methods=["POST"])
def add_visit(patient_id: int):
    visit = Visit(
        patient_id=patient_id,
        visit_time=_utc_now(),
        body_temperature=float(request.form["body_temperature"]) if request.form.get("body_temperature") else None,
        blood_pressure=request.form.get("blood_pressure") or None,
        symptoms=request.form.get("symptoms") or None,
    )
    with session_scope() as session:
        session.add(visit)
    flash("Visit saved.")
    room_id = int(request.form["room_id"])
    return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))


@hybrid_bp.route("/admin/medications/<int:medication_id>/taken", methods=["POST"])
def mark_medication_taken(medication_id: int):
    room_id = int(request.form["room_id"])
    with session_scope() as session:
        medication = session.get(Medication, medication_id)
        if medication is None:
            flash("Medication not found.")
        else:
            is_taken = request.form.get("taken") == "true"
            medication.status = "taken" if is_taken else "pending"
            if is_taken:
                medication.medication_time = _utc_now()
            flash("Medication updated.")
    return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))


@hybrid_bp.route("/admin/patients/<int:patient_id>/medications", methods=["POST"])
def add_medication(patient_id: int):
    room_id = int(request.form["room_id"])
    drug_name = (request.form.get("drug_name") or "").strip()
    medication_time = _parse_form_datetime(request.form.get("medication_time"))
    if not drug_name or medication_time is None:
        flash("Drug name and scheduled time are required.")
        return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))

    repeat_every_hours = _parse_positive_int(request.form.get("repeat_every_hours"))
    repeat_times = _parse_positive_int(request.form.get("repeat_times"))
    if (repeat_every_hours is None) != (repeat_times is None):
        flash("Repeat interval and repeat count must both be set, or both left empty.")
        return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))

    occurrences = repeat_times if repeat_times is not None else 1
    step = timedelta(hours=repeat_every_hours) if repeat_every_hours is not None else timedelta(0)

    route = (request.form.get("route") or "").strip() or None
    dose = (request.form.get("dose") or "").strip() or None
    status = (request.form.get("status") or "pending").strip() or "pending"
    medications = [
        Medication(
            patient_id=patient_id,
            drug_name=drug_name,
            medication_time=medication_time + (step * idx),
            route=route,
            dose=dose,
            status=status,
        )
        for idx in range(occurrences)
    ]
    with session_scope() as session:
        session.add_all(medications)
    if occurrences == 1:
        flash("Medication added.")
    else:
        flash(f"Medication series added: {occurrences} doses every {repeat_every_hours} hour(s).")
    return redirect(url_for("hybrid_bp.admin_room", room_id=room_id))


@hybrid_bp.route("/patient")
def patient_home():
    with session_scope() as session:
        assignments = (
            session.query(RoomAssignment)
            .filter(RoomAssignment.room_id.in_(list(DEFAULT_ROOM_MAP.values())))
            .order_by(RoomAssignment.room_id.asc(), RoomAssignment.start_time.desc())
            .all()
        )
        active = []
        seen_room_ids: set[int] = set()
        now = _utc_now()
        for assignment in assignments:
            if assignment.room_id in seen_room_ids:
                continue
            start_time = _as_utc(assignment.start_time)
            end_time = _as_utc(assignment.end_time)
            if start_time and start_time > now:
                continue
            if end_time and end_time < now:
                continue
            seen_room_ids.add(assignment.room_id)
            active.append(assignment)
        return render_template("patient_home.html", assignments=active)


@hybrid_bp.route("/patient/rooms/<int:room_id>")
def patient_room(room_id: int):
    ai_config = load_ai_config()
    ai_enabled = is_room_ai_enabled(room_id)
    with session_scope() as session:
        room = session.get(Room, room_id)
        if room is None:
            return "Room not found", 404
        assignment = _active_assignment_for_room(session, room_id)
        patient = assignment.patient if assignment else None
        latest_comfort = (
            session.query(ComfortPreference)
            .filter(ComfortPreference.room_id == room_id)
            .order_by(ComfortPreference.timestamp.desc())
            .first()
        )
        sensor_rows = _latest_sensor_rows(session, room_id)
        actuator_rows = _actuator_rows(session, room_id)
        return render_template(
            "patient_room.html",
            room=room,
            patient=patient,
            latest_comfort=latest_comfort,
            sensor_rows=sensor_rows,
            actuator_rows=actuator_rows,
            ai_config=ai_config,
            ai_enabled=ai_enabled,
        )


@hybrid_bp.route("/patient/rooms/<int:room_id>/live")
def patient_room_live(room_id: int):
    with session_scope() as session:
        room = session.get(Room, room_id)
        if room is None:
            return jsonify({"error": "Room not found"}), 404
        sensor_rows = _latest_sensor_rows(session, room_id)
        actuator_rows = _actuator_rows(session, room_id)
        latest_comfort = (
            session.query(ComfortPreference)
            .filter(ComfortPreference.room_id == room_id)
            .order_by(ComfortPreference.timestamp.desc())
            .first()
        )
        return jsonify(
            {
                "sensor_rows": [
                    {
                        "device_id": row["device_id"],
                        "location": row["location"],
                        "sensor_type": row["sensor_type"],
                        "uuid": row["uuid"],
                        "value": row["value"],
                        "value_display": row["value_display"],
                        "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                        "unit": row["unit"],
                    }
                    for row in sensor_rows
                ],
                "actuator_rows": actuator_rows,
                "latest_comfort": {
                    "temperature_main": latest_comfort.temperature_main if latest_comfort else None,
                    "temperature_toilet": latest_comfort.temperature_toilet if latest_comfort else None,
                    "light_intensity": latest_comfort.light_intensity if latest_comfort else None,
                    "sound_level": latest_comfort.sound_level if latest_comfort else None,
                    "airflow": latest_comfort.airflow if latest_comfort else None,
                },
            }
        )


@hybrid_bp.route("/patient/rooms/<int:room_id>/toilet", methods=["POST"])
def patient_go_to_toilet(room_id: int):
    with session_scope() as session:
        toilet_light = (
            session.query(Device)
            .filter(Device.room_id == room_id, Device.device_type == "toilet_light")
            .first()
        )
        now = _utc_now().replace(tzinfo=None)
        if toilet_light and toilet_light.toilet_light is not None:
            toilet_light.toilet_light.state = not bool(toilet_light.toilet_light.state)
            toilet_light.toilet_light.timestamp = now
    flash("Toilet light updated.")
    return redirect(url_for("hybrid_bp.patient_room", room_id=room_id))


@hybrid_bp.route("/patient/rooms/<int:room_id>/comfort", methods=["POST"])
def save_patient_comfort(room_id: int):
    patient_id = int(request.form["patient_id"]) if request.form.get("patient_id") else None
    if is_room_ai_enabled(room_id):
        try:
            return _save_ai_comfort_preference(room_id, patient_id, "hybrid_bp.patient_room")
        except Exception as exc:
            flash(f"AI mode failed: {exc}")
            return redirect(url_for("hybrid_bp.patient_room", room_id=room_id))
    return _save_comfort_preference(room_id, patient_id, "hybrid_bp.patient_room")


@hybrid_bp.route("/patient/rooms/<int:room_id>/ai-mode", methods=["POST"])
def toggle_patient_ai_mode(room_id: int):
    next_enabled = request.form.get("enabled") == "true"
    if next_enabled:
        patient_id = int(request.form["patient_id"]) if request.form.get("patient_id") else None
        try:
            response = _save_ai_comfort_preference(room_id, patient_id, "hybrid_bp.patient_room")
        except Exception as exc:
            set_room_ai_enabled(room_id, False)
            flash(f"AI mode could not be enabled: {exc}")
            return redirect(url_for("hybrid_bp.patient_room", room_id=room_id))
        set_room_ai_enabled(room_id, True)
        return response
    set_room_ai_enabled(room_id, False)
    flash(f"AI mode disabled for room {room_id}.")
    return redirect(url_for("hybrid_bp.patient_room", room_id=room_id))


@hybrid_bp.route("/admin/rooms/<int:room_id>/comfort", methods=["POST"])
def save_admin_comfort(room_id: int):
    patient_id = int(request.form["patient_id"]) if request.form.get("patient_id") else None
    return _save_comfort_preference(room_id, patient_id, "hybrid_bp.admin_room")
