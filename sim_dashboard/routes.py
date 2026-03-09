from flask import Blueprint, render_template

from persistence.database import session_scope
from persistence.models.room import Room
from persistence.models.patient import Patient
from persistence.models.admission import Admission
from persistence.models.medication import Medication
from persistence.models.room_assignment import RoomAssignment
from persistence.models.visit import Visit
from persistence.models.data import Data
from persistence.models.utility_usage import UtilityUsage
from persistence.models.comfort_preference import ComfortPreference
from persistence.models.sensor import Sensor
from persistence.models.device import Device
from datetime import timedelta, datetime, time
from statistics import mean, pstdev
import math
from sqlalchemy import func

# import config
from simulation_batch.config import START_DATE, DAYS

sim_bp = Blueprint("sim_bp", __name__)


def _fmt_time(value: datetime | None) -> str:
    if not value:
        return "--:--"
    return value.strftime("%H:%M")


def _fmt_dt(value: datetime | None) -> str:
    if not value:
        return "--"
    return value.strftime("%Y-%m-%d %H:%M")


def _in_window(ts: datetime | None, start: datetime | None, end: datetime | None) -> bool:
    if not ts or not start:
        return False
    if ts < start:
        return False
    if end and ts > end:
        return False
    return True


def _assignment_for_time(assignments: list[dict], ts: datetime | None) -> dict | None:
    if not ts:
        return None
    for a in assignments:
        a_start = a.get("start_time")
        a_end = a.get("end_time")
        if a_start and ts < a_start:
            continue
        if a_end and ts > a_end:
            continue
        return a
    return None


def _build_dist_svg(values: list[int], *, title: str, x_label: str) -> str | None:
    if not values:
        return None

    v_vals = [v for v in values if v is not None]
    if len(v_vals) < 2:
        return None

    w, h = 700, 240
    pad = 30
    bins = 20
    min_v = min(v_vals)
    max_v = max(v_vals)
    if min_v == max_v:
        min_v -= 1
        max_v += 1

    bin_w = (max_v - min_v) / bins
    counts = [0] * bins
    for v in v_vals:
        idx = int((v - min_v) / bin_w)
        if idx == bins:
            idx -= 1
        counts[idx] += 1

    max_count = max(counts) if counts else 1
    plot_w = w - 2 * pad
    plot_h = h - 2 * pad

    def x_for(val: float) -> float:
        return pad + (val - min_v) / (max_v - min_v) * plot_w

    def y_for_count(c: float) -> float:
        return pad + (1 - c / max_count) * plot_h

    mu = mean(v_vals)
    sigma = pstdev(v_vals) or 1.0
    curve_pts = []
    steps = 80
    for i in range(steps + 1):
        x = min_v + (max_v - min_v) * i / steps
        pdf = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
        curve_pts.append((x_for(x), y_for_count(pdf * max_count * sigma)))

    bars = []
    for i, c in enumerate(counts):
        x0 = x_for(min_v + i * bin_w)
        x1 = x_for(min_v + (i + 1) * bin_w)
        y = y_for_count(c)
        bars.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{(x1 - x0 - 1):.1f}" height="{(pad + plot_h - y):.1f}" fill="#8fbcd4" />')

    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in curve_pts)

    svg = f"""
<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff" />
  <rect x="{pad}" y="{pad}" width="{plot_w}" height="{plot_h}" fill="#f8f8f8" stroke="#ddd" />
  {''.join(bars)}
  <polyline fill="none" stroke="#cc2f2f" stroke-width="2" points="{poly}" />
  <text x="{pad}" y="18" font-size="12" fill="#333">{title}</text>
  <text x="{pad}" y="{h - 6}" font-size="12" fill="#333">{x_label}</text>
  <text x="{w - pad - 140}" y="{pad - 8}" font-size="12" fill="#333">Mean={mu:.1f}, SD={sigma:.1f}</text>
</svg>
""".strip()
    return svg


@sim_bp.route("/")
def rooms():
    """
    Display all rooms.
    Uses plain dicts to avoid DetachedInstanceError.
    """
    with session_scope() as session:
        rooms = session.query(Room).all()
        female_heights = [r[0] for r in session.query(Patient.height).filter(Patient.gender == "Female").all()]
        male_heights = [r[0] for r in session.query(Patient.height).filter(Patient.gender == "Male").all()]
        female_ages = [
            r[0]
            for r in session.query(Admission.age)
            .join(Patient, Admission.patient_id == Patient.patient_id)
            .filter(Patient.gender == "Female")
            .all()
        ]
        male_ages = [
            r[0]
            for r in session.query(Admission.age)
            .join(Patient, Admission.patient_id == Patient.patient_id)
            .filter(Patient.gender == "Male")
            .all()
        ]

        # Convert ORM objects → plain dicts (safe for Jinja)
        room_data = [
            {
                "room_id": room.room_id,
                "room_number": room.room_number,
            }
            for room in rooms
        ]
    
        # calcuate simualtion period with start date + days USE timedelta
        simulation_period = f"{START_DATE} to {(START_DATE + timedelta(days=DAYS)).strftime('%Y-%m-%d')}"

        readmission_patients = (
            session.query(Admission.patient_id)
            .group_by(Admission.patient_id)
            .having(func.count(Admission.admission_id) > 1)
            .count()
        )

        # get simulaton info, how many patients, how many rooms, how many devices etc, the date start etc
        simulation_info = {
            "total_rooms": len(room_data),
            "total_patients": session.query(Patient).count(),
            "total_admissions": session.query(Admission).count(),
            "readmissions": readmission_patients,
            "total_medications": session.query(Medication).count(),
            "total_visits": session.query(Visit).count(),
            "simulation_period": simulation_period,
        }
        height_svg_female = _build_dist_svg(
            female_heights, title="Height Distribution - Female", x_label="Heights (cm)"
        )
        height_svg_male = _build_dist_svg(
            male_heights, title="Height Distribution - Male", x_label="Heights (cm)"
        )
        age_svg_female = _build_dist_svg(
            female_ages, title="Age Distribution - Female", x_label="Age (years)"
        )
        age_svg_male = _build_dist_svg(
            male_ages, title="Age Distribution - Male", x_label="Age (years)"
        )

    return render_template(
        "rooms.html",
        rooms=room_data,
        simulation_info=simulation_info,
        height_svg_female=height_svg_female,
        height_svg_male=height_svg_male,
        age_svg_female=age_svg_female,
        age_svg_male=age_svg_male,
    )

@sim_bp.route("/rooms/<int:room_id>")
def room_detail(room_id):
    """
    Display details for a specific room, including patient assignments.
    """

    print(f"Fetching details for room_id={room_id}")  # Debug log
    with session_scope() as session:
        room = session.query(Room).get(room_id)

        if not room:
            return "Room not found", 404


        # get room devices, sensors etc
        devices = room.devices  # Assuming a relationship is defined
        print(f"Devices in room {room_id}: {devices}")  # Debug log

        # Convert devices to match template expectations
        device_data = [
            {
                "device_id": device.device_id,
                "device_type": device.device_type,
            }
            for device in devices
        ]

        sensors = []
        for device in devices:
            for sensor in device.sensors:
                sensors.append(
                    {
                        "sensor_id": sensor.sensor_id,
                        "sensor_type": sensor.sensor_type,
                        "unit": sensor.unit,
                        "device_id": device.device_id,
                    }
                )

        comfort_rows = (
            session.query(ComfortPreference)
            .filter(ComfortPreference.room_id == room_id)
            .order_by(ComfortPreference.timestamp.desc())
            .all()
        )
        comfort_preferences = []
        for pref in comfort_rows:
            window_start = pref.timestamp - timedelta(hours=1)
            window_end = pref.timestamp + timedelta(hours=1)
            sensor_windows = []
            for s in sensors:
                before_rows = (
                    session.query(Data)
                    .filter(
                        Data.sensor_id == s["sensor_id"],
                        Data.timestamp >= window_start,
                        Data.timestamp <= pref.timestamp,
                    )
                    .order_by(Data.timestamp.desc())
                    .limit(10)
                    .all()
                )
                after_rows = (
                    session.query(Data)
                    .filter(
                        Data.sensor_id == s["sensor_id"],
                        Data.timestamp >= pref.timestamp,
                        Data.timestamp <= window_end,
                    )
                    .order_by(Data.timestamp.asc())
                    .limit(10)
                    .all()
                )
                sensor_windows.append(
                    {
                        **s,
                        "before_rows": [
                            {"value": r.value, "timestamp": r.timestamp}
                            for r in before_rows
                        ],
                        "after_rows": [
                            {"value": r.value, "timestamp": r.timestamp}
                            for r in after_rows
                        ],
                    }
                )
            comfort_preferences.append(
                {
                    "comfort_pref_id": pref.comfort_pref_id,
                    "timestamp": pref.timestamp,
                    "temperature_main": pref.temperature_main,
                    "temperature_toilet": pref.temperature_toilet,
                    "light_intensity": pref.light_intensity,
                    "sound_level": pref.sound_level,
                    "airflow": pref.airflow,
                    "source": pref.source,
                    "patient_name": pref.patient.name if pref.patient else None,
                    "sensor_windows": sensor_windows,
                }
            )

        utility_rows = (
            session.query(UtilityUsage)
            .filter(UtilityUsage.room_id == room_id)
            .order_by(UtilityUsage.start_time.desc())
            .limit(10)
            .all()
        )
        utility_usages = [
            {
                "category": row.category,
                "power_consumption": row.power_consumption,
                "water_consumption": row.water_consumption,
                "start_time": row.start_time,
                "end_time": row.end_time,
                "device_id": row.device_id,
            }
            for row in utility_rows
        ]

        ventilation_data = []
        for device in devices:
            vent = device.ventilation
            if vent:
                ventilation_data.append(
                    {
                        "device_id": device.device_id,
                        "mode": vent.mode,
                        "level": vent.level,
                        "timestamp": vent.timestamp,
                    }
                )

        ventilation_data.sort(
            key=lambda v: v["timestamp"] or 0,
            reverse=True,
        )

        room_data = {
            "room_id": room.room_id,
            "room_number": room.room_number,
            "devices": device_data,
        }
        
        # Convert assignments separately to match template expectations
        assignments = [
            {
                "assignment_id": assignment.assignment_id,
                "patient": {
                    "patient_id": assignment.patient.patient_id,
                    "name": assignment.patient.name,
                },
                "start_time": assignment.start_time,
                "end_time": assignment.end_time,
            }
            for assignment in room.assignments
        ]

    return render_template(
        "room_detail.html",
        room=room_data,
        assignments=assignments,
        comfort_preferences=comfort_preferences,
        utility_usages=utility_usages,
        ventilation_data=ventilation_data,
    )


@sim_bp.route("/patients")
def patients():
    """
    Display all patients.
    """
    with session_scope() as session:
        patients = session.query(Patient).all()

        # Convert ORM objects → plain dicts (safe for Jinja)
        patient_data = []
        for patient in patients:
            latest_adm = None
            if patient.admissions:
                latest_adm = max(patient.admissions, key=lambda a: a.admitted_at or 0)
            patient_data.append(
                {
                    "patient_id": patient.patient_id,
                    "name": patient.name,
                    "age": latest_adm.age if latest_adm else None,
                    "admission_date": latest_adm.admitted_at if latest_adm else None,
                    "release_date": latest_adm.discharged_at if latest_adm else None,
                }
            )

    return render_template("patients.html", patients=patient_data)


@sim_bp.route("/ai-suggestion")
def ai_suggestion():
    """
    Static explainer page for AI comfort suggestion input/output.
    """
    return render_template("ai_suggestion.html")


@sim_bp.route("/patients/<int:patient_id>")
def patient_detail(patient_id):
    """
    Display details for a specific patient, including comfort preferences.
    """

    with session_scope() as session:
        patient = session.query(Patient).get(patient_id)
        if not patient:
            return "Patient not found", 404

        latest_adm = None
        if patient.admissions:
            latest_adm = max(patient.admissions, key=lambda a: a.admitted_at or 0)

        # Convert ORM object → plain dict (safe for Jinja)
        patient_data = {
            "patient_id": patient.patient_id,
            "name": patient.name,
            "gender": patient.gender,
            "height": patient.height,
            "ethnicity": patient.ethnicity,
            "age": latest_adm.age if latest_adm else None,
            "weight": latest_adm.weight if latest_adm else None,
            "current_diagnosis": latest_adm.current_diagnosis if latest_adm else None,
            "admission_date": latest_adm.admitted_at if latest_adm else None,
            "release_date": latest_adm.discharged_at if latest_adm else None,
        }

        room_lookup = {room.room_id: room.room_number for room in session.query(Room).all()}

        admissions_sorted = sorted(
            patient.admissions,
            key=lambda a: (a.admitted_at is not None, a.admitted_at),
            reverse=True,
        )
        admissions = [
            {
                "admission_id": adm.admission_id,
                "admitted_at": adm.admitted_at,
                "discharged_at": adm.discharged_at,
                "age": adm.age,
                "weight": adm.weight,
                "current_diagnosis": adm.current_diagnosis,
            }
            for adm in admissions_sorted
        ]

        medications = sorted(
            patient.medications,
            key=lambda m: (m.medication_time is not None, m.medication_time),
            reverse=True,
        )
        visits = sorted(
            patient.visits,
            key=lambda v: (v.visit_time is not None, v.visit_time),
            reverse=True,
        )
        comforts = sorted(
            patient.comfort_preferences,
            key=lambda c: (c.timestamp is not None, c.timestamp),
            reverse=True,
        )
        room_assignments = sorted(
            patient.assignments,
            key=lambda r: (r.start_time is not None, r.start_time),
            reverse=True,
        )

        admission_views = []
        for adm in admissions_sorted:
            adm_assignments = sorted(
                [a for a in room_assignments if a.admission_id == adm.admission_id],
                key=lambda r: (r.start_time is not None, r.start_time),
            )
            admission_start = adm.admitted_at
            admission_end = adm.discharged_at

            # Determine room-context utility events during this stay.
            utility_events = []
            for assign in adm_assignments:
                q = session.query(UtilityUsage).filter(UtilityUsage.room_id == assign.room_id)
                assign_start = max(filter(None, [assign.start_time, admission_start]))
                assign_end = min(
                    [x for x in [assign.end_time, admission_end] if x is not None],
                    default=None,
                )
                if assign_start:
                    q = q.filter(UtilityUsage.start_time >= assign_start)
                if assign_end:
                    q = q.filter(UtilityUsage.start_time <= assign_end)
                utility_events.extend(q.order_by(UtilityUsage.start_time.asc()).all())

            # Compute day span from admission window and all event timestamps.
            event_times = [t for t in [admission_start, admission_end] if t is not None]
            event_times += [m.medication_time for m in medications if _in_window(m.medication_time, admission_start, admission_end)]
            event_times += [v.visit_time for v in visits if _in_window(v.visit_time, admission_start, admission_end)]
            event_times += [c.timestamp for c in comforts if _in_window(c.timestamp, admission_start, admission_end)]
            event_times += [a.start_time for a in adm_assignments if a.start_time]
            event_times += [a.end_time for a in adm_assignments if a.end_time]
            event_times += [u.start_time for u in utility_events if u.start_time]
            event_times = [t for t in event_times if t is not None]
            if not event_times:
                continue

            span_start = min(event_times).date()
            span_end = max(event_times).date()
            day_map = {}
            cursor = span_start
            while cursor <= span_end:
                day_map[cursor] = {
                    "date_key": cursor.isoformat(),
                    "label": cursor.strftime("%A, %d %b %Y"),
                    "events": [],
                    "comfort_events": [],
                    "environment_events": [],
                    "clinical_events": [],
                    "room_events": [],
                }
                cursor += timedelta(days=1)

            def add_event(ts, category, title, details, target_day=None, badge_class=None, badge_label=None):
                day_key = target_day if target_day else (ts.date() if ts else span_start)
                if day_key not in day_map:
                    return
                event = {
                    "time": _fmt_time(ts),
                    "timestamp_str": _fmt_dt(ts),
                    "timestamp": ts,
                    "category": category,
                    "badge_class": badge_class if badge_class else category,
                    "badge_label": badge_label if badge_label else category,
                    "title": title,
                    "details": details,
                }
                day_map[day_key]["events"].append(event)
                bucket = f"{category}_events"
                if bucket in day_map[day_key]:
                    day_map[day_key][bucket].append(event)

            add_event(admission_start, "room", "Admission started", _fmt_dt(admission_start))
            if admission_end:
                add_event(admission_end, "room", "Discharged", _fmt_dt(admission_end))

            for assign in adm_assignments:
                room_number = room_lookup.get(assign.room_id, f"Room {assign.room_id}")
                add_event(
                    assign.start_time,
                    "room",
                    f"Moved to {room_number}",
                    f"Assignment #{assign.assignment_id} (start)",
                )
                if assign.end_time:
                    add_event(
                        assign.end_time,
                        "room",
                        f"Left {room_number}",
                        f"Assignment #{assign.assignment_id} (end)",
                    )

            for c in comforts:
                if not _in_window(c.timestamp, admission_start, admission_end):
                    continue
                active_assignment = _assignment_for_time(
                    [
                        {"start_time": a.start_time, "end_time": a.end_time, "room_id": a.room_id}
                        for a in adm_assignments
                    ],
                    c.timestamp,
                )
                room_hint = ""
                if active_assignment:
                    room_id = active_assignment["room_id"]
                    room_name = room_lookup.get(room_id, f"Room {room_id}")
                    room_hint = f" in {room_name}"
                comfort_text = (
                    f"Main {c.temperature_main} C, Toilet {c.temperature_toilet if c.temperature_toilet is not None else '--'} C, "
                    f"Light {c.light_intensity if c.light_intensity is not None else '--'}, "
                    f"Sound {c.sound_level if c.sound_level is not None else '--'}, "
                    f"Airflow {'On' if c.airflow else 'Off'}{room_hint}"
                )
                add_event(c.timestamp, "comfort", "Comfort setting updated", comfort_text)

            for m in medications:
                if not _in_window(m.medication_time, admission_start, admission_end):
                    continue
                med_text = (
                    f"{m.drug_name} | Dose {m.dose if m.dose else '--'} | "
                    f"Route {m.route if m.route else '--'} | Status {m.status if m.status else '--'}"
                )
                add_event(
                    m.medication_time,
                    "clinical",
                    "Medication",
                    med_text,
                    badge_class="clinical-medication",
                    badge_label="medication",
                )

            for v in visits:
                if not _in_window(v.visit_time, admission_start, admission_end):
                    continue
                visit_text = (
                    f"Temp {v.body_temperature if v.body_temperature is not None else '--'} C | "
                    f"BP {v.blood_pressure if v.blood_pressure else '--'} | "
                    f"Symptoms {v.symptoms if v.symptoms else '--'}"
                )
                add_event(
                    v.visit_time,
                    "clinical",
                    "Visit",
                    visit_text,
                    badge_class="clinical-visit",
                    badge_label="visit",
                )

            for u in utility_events:
                room_number = room_lookup.get(u.room_id, f"Room {u.room_id}")
                env_text = (
                    f"{room_number} | Power {u.power_consumption if u.power_consumption is not None else '--'} kWh | "
                    f"Water {u.water_consumption if u.water_consumption is not None else '--'} L"
                )
                add_event(
                    u.start_time,
                    "environment",
                    f"Environment usage: {u.category}",
                    env_text,
                )

            days = []
            for d in sorted(day_map.keys()):
                day = day_map[d]
                day["events"] = sorted(
                    day["events"],
                    key=lambda e: (e["timestamp"] is None, e["timestamp"]),
                )

                day_start = datetime.combine(d, time.min)
                day_end = datetime.combine(d, time.max)
                room_ids_for_day = set()
                for assign in adm_assignments:
                    assign_start = assign.start_time
                    assign_end = assign.end_time
                    if not assign_start:
                        continue
                    if admission_end and (assign_end is None or admission_end < assign_end):
                        assign_end = admission_end
                    if assign_start > day_end:
                        continue
                    if assign_end and assign_end < day_start:
                        continue
                    room_ids_for_day.add(assign.room_id)

                sensor_rows = []
                if room_ids_for_day:
                    data_rows = (
                        session.query(Data, Sensor, Device)
                        .join(Sensor, Data.sensor_id == Sensor.sensor_id)
                        .join(Device, Sensor.device_id == Device.device_id)
                        .filter(
                            Device.room_id.in_(room_ids_for_day),
                            Data.timestamp >= day_start,
                            Data.timestamp <= day_end,
                        )
                        .order_by(Data.timestamp.desc())
                        .all()
                    )
                    sensor_rows = [
                        {
                            "timestamp": _fmt_dt(row.timestamp),
                            "room_id": device.room_id,
                            "sensor_type": sensor.sensor_type,
                            "value": row.value,
                            "unit": sensor.unit,
                        }
                        for row, sensor, device in data_rows
                    ]

                day["sensor_rows"] = sensor_rows
                day["sensor_count"] = len(sensor_rows)
                day["sensor_types"] = sorted({row["sensor_type"] for row in sensor_rows if row.get("sensor_type")})
                day["sensor_rooms"] = sorted({row["room_id"] for row in sensor_rows if row.get("room_id") is not None})
                days.append(day)

            admission_views.append(
                {
                    "admission_id": adm.admission_id,
                    "admitted_at": adm.admitted_at,
                    "discharged_at": adm.discharged_at,
                    "age": adm.age,
                    "weight": adm.weight,
                    "current_diagnosis": adm.current_diagnosis,
                    "days": days,
                    "assignment_count": len(adm_assignments),
                }
            )

        active_admission_id = admission_views[0]["admission_id"] if admission_views else None

    return render_template(
        "patient_detail.html",
        patient=patient_data,
        comforts=comforts,
        admissions=admissions,
        medications=medications,
        visits=visits,
        room_assignments=room_assignments,
        admission_views=admission_views,
        active_admission_id=active_admission_id,
    )
