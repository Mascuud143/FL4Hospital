# Federated Comfort Training Guide (Simple + Concrete)

This file explains exactly how to prepare your current simulated data and train federated models.

## 1) Big Picture: What Federated Learning Is

Federated Learning (FL) is a way to train one shared model across many data owners without moving raw data to one central database.

In classic centralized ML:

1. All hospital data is copied into one place.
2. One model is trained on that centralized data.

In federated ML:

1. Data stays local at each client (here: room or room-group).
2. Server sends current model to clients.
3. Clients train locally on their own rows.
4. Clients send model updates (weights/gradients), not raw patient records.
5. Server aggregates updates into a new global model.

This cycle repeats in rounds.

### 1.1) Why FL suits your hospital comfort problem

Your use case is sensitive clinical context:

1. Patient profile
2. Symptoms (from visits)
3. Medication timing
4. Comfort choices over time

FL fits because:

1. Privacy: raw patient-level data does not need to leave local client datasets.
2. Governance: aligns better with hospital data sharing constraints.
3. Realism: different rooms/wards have different patterns, and FL is designed for multiple heterogeneous data silos.

Even though your current data is simulated, using FL now gives you a pipeline that can later transfer to real hospital deployments.

### 1.2) What Flower is (and why use it)

Flower is the FL framework/orchestration layer.

Think of Flower as the training coordinator:

1. Starts server and clients.
2. Distributes model parameters to selected clients each round.
3. Collects client updates.
4. Applies aggregation strategy (FedAvg, FedProx, etc.).

Flower does not force your model type. You can use PyTorch models (Model A, Model B) and Flower handles federated communication and round control.

### 1.3) What a "strategy" means in FL

A strategy is the server-side rulebook for:

1. Which clients participate each round.
2. How local updates are aggregated.
3. Optional constraints/regularization for local training.

So when you pick a strategy, you are choosing how collaboration happens across clients.

### 1.4) FedAvg vs FedProx (bigger picture)

FedAvg:

1. Most common baseline strategy.
2. Server averages client updates (usually weighted by client data size).
3. Works well when clients are similar.

FedProx:

1. Extension of FedAvg for heterogeneous (non-IID) clients.
2. Adds a proximal term in local training to keep each client from drifting too far from global weights.
3. Often more stable when clients differ a lot.

For your setup, room clients differ in patient mix, symptom timelines, and event frequencies, so FedProx is a strong default with FedAvg as baseline comparison.

### 1.5) Your prediction objective in this FL setup

At each decision time `t` (for example every 30 minutes), predict:

1. Model A: Should comfort settings change soon?
2. Model B: If yes, what should the new settings be, and when should they happen?

Simple example:

1. At `14:00`, inputs say symptom pain, last medication 40 minutes ago, and current comfort state.
2. Model A predicts change needed in next 60 minutes.
3. Model B predicts new settings and `y_when_minutes=20`.
4. System applies change at `14:20`.

That is the full big-picture loop before we get into table-level data preparation.

## 2) Tables You Already Have

Use these CSV files:

1. `filestorage/room_assignments.csv`
2. `filestorage/comfort_preferences.csv`
3. `filestorage/patients.csv`
4. `filestorage/medications.csv`
5. `filestorage/visits.csv` (symptoms source)

### What each table contributes

1. `room_assignments.csv`: tells who is in which room and when.
2. `comfort_preferences.csv`: tells the true comfort changes (temp/light/sound/airflow + timestamp).
3. `patients.csv`: stable patient profile.
4. `medications.csv`: medication timing and type.
5. `visits.csv`: symptom information over time.

## 3) Build Decision-Time Rows (Every 30 min)

For each assignment row, create timestamps from `start_time` to `end_time` every 30 minutes.

At each timestamp `t`, build one input row using only information known at or before `t`.

### Input features at time `t`

1. Patient features: age, height, gender, etc.
2. Time features: hour of day, weekday, day-of-stay.
3. Medication features: time since last medication, last medication type.
4. Symptom features: latest symptom from visits before `t`.
5. Current comfort state: latest comfort values before `t`:
   - `curr_temp`
   - `curr_light`
   - `curr_sound`
   - `curr_airflow`
   - `minutes_since_last_change`

## 4) Labels for Model A and B

Choose look-ahead window `H = 60` minutes.

For each row at time `t`, search for comfort changes in `(t, t+60m]`.

### Model A label (`y_event`)

1. If at least one comfort change is found: `y_event = 1`
2. Otherwise: `y_event = 0`

### Model B labels (only when `y_event = 1`)

Take the first comfort change in the window and set:

1. `y_temp` = event temperature
2. `y_light` = event light
3. `y_sound` = event sound
4. `y_airflow` = event airflow
5. `y_when_minutes` = minutes from `t` to event time

If `y_event = 0`, Model B labels are empty and ignored.

## 5) Datasets To Save (Expanded)

You will save two training files:

1. `model_a_rows.csv`
2. `model_b_rows.csv`

Also include `client_id = room_id` in both files.

### `model_a_rows.csv` (all decision rows)

Contains every decision timestamp row, including no-change rows.

Example row:

1. `client_id=12`
2. `patient_id=17`
3. `t=2026-01-10 14:00`
4. `age=67`, `height=172`, `symptom=pain`, `minutes_since_last_med=40`
5. `curr_temp=22.0`, `curr_light=20`, `curr_sound=18`, `curr_airflow=False`
6. `y_event=1`

Another example where no change happens:

1. same type of inputs
2. `y_event=0`

This file trains Model A.

### `model_b_rows.csv` (only event rows)

Contains only rows where `y_event=1`.

Example row:

1. same input features as above
2. `y_temp=21.5`
3. `y_light=15`
4. `y_sound=12`
5. `y_airflow=True`
6. `y_when_minutes=20`

This file trains Model B.

### Why two files?

1. Model A needs both positives and negatives.
2. Model B should learn only from true change events.

## 6) Train/Validation Split (Expanded)

Do split per client (per room), by time order.

For each `room_id`:

1. sort rows by timestamp `t`
2. first 80% -> train
3. last 20% -> validation/test

Do **not** random shuffle across time.

Reason: your problem is temporal. Random shuffling leaks future patterns into training.

Simple example for room 12:

1. rows from Jan to Oct -> train
2. rows from Nov to Dec -> validation/test

## 7) Federated Training Flow (Expanded)

Client = room (`room_id`).

Server has global Model A and global Model B.

One round means:

1. Server picks some room clients (for example 20%).
2. Server sends current global weights to those rooms.
3. Each selected room trains locally:
   - Model A on that room's `model_a_rows` train split
   - Model B on that room's `model_b_rows` train split
4. Each room sends updated weights to server.
5. Server aggregates updates (FedProx recommended).
6. New global models are produced.

Repeat this for `N` rounds (for example 100).

## 7.1) Why FedProx (Strategy Explanation)

FedProx is a federated optimization strategy that modifies local training so each client does not drift too far from the global model.

In simple terms:

1. Normal local training (FedAvg) lets each room optimize freely on its own data.
2. With non-IID rooms, local models can move in very different directions.
3. FedProx adds a small penalty that keeps local updates closer to the current global model.

### Why this matters in your project

Your room clients are naturally different:

1. Different patient mixes
2. Different symptom patterns
3. Different medication timing distributions
4. Different number of comfort events

So room data is non-IID. FedProx usually stabilizes training in this situation.

### Practical effect

1. Fewer unstable jumps between rounds
2. Better convergence when client data is heterogeneous
3. Often better global performance than plain FedAvg in non-IID settings

### FedProx parameter (`mu`)

`mu` controls how strongly local models are pulled toward global weights.

1. `mu = 0` means no proximal effect (equivalent to FedAvg behavior).
2. Small `mu` means light stabilization.
3. Large `mu` can over-restrict local learning.

Recommended start:

1. Try `mu` in `{0.001, 0.01, 0.05}`.
2. Keep the one with best validation performance.
3. If training is unstable, increase `mu`.
4. If learning is too slow/underfitting, decrease `mu`.

### Suggested baseline comparison

Run two experiments:

1. FedAvg baseline
2. FedProx (same settings + chosen `mu`)

Use the same data split and metrics, then compare fairly.

## 8) What Gets Saved

After final round save:

1. `model_a_global.pt`
2. `model_b_global.pt`
3. preprocessing objects:
   - encoder mappings
   - scalers
   - category maps

Local room models are temporary by default unless you explicitly save checkpoints.

## 9) Runtime Inference (After Training)

Every 30 minutes in each room:

1. Build current feature row.
2. Run Model A.
3. If `A=0` -> keep current settings.
4. If `A=1` -> run Model B.
5. Apply/schedule changes at `now + y_when_minutes`.

Simple runtime example:

1. now = `14:00`
2. Model A says change needed.
3. Model B predicts:
   - temp `21.5`
   - light `15`
   - sound `12`
   - airflow `True`
   - `y_when_minutes=20`
4. System schedules change at `14:20`.

## 10) Suggested First Experiment Config

1. Decision interval: 30 min
2. Horizon `H`: 60 min
3. Client: room
4. Strategy: FedProx
5. Rounds: 100
6. Local epochs: 2
7. Batch size: 64

This is a strong first baseline for your current simulated dataset.
