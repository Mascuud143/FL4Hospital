# Cleanup Notes

The thesis AI story should stay narrow:

1. `ai_forecast`: time-based next-hour comfort prediction
2. `ai_state_to_outcome`: event-based comfort prediction after a trigger
3. Federated learning is the main method
4. Local baselines are optional comparison models, not the main contribution

## Main question

Do we need many AI files?

Probably no. We only need the files required to:

1. build rows
2. split train/test data
3. run federated training
4. optionally run one baseline for comparison
5. optionally run prediction/evaluation for thesis results

## `ai_forecast` keep

These are the core files for the main forecast idea:

- `ai_forecast/build_row.py`
- `ai_forecast/build_next_hour_rows.py`
- `ai_forecast/next_hour_schema.py`
- `ai_forecast/split_next_hour_by_room.py`
- `ai_forecast/fl_client.py`
- `ai_forecast/fl_server.py`
- `ai_forecast/fl_simulation.py`
- `ai_forecast/predict_from_weights.py`
- `ai_forecast/error_analysis.py`
- `ai_forecast/feature_importance.py`
- `ai_forecast/train_baseline_next_hour.py`

## `ai_forecast` maybe keep only if thesis explicitly compares architectures

These add extra model complexity. Keep them only if they are part of the final evaluation chapter:

- `ai_forecast/fl_client_lstm.py`
- `ai_forecast/fl_server_lstm.py`
- `ai_forecast/fl_simulation_lstm.py`
- `ai_forecast/fl_client_lstm_mlp.py`
- `ai_forecast/fl_server_lstm_mlp.py`
- `ai_forecast/fl_simulation_lstm_mlp.py`
- `ai_forecast/feature_importance_lstm_mlp.py`

If the thesis only needs one federated forecast model, these are unnecessary and should be removed.

## `ai_forecast` likely unnecessary

These remaining items do not look necessary for the final thesis codebase unless they are needed as saved results:

- `ai_forecast/fl_weights_sim/`
- `ai_forecast/fl_weights_sim_lstm_mlp/`
- `ai_forecast/splits_next_hour_3/`

These are artifacts, not core source files. They can be:

1. removed from the repo, or
2. moved to an archive/results folder, or
3. regenerated when needed

## Recommended minimal `ai_forecast` version

If we want the simplest version of the thesis repo, `ai_forecast` should contain only:

- row building
- schema
- room-based split
- one federated model pipeline
- one baseline
- one prediction/evaluation path

That means the ideal minimal set is:

- `build_row.py`
- `build_next_hour_rows.py`
- `next_hour_schema.py`
- `split_next_hour_by_room.py`
- `fl_client.py`
- `fl_server.py`
- `fl_simulation.py`
- `train_baseline_next_hour.py`
- `predict_from_weights.py`
- `error_analysis.py`
- `feature_importance.py`

In that version, the LSTM and hybrid LSTM+MLP files should be deleted.

## `ai_state_to_outcome` keep

This folder is already much cleaner. Keep:

- `ai_state_to_outcome/build_event_rows.py`
- `ai_state_to_outcome/schema.py`
- `ai_state_to_outcome/split_by_patient_stay.py`
- `ai_state_to_outcome/fl_client.py`
- `ai_state_to_outcome/fl_server.py`
- `ai_state_to_outcome/fl_simulation.py`
- `ai_state_to_outcome/train_baseline.py`
- `ai_state_to_outcome/predict_from_weights.py`
- `ai_state_to_outcome/error_analysis.py`
- `ai_state_to_outcome/feature_importance.py`

Generated folders here may also be archived later:

- `ai_state_to_outcome/fl_predictions/`
- `ai_state_to_outcome/fl_weights/`
- `ai_state_to_outcome/rows/`
- `ai_state_to_outcome/splits/`
- `ai_state_to_outcome/error_analysis/`
- `ai_state_to_outcome/feature_importance/`

## Important note about `main.py`

`main.py` still contains references to deleted or old AI flows.

It should be cleaned so it supports only:

1. `ai_forecast` current pipeline
2. `ai_state_to_outcome` current pipeline
3. the federated runs that are actually part of the thesis

## Recommended next cleanup step

Choose one of these directions:

1. Conservative cleanup:
   Keep LSTM and hybrid forecast files for now, only clean `main.py` and move artifacts out of the source folders.

2. Thesis-focused cleanup:
   Keep only one federated forecast implementation and remove the LSTM and hybrid forecast variants.

3. Final-minimal repo:
   Keep only the two thesis pipelines, one FL implementation per pipeline, and one baseline per pipeline.
