# Thesis Figure Prompts

Use the attached thesis figure as the visual style reference.

Style requirements for all prompts:

- match the attached figure's academic style
- light gray boxes with subtle borders
- thin dark connector lines and arrows
- clean white background
- restrained grayscale palette
- rounded rectangles
- minimal, professional, thesis-friendly layout
- no bright colors, no 3D effects, no decorative icons unless very subtle
- clean typography similar to the attached diagram
- balanced spacing and high readability
- landscape layout

## Prompt 1: Input -> Output Figure

Create a thesis-style academic diagram that matches the visual language of the attached reference figure.

The diagram should explain the next-hour comfort prediction task in a smart hospital.

Layout:

- left: one large rounded rectangle titled `Patient and Context at Time t`
- center: one large rounded rectangle titled `Federated Prediction Model`
- right: one large rounded rectangle titled `Predicted Comfort Settings at Time t + 1 hour`

Inside the left box, list:

- age
- gender
- diagnosis
- symptoms
- medication context
- time of day
- current room state
- recent comfort history

Inside the center box, list:

- MLP
- LSTM
- Hybrid LSTM + MLP
- Federated Learning

Inside the right box, list:

- main temperature
- toilet temperature
- light intensity
- sound level
- airflow

Use thin arrows from left to center and center to right.

Add a small caption at the bottom:
`The model predicts personalized room comfort settings from patient condition and contextual data.`

Keep the diagram minimal and structured, like the attached thesis ER-style figure.

## Prompt 2: Data Pipeline Figure

Create a clean thesis-style pipeline diagram matching the attached reference figure.

The goal is to explain how raw hospital data becomes federated training data for next-hour comfort prediction.

Layout from left to right.

Left side: stacked rounded boxes for raw data sources:

- patients.csv
- admissions.csv
- room_assignments.csv
- medications.csv
- visits.csv
- comfort_preferences.csv

Arrow to a central rounded box titled:
`Row Builder`

Under that title, small subtitle:
`build_next_hour_rows.py`

Inside or below the box, add:

- builds one row every 30 minutes
- input at time t
- target at time t + 1 hour

Then arrow to another box titled:
`Example Training Row`

Inside, show two sections:

Left subsection:
`Input at 14:30`
- patient profile
- symptom
- medication context
- time features
- current room state

Right subsection:
`Target at 15:30`
- y_temp_main
- y_temp_toilet
- y_light
- y_sound
- y_airflow
- y_any_change

Then arrow to another box titled:
`Time-Based Split by Room`

Subtitle:
`split_next_hour_by_room.py`

Inside list:

- sort rows by time
- train = first 80%
- test = last 20%

Then arrow to 4 small boxes:

- Room 1
- Room 2
- Room 3
- Room N

Add a small note below:
`Each room becomes one federated client`

Make it look like a thesis systems diagram in the same family as the attached figure.

## Prompt 3: Model Comparison Figure

Create a thesis-style comparison diagram matching the grayscale academic style of the attached reference image.

The diagram should compare three model architectures used in federated next-hour comfort prediction.

Use three columns of rounded boxes with consistent spacing.

Column 1 title:
`MLP`

Inside the MLP column, show:

- input: one single feature row at time t
- feedforward neural network
- predicts:
  - main temperature
  - toilet temperature
  - light
  - sound
  - airflow

Add small note:
`Uses one snapshot row`

Column 2 title:
`LSTM`

Inside the LSTM column, show:

- input sequence:
  - t-90m
  - t-60m
  - t-30m
  - t
- LSTM network
- predicts:
  - main temperature
  - toilet temperature
  - light
  - sound
  - airflow

Add small note:
`Uses short temporal history`

Column 3 title:
`Hybrid LSTM + MLP`

Inside the Hybrid column, show two internal branches:

- branch 1:
  - current row
  - MLP
  - predicts y_any_change

- branch 2:
  - past sequence
  - LSTM
  - predicts target comfort settings

Then merge the branches into:

- decision to change
- predicted next-hour settings

Use thin arrows and subtle labels.
Avoid colorful deep-learning artwork.
Keep it formal, minimal, and consistent with the attached thesis figure.

## Prompt 4: Federated Learning Figure

Create a detailed thesis-style federated learning workflow diagram using the attached figure as the exact visual inspiration for tone, spacing, grayscale palette, and box style.

Top center: rounded box titled:
`Global Federated Model`

Below it, horizontally arrange 4 client boxes:

- Room 1
- Room 2
- Room 3
- Room N

Inside each room box include:

- local train split
- local test split
- private room data
- local model training

Draw arrows from the global model down to each room labeled:
`send global weights`

Under each room box, add a smaller box:
`fit()`

Inside list:

- local epochs
- train on room-specific data
- no raw data shared

Draw arrows from all room boxes back to a central lower box titled:
`FedAvg Aggregation`

Inside list:

- collect local model updates
- average parameters
- form new global model

Then arrow back upward to:
`Updated Global Model`

Add a side box titled:
`Evaluation`

Inside list:

- aggregate MAE and RMSE
- aggregate airflow metrics
- aggregate change detection metrics
- save global weights

Add a small loop note:
`Repeat for multiple communication rounds`

Keep the appearance close to the attached thesis database-style figure, but adapted for an AI workflow diagram.

## Prompt 5: MLP Training Flow Figure

Create a thesis-style explanatory diagram in the same grayscale, rounded-box style as the attached reference figure.

The diagram should explain how one federated room client trains the MLP forecast model.

Layout left to right.

Box 1 title:
`Room k Local Data`

Inside:

- next_hour_train.csv
- rows for one room only
- X = input features
- y = target comfort settings

Arrow to box 2:

`MLPRegressor`

Inside:

- hidden layers: 128, 64, 32
- ReLU activation
- multi-output prediction

Arrow to box 3:

`Forward Pass`

Inside:

- input row at time t
- predict:
  - y_temp_main
  - y_temp_toilet
  - y_light
  - y_sound
  - y_airflow

Arrow to box 4:

`Loss Computation`

Inside:

- compare y_pred with y_true
- squared error loss

Arrow to box 5:

`Backpropagation and Update`

Inside:

- compute gradients
- Adam optimizer
- update local weights

Arrow to box 6:

`Send Updated Weights to Server`

Add a small note below the whole diagram:
`Each room trains locally, then shares only model parameters with the federated server.`

Keep it highly readable and thesis-appropriate.

## Prompt 6: LSTM Sequence Training Figure

Create a thesis-style academic diagram matching the attached grayscale figure style.

The diagram should explain how the LSTM model is trained using short sequences of past rows.

Layout left to right.

Box 1 title:
`Ordered Room Rows`

Inside:

- 14:00
- 14:30
- 15:00
- 15:30
- 16:00

Arrow to box 2:

`Sequence Builder`

Subtitle:
`build_sequence_arrays()`

Inside show:

- sequence length = 4
- sample 1 = [14:00, 14:30, 15:00, 15:30]
- sample 2 = [14:30, 15:00, 15:30, 16:00]

Arrow to box 3:

`LSTM Model`

Inside:

- sequence input
- temporal hidden state
- output head

Arrow to box 4:

`Prediction`

Inside:

- next-hour temperature
- next-hour toilet temperature
- next-hour light
- next-hour sound
- next-hour airflow

Arrow to box 5:

`Training Step`

Inside:

- forward pass
- MSE loss
- backward pass
- Adam update

Add a small note:
`The LSTM uses a fixed recent history, not the entire past.`

Maintain the same formal thesis-diagram appearance as the attached reference.
