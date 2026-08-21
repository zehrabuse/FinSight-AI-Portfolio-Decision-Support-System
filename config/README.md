# Configuration

This directory contains the configuration and metadata files required to understand and correctly interface with the final DQN model.

The configuration files document the model structure, input state definition, preprocessing information, and action mapping.

---

## Directory Structure

```text
config/
├── metadata.json
├── actions.json
└── features.json
```

---

## `metadata.json`

`metadata.json` contains the general technical information of the final model.

It includes:

* Model name
* Model version
* Model type
* State size
* Action size
* Framework
* Export format
* ONNX opset
* Training information
* Network architecture
* Portfolio asset categories
* Model purpose

### Model Summary

| Property      | Value                |
| ------------- | -------------------- |
| Model         | `final_model`        |
| Model Type    | Deep Q-Network (DQN) |
| State Size    | 16                   |
| Action Size   | 9                    |
| Framework     | PyTorch              |
| Export Format | ONNX                 |
| ONNX Opset    | 17                   |
| Input Type    | `float32`            |

The DQN architecture is:

```text
16
↓
128
↓
128
↓
9
```

---

## `features.json`

`features.json` defines the 16-dimensional state vector used as input to the DQN model.

For each feature, the configuration specifies:

* Feature index
* Feature name
* Category
* Applied transformation

The feature order is part of the model interface and **must not be changed**.

The model expects:

```text
[feature_0, feature_1, ..., feature_15]
```

with an input shape of:

```text
(batch_size, 16)
```

### Feature Categories

The state consists of three main groups:

#### Macroeconomic Features

* USD Return
* Gold Return
* Brent Return
* US 10Y Return
* Türkiye CDS
* Annual Inflation
* TCMB Policy Rate

#### Fund and Portfolio Features

* Fund Return
* Portfolio Growth
* Active Value
* Cash Value
* Investor Count

#### Portfolio Allocation Features

* Stock Weight
* Repo Weight
* Collateral Weight
* Fund Weight

The exact feature order and preprocessing transformations are defined in `features.json`.

### Important

The feature order, dimensionality, and preprocessing transformations must remain consistent with the configuration used during model training.

Changing the order or preprocessing of the features may cause the ONNX model to interpret the input incorrectly.

---

## `actions.json`

`actions.json` defines the mapping between the DQN output indices and portfolio actions.

The model produces **9 Q-values**, one for each possible action.

| Index | Action          |
| ----: | --------------- |
|     0 | HOLD            |
|     1 | BUY_STOCK       |
|     2 | SELL_STOCK      |
|     3 | BUY_REPO        |
|     4 | SELL_REPO       |
|     5 | BUY_COLLATERAL  |
|     6 | SELL_COLLATERAL |
|     7 | BUY_FUND        |
|     8 | SELL_FUND       |

The action corresponding to the highest Q-value is selected as the model's action.

For example:

```text
Q-values:
[0.24, 0.19, 0.25, 0.24, 0.25, 0.25, 0.24, 0.26, 0.25]

Highest Q-value:
Index 7

Selected Action:
BUY_FUND
```

The action index mapping must remain consistent with the trained model.

---

## Model Interface

The configuration files describe the interface between the ONNX model and the application layer.

### Input

```text
Name: state
Shape: [batch_size, 16]
Type: float32
```

### Output

```text
Name: q_values
Shape: [batch_size, 9]
Type: float32
```

The application can determine the selected action by finding the index of the highest Q-value and mapping that index using `actions.json`.

---

## Relationship Between Configuration Files

The files are used together:

```text
features.json
      ↓
16-dimensional state
      ↓
final_model.onnx
      ↓
9 Q-values
      ↓
actions.json
      ↓
Selected Action
```

`metadata.json` provides the general technical information required to understand the model.

---

## Stock Portfolio Breakdown

The portfolio environment also maintains an internal breakdown of the stock allocation.

This stock-level breakdown is **not part of the 16-dimensional DQN state**. It is maintained by the environment and can be updated after portfolio actions such as `BUY_STOCK` or `SELL_STOCK`.

For the public demonstration, the initial stock breakdown values are **mock/synthetic values**. They do not represent actual current holdings, official portfolio composition, or production portfolio data.

The real source data used during development is not included in the public repository due to data confidentiality considerations.

The stock breakdown logic is therefore documented conceptually without exposing confidential source data or derived values.

---

## Data Privacy and Public Repository

The model was developed using multiple market-condition scenarios and corresponding datasets.

The original training datasets, including the 16 scenario datasets used during model development, are **not included in this public repository** due to data confidentiality considerations.

The public repository instead provides:

* The final ONNX model
* Model configuration files
* Feature and action definitions
* Development notebooks demonstrating the methodology
* Mock/synthetic demonstration data where necessary
* Portfolio demonstration components

The notebooks are provided to demonstrate the **development methodology, model structure, and workflow**. They do not expose the original confidential training datasets.

Similarly, any portfolio demonstration data included in the public repository is mock/synthetic and is intended only to demonstrate the system architecture and integration flow.

There is not any API keys, credentials, confidential raw datasets, or private company information in this directory.

---

## Important Notes

* The state size is **16**.
* The action size is **9**.
* Feature ordering must not be changed.
* Feature preprocessing must remain consistent with the training configuration.
* Action mapping must not be changed.
* `actions.json` should be used when converting model output indices into human-readable action names.
* The ONNX model produces Q-values and does not directly perform portfolio accounting or portfolio allocation updates.
* Environment-specific portfolio calculations are handled outside the ONNX model.
* The stock portfolio breakdown is maintained by the environment and is not directly produced by the ONNX model.
* Public demonstration values are mock/synthetic and do not represent actual portfolio holdings.
* Original training datasets are intentionally excluded from the public repository.
* Confidential raw datasets, credentials, and private company information are not included in this configuration directory.

