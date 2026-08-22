# AI-Based Fund Strategy & Portfolio Decision Support System

A **Deep Q-Network (DQN)**-based reinforcement learning system developed to support portfolio allocation decisions under different market conditions.

The system evaluates a 16-dimensional financial state vector and selects one of 9 different portfolio actions. The trained model is converted into **ONNX** format for portable model inference.

An optional **LLM-based explainability (XAI) layer** is also included to explain model decisions in natural language.

---

## Project Overview

The system manages portfolio allocation across four main asset groups:

* Stocks
* Repo / TPP
* Collateral
* Fund

The DQN agent receives market, fund, and portfolio information as the state and generates Q-values for 9 different actions.

The action with the highest Q-value is selected as the model's decision.

### System Architecture

```text
Market and Portfolio State
          ↓
       DQN Model
          ↓
     Q-values (9)
          ↓
    Selected Action
          ↓
      Environment
          ↓
   Portfolio Update
          ↓
    Decision Context
          ↓
       LLM / XAI
          ↓
   Natural Language Explanation
```

The decision-making and explanation layers are kept separate from each other.

---

## Model

| Property    | Value                |
| ----------- | -------------------- |
| Model       | Deep Q-Network (DQN) |
| Model Name  | `final_model`        |
| State Size  | 16                   |
| Action Size | 9                    |
| Input Type  | `float32`            |
| Output      | 9 Q-values           |
| ONNX Opset  | 17                   |
| Format      | ONNX                 |

### Model Architecture

```text
Input: 16
   ↓
Linear: 128
   ↓
ReLU
   ↓
Linear: 128
   ↓
ReLU
   ↓
Linear: 9
   ↓
Q-values
```

The model generates one Q-value for each possible action given a state.

The action with the highest Q-value is selected.

Example:

```text
Q-values:
[0.24, 0.19, 0.25, 0.24, 0.25, 0.25, 0.24, 0.26, 0.25]

Highest Q-value:
Action 7

Selected Action:
BUY_FUND
```

---

## State

The model uses a 16-dimensional state vector.

The order of the features in the state must not be changed.

### Macroeconomic Features

1. USD Return
2. Gold Return
3. Brent Return
4. US 10Y Return
5. Türkiye CDS
6. Inflation
7. TCMB Policy Rate

### Fund and Portfolio Features

8. Fund Return
9. Portfolio Growth
10. Active Value
11. Cash Value
12. Investor Count

### Portfolio Allocation

13. Stock Weight
14. Repo Weight
15. Collateral Weight
16. Fund Weight

Definitions, ordering, and applied transformations of the state features are provided in:

```text
config/features.json
```

> **Important:** Changing the order of the state features changes the meaning of the input provided to the ONNX model and may lead to incorrect inference results.

---

## Actions

The model produces 9 different actions.

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

Action mappings are stored in:

```text
config/actions.json
```

---

## Portfolio Structure

The Environment manages four main asset groups:

* Stocks
* Repo / TPP
* Collateral
* Fund

The last four features of the DQN state represent these portfolio weights:

```text
stock_weight
repo_weight
collateral_weight
fund_weight
```

Constraints related to portfolio allocation are applied within the Environment.

Institution-specific portfolio allocation details and proprietary calculation logic are not included in the public repository.

---

## Environment

The Environment is developed to apply the action selected by the DQN to the portfolio and calculate the resulting outcomes.

The Environment manages:

* Portfolio allocation
* Asset-to-asset transfers
* Reward calculation
* Risk calculation
* Deviation
* Portfolio multiplier
* Transaction cost
* Episode termination

For example, when the DQN selects the `BUY_STOCK` action:

```text
DQN
 ↓
BUY_STOCK
 ↓
Environment
 ↓
Portfolio Allocation Updated
 ↓
Required Amount Transferred
 ↓
New Portfolio State
 ↓
Reward / Risk Calculated
```

Therefore, the DQN and Environment should be considered together when evaluating the system's decision-making process.

Institution-specific portfolio calculation logic is not included in the public model package.

---

## Scenario-Based Training

The model is trained using various scenarios designed to represent different market and crisis conditions.

The scenarios used include:

1. Normal Market
2. COVID Period
3. Severe Market Crash
4. Bubble / Strong Bull Market
5. Historical Crisis Periods
6. High Uncertainty / High Volatility
7. Synthetic Crises
8. Flash Crash
9. High Volatility
10. Long Bear Market
11. Long Bull Market
12. Sideways Market
13. Inflation Shock
14. Interest Rate Shock
15. Geopolitical Crisis
16. Black Swan

These scenarios are designed to evaluate the model's behavior under different market conditions.

The scenarios cover conditions such as:

* Market crashes
* Strong upward trends
* Long-term downward trends
* High volatility
* Macroeconomic shocks
* Unexpected extreme events

Detailed scenario generation processes and proprietary training data are not included in the public repository.

---

## ONNX Model

The trained DQN model has been converted into ONNX format.

### Model File

```text
model/final_model.onnx
```

### Input

```text
Name:
state

Shape:
[batch_size, 16]

Type:
float32
```

### Output

```text
Name:
q_values

Shape:
[batch_size, 9]
```

The ONNX model is a portable representation of the trained DQN model and enables inference using ONNX Runtime.

---

## Model Inference Process

The basic inference process is:

```text
1. Prepare the current state
        ↓
2. Convert the state to float32 format
        ↓
3. Send the state to the ONNX model
        ↓
4. Receive 9 Q-values
        ↓
5. Select the highest Q-value
        ↓
6. Convert the action index to an action name
        ↓
7. Apply the action through the Environment
        ↓
8. Update the portfolio state
        ↓
9. Create the decision context
        ↓
10. Optionally generate an LLM explanation
```

---

## Explainable AI / LLM Integration

The LLM is **not the decision-making model** of the system.

The responsibilities of the system are separated into three different layers.

### DQN

Selects the portfolio action.

### Environment

Applies the selected action and calculates the resulting portfolio state.

### LLM

Explains the model decision in natural language.

```text
DQN
 ↓
Action
 ↓
Environment
 ↓
Portfolio and Market Context
 ↓
Prompt
 ↓
LLM API
 ↓
Natural Language Explanation
```

The LLM:

* Must not select a new action.
* Must not change the DQN decision.
* Must not change portfolio weights.
* Must not replace the DQN decision.
* Must not generate an independent investment decision.

The purpose of the LLM is to explain the existing decision based only on the actual outputs provided by the DQN and Environment.

---

## Scenario Information for the LLM

When using the LLM explanation layer, it is recommended not to provide only the scenario number.

For example, instead of:

```text
scenario = 8
```

provide contextual information such as:

```text
Scenario Name:
Flash Crash

Scenario Description:
A rapid market decline followed by a recovery period.
```

Providing the scenario name and description helps the LLM relate the DQN decision to the market conditions under which the decision was generated.

---

## Configuration

The `config/` folder contains supporting information required for using and understanding the model.

```text
config/
├── actions.json
├── features.json
└── metadata.json
```

### actions.json

Defines the mapping between action indices and action names.

For example:

```text
0 → HOLD
1 → BUY_STOCK
2 → SELL_STOCK
```

### features.json

Defines the 16 state features and their related information:

* Feature order
* Feature name
* Category
* Data source
* Applied transformation

### metadata.json

Contains general information about the model:

* Model name
* State size
* Action size
* Model architecture
* ONNX information

Proprietary portfolio allocation details and institution-specific data configurations are not included in the public configuration.

---

## Logging

During training and experimentation, model behavior can be recorded using JSONL logs.

Example log structure:

```text
logs/
├── step_logs.jsonl
└── episode_logs.jsonl
```

### step_logs.jsonl

The following information can be recorded at the step level:

* Episode
* Scenario
* Step
* Date
* Action
* Action name
* Reward
* Risk
* Fund return
* Deviation
* Portfolio multiplier
* Portfolio value
* Transaction cost
* Portfolio allocation

### episode_logs.jsonl

The following information can be recorded at the episode level:

* Total reward
* Average reward
* Average loss
* Positive reward count
* Negative reward count
* Action distribution
* Exploration / exploitation ratio
* Action change rate
* Transaction cost
* Termination reason
* Epsilon

These logs can be used for model analysis, debugging, system evaluation, and future LLM explainability studies.

Private training logs and sensitive financial information are not included in the public repository.

---

## LLM Prompt Design

Sufficient context should be provided to the LLM so that it can explain why the DQN selected a particular action.

The prompt may include the following information.

### Model Decision

* Selected action
* Action name
* Q-values
* Exploration / exploitation information

### Portfolio Information

* Total portfolio value
* Stock weight
* Repo / TPP weight
* Collateral weight
* Fund weight

### Market Information

* USD Return
* Gold Return
* Brent Return
* US 10Y Return
* Türkiye CDS
* Inflation
* TCMB Policy Rate
* Fund Return
* Portfolio Growth
* Active Value
* Cash Value
* Investor Count

### Risk and Decision Information

* Reward
* Risk
* Deviation
* Portfolio Multiplier
* Transaction Cost
* Previous Action
* Action Change

### Scenario Information

* Scenario name
* Scenario description

Example task:

```text
Explain the DQN model's selected action in relation to
the current market conditions, scenario, portfolio state,
risk information and model outputs.

Do not change the model's decision.
Do not recommend a new investment action.
Base the explanation only on the provided data.
```

---

## LLM Responsibilities

The LLM should only be used as an explanation layer.

The LLM should:

* Explain the selected action.
* Relate the decision to the current market conditions.
* Explain the context of the scenario in relation to the decision.
* Interpret the portfolio allocation.
* Consider reward and risk information.
* Generate a natural language explanation.

The LLM should not:

* Select another action.
* Change the DQN decision.
* Change portfolio weights.
* Calculate an independent portfolio allocation.
* Generate financial information that is not present in the system.

---

## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

The main dependencies used by the public inference and XAI components are:

```text
numpy
onnxruntime
google-genai
python-dotenv
```

---

## Environment Variables

The LLM integration uses an API key through an environment variable.

```text
GEMINI_API_KEY
```

For local development, a `.env` file can be created:

```text
GEMINI_API_KEY=your_api_key_here
```

The `.env` file must not be uploaded to the GitHub repository.

The `.gitignore` file should contain:

```text
.env
__pycache__/
*.pyc
.venv/
venv/
.ipynb_checkpoints/
```

API keys should never be stored in:

* Source code
* Model files
* Configuration files
* Public repositories

---

## Portfolio Demo

`portfolio_demo.py` demonstrates how the trained ONNX model can be used within a portfolio decision-support process.

The demo uses an example state to demonstrate the model inference process and the resulting portfolio decision context. Private financial data and institution-specific portfolio calculation logic are not included in the demo.

### General Flow

```text
Example State
      ↓
 ONNX Model
      ↓
   Q-values
      ↓
Selected Action
      ↓
Portfolio Decision Context
```

### Running the Demo

```bash
python portfolio_demo.py
```

The Portfolio Demo is designed to demonstrate the model's operating logic and the portfolio decision-support process.

---

## Run Model

`run_model.py` demonstrates the inference process of the trained ONNX model using a mock state.

### Running

```bash
python run_model.py
```

The script displays:

* State values
* Q-values
* Selected action index
* Selected action name

### Example Output

```text
MODEL DECISION
----------------------------------------
Action Index : 7
Action       : BUY_FUND
```

---

## Model Check

`check_model.py` is used to verify the structure and input/output properties of the ONNX model.

### Running

```bash
python check_model.py
```

The script displays:

* Input name
* Input shape
* Input type
* Output name
* Output shape
* Output type

### Expected Model Structure

```text
Input:
state

Shape:
[batch_size, 16]

Type:
float32

Output:
q_values

Shape:
[batch_size, 9]
```

---

## Run Full Demo

`demo.py` demonstrates the complete workflow combining ONNX model inference with the LLM-based explanation layer.

The demo combines:

* Mock state
* ONNX model inference
* Q-value generation
* Action selection
* LLM-based explanation

### Running

```bash
python demo.py
```

### General Flow

```text
mock_state.json
       ↓
   ONNX Model
       ↓
    Q-values
       ↓
 Selected Action
       ↓
   Gemini API
       ↓
Natural Language Explanation
```

The demo uses the following example state:

```text
mock/mock_state.json
```

The prompt used for the LLM explanation is loaded from:

```text
xai/xai_prompt.txt
```

---

## Project Structure

```text
FINAL_MODEL/
│
├── README.md
│
├── model/
│   └── final_model.onnx
│
├── config/
│   ├── actions.json
│   ├── features.json
│   └── metadata.json
│
├── mock/
│   └── mock_state.json
│
├── xai/
│   └── xai_prompt.txt
│
├── check_model.py
├── run_model.py
├── portfolio_demo.py
├── demo.py
│
├── requirements.txt
└── .gitignore
```

---

## Public Repository Scope

This public repository focuses on the developed **reinforcement learning-based portfolio decision-support system**, the trained model, and the explainability process.

### Included

* Trained DQN model
* ONNX model
* Model configuration required for inference
* Action mapping
* State feature definitions
* Mock state
* ONNX inference scripts
* Portfolio Demo
* Scenario-based model design
* LLM-based explainability integration
* Example XAI prompt

### Not Included

The following information is not included in the public repository:

* Private financial datasets
* Institution-specific portfolio details
* Internal portfolio allocation logic
* Sensitive training logs
* Confidential data sources
* Other confidential development resources

The public package is intended to demonstrate the model's decision-making and explainability process without exposing private source data.

---

## Important Notes

* The model expects a 16-dimensional state.
* The state feature order must not be changed.
* The action mapping must not be changed.
* The ONNX model produces Q-values for 9 different actions.
* The action with the highest Q-value determines the selected action.
* Institution-specific portfolio allocation logic is not included in the public model package.
* Portfolio updates should be performed through the corresponding Environment logic.
* The LLM is only an explanation layer and is not the decision-making model.
* The LLM must not change the DQN decision.
* The scenario name and description should be provided to the LLM whenever possible.
* LLM explanations should be generated only from actual system outputs.
* API keys must not be uploaded to the repository.
* The `.env` file should be kept local.
* Prompt-based LLM integration does not mean that the LLM has been fine-tuned.
* Private training datasets are not included in the public repository.

---

## Summary

This project presents a **Deep Q-Network-based reinforcement learning portfolio decision-support system**.

The overall system structure is:

```text
DQN
 ↓
Decision

Environment
 ↓
Portfolio Decision Context

LLM
 ↓
Decision Explanation
```

By separating the decision-making, portfolio simulation, and natural language explanation layers, the system preserves the original DQN decision while enabling that decision to be explained in the context of relevant market and portfolio information.

The trained model is provided in ONNX format, together with model configuration files, inference scripts, a Portfolio Demo, and an optional LLM-based explainability component.
