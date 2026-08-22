# AI-Based Fund Strategy & Portfolio Decision Support System

A Deep Q-Network (DQN) based reinforcement learning system designed to support portfolio allocation decisions under different market conditions.

The system evaluates a 16-dimensional financial state and selects one of 9 portfolio actions. The trained model is exported to ONNX format for portable inference.

An optional LLM-based explainability layer is also included to convert the model's decisions into natural-language explanations.

---

## Project Overview

The system focuses on portfolio allocation across four main asset groups:

* Stock
* Repo / TPP
* Collateral
* Fund

The DQN agent receives market, fund, and portfolio information as its state and produces Q-values for nine possible actions.

The action with the highest Q-value is selected as the model decision.

### System Architecture

```text
Market & Portfolio State
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

The decision-making and explanation layers are intentionally separated.

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

For a given state, the model produces one Q-value for each possible action.

The action with the highest Q-value is selected.

For example:

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

The order of the state features must not be changed.

### Macroeconomic Features

1. USD Return
2. Gold Return
3. Brent Return
4. US 10Y Return
5. Türkiye CDS
6. Inflation
7. TCMB Policy Rate

### Fund & Portfolio Features

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

The state feature definitions, ordering and transformations are available in:

```text
config/features.json
```

> **Important:** Changing the feature order changes the meaning of the input received by the ONNX model and can lead to incorrect inference results.

---

## Actions

The model produces 9 possible actions.

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

The action mapping is also stored in:

```text
config/actions.json
```

---

## Portfolio Structure

The environment manages four main asset groups:

* Stock
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

Portfolio allocation constraints are handled by the Environment.

Internal portfolio allocation details are intentionally not exposed in this public repository.

---

## Environment

The Environment is responsible for applying the action selected by the DQN model and calculating its effect on the portfolio.

The Environment handles:

* Portfolio allocation
* Asset transfers
* Reward calculation
* Risk calculation
* Deviation
* Portfolio multiplier
* Transaction costs
* Episode termination

For example, when the DQN selects `BUY_STOCK`:

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

The DQN and Environment should therefore be evaluated together when considering the complete portfolio decision system.

Internal portfolio calculation logic is not included in the public model package.

---

## Scenario-Based Training

The model was trained using different market and crisis scenarios designed to represent a range of market conditions.

The scenarios include:

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

These scenarios are designed to evaluate how the model behaves under different market conditions, including:

* Market crashes
* Strong upward trends
* Long-term downward trends
* High volatility
* Macroeconomic shocks
* Unexpected extreme events

Detailed internal scenario-generation procedures and private training data are not included in the public repository.

---

## ONNX Model

The trained model has been exported to ONNX format.

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

The ONNX model provides a portable representation of the trained DQN and can be used for inference with ONNX Runtime.

---

## Inference Flow

The basic inference process is:

```text
1. Prepare current state
        ↓
2. Convert state to float32
        ↓
3. Send state to ONNX model
        ↓
4. Receive 9 Q-values
        ↓
5. Select highest Q-value
        ↓
6. Map action index to action name
        ↓
7. Apply action through Environment
        ↓
8. Update portfolio state
        ↓
9. Generate decision context
        ↓
10. Optional LLM explanation
```

---

## Explainable AI / LLM Integration

The LLM is **not** the decision-making model.

The responsibilities are separated into three layers.

### DQN

Makes the portfolio action decision.

### Environment

Applies the action and calculates the resulting portfolio state.

### LLM

Explains the decision in natural language.

```text
DQN
 ↓
Action
 ↓
Environment
 ↓
Portfolio & Market Context
 ↓
Prompt
 ↓
LLM API
 ↓
Natural Language Explanation
```

The LLM should not:

* Select a new action
* Override the DQN decision
* Modify portfolio weights
* Replace the DQN decision
* Provide an independent investment decision

Its purpose is to explain the actual model decision based on the information provided by the DQN and Environment.

---

## Scenario Information for LLM

When the LLM explanation layer is used, providing only a scenario number is not recommended.

Instead of sending:

```text
scenario = 8
```

the system should provide contextual information such as:

```text
Scenario Name:
Flash Crash

Scenario Description:
A rapid market decline followed by a recovery period.
```

Providing the scenario name and description allows the LLM to relate the DQN decision to the market conditions in which it was made.

---

## Configuration

The `config/` directory contains supporting information required for using and understanding the model.

```text
config/
├── actions.json
├── features.json
└── metadata.json
```

### actions.json

Defines the mapping between action indices and action names.

Example:

```text
0 → HOLD
1 → BUY_STOCK
2 → SELL_STOCK
```

### features.json

Defines the 16 state features, including:

* Feature order
* Feature name
* Category
* Data source
* Applied transformation

### metadata.json

Contains general model metadata such as:

* Model name
* State size
* Action size
* Model architecture
* ONNX information

Private portfolio allocation details and private source-data configurations are not included in the public configuration package.

---

## Logging

During training and experimentation, model behavior can be recorded using JSONL logs.

Typical log files include:

```text
logs/
├── step_logs.jsonl
└── episode_logs.jsonl
```

### step_logs.jsonl

Step-level information may include:

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

Episode-level information may include:

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

These logs can be useful for model analysis, debugging, system evaluation and future LLM explainability studies.

Private training logs and sensitive financial information are not included in the public repository.

---

## LLM Prompt Design

The LLM should receive sufficient context to explain why the DQN selected a particular action.

The prompt may contain:

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

### Risk & Decision Information

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

## LLM Responsibility

The LLM is an explanation layer and should not act as another portfolio decision-maker.

The LLM should:

* Explain the selected action
* Relate the decision to market conditions
* Explain the relevance of the scenario
* Interpret the portfolio allocation
* Consider reward and risk information
* Produce a natural-language explanation

The LLM should not:

* Choose another action
* Override the DQN
* Modify portfolio weights
* Independently calculate portfolio allocations
* Generate unsupported financial information

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies used by the public inference and XAI components include:

```text
numpy
onnxruntime
google-genai
python-dotenv
```

---

## Environment Variables

The LLM integration uses an environment variable for the API key.

```text
GEMINI_API_KEY
```

For local development, create a `.env` file:

```text
GEMINI_API_KEY=your_api_key_here
```

The `.env` file must not be committed to GitHub.

The repository `.gitignore` should contain:

```text
.env
__pycache__/
*.pyc
.venv/
venv/
.ipynb_checkpoints/
```

API keys should never be stored inside:

* Source code
* Model files
* Configuration files
* Public repositories

---

## Model Check

The ONNX model can be checked using:

```bash
python check_model.py
```

This script displays:

* Input name
* Input shape
* Input type
* Output name
* Output shape
* Output type

Expected model specification:

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

## Run Model

The provided `run_model.py` script runs the ONNX model using the mock state.

```bash
python run_model.py
```

The script displays:

* State values
* Q-values
* Selected action index
* Selected action name

Example:

```text
MODEL DECISION
----------------------------------------
Action Index : 7
Action       : BUY_FUND
```

---

## Portfolio Demo

The `portfolio_demo.py` script demonstrates how the trained ONNX model can be used within a portfolio decision-support workflow.

The demo uses an example state and shows the model inference and resulting portfolio decision context without exposing private financial datasets or proprietary internal portfolio logic.

General flow:

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

Run:

```bash
python portfolio_demo.py
```

The portfolio demo is intended for demonstration and model integration purposes.

---

## Run Full Demo

The `demo.py` script combines:

* Mock state
* ONNX model inference
* Q-value generation
* Action selection
* LLM-based explanation

Run:

```bash
python demo.py
```

The general flow is:

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

The demo uses:

```text
mock/mock_state.json
```

as the example input state and:

```text
xai/xai_prompt.txt
```

as the explanation prompt.

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

This public repository focuses on the components developed for the reinforcement learning portfolio decision-support system and its explainability workflow.

### Included

* Trained DQN model
* ONNX model
* Model configuration required for inference
* Action mapping
* State feature definitions
* Mock state
* ONNX inference scripts
* Portfolio demonstration
* Scenario-based model design
* LLM explainability integration
* Example XAI prompt

### Not Included

Private financial datasets, proprietary portfolio details, internal portfolio allocation logic, sensitive training logs and other confidential development resources are not included in the public repository.

The public package is intended to demonstrate the trained model, its decision-making workflow and its explainability pipeline without exposing private source data.

---

## Important Notes

* The model expects a 16-dimensional state.
* The state feature order must not be changed.
* The action mapping must remain unchanged.
* The ONNX model produces Q-values for the 9 available actions.
* The highest Q-value determines the selected action.
* Detailed internal portfolio allocation logic is not part of the public model package.
* Portfolio updates require the corresponding Environment logic.
* The LLM is an explanation layer and not the decision-making model.
* The LLM must not override the DQN decision.
* Scenario name and scenario description should be provided to the LLM when available.
* LLM explanations should be based only on actual system outputs.
* API keys must not be committed to the repository.
* `.env` must remain local.
* Prompt-based LLM integration is not equivalent to LLM fine-tuning.
* Private training datasets are not included in the public repository.

---

## Summary

This project provides a reinforcement learning based portfolio decision-support system using a Deep Q-Network.

The overall architecture is:

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

The separation between decision-making, portfolio simulation and natural-language explanation allows the system to expose the actual DQN decision while providing an interpretable explanation based on the corresponding market and portfolio context.

The final trained model is provided in ONNX format, together with the configuration, inference scripts, portfolio demonstration and optional LLM-based explainability workflow.
