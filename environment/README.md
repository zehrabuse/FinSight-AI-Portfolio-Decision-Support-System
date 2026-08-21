# Portfolio Environment

This directory contains the public version of the portfolio simulation environment used during the development of the DQN-based portfolio decision-support system.

The environment is implemented using **Gymnasium** and provides the interaction structure between the portfolio state, market data, DQN agent actions, portfolio constraints, reward calculation, and episode termination conditions.

> **Important:** The public environment is a demonstration version. Confidential source datasets and real portfolio composition data are not included.

## Environment Overview

The environment follows the standard reinforcement learning interaction loop:

```text
Market / Scenario Data
        ↓
PortfolioEnv
        ↓
16-dimensional State
        ↓
DQN Agent
        ↓
9 Possible Actions
        ↓
Portfolio Update
        ↓
Reward Calculation
        ↓
Next State
```

The same environment structure was used during model development with multiple market-condition scenarios.
The public repository contains the environment logic and mock demonstration values, but does not contain the original confidential scenario datasets.

## Environment Class

The main environment class is:

`PortfolioEnv(gym.Env)`

It follows the Gymnasium environment interface and provides:

- `reset()`
- `step(action)`
- `render()`

The environment receives scenario datasets through:

`env = PortfolioEnv(scenarios)`

where `scenarios` is a collection of pandas DataFrames.

## Scenario Selection

During training, each episode randomly selects one of the available scenarios.

The public environment keeps the scenario names for documentation and demonstration purposes.

The original scenario datasets are **not included in this repository**.

The environment supports the following scenario categories:

| ID | Scenario |
|---:|---|
| 1 | Normal Piyasa |
| 2 | COVID-19 Krizi |
| 3 | 2008 Benzeri Sert Çöküş |
| 4 | Spekülatif Balon ve Ani Yükseliş |
| 5 | Tarihsel ve Jeopolitik Krizler |
| 6 | Belirsiz Yüksek Volatilite |
| 7 | Sentetik Rastgele Krizler |
| 8 | Flash Crash |
| 9 | Aşırı Volatilite ve Overtrading |
| 10 | Uzun Ayı Piyasası |
| 11 | Uzun Boğa Piyasası |
| 12 | Yatay Piyasa ve Overtrading |
| 13 | Enflasyon Şoku |
| 14 | Faiz Şoku |
| 15 | Jeopolitik Kriz |
| 16 | Black Swan / Kara Kuğu |

These names describe the types of market conditions considered during model development.

The original scenario datasets and their underlying confidential source data are intentionally excluded from the public repository.

## Initial Portfolio

The public environment initializes the portfolio using mock/synthetic allocation values:

| Asset | Initial Weight |
|---|---:|
| Stock | 85% |
| Repo | 7% |
| Collateral | 3% |
| Fund | 5% |

These values are used only to demonstrate the environment logic.

They do **not** represent an actual portfolio, current holdings, or confidential company data.

Initial portfolio value:

`100,000 TL`

## Mock Stock Breakdown

The environment also maintains an internal stock-level breakdown.

The public version uses synthetic identifiers:

`MOCK_STOCK_01`  
`MOCK_STOCK_02`  
`...`  
`MOCK_STOCK_10`  
`OTHER`

These values are mock/synthetic data and do not represent actual securities or portfolio holdings.

The stock-level breakdown is maintained by the environment but is **not part of the 16-dimensional DQN state**.

## State Space

The DQN receives a 16-dimensional state vector.

**State Size = 16**

The state consists of:

### Macroeconomic Features

1. USD Return
2. Gold Return
3. Brent Return
4. US 10Y Return
5. Türkiye CDS
6. Annual Inflation
7. TCMB Policy Rate

### Fund and Portfolio Features

8. Fund Return
9. Portfolio Growth
10. Active Value
11. Cash Value
12. Investor Count

### Portfolio Allocation Features

13. Stock Weight
14. Repo Weight
15. Collateral Weight
16. Fund Weight

The state is returned as:

`np.ndarray`

with:

- `shape = (16,)`
- `dtype = float32`

The feature order must remain consistent with the model configuration.

See `config/features.json` for the complete feature definition and preprocessing information.

## Action Space

The environment provides 9 discrete actions:

| Index | Action |
|---:|---|
| 0 | HOLD |
| 1 | BUY_STOCK |
| 2 | SELL_STOCK |
| 3 | BUY_REPO |
| 4 | SELL_REPO |
| 5 | BUY_COLLATERAL |
| 6 | SELL_COLLATERAL |
| 7 | BUY_FUND |
| 8 | SELL_FUND |

The DQN outputs 9 Q-values.

The action with the highest Q-value is selected:

`action = np.argmax(q_values)`

The action mapping must remain consistent with the trained model.

See `config/actions.json` for the action mapping.

## Portfolio Constraints

The environment applies several portfolio constraints.

### Maximum Portfolio Weight Change

A single action can change the portfolio allocation by a maximum of:

**10 percentage points**

`max_weight_change = 0.10`

### Minimum Stock Weight

The stock allocation cannot fall below:

**80%**

`min_stock_weight = 0.80`

### Asset Weight Limits

The public environment applies the following limits:

| Asset | Minimum | Maximum |
|---|---:|---:|
| Stock | 80% | 100% |
| Repo | 0% | 20% |
| Collateral | 0% | 20% |
| Fund | 0% | 20% |

## Stock-Level Constraints

The environment also maintains a stock-level portfolio breakdown.

Each stock can change by a maximum of:

**5 percentage points per step**

`max_stock_breakdown_change = 0.05`

When buying stocks, the environment updates the internal stock breakdown while maintaining the defined per-stock change constraint.

When selling stocks, the stock-level positions are reduced proportionally.

The stock breakdown is normalized so that its total remains approximately:

**100%**

The stock-level breakdown is used internally by the environment and is not included in the DQN state.

## Reward Function

The environment uses a risk-aware reward structure.

The reward considers:

- Portfolio return
- Portfolio risk
- Transaction cost

The general structure is:

`Reward = Return Component - Risk Component + Transaction Cost Component`

The current implementation uses:

- `return_weight = 10.0`
- `risk_weight = 0.5`
- `transaction_weight = 0.5`

Portfolio risk is estimated using the standard deviation of historical portfolio returns within the current episode.

The final reward is clipped to:

`[-1.0, 1.0]`

## Portfolio Return

The environment calculates portfolio growth using the scenario's fund return and a portfolio-allocation deviation factor.

The deviation is calculated between the agent's current allocation and the scenario's target allocation:

- Stock
- Repo
- Collateral
- Fund

The deviation is then used to calculate a portfolio multiplier.

The resulting value is used to determine the simulated daily portfolio return.

This mechanism is part of the simulation environment and should not be interpreted as a production portfolio accounting system.

## Transaction Cost

Each executed portfolio transaction incurs a simulated transaction cost.

The current implementation uses:

`transaction_cost = -0.01 * trade_amount`

The transaction cost is incorporated into the reward calculation.

This is a simplified simulation mechanism intended for reinforcement learning experimentation.

## Episode Termination

An episode can terminate under several conditions.

### End of Dataset

`End_of_Dataset`

The scenario data has been completely consumed.

### Portfolio Loss Limit

`Portfolio_Below_Limit`

The episode terminates if the portfolio falls below:

**55% of initial portfolio value**

This corresponds to a maximum simulated loss of approximately:

**45%**

### Target Return

`Target_Return_Reached`

The episode terminates if the portfolio reaches:

**200% of initial portfolio value**

### Cash Condition

`Cash_Zero`

The environment also checks whether the simulated cash value reaches zero as an episode termination condition.

## Environment Output

The `step()` function follows the Gymnasium API:

`state, reward, terminated, truncated, info`

The `info` dictionary contains additional information about the simulation.

Examples include:

- `scenario`
- `scenario_name`
- `date`
- `step`
- `action`
- `action_name`
- `trade_amount`
- `trade_source`
- `trade_target`
- `reward`
- `portfolio_return`
- `risk`
- `daily_return`
- `fund_return`
- `deviation`
- `portfolio_multiplier`
- `portfolio_value`
- `portfolio_allocation`
- `transaction_cost`
- `termination_reason`

This information can be used for:

- Debugging
- Training analysis
- Action tracking
- Logging
- Explainability
- Demonstration purposes

## Example

A simplified environment interaction looks like:

`env = PortfolioEnv(scenarios)`

`state, info = env.reset()`

`action = 0`

`next_state, reward, terminated, truncated, info = env.step(action)`

The DQN can then use the following reinforcement learning loop:

`state → action → reward → next_state`

to learn a portfolio decision policy.

## Relationship with the DQN Model

The environment and the ONNX model have different responsibilities.

`PortfolioEnv`  
↓  
`16-dimensional state`  
↓  
`DQN / ONNX Model`  
↓  
`9 Q-values`  
↓  
`Selected Action`  
↓  
`PortfolioEnv`

### PortfolioEnv Responsibilities

The environment is responsible for:

- Maintaining portfolio state
- Applying portfolio constraints
- Executing simulated actions
- Calculating portfolio returns
- Calculating rewards
- Tracking episode state
- Maintaining the mock stock breakdown

### DQN / ONNX Model Responsibilities

The model is responsible for:

- Receiving the 16-dimensional state
- Calculating Q-values
- Selecting the action with the highest Q-value

The ONNX model itself does **not** perform portfolio accounting.

## Data Privacy

The original development environment used multiple market-condition scenarios and corresponding datasets.

The following are **not included** in the public repository:

- Original scenario datasets
- Confidential raw data
- Company-specific portfolio data
- Actual stock-level portfolio composition
- Private source data
- API credentials
- Authentication information

The public environment instead uses:

- Mock portfolio weights
- Mock stock identifiers
- Synthetic demonstration values
- Publicly documented environment logic

The purpose is to demonstrate the architecture, reinforcement learning environment design, action mechanism, state structure, and reward methodology without exposing confidential information.

## Important Notes

- The environment uses a 16-dimensional state.
- The action space contains 9 discrete actions.
- Feature ordering must remain consistent with `config/features.json`.
- Action mapping must remain consistent with `config/actions.json`.
- Mock portfolio weights are not real portfolio holdings.
- Mock stock identifiers do not represent actual securities.
- The stock-level breakdown is not part of the DQN input state.
- Original scenario datasets are intentionally excluded.
- The public environment is intended for demonstration and reproducibility of the system architecture.
- The simulation logic should not be interpreted as a production trading or investment system.
