import json
import numpy as np
import onnxruntime as ort


# --------------------------------------------------
# DOSYA YOLLARI
# --------------------------------------------------

MODEL_PATH = "model/final_model.onnx"
STATE_PATH = "mock/mock.json"


# --------------------------------------------------
# ACTION MAPPING
# --------------------------------------------------

ACTION_NAMES = {
    0: "HOLD",
    1: "BUY_STOCK",
    2: "SELL_STOCK",
    3: "BUY_REPO",
    4: "SELL_REPO",
    5: "BUY_COLLATERAL",
    6: "SELL_COLLATERAL",
    7: "BUY_FUND",
    8: "SELL_FUND"
}


# --------------------------------------------------
# STATE SIRASI
# --------------------------------------------------

STATE_FEATURES = [
    "usd_return",
    "gold_return",
    "brent_return",
    "us_10y_return",
    "turkiye_cds",
    "inflation",
    "tcmb_policy_rate",
    "fund_return",
    "portfolio_growth",
    "active_value",
    "cash_value",
    "investor_count",
    "stock_weight",
    "repo_weight",
    "collateral_weight",
    "fund_weight"
]


# --------------------------------------------------
# MOCK STATE OKU
# --------------------------------------------------

with open(STATE_PATH, "r", encoding="utf-8") as file:
    state_data = json.load(file)


# --------------------------------------------------
# STATE VECTOR OLUŞTUR
# --------------------------------------------------

state = np.array(
    [state_data[feature] for feature in STATE_FEATURES],
    dtype=np.float32
)

state = state.reshape(1, 16)


# --------------------------------------------------
# ONNX MODEL
# --------------------------------------------------

session = ort.InferenceSession(MODEL_PATH)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# --------------------------------------------------
# MODEL ÇALIŞTIR
# --------------------------------------------------

q_values = session.run(
    [output_name],
    {input_name: state}
)[0]

q_values = q_values[0]


# --------------------------------------------------
# EN İYİ ACTION
# --------------------------------------------------

action_index = int(np.argmax(q_values))
action_name = ACTION_NAMES[action_index]


# --------------------------------------------------
# SONUÇLARI YAZDIR
# --------------------------------------------------

print("\n========================================")
print("       PORTFOLIO AI MODEL TEST")
print("========================================")

print("\nSTATE:")
print("----------------------------------------")

for feature in STATE_FEATURES:
    print(f"{feature:<22}: {state_data[feature]}")


print("\nQ-VALUES:")
print("----------------------------------------")

for i, q_value in enumerate(q_values):
    print(f"{i} - {ACTION_NAMES[i]:<18}: {q_value:.6f}")


print("\nMODEL DECISION:")
print("----------------------------------------")
print(f"Action Index : {action_index}")
print(f"Action       : {action_name}")

print("\n========================================")
