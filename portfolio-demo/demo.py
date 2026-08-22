import json
import numpy as np
import onnxruntime as ort
import os
from google import genai
from dotenv import load_dotenv


# ==================================================
# ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()


# ==================================================
# DOSYA YOLLARI
# ==================================================

MODEL_PATH = "model/final_model.onnx"
STATE_PATH = "mock/mock_state.json"
PROMPT_PATH = "xai/xai_prompt.txt"


# ==================================================
# ACTION MAPPING
# ==================================================

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


# ==================================================
# STATE FEATURE SIRASI
# ==================================================

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


# ==================================================
# BAŞLANGIÇ
# ==================================================

print("\n========================================")
print("       PORTFOLIO AI DECISION DEMO")
print("========================================")


# ==================================================
# MOCK STATE OKU
# ==================================================

with open(STATE_PATH, "r", encoding="utf-8") as file:
    state_data = json.load(file)


state = np.array(
    [state_data[feature] for feature in STATE_FEATURES],
    dtype=np.float32
)

state = state.reshape(1, 16)


# ==================================================
# STATE GÖSTER
# ==================================================

print("\nSTATE")
print("----------------------------------------")

for feature in STATE_FEATURES:
    print(f"{feature:<22}: {state_data[feature]}")


# ==================================================
# ONNX MODEL
# ==================================================

session = ort.InferenceSession(MODEL_PATH)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# ==================================================
# MODEL TAHMİNİ
# ==================================================

q_values = session.run(
    [output_name],
    {input_name: state}
)[0][0]


# ==================================================
# ACTION SEÇ
# ==================================================

action_index = int(np.argmax(q_values))
action_name = ACTION_NAMES[action_index]


# ==================================================
# Q VALUES
# ==================================================

print("\nQ-VALUES")
print("----------------------------------------")

for i, q_value in enumerate(q_values):
    print(f"{i} - {ACTION_NAMES[i]:<18}: {q_value:.6f}")


print("\nMODEL DECISION")
print("----------------------------------------")
print(f"Action Index : {action_index}")
print(f"Action       : {action_name}")


# ==================================================
# GEMINI
# ==================================================

print("\n========================================")
print("             AI EXPLANATION")
print("========================================")


api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY bulunamadı. .env dosyanızı kontrol edin."
    )


client = genai.Client(api_key=api_key)


# ==================================================
# PROMPT OKU
# ==================================================

with open(PROMPT_PATH, "r", encoding="utf-8") as file:
    system_prompt = file.read()


# ==================================================
# GEMINI'YE GÖNDERİLECEK VERİ
# ==================================================

model_data = {
    "state": state_data,
    "q_values": {
        ACTION_NAMES[i]: float(q_values[i])
        for i in range(len(q_values))
    },
    "selected_action": action_name,
    "selected_action_index": action_index
}


final_prompt = f"""
{system_prompt}

Aşağıdaki veriler reinforcement learning modelinin
gerçek çıktılarıdır.

MODEL DATA:

{json.dumps(model_data, ensure_ascii=False, indent=2)}

Bu verilere dayanarak modelin seçtiği aksiyonu açıkla.
"""


# ==================================================
# GEMINI ÇAĞRISI
# ==================================================

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=final_prompt
)


# ==================================================
# XAI SONUCU
# ==================================================

print(response.text)


print("\n========================================")
print("           DEMO COMPLETED")
print("========================================")
