import os
from google import genai


# --------------------------------------------------
# GEMINI API KEY
# --------------------------------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY bulunamadı. "
        "Lütfen ortam değişkenini tanımlayın."
    )


# --------------------------------------------------
# GEMINI CLIENT
# --------------------------------------------------

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# CONNECTION TEST
# --------------------------------------------------

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Kendini kısaca tanıt. Bu bir bağlantı testidir."
)


# --------------------------------------------------
# RESULT
# --------------------------------------------------

print("\n========================================")
print("        GEMINI CONNECTION TEST")
print("========================================\n")

print(response.text)
