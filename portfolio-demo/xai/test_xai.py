from gemini_client import ask_gemini


prompt = """
Bu bir XAI bağlantı testidir.

Modelin seçtiği aksiyon: SELL_STOCK

Bunu Türkçe olarak tek cümleyle açıkla.
Yatırım tavsiyesi verme.
"""


response = ask_gemini(prompt)

print("\n========================================")
print("             XAI TEST")
print("========================================\n")

print(response)
