import os
from google import genai


def ask_gemini(prompt: str) -> str:
    """
    Gemini'ye verilen prompt'u gönderir
    ve modelin metin cevabını döndürür.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY bulunamadı.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    return response.text
