# # exe3/stage4_llm_rag.py
# import requests

# # Ollama settings
# OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
# OLLAMA_MODEL = "llama3:8b"

# def call_ollama(prompt: str) -> str:
#     """
#     Calls Ollama /api/generate and returns the generated text.
#     Tuned for higher compliance and fewer timeouts.
#     """
#     payload = {
#         "model": OLLAMA_MODEL,
#         "prompt": prompt,
#         "stream": False,
#         "options": {
#             "temperature": 0.0,     # important for instruction following
#             "top_p": 0.9,
#             "top_k": 40,
#             "num_predict": 450,     # shorter output => fewer drifts/timeouts
#             "repeat_penalty": 1.1,
#             "seed": 42
#         }
#     }

#     try:
#         # העליתי timeout כדי שלא ייפול ב-300 שניות
#         resp = requests.post(OLLAMA_URL, json=payload, timeout=900)
#         resp.raise_for_status()
#         data = resp.json()
#         return (data.get("response") or "").strip()
#     except requests.exceptions.RequestException as e:
#         return f"[OLLAMA ERROR] {e}"
#     except ValueError:
#         return "[OLLAMA ERROR] Invalid JSON response from Ollama"
