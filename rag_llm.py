import requests


class RAGLLM:
    def __init__(self):
        print("🤖 Using Qwen via Ollama")

    def call_ollama(self, prompt):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen:4b",
                    "prompt": prompt,
                    "stream": False
                }
            )

            data = response.json()

            if "response" in data:
                return data["response"].strip()
            else:
                return f"Error: {data}"

        except Exception as e:
            return f"❌ Ollama error: {str(e)}"

    def generate(self, query, records):

        context_text = "\n".join(
            f"{r['timestamp']} | {r['role']} performing {r['action']}"
            for r in records
        )

        prompt = f"""
You are an intelligent healthcare monitoring assistant.

Answer naturally based ONLY on the observed events.

Rules:
- Do NOT say yes/no directly
- Be descriptive but concise (1–2 sentences)
- If not found → clearly say it was not observed
- For summary → 2–3 sentences max

Context:
{context_text}

Question:
{query}

Answer:
"""

        return self.call_ollama(prompt)