import json
from sentence_transformers import SentenceTransformer, util


class RAGRetriever:
    def __init__(self, file_path):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.records = []
        self.texts = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(item, dict):
                    continue

                persons = item.get("persons")
                if not isinstance(persons, list) or not persons:
                    continue

                timestamp = item.get("timestamp") or item.get("time_sec") or "0"

                for p in persons:
                    if not isinstance(p, dict):
                        continue

                    # --- SAFE ROLE ---
                    role = p.get("role", "unknown")
                    role = str(role).lower() if role is not None else "unknown"

                    # --- SAFE ACTION (FIXED PART) ---
                    action = p.get("action_name", "unknown")

                    if isinstance(action, dict):
                        # try common fallback keys
                        action = action.get("name") or action.get("action") or "unknown"

                    action = str(action).lower() if action is not None else "unknown"

                    record = {
                        "timestamp": str(timestamp),
                        "role": role,
                        "action": action
                    }

                    self.records.append(record)

                    text = f"At {record['timestamp']}, a {record['role']} was performing {record['action']}."
                    self.texts.append(text)

        print(f"✅ Loaded {len(self.records)} records")

        if self.texts:
            self.embeddings = self.model.encode(self.texts, convert_to_tensor=True)
        else:
            self.embeddings = None

    def retrieve(self, query, top_k=5):
        if self.embeddings is None:
            return []

        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.embeddings)[0]

        top_results = scores.topk(k=min(top_k, len(self.records)))

        return [self.records[i] for i in top_results.indices]