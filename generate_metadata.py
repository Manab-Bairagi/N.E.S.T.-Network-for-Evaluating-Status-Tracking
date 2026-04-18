import json
from datetime import datetime, timedelta
from collections import defaultdict

INPUT_FILE = "output_action_action_log.json"
OUTPUT_FILE = "metadata_output.jsonl"

START_TIME = datetime(2026, 3, 25, 10, 0, 0)


def normalize_action(action):
    if not action:
        return None
    action = str(action).strip().lower()
    if action in ["unknown", "", "none", "null"]:
        return None
    return action


def main():
    print("📂 Loading data...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])

    # 🔥 GROUP BY SECOND
    second_bucket = defaultdict(list)

    for frame in frames:

        timestamp_sec = frame.get("timestamp_sec", 0)
        second_key = int(timestamp_sec)   # group per second

        tracks = frame.get("tracks", {})
        actions = frame.get("actions", {})

        for track_id in tracks:

            action = normalize_action(actions.get(track_id))

            if action is None:
                continue

            second_bucket[second_key].append({
                "track_id": track_id,
                "action": action
            })

    print(f"⏱ Total seconds: {len(second_bucket)}")

    # 🔥 WRITE OUTPUT PER SECOND
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for sec, events in sorted(second_bucket.items()):

            timestamp = START_TIME + timedelta(seconds=sec)

            persons = []

            for e in events:

                action_label = e["action"]

                if action_label in ["falling down", "staggering", "sitting", "lying"]:
                    role = "Patient"
                    category = "Patient"
                else:
                    role = "Caregiver"
                    category = "Caregiver"

                persons.append({
                    "id": int(e["track_id"]),
                    "action_name": action_label,
                    "category": category,
                    "role": role
                })

            if not persons:
                continue

            frame_data = {
                "second": sec,
                "timestamp": timestamp.isoformat(),
                "persons": persons
            }

            out.write(json.dumps(frame_data) + "\n")

    print(f"✅ Done → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()