import json
import requests
import os

DATA_FILE = "output_action_log.jsonl"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral:7b"  # Using the Mistral 7B model

def load_raw_data():
    if not os.path.exists(DATA_FILE):
        return "No data found."
    
    events = []
    current_actions = {}  # Tracks ongoing actions per person

    with open(DATA_FILE, "r") as f:
        for line in f:
            try:
                data = json.loads(line)

                if "frame" in data:
                    frame = data["frame"]
                elif "frame_index" in data:
                    frame = data
                else:
                    continue
                
                time_sec = frame.get("timestamp_sec", 0)
                actions = frame.get("actions", {})
                active_persons = set(actions.keys())

                for person_id, action_info in actions.items():
                    action_label = action_info.get("label", "unknown action")

                    if person_id in current_actions and current_actions[person_id]["label"] == action_label:
                        current_actions[person_id]["end"] = time_sec
                    else:
                        if person_id in current_actions:
                            old = current_actions[person_id]
                            events.append(f"From {old['start']}s to {old['end']}s, Person {person_id} was {old['label']}.")
                        current_actions[person_id] = {"label": action_label, "start": time_sec, "end": time_sec}

                for person_id in list(current_actions.keys()):
                    if person_id not in active_persons:
                        old = current_actions[person_id]
                        events.append(f"From {old['start']}s to {old['end']}s, Person {person_id} was {old['label']}.")
                        del current_actions[person_id]
            except Exception:
                pass

    for person_id, old in current_actions.items():
        events.append(f"From {old['start']}s to {old['end']}s, Person {person_id} was {old['label']}.")

    return "\n".join(events)

def generate_answer(query, context):
    prompt = f"""You are an AI assistant that summarizes video events into a natural, flowing narrative paragraph.

Here is the event log of the video:
{context}

Question: {query}

Instructions:
1. Write EXACTLY ONE continuous paragraph summarizing what happened in the video.
2. DO NOT use bullet points, lists, or line breaks.
3. DO NOT output the raw timestamps or say "From X to Y seconds". Focus purely on the actions and tell a natural story.
4. Combine the events logically. Tell a brief story about what the people did.

Answer:"""

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        })
        
        data = response.json()
        if "response" in data:
            return data["response"].strip()
        else:
            return f"Error from LLM: {data}"
    except Exception as e:
        return f"Error contacting LLM: {str(e)}"

def main():
    print("Loading raw JSONL data for pure LLM generation...")
    context = load_raw_data()
    
    if not context or context == "No data found.":
        print("Error: Could not load data from", DATA_FILE)
        return

    print("\nSystem Ready! Ask your questions about the video (type 'exit' to quit).")
    
    while True:
        query = input("\n>> ")
        if query.lower() in ["exit", "quit"]:
            break
            
        print("\nThinking...")
        answer = generate_answer(query, context)
        print("\nResponse:")
        print(answer)
        print("-" * 60)

if __name__ == "__main__":
    main()
