import json
import os

input_file = 'output_action_action_log.json'
output_file = 'output_action_log.jsonl'

def convert_to_jsonl():
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading JSON: {e}")
            return

    with open(output_file, 'w') as f:
        # Write all top-level objects and expand 'frames' into individual JSON objects
        for key, value in data.items():
            if key == "frames" and isinstance(value, list):
                for frame in value:
                    # Write each frame as its own standalone JSON object on a new line
                    f.write(json.dumps(frame) + '\n')
            else:
                # Write other top-level metadata on new lines
                f.write(json.dumps({key: value}) + '\n')
                
    print(f"Successfully created {output_file} including all actions and objects.")

if __name__ == "__main__":
    convert_to_jsonl()
