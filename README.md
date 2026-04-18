First you should run the metadata generation script (for example generate_metadata.py) using your raw input file output_action_action_log.json, 
which will process the frames and create a cleaned structured dataset file called metadata_output.jsonl.

for llm : 
first run 
ollama serve - > ollama run qwen:4b -> python rag_main.py 
