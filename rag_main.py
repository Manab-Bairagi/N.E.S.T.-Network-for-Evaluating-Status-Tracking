from rag_retriever import RAGRetriever
from rag_llm import RAGLLM

DATA_FILE = "metadata_output.jsonl"


def main():
    print("📂 Initializing RAG system...")

    retriever = RAGRetriever(DATA_FILE)
    llm = RAGLLM()

    print("\n🚀 Ready! Ask questions (type 'exit' to quit)\n")

    while True:
        query = input(">> ")

        if query.lower() == "exit":
            break

        results = retriever.retrieve(query)
        answer = llm.generate(query, results)

        print("\n🧠 Answer:")
        print(answer)
        print("-" * 50)


if __name__ == "__main__":
    main()