from src.generation.llm_client import generate_response


def main():
    prompt = "Explain what Machine Learning is in two sentences."

    answer = generate_response(prompt)

    print("\nGemini Response:")
    print(answer)


if __name__ == "__main__":
    main()

    