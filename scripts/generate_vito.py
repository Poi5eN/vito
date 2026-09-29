from inference.generation import generate


def main() -> None:
    prompt = "Fix the database error."

    generated = generate(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
    )

    print("=" * 60)
    print("VITO GENERATION")
    print("=" * 60)
    print("Prompt:", prompt)
    print("Output:", generated)
    print("=" * 60)


if __name__ == "__main__":
    main()