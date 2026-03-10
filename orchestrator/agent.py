import os
import openai


def generate_hypothesis_and_code() -> tuple[str, str]:
    """
    Generates an ML hypothesis and corresponding PyTorch Lightning code.
    Returns:
        tuple: (hypothesis_text, generated_python_code)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    client = openai.OpenAI(api_key=api_key)

    prompt = (
        "You are an expert ML researcher. "
        "Generate a brief hypothesis for improving a computer vision model, "
        "and then provide ONLY the Python code for a PyTorch Lightning `training_step` function "
        "that implements this hypothesis. Do not provide any explanation, only the Python code.")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You output only python code, no markdown formatting like ```python, just the raw code text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1500
    )

    # Simplification, normally we'd parse it out or ask for structured output
    hypothesis_text = "Generated hypothesis via LLM"
    generated_code = response.choices[0].message.content.strip()

    return hypothesis_text, generated_code
