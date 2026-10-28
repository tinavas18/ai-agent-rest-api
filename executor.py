
import os
from dotenv import load_dotenv
from google import genai
import time
from google.genai.errors import ServerError
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found in .env")

client = genai.Client(api_key=api_key)


def execute_task(task, original_query, previous_results=None):
    """
    Execute one subtask using Gemini.
    Include earlier results to maintain context.
    """

    previous_results = previous_results or []

    context = "\n".join(
        f"Previous result {i + 1}: {result}"
        for i, result in enumerate(previous_results)
    )

    prompt = f"""
    You are an AI research assistant.

    Original research question:
    {original_query}

    Current subtask:
    {task}

    Previous results:
    {context if context else "No previous results yet."}

    Instructions:
    - Focus on the current subtask.
    - Give a clear, useful answer.
    - Avoid repeating previous results unnecessarily.
    - Do not invent facts or citations.
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text.strip()


def execute_plan(plan, original_query):
    """Execute the plan's subtasks in order."""

    subtasks = plan.get("subtasks")

    if not isinstance(subtasks, list) or not subtasks:
        raise ValueError("The plan must contain a subtask list.")

    results = []

    for index, task in enumerate(subtasks, start=1):
        print(f"\nProcessing subtask {index}/{len(subtasks)}")
        print(f"Task: {task}")

        result = execute_task(
            task=task,
            original_query=original_query,
            previous_results=results
        )

        results.append(result)

        print("Subtask completed.")

    return results


if __name__ == "__main__":
    question = input("Enter your research question: ")

    # A sample plan for testing the executor independently.
    test_plan = {
        "subtasks": [
            "Explain three applications of AI in healthcare.",
            "Describe benefits and limitations of these applications.",
            "Summarize future opportunities for AI in healthcare."
        ]
    }

    try:
        outputs = execute_plan(test_plan, question)

        print("\n--- EXECUTION RESULTS ---")

        for index, output in enumerate(outputs, start=1):
            print(f"\nResult {index}:\n{output}")

    except Exception as error:
        print("Execution failed:", error)
