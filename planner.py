
import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found in .env")

client = genai.Client(api_key=api_key)


def create_plan(user_query):
    prompt = f"""
    You are an AI task planner.

    Break the user's question into 3 or 4
    small, logical subtasks.

    Return ONLY valid JSON in this format:
    {{
        "topic": "short topic name",
        "subtasks": [
            "First subtask",
            "Second subtask",
            "Third subtask"
        ]
    }}

    User question: {user_query}
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    # Remove Markdown code fences if the model adds them
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    plan = json.loads(text)

    if not isinstance(plan, dict):
        raise ValueError("Planner must return a JSON object")

    subtasks = plan.get("subtasks")

    if not isinstance(subtasks, list) or not 1 <= len(subtasks) <= 4:
        raise ValueError("Expected between 1 and 4 subtasks")

    if not all(isinstance(task, str) and task.strip()
               for task in subtasks):
        raise ValueError("Each subtask must be non-empty text")

    return plan


if __name__ == "__main__":
    question = input("Enter your research question: ")

    try:
        result = create_plan(question)

        print("\n--- Research Plan ---")
        print(json.dumps(result, indent=4, ensure_ascii=False))

    except (json.JSONDecodeError, ValueError) as error:
        print("Could not create a valid plan:", error)
    except Exception as error:
        print("An API or other error occurred:", error)
