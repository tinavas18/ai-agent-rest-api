
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found in .env")

client = genai.Client(api_key=api_key)


def synthesize_results(original_query, plan, results):
    """Combine the subtask results into one research report."""

    research_sections = []

    for index, result in enumerate(results, start=1):
        task = plan["subtasks"][index - 1]

        research_sections.append(
            f"Subtask {index}: {task}\n{result}"
        )

    research_content = "\n\n".join(research_sections)

    prompt = f"""
You are an expert research report writer.

Original research question:
{original_query}

Research findings:
{research_content}

Create a well-organized report containing:
1. A suitable title
2. An introduction
3. Key findings with clear headings
4. Important benefits and limitations, if relevant
5. A conclusion

Use simple, professional language.
Combine related findings and remove unnecessary repetition.
Do not invent facts or citations.
If the findings do not support a claim, do not present it as verified.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    if not response.text:
        raise ValueError("Gemini returned an empty report.")

    return response.text.strip()


if __name__ == "__main__":
    sample_query = "How is artificial intelligence used in healthcare?"

    sample_plan = {
        "subtasks": [
            "Explain applications of AI in healthcare.",
            "Describe benefits and limitations of AI in healthcare.",
            "Discuss future opportunities for AI in healthcare."
        ]
    }

    sample_results = [
        "AI helps analyze medical images and supports clinical decision-making.",
        "AI can improve efficiency, but accuracy, privacy and bias remain concerns.",
        "Future opportunities include personalized medicine and improved monitoring."
    ]

    try:
        report = synthesize_results(
            sample_query,
            sample_plan,
            sample_results
        )

        print("\n========== FINAL RESEARCH REPORT ==========\n")
        print(report)

    except Exception as error:
        print("Report generation failed:", error)
