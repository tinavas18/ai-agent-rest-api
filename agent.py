
from planner import create_plan
from executor import execute_plan
from synthesizer import synthesize_results


def run_research(question: str) -> str:
    """Run the multi-step research pipeline."""

    plan = create_plan(question)
    results = execute_plan(plan, question)
    report = synthesize_results(question, plan, results)

    return str(report)
