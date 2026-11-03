"""A review analysis agent: answers questions by calling analysis tools."""

import ollama

from config import ASPECTS, MODEL_NAME
from tools import aspect_summary, get_evidence, list_aspects, low_rating_drivers

# The tools the model may use, by name
TOOLS = {
    "list_aspects": list_aspects,
    "aspect_summary": aspect_summary,
    "low_rating_drivers": low_rating_drivers,
    "get_evidence": get_evidence,
}
MAX_STEPS = 6  # safety limit so the loop can never run forever

SYSTEM_PROMPT = f"""You are a consumer insights analyst. You answer questions about
customer reviews of beauty products, for a busy manager.

Valid aspect names are exactly: {", ".join(ASPECTS)}.

Playbook (which tools to use):
- Question about ONE aspect (for example "how do people feel about scent?"):
  1. call aspect_summary for that aspect
  2. call get_evidence for that aspect with the most common sentiment, n=3
- Question about complaints, problems or bad ratings:
  1. call low_rating_drivers
  2. call get_evidence (sentiment "negative", n=2) for the top aspect
- Question about what customers talk about in general: call list_aspects.
- ALWAYS use tools for numbers. Never invent or guess numbers.

Answer format:
- Start with ONE headline sentence that includes the key numbers as percentages.
- Then at most 3 bullets with short quotes. Quote ONLY the "evidence" field,
  never the full review, and add the star rating, like: "smells like chemicals" (2 stars)
- Keep the whole answer under 80 words.
- If the data cannot answer the question, say so honestly."""


def run_agent(messages: list, verbose: bool = True) -> str:
    """Let the model call tools until it has a final answer."""
    for _ in range(MAX_STEPS):
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=list(TOOLS.values()),
            options={"temperature": 0},
        )
        message = response.message
        messages.append(message)

        # No tool calls means the model is ready to answer
        if not message.tool_calls:
            return message.content

        # Run every tool the model asked for and give it the results
        for call in message.tool_calls:
            name, args = call.function.name, call.function.arguments
            if verbose:
                print(f"  [tool] {name}({args})")

            tool = TOOLS.get(name)
            try:
                output = tool(**args) if tool else f"Unknown tool '{name}'"
            except Exception as e:
                output = f"Tool error: {e}"

            messages.append({"role": "tool", "content": output, "tool_name": name})

    return "Sorry, I could not finish the analysis within the step limit."


def main() -> None:
    print("Ask about the reviews. Type 'exit' to stop.\n")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        messages.append({"role": "user", "content": question})
        answer = run_agent(messages)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()