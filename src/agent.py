"""A review analysis agent: answers questions by calling analysis tools."""
import json
import ollama

from config import ASPECTS, MODEL_NAME
from tools import aspect_report, list_aspects, low_rating_drivers

# The tools the model may use, by name
TOOLS = {
    "list_aspects": list_aspects,
    "aspect_report": aspect_report,
    "low_rating_drivers": low_rating_drivers,
}
MAX_STEPS = 6  # safety limit so the loop can never run forever

SYSTEM_PROMPT = f"""You are a consumer insights analyst. You answer questions about
customer reviews of beauty products, for a busy manager.

Valid aspect names are exactly: {", ".join(ASPECTS)}.

Which tool to use:
- Question about ONE aspect (scent, price, packaging...): call aspect_report.
- Question about complaints, problems or bad ratings: call low_rating_drivers.
- Question about what customers talk about in general: call list_aspects.
- ALWAYS use tools for numbers. Never invent numbers.

Answer format:
- Start your answer with the "headline" from the tool result, copied exactly.
- Then up to 3 bullets with quotes, copied word for word from the "quote" field,
  with the star rating, like: "smells like chemicals" (2 stars)
- Keep the answer under 80 words."""

def run_agent(messages: list, verbose: bool = True, on_tool=None) -> str:
    """Let the model call tools until it has a final answer."""
    headlines = []  # headlines returned by tools in this turn

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
            return add_headlines(message.content, headlines)

        # Run every tool the model asked for and give it the results
        for call in message.tool_calls:
            name, args = call.function.name, call.function.arguments
            if verbose:
                print(f"  [tool] {name}({args})")
            if on_tool:
                on_tool(name, args)

            tool = TOOLS.get(name)
            try:
                output = tool(**args) if tool else f"Unknown tool '{name}'"
            except Exception as e:
                output = f"Tool error: {e}"

            headline = get_headline(output)
            if headline:
                headlines.append(headline)

            messages.append({"role": "tool", "content": output, "tool_name": name})

    return "Sorry, I could not finish the analysis within the step limit."


def get_headline(tool_output: str):
    """Return the 'headline' field from a tool result, if there is one."""
    try:
        return json.loads(tool_output).get("headline")
    except (json.JSONDecodeError, AttributeError):
        return None


def add_headlines(answer: str, headlines: list) -> str:
    """Guardrail: make sure every tool headline appears at the top of the answer."""
    missing = [h for h in headlines if h not in answer]
    if not missing:
        return answer
    return "\n\n".join(f"**{h}**" for h in missing) + "\n\n" + answer


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