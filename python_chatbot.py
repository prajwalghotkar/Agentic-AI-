import datetime
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# Optional colored output — falls back to plain text if colorama isn't installed
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLOR = True
except ImportError:
    COLOR = False


def c(text, color=None):
    return f"{color}{text}{Style.RESET_ALL}" if (COLOR and color) else text

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '23 * 7 + 1'.
    Only digits and + - * / ( ) . are allowed."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: expression contains disallowed characters."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def get_current_datetime(_: str = "") -> str:
    """Return the current date and time. Call this if the user asks what time/date it is."""
    return datetime.datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


@tool
def save_note(note: str) -> str:
    """Save a short note or reminder the user asks you to remember, to notes.txt."""
    with open("notes.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M}] {note}\n")
    return f"Saved note: {note}"


TOOLS = [calculator, get_current_datetime, save_note]


class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOllama(model="llama3.1", temperature=0.4)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful, concise AI assistant. "
    "Use the available tools (calculator, get_current_datetime, save_note) "
    "whenever they would give a more accurate or useful answer than guessing. "
    "Otherwise, answer directly and clearly."
))


def chatbot(state: State):
    full_context = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(full_context)
    return {"messages": [response]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(TOOLS))

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)  # routes to "tools" or END
graph_builder.add_edge("tools", "chatbot")

# Persistent memory: keeps the full conversation per thread_id
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

LOG_FILE = f"transcript_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"


def log(line: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def stream_turn(user_input: str, config: dict):
    """Stream the assistant's reply token-by-token, print it, and log the turn."""
    print(c("Assistant: ", Fore.CYAN if COLOR else None), end="", flush=True)
    final_text = ""
    try:
        for chunk, metadata in graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="messages",
        ):
            # Only print tokens from the chatbot node itself, not internal tool chatter
            if metadata.get("langgraph_node") == "chatbot" and chunk.content:
                print(chunk.content, end="", flush=True)
                final_text += chunk.content
        print()
    except Exception as e:
        print(c(f"\n[Error while generating a response: {e}]", Fore.RED if COLOR else None))
        final_text = f"[error: {e}]"

    log(f"User: {user_input}")
    log(f"Assistant: {final_text}")


def main():
    thread_id = "cli-session-1"  # give each user/session a unique id for multi-user support
    config = {"configurable": {"thread_id": thread_id}}

    print(c("Agentic Chatbot ready. Type 'quit', 'exit' or 'q' to stop.", Fore.YELLOW if COLOR else None))
    print(c(f"Transcript will be saved to: {LOG_FILE}\n", Fore.YELLOW if COLOR else None))

    while True:
        try:
            user_input = input(c("User: ", Fore.GREEN if COLOR else None)).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_turn(user_input, config)


if __name__ == "__main__":
    main()