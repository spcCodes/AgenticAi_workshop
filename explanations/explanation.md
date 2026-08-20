# LangGraph Concepts

Quick notes from Session 1, Session 2, and Session 3.

---

## Nodes, Edges, State, and Graphs

- **State** — the shared data that flows through the graph (`TypedDict`).
- **Node** — a function that reads state and returns updates.
- **Edge** — the path from one node to the next.
- **Graph** — nodes + edges compiled into a workflow you can `invoke`.
- **START / END** — where the run begins and finishes.

Example: BMI calculator. Input `weight` and `height`, node calculates `bmi`, graph returns the updated state.

---

## Shared State Schema

The state schema is the contract for the whole graph.

- Every node sees the same schema.
- A node should return only the fields it updates, for example `{'bmi': 22.5}`.
- Use `TypedDict` so keys stay clear: `question`, `answer`, `discriminant`, and so on.

If a field is missing from the schema, the graph cannot store it.

---

## Sequential Workflows (Prompt Chaining)

Nodes run one after another.

`START → outline → blog → END`

Each node uses the previous node's output. Notebook `3_prompt_chaining` writes an outline first, then the blog from that outline.

Use this when step 2 needs step 1.

---

## Parallel Workflows

Several nodes run at the same time from the same point.

Example: essay scoring. Language, analysis, and clarity all evaluate the same essay, then a final node averages the scores.

If multiple nodes write to the same list field, use:

```python
Annotated[list[int], operator.add]
```

That **adds** the values instead of overwriting them.

---

## Conditional Workflows and Routing

A function looks at state and chooses the next node.

```python
graph.add_conditional_edges('calculate_discriminant', check_condition)
```

Quadratic example:

- discriminant `> 0` → two real roots
- discriminant `== 0` → one repeated root
- discriminant `< 0` → no real roots

Review example: positive review → thank-you message, negative review → diagnose, then reply.

**Routing** is this idea with an LLM: classify the input, then send it down the matching branch.

---

## Iterative Workflows

The graph loops until a condition is met.

Tweet example: generate → evaluate → if not good enough, optimize → evaluate again.

Stop when:

- the evaluator says `approved`, or
- `iteration >= max_iteration`

This is a loop **inside one run**, not memory across runs.

---

## Specialist vs Generalist Agents

A **generalist** is one agent that tries to handle everything. One node, one prompt. Billing, crashes, and office hours all go through the same path.

A **specialist** setup splits the work. A router first classifies the query (`billing` / `technical` / `general`), then `add_conditional_edges` sends it to the matching node. Each specialist has its own prompt, so the billing answer sounds like billing, and the tech answer sounds like tech support.

Same idea as the quadratic notebook: one condition, then different branches. Here the condition is the query type, not the discriminant.

**When to use which**

- **Generalist** — simple demos, mixed questions, you want one prompt.
- **Specialist** — different skills, tone, or tools per category, and you want tighter control.

---

## Persistence (Stateless vs Stateful)

LangGraph calls this **persistence**.

**Stateless** — `graph.compile()` with no checkpointer. After `invoke` finishes, the state is gone. Ask "my name is Suman", then "what is my name?" and it will not know.

**Stateful** — `graph.compile(checkpointer=MemorySaver())` plus a `thread_id`. LangGraph saves state after each run and reloads it for the same thread.

```python
config = {'configurable': {'thread_id': 'user-1'}}
workflow.invoke({'question': 'Hi, my name is Suman'}, config=config)
workflow.invoke({'question': 'What is my name?'}, config=config)
```

`thread_id` is like a chat id:

- `user-1` remembers Suman
- `user-2` remembers Rahul
- they do not share memory

Earlier notebooks were stateless. The graph had state **during** one run, but nothing was kept for the next run.

Iterative workflows loop inside one `invoke`. Persistence remembers across many `invoke` calls.

---

## Streaming

`invoke` waits for the full answer. `stream` yields chunks as they arrive, which is what chat UIs need.

```python
for message_chunk, metadata in chatbot.stream(
    {'messages': [HumanMessage(content=user_input)]},
    config=config,
    stream_mode='messages',
):
    ...
```

`stream_mode='messages'` streams LLM tokens (and related message updates) instead of waiting for the finished state.

Use streaming when the user should see the reply grow token by token.

---

## SQLite Checkpointer (Durable Memory)

`MemorySaver` / `InMemorySaver` keeps checkpoints in process memory. Restart the app and the memory is gone.

`SqliteSaver` writes checkpoints to a SQLite file, so threads survive restarts:

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect('chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
chatbot = graph.compile(checkpointer=checkpointer)
```

Same `thread_id` idea as before. The difference is where the checkpoints live: RAM vs disk.

Local files like `chatbot.db`, `chatbot.db-shm`, and `chatbot.db-wal` are runtime artifacts and should stay out of git.

---

## Observability (LangSmith)

Observability means you can see what the graph did: inputs, outputs, latency, tool calls, and failures.

With LangSmith enabled (`LANGCHAIN_TRACING_V2=true` and `LANGSMITH_API_KEY`), LangGraph / LangChain runs show up as traces. Thread-aware configs help you follow one conversation across multiple turns.

Use this when debugging routing, tools, or slow steps — the UI shows the path the graph actually took.

---

## Tools (`ToolNode` and `tools_condition`)

A **tool** is a function the LLM can call (search, calculator, APIs). In LangGraph:

1. Bind tools to the model.
2. Add a `ToolNode` that runs the chosen tool(s).
3. Use `tools_condition` so the graph either calls tools or finishes.

```python
from langgraph.prebuilt import ToolNode, tools_condition

tool_node = ToolNode(tools)
graph.add_node('tools', tool_node)
graph.add_conditional_edges('chat_node', tools_condition)
```

Typical loop: chat node → (if tool call) tools node → back to chat node → END when done.

Session 2 uses Tavily search as an example tool. The LLM decides when search is needed; the graph runs the tool and feeds the result back into the conversation.

---

## RAG (Retrieval-Augmented Generation)

RAG means: retrieve relevant text from your documents, then let the LLM answer using that context — instead of guessing from training data alone.

Typical pipeline in Session 3:

1. **Load** a PDF (`PyPDFLoader`)
2. **Split** into chunks (`RecursiveCharacterTextSplitter`)
3. **Embed** and store in a vector DB (`OpenAIEmbeddings` + `FAISS`)
4. **Retrieve** top-k similar chunks for a query
5. Expose retrieval as a **tool** (`rag_tool`) inside a LangGraph chatbot with `ToolNode` / `tools_condition`

```python
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

@tool
def rag_tool(query: str) -> str:
    """Retrieve relevant information from the PDF."""
    docs = retriever.invoke(query)
    ...
```

The graph looks like the tools chatbot: chat node may call `rag_tool`, tool results go back to the chat node, then the model writes the final answer grounded in retrieved chunks.

Use RAG when answers must come from a specific document or knowledge base.

---

## Human-in-the-Loop (HITL)

HITL pauses the graph so a human can approve, edit, or reject before the agent continues.

In LangGraph:

1. Call `interrupt(...)` inside a node — the run stops and returns `__interrupt__`
2. A checkpointer is **required** so state can resume later
3. Resume with `Command(resume=...)` on the same `thread_id`

```python
def chat_node(state: ChatState):
    decision = interrupt({
        "type": "approval",
        "question": state["messages"][-1].content,
        "instruction": "Approve this question? yes/no",
    })
    if decision["approved"] == "no":
        return {"messages": [AIMessage(content="Not approved.")]}
    response = model.invoke(state["messages"])
    return {"messages": [response]}

app = builder.compile(checkpointer=MemorySaver())

# later
app.invoke(Command(resume={"approved": "yes"}), config=config)
```

Use HITL for risky actions: sending emails, spending money, deleting data, or any step that needs a human gate.

---

## Subgraphs (Private vs Shared State)

A **subgraph** is a compiled graph used inside a parent graph. Session 3 uses a parent that answers in English, then a subgraph that translates to Hindi.

**Private state** — parent and subgraph have different keys.

```python
class SubState(TypedDict):
    input_text: str
    translated_text: str

class ParentState(TypedDict):
    question: str
    answer_eng: str
    answer_hin: str
```

You call the subgraph yourself and map fields:

```python
result = subgraph.invoke({"input_text": state["answer_eng"]})
return {"answer_hin": result["translated_text"]}
```

The subgraph never sees `question` or `answer_eng`. Its keys never land on the parent unless you copy them.

**Shared state** — both graphs use the same schema. Nest the compiled subgraph as a node:

```python
parent_builder.add_node("translate", subgraph)
```

Overlapping keys flow automatically. `translate_text` can read `answer_eng` and write `answer_hin`.

**If a run is interrupted** (Jupyter stop, crash, delay):

- A checkpointer still saves completed **parent** nodes in both patterns.
- Private state: subgraph keys stay off the parent. You can wrap `subgraph.invoke(...)` in `try/except`.
- Shared state: the subgraph **is** a parent node, so its error **is** the parent `invoke()` error. There is no separate helper call to catch.

Use private state for a reusable helper. Use shared state when the subgraph is just a nested piece of the same workflow.

---

## Short-Term Memory (STM)

**Short-term memory** is what the agent remembers in the current conversation (one `thread_id`). In LangGraph that is the checkpointed graph state: messages, slots, and anything else in the schema.

It is not a second database. It is the same persistence idea from Session 2. Session 3 walks the ladder:

1. **No checkpointer** — each `invoke` starts fresh
2. **`InMemorySaver`** — remembers across turns until the process exits
3. **`SqliteSaver`** — same `thread_id` memory survives restarts on disk

```python
config = {"configurable": {"thread_id": "user-1"}}
graph.invoke({"messages": [HumanMessage(content="My name is Suman")]}, config=config)
graph.invoke({"messages": [HumanMessage(content="What is my name?")]}, config=config)
```

Turn 2 reloads the thread checkpoint, so the model still sees the earlier messages.

You can also **read** a saved thread without calling the LLM. Open the DB, compile with the same checkpointer, and call `get_state`:

```python
with SqliteSaver.from_conn_string("chatbot.db") as cp:
    outside_graph = builder.compile(checkpointer=cp)
    snap = outside_graph.get_state({"configurable": {"thread_id": "thread-1"}})
```

`get_state` only loads the checkpoint. Use `invoke` when you want a new turn.

Compared with other memory types:

- **During one `invoke`** — state fields the nodes pass around (no checkpointer needed)
- **Short-term** — across turns in the same thread (checkpointer + `thread_id`)
- **Long-term** — across threads or sessions (a Store, files, or an external DB)

STM dies if you change `thread_id`, compile without a checkpointer, or use an in-memory checkpointer and restart the process. A SQLite checkpointer keeps that thread memory on disk.

---

## STM Trimming and Deletion

Checkpointed conversations grow. If you pass the full history into the model every turn, cost and context length climb. Two common controls:

### Trimming (`trim_messages`)

Keep only recent messages that fit a **token budget** before calling the model. The full thread can still live in the checkpointer; you trim a copy for the LLM call.

```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

messages = trim_messages(
    state["messages"],
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=300,
    start_on="human",
    include_system=True,
)
```

### Deletion (`RemoveMessage`)

Actually remove messages from state (and therefore from the checkpoint on the next write). Useful when history should shrink permanently.

```python
from langchain.messages import RemoveMessage

def delete_old_messages(state):
    if len(state["messages"]) <= 10:
        return {}
    to_remove = state["messages"][:-4]  # keep the latest few
    return {"messages": [RemoveMessage(id=m.id) for m in to_remove]}
```

Wire a cleanup node after chat so deletion runs each turn. Trimming limits what the model sees; deletion limits what the thread stores.

---

## STM Summarisation

When history is long, **summarise** older turns into one `summary` string in state, then delete the raw messages you no longer need. The next chat turn can inject that summary as context.

Typical pattern:

1. Extend state with `summary: str` (on top of `MessagesState`)
2. After chat, conditionally route to a `summarize` node when message count is high
3. Summarize (or extend an existing summary), write `summary`, and return `RemoveMessage` for old ids

```python
class ChatState(MessagesState):
    summary: str

def should_summarize(state: ChatState) -> bool:
    return len(state["messages"]) > 6

# chat -> (should_summarize?) -> summarize -> END
# chat -> END
```

Compared with trimming/deletion alone:

- **Trim** — full history may still sit in the checkpointer; model only sees a window
- **Delete** — drop messages with no replacement
- **Summarise** — keep a compressed memory of what was said, then drop the verbose turns

Use summarisation when you need long-running threads without blowing the context window or losing the gist.

---

## Long-Term Memory (LTM) and Stores

**Short-term memory** is thread state (checkpointer + `thread_id`).  
**Long-term memory** lives in a **Store** — facts keyed by namespace that can outlive a single chat thread.

`InMemoryStore` is the in-process Store used in Session 3:

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
namespace = ("user", "u1")

store.put(namespace, "1", {"data": "User likes pizza"})
item = store.get(namespace, "1")
items = store.search(namespace)
```

- **Namespace** — a tuple that scopes memories (for example per user)
- **Key** — id inside that namespace
- **Value** — a JSON-like dict you store

Different users use different namespaces so memories do not mix.

### Semantic search

With an embedding index, `search` can find memories by meaning, not only by key:

```python
store = InMemoryStore(index={"embed": embedding_model, "dims": 1536})
store.put(namespace, "5", {"data": "User is learning machine learning"})
hits = store.search(namespace, query="what is the user currently learning", limit=1)
```

STM answers “what did we say in this thread?”  
LTM answers “what do we know about this user across sessions?”

---

## LTM in a Chatbot Graph

Session 3 next wires the Store into graph nodes. Pass `store=` at compile time and take `store: BaseStore` in the node signature. Scope memories with `user_id` from config:

```python
def chat_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("user", user_id, "details")
    items = store.search(namespace)
    # inject memories into the system prompt, then call the model
    ...

graph = builder.compile(store=store)
graph.invoke(..., config={"configurable": {"user_id": "u1"}})
```

Common teaching steps in the notebook:

1. **Read** — seed the Store, search the user namespace, put facts into the prompt
2. **Write** — an LLM with structured output decides `should_write` + memory strings, then `store.put`
3. **Avoid duplicates** — search existing memories first; only store items marked new
4. **Merged workflow** — chat that both reads and writes in one graph

Checkpointers still handle STM (thread messages). The Store handles LTM (user facts across turns/threads).
