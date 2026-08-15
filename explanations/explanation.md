# LangGraph Concepts

Quick notes from Session 1 and Session 2.

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
