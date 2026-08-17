# Agentic AI Workshop

Hands-on LangGraph notebooks for building agentic workflows: state, nodes, edges, routing, specialists, chatbots, and persistence.

Concept notes: [explanations/explanation.md](explanations/explanation.md)

## Setup

Python 3.13+ and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
cp .env.example .env
```

Add your keys to `.env`:

```
OPENAI_API_KEY=your_openai_api_key_here
```

Then open the notebooks in Jupyter or Cursor.

```bash
uv run jupyter lab
```

## Session 1: LangGraph Basics

| Notebook | Topic |
| --- | --- |
| [1_bmi_calculator.ipynb](session1_langgraph_basics/1_bmi_calculator.ipynb) | First graph: state, nodes, edges |
| [2_simple_llm_workflow.ipynb](session1_langgraph_basics/2_simple_llm_workflow.ipynb) | Single LLM node |
| [3_prompt_chaining.ipynb](session1_langgraph_basics/3_prompt_chaining.ipynb) | Sequential workflow |
| [4_parallel_workflows.ipynb](session1_langgraph_basics/4_parallel_workflows.ipynb) | Parallel nodes |
| [5_llm_parallel_workflows.ipynb](session1_langgraph_basics/5_llm_parallel_workflows.ipynb) | Parallel LLM evaluators |
| [6_condition_workflow.ipynb](session1_langgraph_basics/6_condition_workflow.ipynb) | Conditional edges |
| [7_llm_conditional_workflows.ipynb](session1_langgraph_basics/7_llm_conditional_workflows.ipynb) | LLM routing |
| [8_iterative_Workflows.ipynb](session1_langgraph_basics/8_iterative_Workflows.ipynb) | Loops until a stop condition |
| [9_specialist_vs_generalist.ipynb](session1_langgraph_basics/9_specialist_vs_generalist.ipynb) | One agent vs routed specialists |

## Session 2: Chatbots and Persistence

| Notebook | Topic |
| --- | --- |
| [1_basic_chatbot_llm.ipynb](session2/1_basic_chatbot_llm.ipynb) | Message-based chatbot, then memory with `MemorySaver` |
| [2_persistence.ipynb](session2/2_persistence.ipynb) | Checkpoints, `thread_id`, state history, and fault tolerance |

## Concepts covered

- Nodes, edges, state, and graphs
- Shared state schema (`TypedDict`)
- Sequential and parallel workflows
- Conditional routing
- Iterative loops
- Specialist vs generalist agents
- Chatbots with `add_messages`
- Persistence: checkpointers, `thread_id`, `get_state_history`, fault tolerance
