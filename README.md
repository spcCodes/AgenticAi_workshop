# Agentic AI Workshop

Hands-on LangGraph notebooks for building agentic workflows: state, nodes, edges, routing, specialists, chatbots, persistence, streaming, observability, tools, RAG, and human-in-the-loop.

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
TAVILY_API_KEY=your_tavily_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
```

- `OPENAI_API_KEY` — LLM / embeddings calls
- `TAVILY_API_KEY` — web search tool (Session 2 tools notebook)
- `OPENROUTER_API_KEY` — optional alternate model provider (commented examples in notebooks)
- `LANGSMITH_API_KEY` + `LANGCHAIN_TRACING_V2` — LangSmith tracing (observability)

Then open the notebooks in Jupyter or Cursor.

```bash
uv run jupyter lab
```

### Streamlit chatbot demo

```bash
cd session2_advanced_langgraph_topics
uv run streamlit run chatbot_frontend.py
```

Local SQLite files such as `chatbot.db` (and `-shm` / `-wal`) are gitignored. They are created when you run the database checkpointer notebooks.

Sample PDF for RAG lives in [`data/intro-to-ml.pdf`](data/intro-to-ml.pdf).

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

## Session 2: Advanced LangGraph Topics

| Notebook / file | Topic |
| --- | --- |
| [1_basic_chatbot_llm.ipynb](session2_advanced_langgraph_topics/1_basic_chatbot_llm.ipynb) | Message-based chatbot, then memory with `MemorySaver` |
| [2_persistence.ipynb](session2_advanced_langgraph_topics/2_persistence.ipynb) | Checkpoints, `thread_id`, state history, and fault tolerance |
| [3_streaming_langgraph.ipynb](session2_advanced_langgraph_topics/3_streaming_langgraph.ipynb) | Token streaming with `stream_mode='messages'` |
| [4_database_checkpointer.ipynb](session2_advanced_langgraph_topics/4_database_checkpointer.ipynb) | Durable memory with `SqliteSaver` |
| [5_observability_langgraph.ipynb](session2_advanced_langgraph_topics/5_observability_langgraph.ipynb) | LangSmith tracing for runs and threads |
| [6_tools_langgraph.ipynb](session2_advanced_langgraph_topics/6_tools_langgraph.ipynb) | Tools, `ToolNode`, `tools_condition`, Tavily search |
| [chatbot_backend.py](session2_advanced_langgraph_topics/chatbot_backend.py) | Compiled chatbot graph used by the UI |
| [chatbot_frontend.py](session2_advanced_langgraph_topics/chatbot_frontend.py) | Streamlit chat UI |

## Session 3: Reliable Agent Execution

| Notebook | Topic |
| --- | --- |
| [1_rag_langgraph.ipynb](session3_relaible_agent_execution/1_rag_langgraph.ipynb) | PDF → chunks → FAISS → RAG tool in a LangGraph chatbot |
| [2_hitl_langgraph.ipynb](session3_relaible_agent_execution/2_hitl_langgraph.ipynb) | Human-in-the-loop with `interrupt` and `Command(resume=...)` |

## Concepts covered

- Nodes, edges, state, and graphs
- Shared state schema (`TypedDict`)
- Sequential and parallel workflows
- Conditional routing
- Iterative loops
- Specialist vs generalist agents
- Chatbots with `add_messages`
- Persistence: checkpointers, `thread_id`, `get_state_history`, fault tolerance
- Streaming responses
- SQLite / durable checkpointers
- Observability with LangSmith
- Tool calling with `ToolNode` and `tools_condition`
- RAG: load, split, embed, retrieve, answer from documents
- Human-in-the-loop: pause for approval, then resume
