<div align="center">

# Flow

**AI agents for Frappe.**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

</div>

<img width="1470" height="923" alt="image" src="https://github.com/user-attachments/assets/59749f10-ccc7-49ce-aa30-9d0dc46c117a" />

---

Flow puts an AI assistant inside your Frappe site. It knows your DocTypes, respects your permissions, and can operate your site in plain English — creating records, sending emails, building DocTypes, creating agents and tools, and setting up RAG chatbots.

**Set up a support agent:**
> *"Create an agent that resolves customer support issues. It should use the Support Ticket doctype and these files as its knowledge: [attach files]"*

Flow sets up the RAG pipeline, embeds and indexes the files, and keeps the DocType source in sync — automatically.


**Put reporting on autopilot:**
> *"Every weekday at 9am, email me a summary of yesterday's new leads with their contact details and source."*

**Do anything a user can do:**
> *"Find all overdue sales orders from last month and mark them as closed."*

---

## Quick Access

A blue chat launcher sits at the bottom right of every Desk page; on each page load it peeks with your last chat. Open the panel by clicking it, or press:

<kbd>Cmd</kbd> + <kbd>I</kbd> *(or <kbd>Ctrl</kbd> + <kbd>I</kbd> on Windows)*

---

## How a message is answered

The chat model (Qwen 2.5 3B by default) runs **in the user's browser** through WebLLM; the agent loop and every tool run **on the server, as the logged-in user**. For each message the first match wins:

1. **Exact routes** — each tool can declare a regex route (greetings, "how many X", "who can create X", "which users have the X role", "what roles do I have", pasted errors, "how to create X"). The tool runs directly with no model call.
2. **Laya** — a small intent classifier. At confidence ≥ 0.85 it may route to a **read-only** tool; a how-to that no site tool takes goes to `docs_howto`, which answers from docs.frappe.io. Otherwise the message falls through.
3. **The chat model** — the server asks the browser to run the model and posts its reply back.

Write tools (`create`, `update`, `delete`, `run_action`, `execute`) always go through prechecks and an approval card before anything is saved. Answers are built from live metadata and records, not canned text.

The first question can wait while the browser model downloads; the panel shows "Model is loading, please wait…" and Stop or a new send clears a stuck run.

---

## Framework

Flow also ships as a Python framework for building agents in code.

### DocTypes

| DocType | Purpose |
| --- | --- |
| `Flow Provider` | Provider-level credentials and endpoint settings |
| `Flow Model` | Model configuration — set a `model_id` like `anthropic/claude-sonnet-4-6` |
| `Flow Tool` | Reusable tools, defined as a module path or inline script |
| `Flow Agent` | Instructions, model, and tools — configurable from the Desk or in code |
| `Flow Trigger` | Run an agent on a DocType event or a schedule |
| `Flow Knowledge Base` / `Flow Knowledge Source` | Knowledge sources (files, URLs, DocTypes) for retrieval-augmented answers |
| `Flow Session` / `Flow Run` | Persisted conversations and execution history |

### Define a tool

```python
import frappe
from flow import tool

@tool
def get_open_todos(allocated_to: str) -> list[dict]:
	"""List open ToDo items for a user."""
	return frappe.get_list(
		"ToDo",
		filters={"allocated_to": allocated_to, "status": "Open"},
		fields=["name", "description", "status"],
	)
```

### Run an agent

```python
from flow import Agent

agent = Agent(
	model="anthropic/claude-sonnet-4-6",
	instructions="You help with Frappe tasks.",
	tools=[get_open_todos],
)

result = agent.run("List open ToDo items assigned to me")
print(result.output)
```

### Persist a conversation

```python
session = agent.new_session(title="Todo assistant")

session.chat("What tasks do I have open?")
session.chat("Close the ones due before last Friday.")
```

Every turn is saved as a `Flow Run` linked to the session — so conversation history, tool calls, and outputs are all in the database.

### Built-in tools

| Group | Tools |
| --- | --- |
| Answers (shown as written) | `small_talk`, `count`, `show_records`, `explain_doctype`, `document_flow`, `creation_steps`, `required_values`, `assignment_steps`, `status_steps`, `error_diagnosis`, `docs_howto` |
| Roles and access | `role_permissions` (what a role can do), `doctype_access` (which roles and users can create/read/write a DocType), `role_users` (users holding a role), `user_roles` (roles of a user, or of the asker) |
| Lookup (for the model) | `find_doctypes`, `describe`, `read`, `search_knowledge`, `frappe_docs`, `web_search` |
| Memory | `update_memory` |
| Write (approval required) | `create`, `update`, `delete`, `run_action`, `execute` |

The default "Flow Lite" agent offers a subset (no `delete`, `run_action`, `execute`, `search_knowledge` or `update_memory`).

Tool calls can require human confirmation before executing, and every run is persisted for audit.

---

## Installation

Install into an existing [bench](https://github.com/frappe/bench):

```bash
cd $PATH_TO_YOUR_BENCH
bench get-app flow
bench --site site-name install-app flow
```

Then add a `Flow Provider` with your API key and a `Flow Model` with a model ID such as `anthropic/claude-sonnet-4-6`, and open the Desk Flow panel to start. For local providers like Ollama or LM Studio, set `Base URL` on the provider or model.

### Laya decision layer

[Laya](https://github.com/NandhaKishorM/laya) is a small classifier that runs inside Flow's Python code, before the chat model. It labels each message (how-to, record lookup, troubleshooting, create, definition, workflow, small talk) with a calibrated confidence; when it is confident, Flow answers with a read-only tool directly, and otherwise the chat model handles the message as usual. Laya never creates, changes or deletes data.

It is **on by default**. The default checkpoint is the int8 ONNX export `aditya-walue/flow-laya-intent-int8`, downloaded from Hugging Face into the site's `private/laya` folder and run by Flow's own ONNX runner (no PyTorch needed). `bench migrate` queues that download so the first message does not wait for it; a message that arrives while the model is still loading goes to the chat model.

```bash
bench --site site-name set-config flow_laya_enabled 0 --parse        # turn it off
bench --site site-name set-config flow_laya_threshold 0.85 --parse   # default 0.85
bench --site site-name set-config flow_laya_cache_dir /path          # where the checkpoint is kept
```

Every decision is recorded in `Flow Route Log`; `flow.api.laya_metrics` summarises routed, fallback and error rates and `flow.api.laya_status` shows whether the model is loaded (System Manager only). The PyTorch route (`pip install -e "apps/flow[laya]"`) is only needed to fine-tune or to run a checkpoint without `laya.onnx`.

#### Your own fine-tuned checkpoint

A Laya checkpoint fine-tuned on Flow (`scripts/laya/`) can be exported to ONNX and run by Flow's own runner (`flow/lib/laya_onnx.py`), which needs only packages Flow already installs (onnxruntime, tokenizers, numpy, huggingface_hub). No `laya` extra, no PyTorch, so it works on hosts where you can't install extra packages, such as Frappe Cloud.

```bash
# on a machine with the laya extra installed
./env/bin/pip install "laya[onnx]==0.3.27"
python <laya repo>/scripts/export_onnx.py --model <checkpoint> --output <out>/laya.onnx
cp -R <checkpoint>/rl_agent_config.json <checkpoint>/tokenizer <out>/
python apps/flow/scripts/laya/check_onnx_parity.py --torch-model <checkpoint> --onnx-model <out> --data <dataset>
```

Then point the site at the folder, or at a Hugging Face repo holding the same files (`laya.onnx`, `laya.onnx.data`, `rl_agent_config.json`, `tokenizer/`):

```bash
bench --site site-name set-config flow_laya_model_path <folder or org/repo>
bench --site site-name set-config flow_laya_threshold 0.85 --parse
bench --site site-name set-config flow_laya_enabled 1 --parse
```

The default repo is public. A private repo needs a read token: the `HF_TOKEN` environment variable, or the `flow_hf_token` config key where environment variables can't be set. On Frappe Cloud, set the keys above in the site's config and `flow_hf_token` in the bench group's config (so site managers don't see it). Each web worker keeps its own copy of the model in memory (about 1.7 GB for the fp32 export), so this suits a dedicated server; on a small plan leave Laya off. `flow_laya_backend` (`onnx` or `laya`) forces the runtime when needed.

## Site config

| Key | Purpose |
| --- | --- |
| `flow_laya_enabled`, `flow_laya_threshold`, `flow_laya_model_path`, `flow_laya_backend`, `flow_laya_cache_dir`, `flow_hf_token` | Laya (see above) |
| `flow_webllm_strict_json`, `flow_webllm_compact_context` | Browser-model prompting (both default on) |
| `flow_scope_check` | Ask the browser model one IN / OUT question before answering a message no route took, so jokes, poems and general knowledge get "ask ERP questions" (default on) |
| `flow_docs_pick` | Let the browser model choose, among the best-matching docs sections of a how-to answer, the one that answers the question, or say none does (default on) |
| `flow_web_search_domains` | Sources `web_search` may return (default: Frappe / ERPNext sites) |
| `flow_opik_enabled`, `flow_opik_project`, `flow_opik_url`, `flow_opik_api_key`, `flow_opik_workspace`, `flow_opik_capture_content` | Optional Opik tracing (`pip install -e "apps/flow[opik]"`) |

## Development

```bash
bench --site site-name run-tests --app flow --module flow.tests.test_ai_agent
bench --site site-name migrate        # also re-syncs the builtin tools and the default agent
cd apps/flow && yarn install && yarn build   # rebuild the panel after any frontend/src change
```

The built panel (`flow/public/flow_panel/`) is not in git. See `CLAUDE.md` for the architecture notes and rules for changes.

## License

[GNU AGPLv3](license.txt)
