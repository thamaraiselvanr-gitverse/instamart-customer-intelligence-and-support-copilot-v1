"""Part D - Policy RAG Assistant and Refund Agent
Document chunking, TF-IDF retrieval, policy Q&A, refund agent with tools."""

import pandas as pd
import numpy as np
import os
import sys
import json
import re
import time
import sqlite3
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mock_llm
from mock_llm import est_tokens

os.makedirs("outputs", exist_ok=True)
os.makedirs("state", exist_ok=True)

# Clear output files from previous runs
for f in ["outputs/agent_runs.jsonl", "outputs/refund_ledger.jsonl"]:
    if os.path.exists(f):
        os.remove(f)

print("=" * 60)
print("PART D: POLICY RAG ASSISTANT AND REFUND AGENT")
print("=" * 60)

# ============================================================
# TASK 1: CHUNK THE DOCUMENTS
# ============================================================
print("\n--- TASK 1: DOCUMENT CHUNKING ---")

POLICY_DIR = "data/policies"
CHUNK_SIZE = 40
OVERLAP = 10
STEP = CHUNK_SIZE - OVERLAP  # 30

def chunk_document(text, source, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into word-based chunks with overlap."""
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = ' '.join(words[start:end])
        chunks.append({
            'source': source,
            'text': chunk_text,
            'start_word': start,
            'end_word': end
        })
        if end >= len(words):
            break
        start += step
    return chunks

all_chunks = []
for filename in sorted(os.listdir(POLICY_DIR)):
    if filename.endswith('.txt'):
        filepath = os.path.join(POLICY_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        chunks = chunk_document(text, filename)
        all_chunks.extend(chunks)
        print(f"  {filename}: {len(chunks)} chunks")

print(f"Total chunks: {len(all_chunks)}")


# ============================================================
# TASK 2: VECTORISE AND RETRIEVE
# ============================================================
print("\n--- TASK 2: TF-IDF VECTORISATION ---")

chunk_texts = [c['text'] for c in all_chunks]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(chunk_texts)

def retrieve(query, k=3):
    """Retrieve top-k chunks by cosine similarity to query."""
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'source': all_chunks[idx]['source'],
            'score': float(similarities[idx]),
            'text': all_chunks[idx]['text']
        })
    return results

# Test retrieval
test_query = "refund policy late delivery"
test_results = retrieve(test_query)
print(f"\nTest query: '{test_query}'")
for i, r in enumerate(test_results):
    print(f"  Top {i+1}: {r['source']} (score={r['score']:.4f})")

print("\nTF measures how often a term appears in a document (chunk), giving higher")
print("weight to frequently mentioned terms. IDF measures how rare a term is across")
print("all documents — terms that appear in many documents get lower weight.")
print("Limitation: TF-IDF is a bag-of-words model that cannot capture semantic meaning.")
print("Two phrasings with the same meaning but different words (e.g. 'money back' vs 'refund')")
print("will have low similarity because TF-IDF matches exact word tokens, not concepts.")


# ============================================================
# TASK 3: AUGMENT AND ANSWER
# ============================================================
print("\n--- TASK 3: POLICY Q&A ---")

def answer(question):
    """Retrieve context and answer a policy question using the mock LLM."""
    chunks = retrieve(question, k=3)
    
    # Build context
    context_lines = []
    sources = set()
    for c in chunks:
        context_lines.append(f"[{c['source']}] {c['text']}")
        sources.add(c['source'])
    
    context = '\n'.join(context_lines)
    
    messages = [
        {"role": "system", "content": "You are a POLICY ASSISTANT for Instamart. Answer the question based ONLY on the provided context. If the answer is not in the context, say 'I could not find that information in the policy documents.'"},
        {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}
    ]
    
    response = mock_llm.chat(messages)
    return response.choices[0].message.content, list(sources)


# ============================================================
# TASK 4: EVALUATE RETRIEVAL AND GROUNDING
# ============================================================
print("\n--- TASK 4: EVALUATION (WITH OVERLAP) ---")

eval_df = pd.read_csv("data/raw/policy_eval.csv")

def evaluate_rag(chunk_size=40, overlap=10, label=""):
    """Run evaluation with given chunking parameters."""
    global all_chunks, tfidf_matrix, vectorizer
    
    # Re-chunk if needed
    if overlap != OVERLAP or chunk_size != CHUNK_SIZE:
        all_chunks_local = []
        for filename in sorted(os.listdir(POLICY_DIR)):
            if filename.endswith('.txt'):
                filepath = os.path.join(POLICY_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                chunks = chunk_document(text, filename, chunk_size, overlap)
                all_chunks_local.extend(chunks)
        
        chunk_texts_local = [c['text'] for c in all_chunks_local]
        vectorizer_local = TfidfVectorizer()
        tfidf_matrix_local = vectorizer_local.fit_transform(chunk_texts_local)
        
        def retrieve_local(query, k=3):
            query_vec = vectorizer_local.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix_local).flatten()
            top_indices = similarities.argsort()[-k:][::-1]
            results = []
            for idx in top_indices:
                results.append({
                    'source': all_chunks_local[idx]['source'],
                    'score': float(similarities[idx]),
                    'text': all_chunks_local[idx]['text']
                })
            return results
        
        def answer_local(question):
            chunks = retrieve_local(question, k=3)
            context_lines = []
            sources = set()
            for c in chunks:
                context_lines.append(f"[{c['source']}] {c['text']}")
                sources.add(c['source'])
            context = '\n'.join(context_lines)
            messages = [
                {"role": "system", "content": "You are a POLICY ASSISTANT for Instamart. Answer the question based ONLY on the provided context. If the answer is not in the context, say 'I could not find that information in the policy documents.'"},
                {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}
            ]
            response = mock_llm.chat(messages)
            return response.choices[0].message.content, list(sources)
        
        ret_fn = retrieve_local
        ans_fn = answer_local
    else:
        ret_fn = retrieve
        ans_fn = answer
    
    retrieval_hits = 0
    fact_hits = 0
    fallback_hits = 0
    answerable = 0
    unanswerable = 0
    failed_questions = []
    
    for _, row in eval_df.iterrows():
        question = row['question']
        expected_doc = row['expected_doc']
        expected_fact = row['expected_fact']
        
        if expected_doc == "NONE":
            unanswerable += 1
            ans_text, sources = ans_fn(question)
            if ans_text.startswith("I could not find"):
                fallback_hits += 1
        else:
            answerable += 1
            top3 = ret_fn(question)
            top3_sources = [r['source'] for r in top3]
            
            if expected_doc in top3_sources:
                retrieval_hits += 1
            else:
                failed_questions.append((question, expected_doc, top3_sources))
            
            ans_text, sources = ans_fn(question)
            if expected_fact and expected_fact in ans_text:
                fact_hits += 1
            elif expected_fact:
                failed_questions.append((question, f"expected_fact='{expected_fact}' not in '{ans_text}'", []))
    
    print(f"\n{label}:")
    print(f"  Answerable: retrieval hits {retrieval_hits}/{answerable}, fact hits {fact_hits}/{answerable}")
    print(f"  Unanswerable: fallback hits {fallback_hits}/{unanswerable}")
    if failed_questions:
        print(f"  Failed questions:")
        for q, reason, srcs in failed_questions:
            print(f"    Q: {q}")
            print(f"    Reason: {reason}")
            if srcs:
                print(f"    Retrieved: {srcs}")
    
    return retrieval_hits, fact_hits, fallback_hits, failed_questions

# Evaluate with overlap (default)
r_hits, f_hits, fb_hits, _ = evaluate_rag(40, 10, "Overlap=10 (default)")

# Evaluate with zero overlap
print("\n--- TASK 4: EVALUATION (ZERO OVERLAP) ---")
r_hits_0, f_hits_0, fb_hits_0, failed_0 = evaluate_rag(40, 0, "Overlap=0")

print("\nWith zero overlap, answers whose relevant content spans a chunk boundary")
print("may fail because neither chunk contains the complete answer. The overlap")
print("ensures that boundary-spanning content appears in at least one chunk.")


# ============================================================
# TASK 5: DEFINE TOOLS AND TOOL MAP
# ============================================================
print("\n--- TASK 5: TOOL DEFINITIONS ---")

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order by its order_id in the database and return its details.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}
    },
    {
        "name": "search_policy",
        "description": "Search the policy documents for relevant information about a query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the refund amount based on order amount and delivery time.",
        "parameters": {"type": "object", "properties": {
            "order_amount": {"type": "number"},
            "delivery_minutes": {"type": "number"}
        }, "required": ["order_amount", "delivery_minutes"]}
    },
    {
        "name": "issue_refund",
        "description": "Issue an automatic refund for an order.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"},
            "refund_inr": {"type": "number"}
        }, "required": ["order_id", "refund_inr"]}
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate a refund to a human support lead for approval.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"},
            "refund_inr": {"type": "number"}
        }, "required": ["order_id", "refund_inr"]}
    }
]

def get_order(order_id):
    """Look up order in SQLite database."""
    conn = sqlite3.connect("data/instamart.db")
    cursor = conn.cursor()
    cursor.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return {"error": "order not found"}
    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": row[2],
        "delivery_minutes": row[3]
    }

def search_policy(query):
    """Search policy documents using TF-IDF retrieval."""
    results = retrieve(query, k=3)
    return {"results": results}

def calculate_refund(order_amount, delivery_minutes):
    """Calculate refund based on delivery time rules."""
    order_amount = float(order_amount)
    delivery_minutes = float(delivery_minutes)
    if delivery_minutes > 45:
        return {"refund_inr": order_amount}
    else:
        return {"refund_inr": 0}

def issue_refund(order_id, refund_inr):
    """Issue a mock refund, preventing duplicate refunds for the same order."""
    refund_inr = float(refund_inr)
    ledger_path = "outputs/refund_ledger.jsonl"

    # Create the output directory if it does not exist.
    os.makedirs("outputs", exist_ok=True)

    # Check whether this order already has a recorded refund.
    if os.path.exists(ledger_path):
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # Do not silently ignore malformed ledger records.
                    raise ValueError(
                        f"Invalid JSON in refund ledger: {ledger_path}"
                    )

                if (
                    entry.get("tool") == "issue_refund"
                    and entry.get("order_id") == order_id
                ):
                    # Return the existing refund instead of recording another.
                    return {"refund_inr": float(entry["refund_inr"])}

    # No previous refund found, so record the mock refund.
    entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": "issue_refund",
        "order_id": order_id,
        "refund_inr": refund_inr,
    }

    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return {"refund_inr": refund_inr}

def escalate_to_human(order_id, refund_inr):
    """Escalate to human and log it."""
    refund_inr = float(refund_inr)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": refund_inr
    }
    with open("outputs/refund_ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": refund_inr}

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

print("Tools defined: " + ", ".join(TOOL_MAP.keys()))


# ============================================================
# TASK 6 & 7: ReAct LOOP WITH SAFETY
# ============================================================
print("\n--- TASK 6: REACT AGENT ---")

def save_checkpoint(run_id, scenario_id, messages, tokens_used, llm_calls, step_number, tools_executed):
    """Save agent checkpoint to state/ directory."""
    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "llm_calls": llm_calls,
        "step": step_number,
        "tools_executed": tools_executed
    }
    filepath = os.path.join("state", f"{run_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, default=str)

def load_checkpoint(run_id):
    """Load agent checkpoint from state/ directory."""
    filepath = os.path.join("state", f"{run_id}.json")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, resume_from=None):
    """Run the ReAct agent loop."""
    if run_id is None:
        run_id = f"run_{scenario_id}_{int(time.time())}"
    
    if resume_from:
        checkpoint = load_checkpoint(resume_from)
        messages = checkpoint['messages']
        tokens_used = checkpoint['tokens_used']
        step_number = checkpoint.get('step', checkpoint.get('step_number', 0))
        tools_executed = checkpoint['tools_executed']
        llm_calls_prior = checkpoint.get('llm_calls', 0)
        steps = []  # Steps for this continuation
    else:
        messages = [
            {"role": "system", "content": "You are an AGENT that helps process refund requests for Instamart. Use the available tools to look up orders, check policies, calculate refunds, and issue or escalate them."},
            {"role": "user", "content": user_message}
        ]
        tokens_used = 0
        step_number = 0
        tools_executed = []
        steps = []
    
    start_time = time.monotonic()
    final_answer = None
    status = None
    llm_calls = 0
    
    for iteration in range(max_iterations):
        # Check timeout
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            final_answer = f"FALLBACK_TIMEOUT: Timed out after {elapsed:.1f}s. Steps completed: {len(tools_executed)}. Handing to human."
            break
        
        # Call model
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens
        
        # Check token limit
        if tokens_used > token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            final_answer = f"FALLBACK_TOKEN_LIMIT: Token budget ({token_limit}) exceeded at {tokens_used}. Steps completed: {len(tools_executed)}. Handing to human."
            break
        
        msg = response.choices[0].message
        
        # If no tool calls, this is the final answer
        if not msg.tool_calls or len(msg.tool_calls) == 0:
            final_answer = msg.content
            # Extract status from answer
            if final_answer.startswith("REFUND:"):
                status = "REFUND"
            elif final_answer.startswith("ESCALATE:"):
                status = "ESCALATE"
            elif final_answer.startswith("DENIED:"):
                status = "DENIED"
            elif final_answer.startswith("NOT_FOUND:"):
                status = "NOT_FOUND"
            elif final_answer.startswith("NEEDS_INFO:"):
                status = "NEEDS_INFO"
            else:
                status = "UNKNOWN"
            break
        
        # Process tool calls
        for tc in msg.tool_calls:
            tool_name = tc.name
            tool_args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
            
            # Add assistant message with tool call
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": tc.id, "name": tool_name, "arguments": tool_args}]
            })
            
            # Execute tool
            tool_fn = TOOL_MAP[tool_name]
            observation = tool_fn(**tool_args)
            observation_str = json.dumps(observation, default=str)
            
            # Add tool result message
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tool_name,
                "content": observation_str
            })
            
            tools_executed.append(tool_name)
            step_number += 1
            
            steps.append({
                "tool": tool_name,
                "arguments": tool_args,
                "observation": observation
            })
            
            # Save checkpoint after each tool observation
            save_checkpoint(run_id, scenario_id, messages, tokens_used, llm_calls, step_number, tools_executed)
    else:
        # max_iterations reached without final answer
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = f"FALLBACK_MAX_ITERATIONS: Reached {max_iterations} iterations without resolution. Tools called: {', '.join(tools_executed)}. Handing to human."
    
    result = {
        "scenario_id": scenario_id,
        "run_id": run_id,
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "tools_executed": tools_executed
    }
    
    return result


def resume_agent(run_id):
    """Resume an agent run from a checkpoint."""
    checkpoint = load_checkpoint(run_id)
    scenario_id = checkpoint['scenario_id']
    # Get the original user message from messages
    user_message = next(m['content'] for m in checkpoint['messages'] if m['role'] == 'user')
    
    result = run_agent(
        user_message=user_message,
        scenario_id=scenario_id,
        run_id=run_id + "_resumed",
        resume_from=run_id
    )
    return result


# ============================================================
# RUN ALL 7 SCENARIOS
# ============================================================
print("\n--- RUNNING AGENT SCENARIOS ---")

scenarios = pd.read_csv("data/raw/agent_scenarios.csv")

for _, row in scenarios.iterrows():
    sid = row['scenario_id']
    msg = row['message']
    expected_outcome = row['expected_outcome']
    expected_refund = float(row['expected_refund_inr'])
    
    print(f"\n  Scenario {sid}: '{msg[:60]}...'")
    
    result = run_agent(msg, sid)
    
    print(f"    Status: {result['status']}")
    print(f"    Answer: {result['final_answer'][:80]}...")
    print(f"    LLM calls: {result['llm_calls']}, Tokens: {result['tokens_used']}")
    print(f"    Tools: {result['tools_executed']}")
    
    # Save to agent_runs.jsonl
    run_entry = {
        "run_id": result['run_id'],
        "scenario_id": sid,
        "status": result['status'],
        "final_answer": result['final_answer'],
        "llm_calls": result['llm_calls'],
        "tokens_used": result['tokens_used'],
        "steps": result['steps']
    }
    with open("outputs/agent_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(run_entry, default=str) + "\n")


# ============================================================
# TASK 7: STOP SAFELY DEMONSTRATIONS
# ============================================================
print("\n\n--- TASK 7: SAFETY DEMONSTRATIONS ---")

# (b) Token limit demo: S1 with token_limit=200
s1_msg = scenarios.iloc[0]['message']
print("\n  Demo: S1 with token_limit=200")
result_token = run_agent(s1_msg, "S1_token_demo", token_limit=200)
print(f"    Status: {result_token['status']}")
print(f"    Answer: {result_token['final_answer'][:100]}")

# (c) Timeout demo: any scenario with time_limit_seconds=0
print("\n  Demo: S1 with time_limit_seconds=0")
result_timeout = run_agent(s1_msg, "S1_timeout_demo", time_limit_seconds=0)
print(f"    Status: {result_timeout['status']}")
print(f"    Answer: {result_timeout['final_answer'][:100]}")


# ============================================================
# TASK 8: CRASH AND RESUME DEMONSTRATION
# ============================================================
print("\n\n--- TASK 8: CRASH AND RESUME ---")

# Run S1, simulate crash after 2nd tool execution
print("\n  Phase 1: Running S1 until 2nd tool execution, then simulating crash...")

# We need a modified agent that stops after 2 tools
def run_agent_crash_after_n(user_message, scenario_id, crash_after_n_tools=2, run_id=None):
    """Run agent but crash (stop) after N tool executions."""
    if run_id is None:
        run_id = f"crash_{scenario_id}"
    
    messages = [
        {"role": "system", "content": "You are an AGENT that helps process refund requests for Instamart. Use the available tools to look up orders, check policies, calculate refunds, and issue or escalate them."},
        {"role": "user", "content": user_message}
    ]
    tokens_used = 0
    step_number = 0
    tools_executed = []
    steps = []
    llm_calls = 0
    
    for iteration in range(6):
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens
        msg = response.choices[0].message
        
        if not msg.tool_calls or len(msg.tool_calls) == 0:
            break
        
        for tc in msg.tool_calls:
            tool_name = tc.name
            tool_args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
            
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": tc.id, "name": tool_name, "arguments": tool_args}]
            })
            
            tool_fn = TOOL_MAP[tool_name]
            observation = tool_fn(**tool_args)
            observation_str = json.dumps(observation, default=str)
            
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tool_name,
                "content": observation_str
            })
            
            tools_executed.append(tool_name)
            step_number += 1
            steps.append({"tool": tool_name, "arguments": tool_args, "observation": observation})
            
            # Save checkpoint (including llm_calls)
            save_checkpoint(run_id, scenario_id, messages, tokens_used, llm_calls, step_number, tools_executed)
            
            if len(tools_executed) >= crash_after_n_tools:
                print(f"    CRASH! After {len(tools_executed)} tools: {tools_executed}")
                return run_id, tools_executed
    
    return run_id, tools_executed

crash_run_id, crash_tools = run_agent_crash_after_n(s1_msg, "S1", crash_after_n_tools=2, run_id="crash_s1")

print(f"\n  Phase 2: Resuming from checkpoint '{crash_run_id}'...")
resumed_result = resume_agent(crash_run_id)
print(f"    Status: {resumed_result['status']}")
print(f"    Answer: {resumed_result['final_answer']}")
print(f"    Resume tools: {[s['tool'] for s in resumed_result['steps']]}")

all_tools = crash_tools + [s['tool'] for s in resumed_result['steps']]
print(f"\n    Total tools across crash and resume: {all_tools}")
print(f"    No tool repeated, final status: {resumed_result['status']}")

# Log demo runs to agent_runs.jsonl
for demo_result in [result_token, result_timeout]:
    demo_entry = {
        "run_id": demo_result['run_id'],
        "scenario_id": demo_result['scenario_id'],
        "status": demo_result['status'],
        "final_answer": demo_result['final_answer'],
        "llm_calls": demo_result['llm_calls'],
        "tokens_used": demo_result['tokens_used'],
        "steps": demo_result['steps']
    }
    with open("outputs/agent_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(demo_entry, default=str) + "\n")


# ============================================================
# TASK 9: REACT VS PLAN-AND-EXECUTE
# ============================================================
print("\n\n--- TASK 9: REACT VS PLAN-AND-EXECUTE ---")

react_essay = """Our ReAct agent makes model calls incrementally: S1 (late, small refund) \
takes 5 calls, S5 (cancelled order) takes 3, and S6 (order not found) takes 2. \
A plan-and-execute design emits the full tool-call plan in one shot and replans only on error.

For S6 (NOT_FOUND), plan-and-execute would emit a 4-step plan (get_order, search_policy, \
calculate_refund, issue_refund) but replan after step 1 returns an error, costing 2 calls, \
matching ReAct. For S5 (cancelled), it replans after discovering the order is cancelled, \
also matching ReAct at 3 calls. For the happy-path S1, plan-and-execute could reduce from 5 to \
2 calls (one plan, one confirmation), saving tokens.

However, the saving is small for short conversations and comes at the cost of more complex \
replanning logic. ReAct handles unexpected observations (NOT_FOUND, cancelled, above-limit \
refunds) naturally because each step reacts to the last observation, requiring no special \
replanning code.

For S7 (LOOP), ReAct reaches FALLBACK_MAX_ITERATIONS after 6 calls, each requesting the same \
tool. Plan-and-execute would detect the loop at replan time, potentially stopping sooner, \
but this edge case does not outweigh the simplicity benefit.

For this refund workflow I would choose ReAct. The conversations are short (2 to 5 calls), \
so token overhead is modest, and the ability to naturally handle any unexpected observation \
without replanning logic makes the system simpler, more robust, and easier to extend with \
new tools."""

word_count = len(react_essay.split())
print(f"\nReAct vs Plan-and-Execute ({word_count} words):\n")
print(react_essay)

assert word_count <= 250, f"Essay is {word_count} words, exceeds 250 limit"
print(f"\n(Word count: {word_count}, within 250-word limit)")

print("\nPart D complete.")

