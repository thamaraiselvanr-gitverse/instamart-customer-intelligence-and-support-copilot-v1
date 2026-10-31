import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "outputs" / "live_test_runs"

PIPELINE = [
    ("Generate synthetic data", "generate_data.py"),
    ("Part A: Data analysis", "part_a.py"),
    ("Validate checkpoints", "validate.py"),
    ("Part B: Customer segmentation", "part_b.py"),
    ("Part C: Support ticket classification", "part_c.py"),
    ("Part D: Policy RAG and refund agent", "part_d.py"),
    ("Verify all tests", "tests/verify_all.py"),
    ("Leakage proof", "tests/leakage_proof.py"),
]

st.set_page_config(
    page_title="Instamart Pipeline Dashboard",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Instamart Customer Intelligence")
st.subheader("Pipeline Testing Dashboard")
st.caption("Local project test runner. This dashboard uses the scripts in your repository.")

if "running" not in st.session_state:
    st.session_state.running = False
if "events" not in st.session_state:
    st.session_state.events = []
if "results" not in st.session_state:
    st.session_state.results = []
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "event_queue" not in st.session_state:
    st.session_state.event_queue = queue.Queue()

st.warning(
    "Safety notice: running the complete pipeline can regenerate data and modify files "
    "under outputs/ and state/. Part D may reset its mock refund and agent-run logs. "
    "Do not start a run if you need to preserve the current generated outputs."
)

with st.expander("Pipeline stages", expanded=True):
    for number, (label, script) in enumerate(PIPELINE, start=1):
        st.write(f"{number}. **{label}**  ·  `{script}`")

def run_pipeline(event_queue):
    started = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_logs = []

    def emit(event):
        event_queue.put(event)

    emit({"type": "started", "run_id": run_id})

    for index, (label, relative_script) in enumerate(PIPELINE, start=1):
        script_path = ROOT / relative_script
        stage_start = time.time()

        result = {
            "stage": index,
            "name": label,
            "script": relative_script,
            "status": "RUNNING",
            "duration_seconds": None,
            "return_code": None,
            "log_file": f"stage_{index:02d}.log",
        }
        results.append(result)
        emit({"type": "stage", "result": dict(result)})

        if not script_path.is_file():
            result.update({
                "status": "FAIL",
                "return_code": None,
                "duration_seconds": round(time.time() - stage_start, 2),
                "error": f"Script not found: {relative_script}",
            })
            emit({"type": "stage", "result": dict(result)})
            all_logs.append(f"\n===== {label} =====\n{result['error']}\n")
            break

        command = [sys.executable, "-u", str(script_path)]
        emit({
            "type": "log",
            "text": f"\n===== [{index}/{len(PIPELINE)}] {label} =====\n$ {' '.join(command)}\n",
        })

        log_lines = []
        try:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            for line in iter(process.stdout.readline, ""):
                log_lines.append(line)
                emit({"type": "log", "text": line})

            process.stdout.close()
            return_code = process.wait()
            duration = round(time.time() - stage_start, 2)

            result.update({
                "status": "PASS" if return_code == 0 else "FAIL",
                "return_code": return_code,
                "duration_seconds": duration,
            })

            if return_code != 0:
                result["error"] = f"Command exited with code {return_code}."

        except Exception as exc:
            duration = round(time.time() - stage_start, 2)
            result.update({
                "status": "FAIL",
                "return_code": None,
                "duration_seconds": duration,
                "error": f"{type(exc).__name__}: {exc}",
            })
            log_lines.append(f"\nERROR: {result['error']}\n")

        log_text = "".join(log_lines)
        all_logs.append(f"\n===== {label} =====\n{log_text}")
        (run_dir / result["log_file"]).write_text(log_text, encoding="utf-8")
        emit({"type": "stage", "result": dict(result)})

        if result["status"] != "PASS":
            emit({
                "type": "log",
                "text": f"\nStopped after failure in: {label}\n",
            })
            break

    overall_status = (
        "PASS"
        if len(results) == len(PIPELINE)
        and all(item["status"] == "PASS" for item in results)
        else "FAIL"
    )

    report = {
        "run_id": run_id,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "overall_status": overall_status,
        "steps_expected": len(PIPELINE),
        "steps_completed": len(results),
        "steps_passed": sum(item["status"] == "PASS" for item in results),
        "steps_failed": sum(item["status"] == "FAIL" for item in results),
        "duration_seconds": round(time.time() - started, 2),
        "results": results,
    }

    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "full_run.log").write_text(
        "".join(all_logs),
        encoding="utf-8",
    )

    emit({
        "type": "done",
        "report": report,
        "log": "".join(all_logs),
        "report_path": str(run_dir / "report.json"),
        "log_path": str(run_dir / "full_run.log"),
    })


def drain_events():
    changed = False
    event_queue = st.session_state.event_queue

    while True:
        try:
            event = event_queue.get_nowait()
        except queue.Empty:
            break

        changed = True
        event_type = event.get("type")

        if event_type == "started":
            st.session_state.events.append(
                f"Run started: {event['run_id']}"
            )
        elif event_type == "log":
            st.session_state.events.append(event["text"])
        elif event_type == "stage":
            result = event["result"]
            found = False
            for position, existing in enumerate(st.session_state.results):
                if existing["stage"] == result["stage"]:
                    st.session_state.results[position] = result
                    found = True
                    break
            if not found:
                st.session_state.results.append(result)
        elif event_type == "done":
            st.session_state.final_report = event["report"]
            st.session_state.full_log = event["log"]
            st.session_state.report_path = event["report_path"]
            st.session_state.log_path = event["log_path"]
            st.session_state.running = False

    return changed


@st.fragment(run_every="1s")
def live_status():
    drain_events()

    report = st.session_state.get("final_report")
    results = st.session_state.results

    if st.session_state.running:
        st.info("⏳ Pipeline running. Live logs update automatically.")
    elif report:
        if report["overall_status"] == "PASS":
            st.success(
                f"✅ All {report['steps_expected']} pipeline commands passed."
            )
        else:
            st.error(
                f"❌ Run finished with failures. "
                f"{report['steps_passed']} passed, "
                f"{report['steps_failed']} failed."
            )
    else:
        st.info("Ready. No pipeline run has been started in this session.")

    if results:
        st.markdown("### Stage results")
        st.dataframe(
            [
                {
                    "Stage": item["stage"],
                    "Name": item["name"],
                    "Script": item["script"],
                    "Status": item["status"],
                    "Duration (s)": item.get("duration_seconds"),
                    "Exit code": item.get("return_code"),
                    "Error": item.get("error", ""),
                }
                for item in results
            ],
            use_container_width=True,
            hide_index=True,
        )

    if report:
        st.markdown("### Run summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall", report["overall_status"])
        col2.metric("Passed", report["steps_passed"])
        col3.metric("Failed", report["steps_failed"])
        col4.metric("Duration", f"{report['duration_seconds']} s")

        if st.session_state.get("report_path"):
            report_path = Path(st.session_state.report_path)
            if report_path.exists():
                st.download_button(
                    "Download JSON report",
                    data=report_path.read_bytes(),
                    file_name=f"pipeline_report_{report['run_id']}.json",
                    mime="application/json",
                    key="download_report",
                )

        if st.session_state.get("log_path"):
            log_path = Path(st.session_state.log_path)
            if log_path.exists():
                st.download_button(
                    "Download full run log",
                    data=log_path.read_bytes(),
                    file_name=f"pipeline_log_{report['run_id']}.log",
                    mime="text/plain",
                    key="download_log",
                )

    with st.expander("Live execution logs", expanded=True):
        log_text = "".join(st.session_state.events)
        st.code(log_text[-30000:] if log_text else "No logs yet.", language="text")


st.markdown("### Run controls")
confirmed = st.checkbox(
    "I understand that running the pipeline can modify generated outputs and state files.",
    value=False,
    disabled=st.session_state.running,
)

col1, col2 = st.columns(2)

with col1:
    run_clicked = st.button(
        "▶ Start full pipeline",
        type="primary",
        disabled=st.session_state.running or not confirmed,
        use_container_width=True,
    )

with col2:
    clear_clicked = st.button(
        "Clear displayed results",
        disabled=st.session_state.running,
        use_container_width=True,
    )

if clear_clicked:
    st.session_state.events = []
    st.session_state.results = []
    st.session_state.final_report = None
    st.session_state.event_queue = queue.Queue()
    for key in ("full_log", "report_path", "log_path"):
        st.session_state.pop(key, None)
    st.rerun()

if run_clicked:
    st.session_state.events = []
    st.session_state.results = []
    st.session_state.final_report = None
    st.session_state.event_queue = queue.Queue()
    st.session_state.running = True

    worker = threading.Thread(
        target=run_pipeline,
        args=(st.session_state.event_queue,),
        daemon=True,
    )
    worker.start()
    st.rerun()

live_status()
