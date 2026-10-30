
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

def load_issue_refund_function():
    """
    Load only issue_refund() from part_d.py.
    Resolve part_d.py relative to the project root,
    regardless of the current working directory.
    """
    import ast

    project_root = Path(__file__).resolve().parent.parent
    source_path = project_root / "part_d.py"
    source = source_path.read_text(encoding="utf-8")

    parsed = ast.parse(source)

    function_node = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "issue_refund"
    )

    isolated_module = ast.Module(
        body=[function_node],
        type_ignores=[],
    )

    namespace = {
        "os": os,
        "json": json,
        "datetime": datetime,
    }

    exec(
        compile(isolated_module, str(source_path), "exec"),
        namespace,
    )

    return namespace["issue_refund"]


class RefundIdempotencyTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)

        os.makedirs("outputs", exist_ok=True)
        self.issue_refund = load_issue_refund_function()

    def tearDown(self):
        os.chdir(self.original_cwd)
        self.temp_dir.cleanup()

    def read_ledger(self):
        path = Path("outputs/refund_ledger.jsonl")

        if not path.exists():
            return []

        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_duplicate_refund_is_not_recorded_twice(self):
        first = self.issue_refund("ORD-TEST-001", 128.25)
        second = self.issue_refund("ORD-TEST-001", 128.25)

        ledger = self.read_ledger()

        self.assertEqual(first["refund_inr"], 128.25)
        self.assertEqual(second["refund_inr"], 128.25)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["order_id"], "ORD-TEST-001")

    def test_different_orders_can_be_refunded(self):
        self.issue_refund("ORD-TEST-001", 128.25)
        self.issue_refund("ORD-TEST-002", 135.00)

        ledger = self.read_ledger()

        self.assertEqual(len(ledger), 2)
        self.assertEqual(
            {entry["order_id"] for entry in ledger},
            {"ORD-TEST-001", "ORD-TEST-002"},
        )

    def test_escalation_does_not_block_a_refund(self):
        path = Path("outputs/refund_ledger.jsonl")
        path.write_text(
            json.dumps({
                "timestamp": datetime.now().isoformat(),
                "tool": "escalate_to_human",
                "order_id": "ORD-TEST-001",
                "refund_inr": 128.25,
            }) + "\n",
            encoding="utf-8",
        )

        self.issue_refund("ORD-TEST-001", 128.25)

        ledger = self.read_ledger()

        self.assertEqual(len(ledger), 2)
        self.assertEqual(
            sum(entry["tool"] == "issue_refund" for entry in ledger),
            1,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
