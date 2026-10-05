"""test_graph.py output: UTF-8 on a cp1252 console, and the runs/ file with report + summary."""
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from support import ROOT

# runs test_graph.main() in a child process with a fake graph, so stdout encoding can be forced
CHILD = textwrap.dedent("""
    import sys
    sys.path.insert(0, {tests!r})
    import support  # noqa: F401  (network guard)
    from pathlib import Path
    import test_graph

    class FakeGraph:
        def stream(self, *args, **kwargs):
            yield {{"merge": {{"all_candidates": [{{"product_name": "Lenovo V15", "price": 61999,
                "price_source": "page_extract", "price_source_url": "https://www.amazon.in/dp/B0TEST0001"}}]}}}}
            yield {{"report": {{"report": "## Report\\nLenovo V15 — ₹61,999", "report_candidates": [1], "is_degraded": True}}}}
            if {fail}:
                raise RuntimeError("pipeline failed after the report")

    test_graph.build_graph = lambda: FakeGraph()
    test_graph.RUNS_DIR = Path({runs!r})
    sys.argv = ["test_graph.py", "laptop query"]
    test_graph.main()
""")


def run_child(runs_dir, fail=False):
    code = CHILD.format(tests=str(Path(__file__).parent), runs=str(runs_dir), fail=fail)
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}  # what a Windows redirect gives Python
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True)


class RunOutputTest(unittest.TestCase):
    def test_report_with_rupee_prints_on_cp1252_console(self):
        with tempfile.TemporaryDirectory() as runs:
            proc = run_child(runs)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
            out = proc.stdout.decode("utf-8")
            self.assertIn("Lenovo V15 — ₹61,999", out)
            self.assertIn("Candidates priced: 1/1", out)

    def test_report_and_summary_saved_as_utf8_file(self):
        with tempfile.TemporaryDirectory() as runs:
            run_child(runs)
            files = list(Path(runs).glob("run-*.txt"))
            self.assertEqual(len(files), 1)
            text = files[0].read_text(encoding="utf-8")
            self.assertIn("Query: laptop query", text)
            self.assertIn("₹61,999", text)
            self.assertIn("LLM requests:", text)
            self.assertIn("price=61999 via page_extract https://www.amazon.in/dp/B0TEST0001", text)

    def test_summary_saved_even_when_pipeline_raises(self):
        with tempfile.TemporaryDirectory() as runs:
            proc = run_child(runs, fail=True)
            self.assertNotEqual(proc.returncode, 0)
            files = list(Path(runs).glob("run-*.txt"))
            self.assertEqual(len(files), 1)
            self.assertIn("Lenovo V15", files[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
