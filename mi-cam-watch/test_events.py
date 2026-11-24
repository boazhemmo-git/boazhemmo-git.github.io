"""Tests for snapshot retention.  python -m unittest test_events -v"""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from watch import EventSink


class PruneTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "cam").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, name, age_days, size=1024):
        p = self.dir / "cam" / name
        p.write_bytes(b"x" * size)
        t = time.time() - age_days * 86400
        os.utime(p, (t, t))
        with (self.dir / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"image": f"cam/{name}", "time": name}) + "\n")

    def names(self):
        return sorted(p.name for p in (self.dir / "cam").glob("*.jpg"))

    def logged(self):
        return [json.loads(x)["image"] for x in
                (self.dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_age_limit_removes_old_and_their_log_lines(self):
        self.add("old.jpg", 20)
        self.add("new.jpg", 1)
        EventSink(self.dir, {}, retain_days=14, max_mb=0).prune()
        self.assertEqual(self.names(), ["new.jpg"])
        self.assertEqual(self.logged(), ["cam/new.jpg"])

    def test_size_limit_removes_oldest_first(self):
        mb = 1024 * 1024
        for i, age in enumerate((5, 4, 3, 2, 1)):
            self.add(f"{i}.jpg", age, size=mb)
        EventSink(self.dir, {}, retain_days=0, max_mb=2.5).prune()
        self.assertEqual(self.names(), ["3.jpg", "4.jpg"])
        self.assertEqual(self.logged(), ["cam/3.jpg", "cam/4.jpg"])

    def test_nothing_to_remove_leaves_everything(self):
        self.add("a.jpg", 1)
        EventSink(self.dir, {}, retain_days=14, max_mb=2000).prune()
        self.assertEqual(self.names(), ["a.jpg"])
        self.assertEqual(self.logged(), ["cam/a.jpg"])


if __name__ == "__main__":
    unittest.main()
