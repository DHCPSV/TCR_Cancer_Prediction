from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from pipeline import reproduction_cache


class RawRepertoirePackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name)
        self.name = "data/raw/internal/cancer/alpha/pooled.tsv"
        self.raw = self.repo / self.name
        self.raw.parent.mkdir(parents=True)
        self.raw.write_bytes(b"LTX_ID\tcdr3\nLTX0001\tCAVF\n")
        self.digest = hashlib.sha256(self.raw.read_bytes()).hexdigest()

    def manifest(self, rows: list[tuple[str, str]]) -> None:
        path = self.repo / "artifacts/manifests/samples.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["raw_file", "raw_sha256"])
            writer.writerows(rows)

    def test_shared_raw_tables_are_included_once(self) -> None:
        self.manifest([(self.name, self.digest)] * 2)
        self.assertEqual(
            reproduction_cache.raw_file_hashes(self.repo), {self.name: self.digest}
        )

    def test_conflicting_or_missing_checksums_are_rejected(self) -> None:
        for rows in (
            [(self.name, self.digest), (self.name, "0" * 64)],
            [(self.name, "")],
            [],
        ):
            with self.subTest(rows=rows):
                self.manifest(rows)
                with self.assertRaises(ValueError):
                    reproduction_cache.raw_file_hashes(self.repo)

    def test_paths_must_stay_within_raw_data(self) -> None:
        for name in (
            "../outside.tsv",
            "data/raw/../outside.tsv",
            "artifacts/runs/predictions.tsv",
            "data\\raw\\sample.tsv",
            "data/raw/unrelated.zip",
            "C:/outside.tsv",
        ):
            with self.subTest(name=name):
                self.manifest([(name, self.digest)])
                with self.assertRaises(ValueError):
                    reproduction_cache.raw_file_hashes(self.repo)

    def test_manifest_rejects_changed_raw_data_before_writing(self) -> None:
        self.manifest([(self.name, "0" * 64)])
        target = self.repo / "contents.json"
        with self.assertRaisesRegex(ValueError, "does not match"):
            reproduction_cache.write_contents_manifest([self.raw], target, self.repo)
        self.assertFalse(target.exists())

    def test_raw_records_are_written_with_size_and_hash(self) -> None:
        self.manifest([(self.name, self.digest)])
        target = self.repo / "contents.json"
        payload = reproduction_cache.write_contents_manifest([self.raw], target, self.repo)
        self.assertEqual(payload["files"], [{
            "path": self.name, "bytes": self.raw.stat().st_size, "sha256": self.digest
        }])


if __name__ == "__main__":
    unittest.main()
