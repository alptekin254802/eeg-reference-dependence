"""Checks for pair weighting, ties and fold score offsets."""
import sys
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis/core"))
from within_fold_auc import summarize, sensitivity


class WithinFoldAucTests(unittest.TestCase):
    def test_pair_weighting_and_ties_match_direct_comparisons(self):
        y = np.array([1, 0, 1, 0, 0, 0])
        q = np.array([.5, .5, .7, .1, .8, .9])
        folds = [{"fold": 0, "test": [0, 1]}, {"fold": 1, "test": [2, 3, 4, 5]}]
        r = summarize(y, q, folds)
        wins = []
        for f in folds:
            ids = np.array(f["test"])
            d = q[ids][y[ids] == 1, None] - q[ids][y[ids] == 0][None, :]
            wins.extend(((d > 0) + .5 * (d == 0)).ravel())
        self.assertAlmostEqual(r["pair_weighted_within_fold_auc"][0], np.mean(wins))
        self.assertNotAlmostEqual(r["macro_within_fold_auc"][0], np.mean(wins))

    def test_fold_offsets_change_pooled_but_not_within_fold_auc(self):
        y = [1, 0, 1, 0]
        folds = [{"fold": 0, "test": [0, 1]}, {"fold": 1, "test": [2, 3]}]
        r = summarize(y, [[.2, .1, .9, .8], [.9, .8, .2, .1]], folds)
        self.assertEqual(r["pair_weighted_within_fold_auc"], [1., 1.])
        self.assertEqual(r["pooled_auc"], [.75, .75])
        self.assertEqual(r["within_fold_delta"], 0.)
        self.assertEqual(summarize(y, [.2, .1, .2, .1], folds)["pooled_auc"], [1.])

    def test_reject_invalid_inputs(self):
        good = [{"fold": 0, "test": [0, 1]}, {"fold": 1, "test": [2, 3]}]
        bad = [good + [good[0]], [{"fold": 0, "test": [0, 2]}, {"fold": 1, "test": [1, 3]}], good[:1]]
        for folds in bad:
            with self.assertRaises(ValueError):
                summarize([1, 0, 1, 0], [.8, .1, .7, .2], folds)
        with self.assertRaises(ValueError):
            summarize([1, 0, 1, 0], [.8, .1, np.nan, .2], good)

    def test_public_report_reads_saved_logits_and_ensembles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, training, output = [root / x for x in ["cache", "training", "report"]]
            cache.mkdir(); training.mkdir()
            np.save(cache / "labels.npy", [1, 0, 1, 0])
            folds = [{"fold": 0, "test": [0, 1]}, {"fold": 1, "test": [2, 3]}]
            (cache / "folds.json").write_text(json.dumps(folds))
            jobs = []
            for arm in ["linked_basic", "average_basic"]:
                for seed in [1109, 2209, 3309]:
                    for fold in folds:
                        job = dict(fold, id=f"{arm}_{seed}_{fold['fold']}", mode="fixed30",
                                   architecture="eegnet", arm=arm, seed=seed, split_seed=20261109)
                        jobs.append(job)
                        logits = [2., -2.] if arm == "linked_basic" else [-2., 2.]
                        result = {"job": job, "test_ids": job["test"], "epochs": 30,
                                  "logits_by_epoch": {"30": logits}}
                        (training / (job["id"] + ".json")).write_text(json.dumps(result))
            plan = root / "plan.json"
            plan.write_text(json.dumps({"training_jobs": jobs}))
            script = Path(__file__).resolve().parents[1] / "analysis/core/report.py"
            run = subprocess.run([sys.executable, "-B", str(script), "--input", str(cache),
                                  "--plan", str(plan), "--training", str(training),
                                  "--output", str(output), "--within-fold-only"],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((output / "within_fold_auc.json").read_text())
            self.assertEqual(report["comparisons"]["main_ensemble"]["within_fold_delta"], 1.)
            self.assertFalse((output / "classification.json").exists())

    def test_incomplete_groups_are_unavailable(self):
        r = sensitivity(np.array([1, 0]), {}, {}, [{"fold": 0, "test": [0, 1]}])
        self.assertEqual(r["comparisons"], {})
        self.assertEqual(len(r["unavailable"]), 6)


if __name__ == "__main__":
    unittest.main()
