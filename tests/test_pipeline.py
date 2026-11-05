import tempfile
import unittest
from pathlib import Path

from pipeline.aggregate import scores_by_window, summarize_windows
from pipeline.config import load_config
from pipeline.final_score import final_score
from pipeline.judge import parse_scores, score_prompt
from pipeline.media import probe_video
from pipeline.run import _is_complete, summarize_run
from pipeline.cases import discover_cases, load_eval_run, samples_for_video
from pipeline.technical import TechnicalUnavailable, choose_gpu
from pipeline.windows import split_windows


class AggregateTests(unittest.TestCase):
    def test_worst_window_and_gap_use_time_order(self) -> None:
        summary = summarize_windows([
            {"index": 1, "start": 2.0, "score": 0.4},
            {"index": 0, "start": 0.0, "score": 0.9},
            {"index": 2, "start": 4.0, "score": 0.5},
        ])
        self.assertEqual(summary["worst_window"]["index"], 1)
        self.assertAlmostEqual(summary["first_last_gap"], 0.4)
        self.assertAlmostEqual(summary["mean"], (0.9 + 0.4 + 0.5) / 3)

    def test_vbench_rows_align_to_cut_windows(self) -> None:
        result = [
            0.8,
            [
                {"video_path": "/tmp/win_001.mp4", "video_results": 0.2},
                {"video_path": "/tmp/win_000.mp4", "video_results": [0.4, 0.6]},
            ],
        ]
        clips = [
            {"index": 0, "start": 0.0, "path": "/data/win_000.mp4"},
            {"index": 1, "start": 2.0, "path": "/data/win_001.mp4"},
        ]
        aligned = scores_by_window(result, clips)
        summary = summarize_windows(aligned)
        self.assertAlmostEqual(summary["windows"][0]["score"], 0.5)
        self.assertEqual(summary["worst_window"]["index"], 1)


class JudgeParseTests(unittest.TestCase):
    def test_three_lines_become_scores(self) -> None:
        text = "visual quality: 4\ntext-to-video alignment: 3\nphysical/common-sense consistency: 5\n"
        self.assertEqual(parse_scores(text), {
            "visual_quality": 4,
            "text_video_alignment": 3,
            "physical_consistency": 5,
        })

    def test_score_outside_one_to_five_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_scores("visual quality: 6\ntext-to-video alignment: 3\nphysical/common-sense consistency: 5")

    def test_prompt_is_inside_the_question(self) -> None:
        question = score_prompt("一只猫跳上桌子")
        self.assertIn("一只猫跳上桌子", question)
        self.assertIn("text-to-video alignment", question)


class CompletionTests(unittest.TestCase):
    def test_complete_requires_both_groups(self) -> None:
        record = {
            "judge": {"status": "ok", "scores": {
                "visual_quality": 4, "text_video_alignment": 3, "physical_consistency": 5,
            }},
            "technical": {"status": "ok", "dimensions": {name: {} for name in (
                "subject_consistency", "background_consistency", "motion_smoothness",
                "dynamic_degree", "aesthetic_quality", "imaging_quality",
            )}},
        }
        self.assertTrue(_is_complete(record))
        record["technical"]["status"] = "unavailable"
        self.assertFalse(_is_complete(record))

    def test_summary_keeps_durations_apart(self) -> None:
        def record(duration: float, technical: float, judge: int) -> dict:
            return {
                "dimension": "整段保持与提示遵循",
                "status": "complete",
                "media": {"duration_seconds": duration},
                "technical": {"dimensions": {"subject_consistency": {"mean": technical}}},
                "judge": {"scores": {
                    "visual_quality": judge,
                    "text_video_alignment": judge,
                    "physical_consistency": judge,
                }},
            }
        with tempfile.TemporaryDirectory() as temp:
            summary = summarize_run([record(15.1, 0.8, 4), record(5.0, 0.2, 2)], Path(temp))
        self.assertEqual(summary["groups"]["15s"]["count"], 1)
        self.assertEqual(summary["groups"]["5s"]["technical_mean"]["subject_consistency"], 0.2)


class FinalScoreTests(unittest.TestCase):
    def test_blocks_mix_on_a_one_to_five_scale(self) -> None:
        config = load_config()
        record = {
            "technical": {"status": "ok", "dimensions": {
                "subject_consistency": {"mean": 0.8, "worst_window": {"score": 0.6}, "first_last_gap": 0.2},
                "background_consistency": {"mean": 1.0, "worst_window": {"score": 1.0}, "first_last_gap": -0.1},
                "motion_smoothness": {"mean": 0.5},
                "dynamic_degree": {"mean": 0.0},
                "aesthetic_quality": {"mean": 0.0},
                "imaging_quality": {"mean": 0.25},
            }},
            "judge": {"status": "ok", "scores": {
                "visual_quality": 4, "text_video_alignment": 2, "physical_consistency": 4,
            }},
        }
        scored = final_score(record, config.final_score)
        self.assertIsNotNone(scored)
        assert scored is not None
        self.assertEqual(scored["technical_components"]["subject_consistency"]["score"], 3.4)
        self.assertEqual(scored["technical_components"]["background_consistency"]["score"], 5.0)
        self.assertEqual(scored["technical_score"], 3.31)
        self.assertEqual(scored["judge_score"], 3.0)
        self.assertEqual(scored["score"], 3.16)
        self.assertEqual(scored["excluded"], ["dynamic_degree"])
        self.assertEqual(config.final_score.technical_weight, 0.5)

    def test_incomplete_record_has_no_final_score(self) -> None:
        config = load_config()
        self.assertIsNone(final_score({"technical": {"status": "error"}, "judge": {"status": "ok"}}, config.final_score))


class GpuChoiceTests(unittest.TestCase):
    def test_default_config_uses_gpu_4_or_5(self) -> None:
        config = load_config()
        self.assertEqual(config.technical.gpu, "auto")
        self.assertEqual(config.technical.gpus, ["4", "5"])

    def test_more_free_memory_is_chosen(self) -> None:
        self.assertEqual(choose_gpu([("4", 1000), ("5", 8000)]), "5")
        self.assertEqual(choose_gpu([("4", 9000), ("5", 1000)]), "4")

    def test_tie_keeps_the_earlier_card(self) -> None:
        self.assertEqual(choose_gpu([("4", 4000), ("5", 4000)]), "4")

    def test_unreadable_card_is_skipped(self) -> None:
        self.assertEqual(choose_gpu([("4", None), ("5", 100)]), "5")
        with self.assertRaises(TechnicalUnavailable):
            choose_gpu([("4", None), ("5", None)])


class CaseSelectionTests(unittest.TestCase):
    def test_listed_ids_keep_order_and_report_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for case_id in ("130101", "200101"):
                case_dir = root / "13_style" / "t2va" / case_id
                case_dir.mkdir(parents=True)
                (case_dir / "prompt_final.txt").write_text(f"提示词 {case_id}", encoding="utf-8")
                (case_dir / "线上模型.mp4").write_bytes(b"video")
            found, missing = discover_cases(root, ["t2va"], ["200101", "130101", "999999"])
            self.assertEqual([path.name for path in found], ["200101", "130101"])
            self.assertEqual(missing, ["999999"])
            samples, skipped = samples_for_video(found, "prompt_final.txt", "线上模型.mp4")
            self.assertEqual([item["id"] for item in samples], ["200101", "130101"])
            self.assertEqual(skipped, [])
            self.assertIn("提示词 200101", samples[0]["prompt"])

    def test_eval_config_reads_copied_cases(self) -> None:
        spec = load_eval_run(Path(__file__).resolve().parents[1] / "configs" / "eval_run.yaml")
        self.assertEqual(spec.cases_root, (Path(__file__).resolve().parents[1] / "data" / "固定case").resolve())
        found, missing = discover_cases(spec.cases_root, spec.input_modes, spec.case_ids)
        self.assertEqual(missing, [])
        self.assertEqual([path.name for path in found], spec.case_ids)
        self.assertEqual(len(found), 10)
        self.assertTrue(all(path.parent.parent.name == "固定case" for path in found))
        for video_name in spec.video_filenames:
            samples, skipped = samples_for_video(found, spec.prompt_filename, video_name)
            self.assertEqual(skipped, [])
            self.assertEqual(len(samples), 10)
            for sample in samples:
                self.assertTrue(str(sample["video"]).startswith(str(spec.cases_root)))


class MediaTests(unittest.TestCase):
    def test_probe_and_split_a_short_clip(self) -> None:
        import subprocess

        from PIL import Image

        from pipeline.ffmpeg import ffmpeg_executable
        ffmpeg = ffmpeg_executable()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frames = root / "src"
            frames.mkdir()
            for index in range(40):
                Image.new("RGB", (320, 240), (index, 0, 0)).save(frames / f"f{index:03d}.jpg")
            video = root / "clip.mp4"
            subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-framerate", "8", "-i", str(frames / "f%03d.jpg"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
                check=True, timeout=60,
            )
            media = probe_video(video)
            self.assertGreater(media.duration_seconds, 4.5)
            self.assertEqual(media.width, 320)
            clips, unused = split_windows(media, root / "windows", window_seconds=2.0, min_tail_seconds=1.0)
            self.assertGreaterEqual(len(clips), 2)
            self.assertEqual(clips[0].start, 0.0)
            self.assertLess(unused, 1.0)
            self.assertTrue(clips[0].path.is_file())


if __name__ == "__main__":
    unittest.main()
