import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.config import load_config
from pipeline.judge import JudgeResponseError, SCORE_AXES, judge_video, parse_scores
from pipeline.media import MediaInfo
from pipeline.run import evaluate_sample


STRICT = "visual quality: 3\ntext-to-video alignment: 4\nphysical/common-sense consistency: 3\n"


def described(scores):
    return "\n".join(
        f"({i}) {label} — {description}{score}"
        for i, ((_, label, description), score) in enumerate(zip(SCORE_AXES, scores), 1)
    )


class JudgeResponseTests(unittest.TestCase):
    def test_missing_opening_think_tag_still_discards_reasoning(self):
        text = STRICT + "</think>\n" + described((4, 5, 4))
        self.assertEqual(parse_scores(text), {
            "visual_quality": 4, "text_video_alignment": 5, "physical_consistency": 4,
        })

    def test_both_observed_formats_ignore_reasoning_scores(self):
        for scores in [(3, 5, 4), (3, 4, 3)]:
            with self.subTest(scores=scores):
                text = "<think>" + STRICT + "</think>\n" + described(scores)
                self.assertEqual(parse_scores(text), dict(zip((axis[0] for axis in SCORE_AXES), scores)))

    def test_invalid_or_ambiguous_answers_are_rejected(self):
        fallback = described((3, 4, 3))
        bad_outputs = [
            "", "<think>" + STRICT, "<think>" + STRICT + "</think>",
            STRICT.rsplit("physical/", 1)[0],
            fallback.rsplit("\n", 1)[0],
            STRICT + STRICT, fallback + "\n" + fallback,
            STRICT + "\n" + fallback,
            STRICT.replace("quality: 3", "quality: 3.5"),
            STRICT.replace("quality: 3", "quality: 3/5"),
            fallback.replace("clip.3", "clip.6"),
            fallback.replace("clip.3", "clip.0"),
            fallback.replace("clip.3", "clip.3.5"),
            fallback.replace("clip.3", "clip.3/5"),
            fallback.replace("clip.3", "clip.3 or 4"),
            fallback.replace("clarity, stability, and visible artifacts across the whole clip.", "some other explanation."),
        ]
        for text in bad_outputs:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_scores(text)

    def _call(self, directory, payload):
        media = MediaInfo(Path("/tmp/video.mp4"), 15.0, 24.0, 720, 1280)
        with patch("pipeline.judge.build_context", return_value=([], [])), \
                patch("pipeline.judge.extract_frames", return_value=[]), \
                patch("pipeline.judge._complete", return_value=payload):
            return judge_video(media, "prompt", load_config().judge, Path(directory))

    def test_success_retains_raw_response_and_metadata(self):
        payload = {"choices": [{"message": {"content": described((3, 5, 4))}, "finish_reason": "stop"}],
                   "usage": {"completion_tokens": 451}}
        with tempfile.TemporaryDirectory() as temp:
            result = self._call(temp, payload)
            audit = json.loads(Path(result["response_file"]).read_text())
            self.assertEqual(audit["response"], payload)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["finish_reason"], "stop")
            self.assertEqual(result["usage"], payload["usage"])
            self.assertEqual(result["raw_text"], described((3, 5, 4)))
            self.assertEqual(result["scores"]["text_video_alignment"], 5)

    def test_failed_responses_are_retained_before_error(self):
        payloads = [
            {"choices": [{"message": {"content": "unparseable"}, "finish_reason": "stop"}]},
            {"choices": [{"message": {"content": STRICT}, "finish_reason": "length"}]},
            {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]},
            {"choices": []}, {"choices": [None]},
        ]
        for payload in payloads:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(JudgeResponseError) as raised:
                    self._call(temp, payload)
                result = raised.exception.result
                audit = json.loads(Path(result["response_file"]).read_text())
                self.assertEqual(audit["response"], payload)
                self.assertEqual(result["status"], "error")
                self.assertIn("parse_error", audit)
                self.assertNotIn("scores", result)

    def test_attempts_keep_separate_response_files(self):
        payload = {"choices": [{"message": {"content": STRICT}, "finish_reason": "stop"}]}
        with tempfile.TemporaryDirectory() as temp:
            first = self._call(temp, payload)
            second = self._call(temp, payload)
            self.assertNotEqual(first["response_file"], second["response_file"])
            self.assertTrue(Path(first["response_file"]).is_file())

    def test_pipeline_preserves_error_diagnostics_and_technical_scores(self):
        diagnostic = {"status": "error", "raw_text": "bad answer", "finish_reason": "stop",
                      "usage": {"completion_tokens": 3}, "error": "invalid scores", "response_file": "audit.json"}
        media = MediaInfo(Path("/tmp/video.mp4"), 15.0, 24.0, 720, 1280)
        technical = {"status": "ok", "dimensions": {str(i): {} for i in range(6)}}
        existing = {"sample": {}, "reasons": [], "technical": technical}
        with tempfile.TemporaryDirectory() as temp, patch("pipeline.run.probe_video", return_value=media), \
                patch("pipeline.run.judge_video", side_effect=JudgeResponseError(diagnostic)):
            for _ in range(2):
                existing = evaluate_sample(
                    sample_id="case", prompt="prompt", video=media.path, output_dir=Path(temp),
                    config=load_config(), run_technical=False, existing_record=existing,
                )
            persisted = json.loads((Path(temp) / "case" / "result.json").read_text())
            self.assertEqual(persisted["judge"], diagnostic)
            self.assertEqual(persisted["technical"], technical)
            self.assertEqual(persisted["reasons"], ["裁判：invalid scores"])
            self.assertEqual(persisted["status"], "incomplete")
            self.assertNotIn("final", persisted)


if __name__ == "__main__":
    unittest.main()
