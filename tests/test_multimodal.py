import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from pipeline.cases import discover_cases, samples_for_video
from pipeline.config import load_config
from pipeline.judge import judge_video, parse_scores
from pipeline.media import MediaInfo
from pipeline.references import discover_references, fingerprint, validate_references


class MultimodalTests(unittest.TestCase):
    def test_service_score_labels_can_repeat_axis_definitions(self):
        text = "<think>visual quality: 1\ntext-to-video alignment: 1\nphysical/common-sense consistency: 1</think>\n(1) visual quality — clarity, stability, and visible artifacts across the whole clip.: 3\n(2) text-to-video alignment — whether the clip shows the subject, action, and scene in the prompt.: 4\n(3) physical/common-sense consistency — whether visible motion and object relations follow everyday physics and common sense.: 3"
        self.assertEqual(parse_scores(text), {"visual_quality": 3, "text_video_alignment": 4, "physical_consistency": 3})

    def test_manifest_reference_index_is_normalized(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "input_01.png"
            Image.new("RGB", (20, 10)).save(image)
            refs = validate_references("r2va", [{"type": "image", "index": "1", "path": str(image)}], "")
            self.assertEqual(refs[0]["index"], 1)

    def test_actual_first_last_frames_and_reference_video(self):
        import subprocess
        from pipeline.context import build_context
        from pipeline.ffmpeg import ffmpeg_executable
        from pipeline.media import probe_video
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, color in enumerate(((255, 0, 0), (0, 0, 255))):
                Image.new("RGB", (48, 32), color).save(root / f"src{index}.png")
            video = root / "video.mp4"
            subprocess.run([ffmpeg_executable(), "-y", "-v", "error", "-framerate", "1", "-i", str(root / "src%d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True, timeout=30)
            config = load_config().judge
            refs = [{"type": "image", "index": i + 1, "path": str(root / f"src{i}.png")} for i in (0, 1)]
            build_context(probe_video(video), "fl2va", refs, config, root / "endpoints")
            for filename, channel in (("generated-first.jpg", 0), ("generated-last.jpg", 2)):
                with Image.open(root / "endpoints" / "reference_media" / filename) as frame:
                    pixel = frame.getpixel((0, 0))
                    self.assertGreater(pixel[channel], 200)
            content, evidence = build_context(probe_video(video), "r2va", [{"type": "video", "index": 1, "path": str(video)}], config, root / "reference")
            self.assertEqual(evidence[0]["coverage"], "visual frames")
            self.assertTrue(any("Reference video 1" in part.get("text", "") for part in content))

    def test_modes_and_reference_mapping_without_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for mode in ("f2va", "l2va", "fl2va", "r2va"):
                case = root / "dimension" / mode / "same"
                case.mkdir(parents=True)
                (case / "prompt.txt").write_text("参考图1", encoding="utf-8")
                (case / "video.mp4").write_bytes(b"video")
                Image.new("RGB", (20, 10)).save(case / "input_01.png")
                if mode == "fl2va":
                    Image.new("RGB", (20, 10)).save(case / "input_02.jpg")
                (case / "metadata.json").write_text("not valid json", encoding="utf-8")
            found, missing = discover_cases(root, ["f2va", "l2va", "fl2va", "r2va"], ["same"])
            samples, skipped = samples_for_video(found, "prompt.txt", "video.mp4")
            self.assertFalse(missing or skipped)
            self.assertEqual({s["mode"] for s in samples}, {"f2va", "l2va", "fl2va", "r2va"})
            self.assertEqual(len(next(s for s in samples if s["mode"] == "fl2va")["references"]), 2)

    def test_missing_ambiguous_and_duplicate_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.new("RGB", (20, 10)).save(root / "input_01.png")
            with self.assertRaisesRegex(ValueError, "端点"):
                discover_references(root, "fl2va", "")
            with self.assertRaisesRegex(ValueError, "不存在"):
                discover_references(root, "r2va", "使用图片5")
            Image.new("RGB", (20, 10)).save(root / "input_01.jpg")
            with self.assertRaisesRegex(ValueError, "重复"):
                discover_references(root, "f2va", "")

    def test_reference_images_reach_existing_three_axis_judge(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "input_01.png"
            Image.new("RGB", (20, 10)).save(image)
            media = MediaInfo(root / "video.mp4", 2, 24, 20, 10)
            frames = [(0.5, image)]
            response = {"choices": [{"message": {"content": "visual quality: 4\ntext-to-video alignment: 3\nphysical/common-sense consistency: 5"}}]}
            config = load_config().judge
            with patch("pipeline.judge.extract_frames", return_value=frames), patch("pipeline.judge._complete", return_value=response) as call:
                result = judge_video(media, "图1中的人物", config, root, mode="r2va", references=[{"type": "image", "index": 1, "path": str(image)}])
            content = call.call_args.args[1]
            self.assertTrue(any("Condition image 1" in part.get("text", "") for part in content))
            self.assertEqual(len(result["scores"]), 3)
            self.assertEqual(result["scores"]["text_video_alignment"], 3)

    def test_disabled_audio_does_not_report_a_complete_judgement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "ref_audio_01.wav"
            audio.write_bytes(b"audio")
            media = MediaInfo(root / "video.mp4", 2, 24, 20, 10)
            with self.assertRaisesRegex(RuntimeError, "支持音频"):
                judge_video(media, "参考音频1", load_config().judge, root, mode="r2va", references=[{"type": "audio", "index": 1, "path": str(audio)}])

    def test_fingerprint_changes_with_media_and_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "video.mp4"
            video.write_bytes(b"one")
            sample = {"id": "1", "video": str(video), "prompt": "x", "references": []}
            original = fingerprint(sample, {"model": "a"})
            self.assertNotEqual(original, fingerprint(sample, {"model": "b"}))
            video.write_bytes(b"two")
            self.assertNotEqual(original, fingerprint(sample, {"model": "a"}))


if __name__ == "__main__":
    unittest.main()
