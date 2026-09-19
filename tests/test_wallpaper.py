import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from caelestia.utils.wallpaper import (
    check_wall,
    extract_video_thumbnail,
    extract_video_thumbnails,
    get_video_thumbnail_path,
    is_valid_wallpaper,
    video_thumbnail_hash,
)


class VideoWallpaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise unittest.SkipTest("ffmpeg and ffprobe are required")

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.video = self.root / "wallpaper.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=320x180:d=1",
                "-pix_fmt",
                "yuv420p",
                str(self.video),
            ],
            check=True,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_video_is_a_valid_wallpaper_and_can_be_size_filtered(self) -> None:
        self.assertTrue(is_valid_wallpaper(self.video))
        self.assertTrue(check_wall(self.video, (320, 180), 1))
        self.assertFalse(check_wall(self.video, (640, 360), 1))

    def test_hash_matches_shell_djb2_implementation(self) -> None:
        self.assertEqual(video_thumbnail_hash("/tmp/video.mp4"), "1879601642")

    def test_extracts_a_256_by_144_jpeg_to_the_shell_cache_path(self) -> None:
        cache = self.root / "cache"
        with patch("caelestia.utils.wallpaper.c_cache_dir", cache):
            thumbnail = extract_video_thumbnail(self.video)
            self.assertEqual(thumbnail, get_video_thumbnail_path(self.video))
            with Image.open(thumbnail) as image:
                self.assertEqual(image.size, (256, 144))
                self.assertEqual(image.format, "JPEG")

    def test_extracts_thumbnails_recursively_and_reuses_cache(self) -> None:
        nested = self.root / "nested"
        nested.mkdir()
        second_video = nested / "second.webm"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=640x360:d=1",
                str(second_video),
            ],
            check=True,
        )

        cache = self.root / "cache"
        with patch("caelestia.utils.wallpaper.c_cache_dir", cache):
            thumbnails = extract_video_thumbnails(self.root)
            mtimes = {thumbnail: thumbnail.stat().st_mtime_ns for thumbnail in thumbnails}
            self.assertEqual(len(thumbnails), 2)
            self.assertEqual(extract_video_thumbnails(self.root), thumbnails)
            self.assertEqual({thumbnail: thumbnail.stat().st_mtime_ns for thumbnail in thumbnails}, mtimes)

    def test_setting_a_video_persists_the_video_and_a_still_thumbnail(self) -> None:
        from caelestia.utils.wallpaper import set_wallpaper

        state = self.root / "state"
        cache = self.root / "cache"
        scheme = Mock(name="default", colours={}, mode="dark")

        with (
            patch("caelestia.utils.wallpaper.c_cache_dir", cache),
            patch("caelestia.utils.wallpaper.wallpapers_cache_dir", cache / "wallpapers"),
            patch("caelestia.utils.wallpaper.wallpaper_path_path", state / "wallpaper/path.txt"),
            patch("caelestia.utils.wallpaper.wallpaper_link_path", state / "wallpaper/current"),
            patch("caelestia.utils.wallpaper.wallpaper_thumbnail_path", state / "wallpaper/thumbnail.jpg"),
            patch("caelestia.utils.wallpaper.get_scheme", return_value=scheme),
            patch("caelestia.utils.wallpaper.apply_colours"),
            patch("caelestia.utils.wallpaper.get_config", return_value={}),
        ):
            set_wallpaper(self.video, no_smart=False)

        self.assertEqual((state / "wallpaper/path.txt").read_text(), str(self.video.resolve()))
        self.assertEqual((state / "wallpaper/current").resolve(), self.video.resolve())
        with Image.open((state / "wallpaper/thumbnail.jpg").resolve()) as image:
            self.assertEqual(image.format, "JPEG")
        scheme.update_colours.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
