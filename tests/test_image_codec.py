"""
Tests for the image codec module: DPCM predictor, DPCM encode/decode
round-trips, and end-to-end ImageEncoder / ImageDecoder.
"""

import unittest
import tempfile
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.image_codec import (
    _med_predictor,
    dpcm_encode,
    dpcm_decode,
    ImageEncoder,
    ImageDecoder,
)
from src.utils import write_image, read_image


# ======================================================================
#  MED Predictor Tests
# ======================================================================

class TestMEDPredictor(unittest.TestCase):
    """Test the Median-Edge-Detector predictor logic."""

    def test_flat_region(self):
        """MED on a flat (constant) region: prediction = value at (left, top)."""
        # a=100, b=100, c=100. Since c >= max(a,b), pred = min(a,b) = 100
        pred = _med_predictor(top=100, left=100, top_left=100)
        self.assertEqual(pred, 100)

    def test_horizontal_edge(self):
        """Horizontal edge: intensity changes row-to-row, not column-to-column."""
        # top row is dark (50), current row is light (200). c=50 <= min(50,200)
        # -> pred = max(a,b) = 200
        pred = _med_predictor(top=50, left=200, top_left=50)
        self.assertEqual(pred, 200)

    def test_vertical_edge(self):
        """Vertical edge: intensity changes column-to-column, not row-to-row."""
        # left col is dark (30), top is light (180). c >= max(180,30)=180
        # -> pred = min(180,30) = 30
        pred = _med_predictor(top=180, left=30, top_left=30)
        self.assertEqual(pred, 180)

    def test_diagonal_edge(self):
        """Smooth diagonal: standard a+b-c predictor."""
        # a=150, b=100, c=120. c is between min(100,150)=100 and max=150
        # -> pred = 100 + 150 - 120 = 130
        pred = _med_predictor(top=150, left=100, top_left=120)
        self.assertEqual(pred, 130)

    def test_corner_case_top_less_than_left(self):
        """c >= max(a,b) branch when top < left."""
        pred = _med_predictor(top=40, left=100, top_left=120)
        # c=120 >= max(40,100)=100 -> min(100,40)=40
        self.assertEqual(pred, 40)

    def test_corner_case_top_greater_than_left(self):
        """c <= min(a,b) branch when top > left."""
        pred = _med_predictor(top=180, left=60, top_left=50)
        # c=50 <= min(60,180)=60 -> max(60,180)=180
        self.assertEqual(pred, 180)


# ======================================================================
#  DPCM Encode / Decode Round-Trip Tests
# ======================================================================

class TestDPCMRoundTrip(unittest.TestCase):
    """Verify DPCM encode followed by decode recovers the original image."""

    def _assert_roundtrip(self, image: np.ndarray, msg: str = ""):
        """Helper: encode then decode and verify lossless reconstruction."""
        mapped, base = dpcm_encode(image)
        reconstructed = dpcm_decode(mapped, base)
        np.testing.assert_array_equal(
            image, reconstructed,
            err_msg=f"DPCM round-trip failed: {msg}"
        )

    def test_constant_image(self):
        """All pixels the same value: 128."""
        img = np.full((16, 16), 128, dtype=np.uint8)
        self._assert_roundtrip(img, "constant 128")

    def test_black_image(self):
        """All pixels 0."""
        img = np.zeros((8, 8), dtype=np.uint8)
        self._assert_roundtrip(img, "black")

    def test_white_image(self):
        """All pixels 255."""
        img = np.full((8, 8), 255, dtype=np.uint8)
        self._assert_roundtrip(img, "white")

    def test_horizontal_gradient(self):
        """Left-to-right gradient from 0 to 255."""
        img = np.tile(np.arange(256, dtype=np.uint8), (16, 1))
        self._assert_roundtrip(img, "horizontal gradient")

    def test_vertical_gradient(self):
        """Top-to-bottom gradient from 0 to 255."""
        img = np.tile(np.arange(256, dtype=np.uint8).reshape(-1, 1), (1, 16)).astype(np.uint8)
        self._assert_roundtrip(img, "vertical gradient")

    def test_checkerboard(self):
        """4x4 checkerboard pattern."""
        img = np.zeros((16, 16), dtype=np.uint8)
        img[::2, ::2] = 255
        img[1::2, 1::2] = 255
        self._assert_roundtrip(img, "checkerboard")

    def test_random_noise(self):
        """Small random image (should still round-trip losslessly)."""
        rng = np.random.RandomState(42)
        img = rng.randint(0, 256, size=(32, 32)).astype(np.uint8)
        self._assert_roundtrip(img, "random noise")

    def test_single_pixel(self):
        """Smallest possible image: 1x1."""
        img = np.array([[127]], dtype=np.uint8)
        self._assert_roundtrip(img, "1x1")

    def test_single_row(self):
        """1xN image."""
        img = np.linspace(0, 255, 50, dtype=np.uint8).reshape(1, -1)
        self._assert_roundtrip(img, "single row")

    def test_single_column(self):
        """Nx1 image."""
        img = np.linspace(0, 255, 50, dtype=np.uint8).reshape(-1, 1)
        self._assert_roundtrip(img, "single column")

    def test_residual_range(self):
        """Verify DPCM residual (before mapping) stays within [-255, 255]."""
        rng = np.random.RandomState(123)
        img = rng.randint(0, 256, size=(50, 50)).astype(np.uint8)
        mapped, _ = dpcm_encode(img)
        residual_min = int(mapped.min()) - 255
        residual_max = int(mapped.max()) - 255
        self.assertGreaterEqual(residual_min, -255,
                                f"Residual too negative: {residual_min}")
        self.assertLessEqual(residual_max, 255,
                             f"Residual too positive: {residual_max}")


# ======================================================================
#  ImageEncoder / ImageDecoder Integration Tests
# ======================================================================

class TestImageCodecRoundTrip(unittest.TestCase):
    """End-to-end ImageEncoder -> ImageDecoder round-trip with real file I/O."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="huf_img_test_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_png(self, name: str, img: np.ndarray) -> str:
        """Write a temporary PNG image and return its path."""
        path = os.path.join(self.tmpdir, name)
        write_image(path, img)
        return path

    def _encode_decode(self, img: np.ndarray, grayscale: bool, mode: str,
                       label: str = ""):
        """Encode a numpy image array, decode it, and verify lossless recovery."""
        in_path = self._write_png(f"input_{label}.png", img)
        huf_path = os.path.join(self.tmpdir, f"compressed_{label}.huf")

        # Encode
        encoder = ImageEncoder()
        report = encoder.encode(in_path, huf_path,
                                grayscale=grayscale, mode=mode)

        # Determine shape for decoder
        shape = img.shape

        # Decode
        decoder = ImageDecoder()
        reconstructed = decoder.decode(huf_path, shape=shape,
                                       grayscale=grayscale, mode=mode)

        # Verify lossless
        np.testing.assert_array_equal(
            img, reconstructed,
            err_msg=f"Image codec round-trip failed (grayscale={grayscale}, "
                    f"mode={mode}, {label})"
        )
        return report

    # ---- Grayscale ----

    def test_gray_direct_constant(self):
        img = np.full((8, 8), 128, dtype=np.uint8)
        self._encode_decode(img, grayscale=True, mode="direct", label="gray_const")

    def test_gray_direct_gradient(self):
        img = np.tile(np.arange(64, dtype=np.uint8), (8, 1))
        self._encode_decode(img, grayscale=True, mode="direct", label="gray_grad")

    def test_gray_dpcm_constant(self):
        img = np.full((16, 16), 100, dtype=np.uint8)
        self._encode_decode(img, grayscale=True, mode="dpcm", label="gray_dpcm_const")

    def test_gray_dpcm_gradient(self):
        img = np.tile(np.arange(128, dtype=np.uint8), (16, 1))
        self._encode_decode(img, grayscale=True, mode="dpcm", label="gray_dpcm_grad")

    def test_gray_dpcm_random(self):
        rng = np.random.RandomState(7)
        img = rng.randint(0, 256, size=(16, 16)).astype(np.uint8)
        self._encode_decode(img, grayscale=True, mode="dpcm", label="gray_dpcm_rand")

    # ---- Color (3-channel) ----

    def test_color_direct_constant(self):
        img = np.full((8, 8, 3), [64, 128, 192], dtype=np.uint8)
        self._encode_decode(img, grayscale=False, mode="direct",
                            label="color_const")

    def test_color_direct_gradient(self):
        h, w = 8, 16
        r = np.tile(np.arange(w, dtype=np.uint8), (h, 1))
        g = np.tile(np.arange(h, dtype=np.uint8).reshape(-1, 1), (1, w))
        b = np.full((h, w), 128, dtype=np.uint8)
        img = np.stack([b, g, r], axis=2)
        self._encode_decode(img, grayscale=False, mode="direct",
                            label="color_grad")

    def test_color_dpcm_constant(self):
        img = np.full((8, 8, 3), [10, 100, 200], dtype=np.uint8)
        self._encode_decode(img, grayscale=False, mode="dpcm",
                            label="color_dpcm_const")

    def test_color_dpcm_gradient(self):
        h, w = 12, 16
        r = np.tile(np.arange(w, dtype=np.uint8), (h, 1))
        g = np.tile(np.arange(h, dtype=np.uint8).reshape(-1, 1), (1, w))
        b = np.full((h, w), 64, dtype=np.uint8)
        img = np.stack([b, g, r], axis=2)
        self._encode_decode(img, grayscale=False, mode="dpcm",
                            label="color_dpcm_grad")

    def test_color_synthetic_realistic(self):
        """Small colorful image with random-ish content."""
        rng = np.random.RandomState(42)
        img = rng.randint(0, 256, size=(16, 16, 3)).astype(np.uint8)
        self._encode_decode(img, grayscale=False, mode="direct",
                            label="color_real")


# ======================================================================
#  ImageEncoder Metrics Verification
# ======================================================================

class TestImageEncoderMetrics(unittest.TestCase):
    """Verify that ImageEncoder returns sensible metrics."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="huf_img_metrics_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _encode_img(self, img: np.ndarray, grayscale: bool, mode: str) -> dict:
        in_path = os.path.join(self.tmpdir, "in.png")
        write_image(in_path, img)
        huf_path = os.path.join(self.tmpdir, "out.huf")
        encoder = ImageEncoder()
        return encoder.encode(in_path, huf_path, grayscale=grayscale, mode=mode)

    def test_metrics_present(self):
        img = np.full((8, 8), 128, dtype=np.uint8)
        r = self._encode_img(img, grayscale=True, mode="direct")
        for key in ("entropy", "avg_length", "efficiency",
                    "compression_ratio", "original_size", "compressed_size"):
            self.assertIn(key, r, f"Missing key: {key}")

    def test_dpcm_compression_better_for_constant(self):
        """For a constant image, DPCM should achieve near-perfect compression."""
        img = np.full((32, 32), 100, dtype=np.uint8)
        r = self._encode_img(img, grayscale=True, mode="dpcm")
        # Residuals are all zero -> near-zero entropy, excellent ratio
        self.assertLess(r["entropy"], 0.1,
                        f"DPCM entropy should be near 0 for constant image, got {r['entropy']}")
        self.assertGreater(r["compression_ratio"], 5.0,
                           f"DPCM ratio should be high for constant image, got {r['compression_ratio']}")

    def test_dpcm_entropy_lower_than_direct_for_smooth(self):
        """For a smooth gradient, DPCM residual entropy < direct pixel entropy."""
        img = np.tile(np.arange(128, dtype=np.uint8), (32, 1))
        r_direct = self._encode_img(img, grayscale=True, mode="direct")
        r_dpcm = self._encode_img(img, grayscale=True, mode="dpcm")
        self.assertLessEqual(
            r_dpcm["entropy"], r_direct["entropy"],
            f"DPCM entropy ({r_dpcm['entropy']}) should be <= direct entropy "
            f"({r_direct['entropy']}) for a smooth gradient"
        )

    def test_color_mode_detection(self):
        """Color image should report channels=3."""
        img = np.full((4, 4, 3), [100, 100, 100], dtype=np.uint8)
        r = self._encode_img(img, grayscale=False, mode="direct")
        self.assertEqual(r["channels"], 3)

    def test_grayscale_image_reported(self):
        """Grayscale image should report channels=1."""
        img = np.full((4, 4), 128, dtype=np.uint8)
        r = self._encode_img(img, grayscale=True, mode="direct")
        self.assertEqual(r["channels"], 1)



# ======================================================================
#  真实图片测试
# ======================================================================

class TestRealImage(unittest.TestCase):
    """用真实图片文件进行端到端编码→解码测试"""

    def setUp(self):
        self.image_path = os.path.join(
            os.path.dirname(__file__), '..',
            'results', 'images', 'original', 'test_image.png'
        )
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _roundtrip(self, mode):
        huf_path = os.path.join(self.tmpdir, f"test_{mode}.huf")
        img = read_image(self.image_path)
        shape = img.shape
        is_gray = (len(shape) == 2)
        encoder = ImageEncoder()
        encoder.encode(self.image_path, huf_path, grayscale=is_gray, mode=mode)
        decoder = ImageDecoder()
        reconstructed = decoder.decode(huf_path, shape=shape, grayscale=is_gray, mode=mode)
        np.testing.assert_array_equal(img, reconstructed)

    def test_real_image_direct(self):
        self._roundtrip("direct")

    def test_real_image_dpcm(self):
        self._roundtrip("dpcm")





# ======================================================================
# 风景图测试（使用 results/images/original/landscape_pexels.png）
# ======================================================================

class TestLandscapeImage(unittest.TestCase):
    """用真实风景图片进行端到端编码→解码测试"""

    def setUp(self):
        self.image_path = os.path.join(
            os.path.dirname(__file__), '..',
            'results', 'images', 'original', 'landscape_pexels.png'
        )
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _roundtrip(self, mode):
        huf_path = os.path.join(self.tmpdir, f"landscape_{mode}.huf")
        img = read_image(self.image_path)
        shape = img.shape
        is_gray = (len(shape) == 2)
        encoder = ImageEncoder()
        encoder.encode(self.image_path, huf_path, grayscale=is_gray, mode=mode)
        decoder = ImageDecoder()
        # 彩色图时解码器自动拼接 _ch0/_ch1/_ch2，用原始路径即可
        decode_path = huf_path
        reconstructed = decoder.decode(decode_path, shape=shape, grayscale=is_gray, mode=mode)
        np.testing.assert_array_equal(img, reconstructed)

    def test_landscape_direct(self):
        self._roundtrip("direct")

    def test_landscape_dpcm(self):
        self._roundtrip("dpcm")



if __name__ == '__main__':
    unittest.main(verbosity=2)



