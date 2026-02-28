import unittest
import cv2
from metrics import MSE

class TestMSE(unittest.TestCase):
    
    def setUp(self):
        image_pairs = [
            ("images/image.jpg", "images/image-MSE-142-SSIM-0662.jpg"),
            ("images/other.jpg", "images/other-distorted.jpg"),
        ]
    
        self.pairs = [
            (cv2.imread(original), cv2.imread(distorted))
            for original, distorted in image_pairs
        ]
    
    def test_identical_images(self):
        self.assertEqual(MSE(self.pairs[0][0], self.pairs[0][0]), 0.0)
    
    def test_matches_skimage(self):
        from skimage.metrics import mean_squared_error
        for original, distorted in self.pairs:
            self.assertAlmostEqual(
                MSE(original, distorted),
                mean_squared_error(original, distorted),
                places=5
            )

if __name__ == "__main__":
    unittest.main(verbosity=2)