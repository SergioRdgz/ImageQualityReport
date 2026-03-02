import unittest
import cv2
from metrics import MSE, PSNR, SSIM_windowed, SSIM_windowed_fast
class TestMSE(unittest.TestCase):
    
    def setUp(self):
        image_pairs = [
            ("images/image.jpg", "images/image-MSE-142-SSIM-0662.jpg"),
            ("images/image.jpg", "images/image.jpg"),
            #("images/other.jpg", "images/other-distorted.jpg"),
        ]
    
        self.pairs = [
            (cv2.imread(original), cv2.imread(distorted))
            for original, distorted in image_pairs
        ]
    
    def test_identical_images_MSE(self):
        self.assertEqual(MSE(self.pairs[0][0], self.pairs[0][0]), 0.0)
    
    def test_identical_images_PSNR(self):
        self.assertEqual(PSNR(self.pairs[0][0],self.pairs[0][0]),float('inf'))

    def test_identical_images_SSIM(self):
        score, _ = SSIM_windowed_fast(self.pairs[0][0],self.pairs[0][0])
        self.assertAlmostEqual(score,1.0,places=4)

    def test_matches_skimage_MSE(self):
        from skimage.metrics import mean_squared_error
        for original, distorted in self.pairs:
            self.assertAlmostEqual(
                MSE(original, distorted),
                mean_squared_error(original, distorted),
                places=5
            )
            
    def test_matches_skimage_PSNR(self):
        from skimage.metrics import peak_signal_noise_ratio
        for original, distorted in self.pairs:
            self.assertAlmostEqual(
                PSNR(original, distorted),
                peak_signal_noise_ratio(original, distorted),
                places=5
            )

    def test_matches_skimage_SSIM(self):
        from skimage.metrics import structural_similarity
        for original, distorted in self.pairs:
            score, _ = SSIM_windowed(original,distorted)
            self.assertAlmostEqual(
                score,
                structural_similarity(original, distorted,channel_axis=2),
                places=4
            )

    def test_SSIM_VS_AI(self):
        from metrics import SSIM
        for original, distorted in self.pairs:
            score, _ = SSIM_windowed(original,distorted)
            print("ai ssim = ", score)

            score = SSIM(original,distorted)
            print("my ssim = ", score)



if __name__ == "__main__":
    unittest.main(verbosity=2)