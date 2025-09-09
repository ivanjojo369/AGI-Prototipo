import unittest
import subprocess

class TestGPU(unittest.TestCase):
    def test_gpu_available(self):
        try:
            result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 0, "GPU no detectada o nvidia-smi no disponible")
        except FileNotFoundError:
            self.skipTest("nvidia-smi no está instalado")

if __name__ == '__main__':
    unittest.main()
