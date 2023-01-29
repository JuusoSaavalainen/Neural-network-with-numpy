import unittest
import os
from data.idx_to_csv import convert_to_csv

class TestConvert(unittest.TestCase):
    def setUp(self):
        self.image = "test_mock_images.idx3-ubyte"
        self.label = "test_mock_labels.idx1-ubyte"
        self.out = "test_output.csv"
        self.n = 10

        with open(self.image, "wb") as f:
            f.write(b"\x00" * 16 + b"".join(b"\x01" * 28 * 28 for _ in range(self.n)))
            
        with open(self.label, "wb") as f:
            f.write(b"\x00" * 8 + b"".join(bytes([i % 10]) for i in range(self.n)))

    def test_convert(self):
        convert_to_csv(self.image, self.label, self.out, self.n)
        with open(self.out, "r") as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), self.n)

        for i, line in enumerate(lines):
            line = list(map(int, line.strip().split(",")))
            
            self.assertEqual(len(line), 28 * 28 + 1)
            self.assertEqual(line[0], i % 10)

    def tearDown(self):
        os.remove(self.image)
        os.remove(self.label)
        os.remove(self.out)
