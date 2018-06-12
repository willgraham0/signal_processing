import unittest
import bases


class FourierTests(unittest.TestCase):

    def inverse_is_transpose(self):
        """Test if the inverse of fourier matrix is the same as the 
        transpose of the fourier matrix."""
        dft_matrix = bases.fourier.dft_matrix(10)
        idft_matrix = bases.fourier.idft_matrix(10)
        self.assertEqual(dft_matrix.T, idft_matrix)


if __name__ == "__main__":
    unittest.main()