import unittest

from fractional_cascading_index import Cascade, search_all


class TestCascade(unittest.TestCase):
    def test_single_layer_basic(self):
        c = Cascade([[1, 3, 5, 7]])
        self.assertEqual(c.search(5), [2])

    def test_single_layer_out_of_range(self):
        c = Cascade([[1, 3, 5]])
        self.assertEqual(c.search(10), [3])

    def test_single_layer_below_all(self):
        c = Cascade([[10, 20, 30]])
        self.assertEqual(c.search(5), [0])

    def test_two_layers_present(self):
        c = Cascade([[1, 3, 5, 7], [2, 4, 6, 8]])
        self.assertEqual(c.search(5), [2, 2])

    def test_two_layers_first_lowerbound(self):
        c = Cascade([[1, 3, 5, 7], [2, 4, 6, 8]])
        self.assertEqual(c.search(4), [2, 1])

    def test_two_layers_out_of_range(self):
        c = Cascade([[1, 3, 5], [2, 4, 6]])
        self.assertEqual(c.search(100), [3, 3])

    def test_three_layers(self):
        c = Cascade([[1, 3, 5, 7, 9], [2, 4, 6, 8], [1, 2, 3, 4]])
        self.assertEqual(c.search(5), [2, 2, 4])

    def test_duplicate_keys_in_layer(self):
        c = Cascade([[1, 1, 3, 3, 5]])
        self.assertEqual(c.search(3), [2])

    def test_query_below_all_layers(self):
        c = Cascade([[10, 20], [5, 15, 25]])
        self.assertEqual(c.search(1), [0, 0])

    def test_query_above_all_layers(self):
        c = Cascade([[10, 20], [5, 15, 25]])
        self.assertEqual(c.search(100), [2, 3])

    def test_empty_layer_first(self):
        c = Cascade([[], [1, 2, 3]])
        self.assertEqual(c.search(2), [0, 1])

    def test_empty_layer_middle(self):
        c = Cascade([[1, 2, 3], [], [1, 2, 3]])
        self.assertEqual(c.search(2), [1, 0, 1])

    def test_empty_layer_last(self):
        c = Cascade([[1, 2, 3], [4, 5, 6], []])
        self.assertEqual(c.search(2), [1, 0, 0])

    def test_empty_layers_all(self):
        c = Cascade([[], [], []])
        self.assertEqual(c.search(5), [0, 0, 0])

    def test_search_all_matches_search(self):
        c = Cascade([[1, 3, 5], [2, 4, 6]])
        self.assertEqual(search_all(c, 4), c.search(4))

    def test_constructor_rejects_empty_layers_list(self):
        with self.assertRaises(ValueError):
            Cascade([])

    def test_constructor_rejects_none_layer(self):
        with self.assertRaises(ValueError):
            Cascade([None, [1, 2, 3]])

    def test_unsorted_input_is_sorted(self):
        c = Cascade([[5, 1, 3], [8, 2, 4]])
        self.assertEqual(c.search(3), [1, 1])

    def test_float_keys(self):
        c = Cascade([[1.0, 2.5, 3.0], [1.25, 2.0, 3.5]])
        self.assertEqual(c.search(2.5), [1, 2])

    def test_num_layers_property(self):
        c = Cascade([[1], [2], [3]])
        self.assertEqual(c.num_layers, 3)

    def test_duplicate_keys_across_layers(self):
        c = Cascade([[1, 2, 3], [2, 3, 4]])
        self.assertEqual(c.search(2), [1, 0])


if __name__ == "__main__":
    unittest.main()
