#!/usr/bin/env python

import unittest
import collections
import math

from smooth import (
    point_to_point_distance, point_to_line_distance, find_neighborhood,
    root_mean_square, root_mean_square_error, max_error,
)


Point = collections.namedtuple('Point', ['x', 'y'])
AnnotatedPoint = collections.namedtuple('Point', ['x', 'y', 'i', 'r'])


class PointToPointDistanceTest(unittest.TestCase):
    def test_it_should_find_the_correct_distance_when_p1_equals_p2(self):
        self.assertEqual(point_to_point_distance(
            Point(0, 0),
            Point(0, 0),
        ), 0)
        self.assertEqual(point_to_point_distance(
            Point(1, 1),
            Point(1, 1),
        ), 0)

    def test_it_should_find_the_correct_distance_when_the_line_slope_is_infinity(self):
        self.assertEqual(point_to_point_distance(
            Point(0, 0),
            Point(0, 10),
        ), 10)
        self.assertEqual(point_to_point_distance(
            Point(123, 0),
            Point(123, -10),
        ), 10)

    def test_it_should_find_the_correct_distance_when_the_line_slope_is_0(self):
        self.assertEqual(point_to_point_distance(
            Point(0, 0),
            Point(10, 0),
        ), 10)
        self.assertEqual(point_to_point_distance(
            Point(0, -10),
            Point(1000, -10),
        ), 1000)

    def test_it_should_find_the_correct_distance_in_a_normal_case(self):
        self.assertAlmostEqual(point_to_point_distance(
            Point(0, 0),
            Point(10, 10),
        ), 14.1421, 4)
        self.assertAlmostEqual(point_to_point_distance(
            Point(10, -5),
            Point(-10, 5),
        ), 22.3607, 4)


class PointToLineDistanceTest(unittest.TestCase):
    def test_it_should_find_the_correct_distance_when_the_line_slope_is_infinity(self):
        self.assertEqual(point_to_line_distance(
            Point(0, 0),
            Point(5, -100),
            Point(0, 10),
        ), 5)
        self.assertEqual(point_to_line_distance(
            Point(-20, -2000),
            Point(5, 0),
            Point(-20, 10),
        ), 25)

    def test_it_should_find_the_correct_distance_when_the_line_slope_is_0(self):
        self.assertEqual(point_to_line_distance(
            Point(0, 0),
            Point(5, 1000),
            Point(10, 0),
        ), 1000)
        self.assertEqual(point_to_line_distance(
            Point(0, -20),
            Point(5, 1000),
            Point(-20, -20),
        ), 1020)

    def test_it_should_find_the_correct_distance_when_the_line_has_length_0(self):
        self.assertEqual(point_to_line_distance(
            Point(0, 0),
            Point(5, 0),
            Point(0, 0),
        ), 5)
        self.assertEqual(point_to_line_distance(
            Point(-10, -10),
            Point(10, 10),
            Point(-10, -10),
        ), math.sqrt(20.0 * 20.0 + 20.0 * 20.0))

    def test_it_should_find_the_correct_distance_when_p1_equals_p2_equals_p3(self):
        self.assertEqual(point_to_line_distance(
            Point(0, 0),
            Point(0, 0),
            Point(0, 0),
        ), 0)
        self.assertEqual(point_to_line_distance(
            Point(1, 1),
            Point(1, 1),
            Point(1, 1),
        ), 0)
        self.assertEqual(point_to_line_distance(
            Point(-1, -1),
            Point(-1, -1),
            Point(-1, -1),
        ), 0)

    def test_it_should_find_the_correct_distance_in_a_normal_case(self):
        self.assertAlmostEqual(point_to_line_distance(
            Point(0, 0),
            Point(0, 1),
            Point(1, 1),
        ), 0.7071, 4)
        self.assertAlmostEqual(point_to_line_distance(
            Point(0, 0),
            Point(5, 1),
            Point(10, 1),
        ), 0.4975, 4)
        self.assertAlmostEqual(point_to_line_distance(
            Point(0, 0),
            Point(6, 1.1),
            Point(10, 1),
        ), 0.4975, 4)
        self.assertAlmostEqual(point_to_line_distance(
            Point(-10, 10),
            Point(-4, 8),
            Point(-3, 5),
        ), 1.8600, 4)


class FindNeighborhoodTestCase(unittest.TestCase):
    def setUp(self):
        self.coords1 = [
            AnnotatedPoint(0, 0, 0, True),
            AnnotatedPoint(0, 0, 1, True),
            AnnotatedPoint(0, 0, 2, False),
            AnnotatedPoint(0, 0, 3, True),
            AnnotatedPoint(0, 0, 4, False),
            AnnotatedPoint(0, 0, 5, False),
            AnnotatedPoint(0, 0, 6, True),
        ]
        self.coords2 = [
            AnnotatedPoint(0, 0, 0, True),
            AnnotatedPoint(0, 0, 1, True),
            AnnotatedPoint(0, 0, 2, False),
            AnnotatedPoint(0, 0, 3, True),
            AnnotatedPoint(0, 0, 4, True),
            AnnotatedPoint(0, 0, 5, False),
            AnnotatedPoint(0, 0, 6, True),
            AnnotatedPoint(0, 0, 7, False),
            AnnotatedPoint(0, 0, 8, False),
            AnnotatedPoint(0, 0, 9, True),
            AnnotatedPoint(0, 0, 10, False),
            AnnotatedPoint(0, 0, 11, True),
            AnnotatedPoint(0, 0, 12, True)
        ]

    def test_it_should_find_the_closest_points_on_the_left_and_right_which_will_not_be_removed(self):
        self.assertEqual(find_neighborhood(self.coords1, self.coords1[3]), (self.coords1[2], self.coords1[4]))
        self.assertEqual(find_neighborhood(self.coords1, self.coords1[4]), (self.coords1[2], self.coords1[5]))
        self.assertEqual(find_neighborhood(self.coords2, self.coords2[5]), (self.coords2[2], self.coords2[7]))
        self.assertEqual(find_neighborhood(self.coords2, self.coords2[8]), (self.coords2[7], self.coords2[10]))

    def test_it_should_return_none_as_the_right_point_if_no_valid_point_on_the_right_is_found(self):
        self.assertEqual(find_neighborhood(self.coords1, self.coords1[5]), (self.coords1[4], None))
        self.assertEqual(find_neighborhood(self.coords2, self.coords2[10]), (self.coords2[8], None))

    def test_it_should_return_none_as_the_left_point_if_no_valid_point_on_the_left_is_found(self):
        self.assertEqual(find_neighborhood(self.coords1, self.coords1[1]), (None, self.coords1[2]))
        self.assertEqual(find_neighborhood(self.coords2, self.coords2[2]), (None, self.coords2[5]))

    def test_it_should_return_none_as_the_right_point_if_the_given_point_is_the_last_point(self):
        self.assertEqual(find_neighborhood(self.coords1, self.coords1[6]), (self.coords1[5], None))
        self.assertEqual(find_neighborhood(self.coords2, self.coords2[12]), (self.coords2[10], None))

    def test_it_should_return_none_as_the_left_point_if_the_given_point_is_the_first_point(self):
        self.assertEqual(find_neighborhood(self.coords1, self.coords1[0]), (None, self.coords1[2]))
        self.assertEqual(find_neighborhood(self.coords2, self.coords2[0]), (None, self.coords2[2]))

    def test_it_should_return_none_for_both_the_left_and_right_points_if_the_given_list_is_empty(self):
        self.assertEqual(find_neighborhood([], None), (None, None))
        self.assertEqual(find_neighborhood([], None), (None, None))

    def test_it_should_return_none_for_both_the_left_and_right_points_if_the_given_has_a_length_of_1(self):
        coords = [AnnotatedPoint(0, 0, 0, False)]
        self.assertEqual(find_neighborhood(coords, coords[0]), (None, None))


class RootMeanSquareTestCase(unittest.TestCase):
    def test_it_should_return_the_root_mean_square_of_values_in_a_list(self):
        self.assertAlmostEqual(root_mean_square([1, 2, 3, 4, 5]), 3.3166, 4)
        self.assertAlmostEqual(root_mean_square([10, 20, 30, 40, 50]), 33.1662, 4)
        self.assertAlmostEqual(root_mean_square([-10, -20, 30, 40, 50]), 33.1662, 4)

    def test_it_should_work_for_single_value_lists(self):
        self.assertEqual(root_mean_square([1]), 1)
        self.assertEqual(root_mean_square([12345]), 12345)
        self.assertEqual(root_mean_square([-12345]), 12345)

    def test_it_should_return_none_for_an_empty_list(self):
        self.assertTrue(root_mean_square([]) is None)


class ErrorTestCase(unittest.TestCase):
    def setUp(self):
        self.points1 = [
            AnnotatedPoint(0, 0, 0, 0),
            AnnotatedPoint(5, 5, 1, 0),
            AnnotatedPoint(10, 0, 2, 0),
            AnnotatedPoint(15, 0, 3, 0),
            AnnotatedPoint(20, 5, 4, 0),
            AnnotatedPoint(25, 0, 5, 0),
            AnnotatedPoint(30, 0, 6, 0),
        ]
        self.points2 = [
            AnnotatedPoint(0, 0, 0, 0),
            AnnotatedPoint(5, 6, 1, 0),
            AnnotatedPoint(10, 10, 2, 0),
            AnnotatedPoint(9, 132, 3, 0),
            AnnotatedPoint(20, 155, 4, 0),
            AnnotatedPoint(25, 120, 5, 0),
            AnnotatedPoint(30, 10, 6, 0),
        ]


class RootMeanSquareError(ErrorTestCase):
    def test_it_should_return_the_root_mean_square_of_deviations_within_a_certain_point_range(self):
        self.assertEqual(root_mean_square_error(self.points1, self.points1[0], self.points1[2]), 5)
        self.assertAlmostEqual(root_mean_square_error(self.points1, self.points1[0], self.points1[3]), 3.5355, 4)
        self.assertAlmostEqual(root_mean_square_error(self.points1, self.points1[0], self.points1[6]), 3.1623, 4)
        self.assertAlmostEqual(root_mean_square_error(self.points2, self.points2[0], self.points2[6]), 95.9779, 4)

    def test_it_should_return_a_correct_value_even_if_the_start_and_end_points_are_swapped(self):
        self.assertAlmostEqual(root_mean_square_error(self.points1, self.points1[6], self.points1[0]), 3.1623, 4)
        self.assertAlmostEqual(root_mean_square_error(self.points2, self.points2[6], self.points2[0]), 95.9779, 4)

    def test_it_should_return_none_for_a_zero_length_range(self):
        self.assertTrue(root_mean_square_error(self.points1, self.points1[0], self.points1[0]) is None)

    def test_it_should_return_none_for_an_empty_list(self):
        self.assertTrue(root_mean_square_error([], None, None) is None)


class MaxErrorTestCase(ErrorTestCase):
    def test_it_should_return_the_maximum_deviation_within_a_certain_point_range(self):
        self.assertEqual(max_error(self.points1, self.points1[0], self.points1[2]), 5)
        self.assertEqual(max_error(self.points1, self.points1[0], self.points1[3]), 5)
        self.assertEqual(max_error(self.points1, self.points1[0], self.points1[6]), 5)
        self.assertAlmostEqual(max_error(self.points2, self.points2[0], self.points2[6]), 140.7214, 4)

    def test_it_should_return_a_correct_value_even_if_the_start_and_end_points_are_swapped(self):
        self.assertEqual(max_error(self.points1, self.points1[6], self.points1[0]), 5)
        self.assertAlmostEqual(max_error(self.points2, self.points2[6], self.points2[0]), 140.7214, 4)

    def test_it_should_return_none_for_a_zero_length_range(self):
        self.assertTrue(max_error(self.points1, self.points1[0], self.points1[0]) is None)

    def test_it_should_return_none_for_an_empty_list(self):
        self.assertTrue(max_error([], None, None) is None)


if __name__ == '__main__':
    unittest.main()
