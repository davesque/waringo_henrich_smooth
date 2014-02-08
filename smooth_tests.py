#!/usr/bin/env python

import unittest
import collections
import math

from smooth import (
    point_to_point_distance, point_to_line_distance, find_neighborhood,
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


class MaxErrorTestCase(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
