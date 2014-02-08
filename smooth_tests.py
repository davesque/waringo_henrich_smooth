#!/usr/bin/env python

import unittest
import collections
import math

from smooth import (
    Point, point_to_point_distance, point_to_line_distance, find_neighborhood,
    root_mean_square, root_mean_square_error, max_error,
    waringo_henrich_smooth,
)


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


class WaringoHenrichSmoothTestCase(unittest.TestCase):
    def setUp(self):
        # Random line
        self.points1 = [
            Point(0, 0),
            Point(62, 4),
            Point(102, 44),
            Point(198, 28),
            Point(214, 100),
            Point(326, 76),
            Point(406, 92),
            Point(438, 124),
            Point(502, 76),
            Point(574, 108),
            Point(638, 140),
        ]
        # Road bike handlebar contour:
        # 2014 Specialized Comp Alloy Tarmac Bend 44cm
        self.points2 = [
            Point(-34, 134),
            Point(-33, 134),
            Point(-31, 133),
            Point(-30, 133),
            Point(-29, 133),
            Point(-28, 133),
            Point(-27, 132),
            Point(-26, 132),
            Point(-25, 132),
            Point(-24, 132),
            Point(-23, 132),
            Point(-21, 131),
            Point(-20, 131),
            Point(-19, 131),
            Point(-18, 131),
            Point(-17, 131),
            Point(-15, 130),
            Point(-14, 130),
            Point(-13, 130),
            Point(-11, 130),
            Point(-10, 129),
            Point(-9, 129),
            Point(-8, 129),
            Point(-7, 129),
            Point(-5, 128),
            Point(-4, 128),
            Point(-3, 128),
            Point(-2, 128),
            Point(0, 128),
            Point(1, 127),
            Point(3, 127),
            Point(4, 126),
            Point(5, 126),
            Point(6, 126),
            Point(7, 126),
            Point(8, 125),
            Point(9, 125),
            Point(11, 125),
            Point(12, 124),
            Point(14, 124),
            Point(15, 123),
            Point(16, 123),
            Point(17, 122),
            Point(18, 122),
            Point(19, 121),
            Point(21, 121),
            Point(22, 120),
            Point(23, 120),
            Point(24, 119),
            Point(26, 118),
            Point(27, 118),
            Point(28, 117),
            Point(29, 116),
            Point(30, 116),
            Point(31, 116),
            Point(32, 115),
            Point(33, 114),
            Point(35, 114),
            Point(36, 113),
            Point(37, 112),
            Point(38, 111),
            Point(40, 110),
            Point(41, 110),
            Point(42, 109),
            Point(42, 108),
            Point(44, 108),
            Point(44, 107),
            Point(45, 106),
            Point(46, 105),
            Point(47, 105),
            Point(48, 104),
            Point(49, 103),
            Point(50, 102),
            Point(51, 101),
            Point(52, 101),
            Point(52, 100),
            Point(53, 99),
            Point(54, 98),
            Point(55, 97),
            Point(56, 96),
            Point(57, 95),
            Point(58, 94),
            Point(59, 94),
            Point(59, 93),
            Point(60, 92),
            Point(61, 91),
            Point(62, 90),
            Point(63, 89),
            Point(63, 88),
            Point(64, 87),
            Point(65, 86),
            Point(66, 84),
            Point(67, 83),
            Point(68, 82),
            Point(68, 81),
            Point(69, 80),
            Point(70, 79),
            Point(70, 78),
            Point(71, 77),
            Point(72, 76),
            Point(72, 75),
            Point(73, 73),
            Point(74, 72),
            Point(74, 71),
            Point(75, 70),
            Point(75, 69),
            Point(76, 68),
            Point(77, 67),
            Point(77, 66),
            Point(77, 65),
            Point(78, 63),
            Point(79, 62),
            Point(79, 60),
            Point(80, 59),
            Point(80, 58),
            Point(81, 56),
            Point(81, 55),
            Point(81, 54),
            Point(81, 53),
            Point(82, 51),
            Point(82, 50),
            Point(82, 48),
            Point(82, 47),
            Point(82, 46),
            Point(82, 45),
            Point(82, 44),
            Point(81, 43),
            Point(81, 41),
            Point(81, 39),
            Point(81, 38),
            Point(80, 37),
            Point(80, 36),
            Point(79, 35),
            Point(79, 33),
            Point(79, 32),
            Point(78, 31),
            Point(78, 30),
            Point(76, 29),
            Point(76, 27),
            Point(75, 26),
            Point(74, 25),
            Point(74, 24),
            Point(73, 23),
            Point(73, 22),
            Point(72, 22),
            Point(71, 21),
            Point(70, 20),
            Point(69, 19),
            Point(68, 19),
            Point(67, 18),
            Point(66, 17),
            Point(65, 17),
            Point(64, 16),
            Point(63, 16),
            Point(62, 15),
            Point(62, 15),
            Point(61, 15),
            Point(60, 15),
            Point(60, 14),
            Point(58, 14),
            Point(57, 13),
            Point(56, 13),
            Point(55, 13),
            Point(53, 12),
            Point(52, 12),
            Point(51, 12),
            Point(50, 12),
            Point(48, 12),
            Point(47, 12),
            Point(45, 12),
            Point(44, 12),
            Point(43, 12),
            Point(42, 12),
            Point(41, 12),
            Point(39, 12),
            Point(39, 12),
            Point(37, 12),
            Point(36, 12),
            Point(34, 12),
            Point(33, 12),
            Point(33, 12),
            Point(32, 12),
            Point(31, 12),
            Point(30, 12),
            Point(30, 12),
        ]

    def test_it_should_smooth_paths_to_within_the_specified_degree_of_error(self):
        self.assertEqual(waringo_henrich_smooth(self.points2, 1), [
            Point(-34, 134),
            Point(0, 128),
            Point(21, 121),
            Point(35, 114),
            Point(42, 108),
            Point(44, 108),
            Point(59, 94),
            Point(77, 67),
            Point(81, 56),
            Point(82, 44),
            Point(78, 30),
            Point(76, 29),
            Point(73, 22),
            Point(62, 15),
            Point(53, 12),
            Point(30, 12),
        ])
        self.assertEqual(waringo_henrich_smooth(self.points2, 1.5), [
            Point(-34, 134),
            Point(0, 128),
            Point(21, 121),
            Point(44, 108),
            Point(59, 94),
            Point(77, 67),
            Point(81, 56),
            Point(82, 44),
            Point(78, 30),
            Point(73, 22),
            Point(53, 12),
            Point(30, 12),
        ])
        self.assertEqual(waringo_henrich_smooth(self.points2, 2), [
            Point(-34, 134),
            Point(21, 121),
            Point(44, 108),
            Point(59, 94),
            Point(77, 67),
            Point(82, 44),
            Point(73, 22),
            Point(53, 12),
            Point(30, 12),
        ])
        self.assertEqual(waringo_henrich_smooth(self.points1, 30), [
            Point(0, 0),
            Point(198, 28),
            Point(214, 100),
            Point(326, 76),
            Point(438, 124),
            Point(502, 76),
            Point(638, 140),
        ])
        self.assertEqual(waringo_henrich_smooth(self.points1, 60), [
            Point(0, 0),
            Point(638, 140),
        ])

    def test_it_should_return_the_given_list_if_the_list_has_a_length_less_than_3(self):
        self.assertEqual(waringo_henrich_smooth([], 0), [])
        self.assertEqual(waringo_henrich_smooth([Point(0, 0)], 0), [Point(0, 0)])
        self.assertEqual(waringo_henrich_smooth([Point(0, 0), Point(1, 1)], 0), [Point(0, 0), Point(1, 1)])

    def test_it_should_always_return_only_the_first_and_last_points_in_a_list_if_a_large_enough_error_limit_is_given(self):
        self.assertEqual(waringo_henrich_smooth(self.points2, 1000), [
            Point(-34, 134),
            Point(30, 12),
        ])
        self.assertEqual(waringo_henrich_smooth(self.points1, 1000), [
            Point(0, 0),
            Point(638, 140),
        ])


if __name__ == '__main__':
    unittest.main()
