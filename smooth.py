from __future__ import division

from math import (
    fabs,
    pow,
    sqrt,
)
import collections


Point = collections.namedtuple('Point', ['x', 'y'])


class PointWrapper(object):
    def __init__(self, x, y, i):
        self.x = x
        self.y = y
        self.r = False
        self.i = i

    def __repr__(self):
        return 'PointWrapper(x={0}, y={1}, i={2})'.format(self.x, self.y, self.i)


def point_to_point_distance(p1, p2):
    """
    Returns the distance between a point `p1` and a point `p2`.
    """
    if p1.x == p2.x and p1.y == p2.y:
        # Special case for p1 == p2
        return 0.0

    if p1.x == p2.x:
        # Special case for slope infinity
        d = p2.y - p1.y
    elif p1.y == p2.y:
        # Special case for slope 0
        d = p2.x - p1.x
    else:
        # Normal case
        d = sqrt(pow(p2.y - p1.y, 2) + pow(p2.x - p1.x, 2))

    return fabs(d)


def point_to_line_distance(p1, p2, p3):
    """
    Returns the distance between a point `p2` and the line formed by two points
    `p1` and `p3`.
    """
    if (
        p1.x == p2.x and
        p1.y == p2.y and
        p2.x == p3.x and
        p2.y == p3.y
    ):
        # Special case for p1 == p2 == p3
        return 0

    if p1.x == p3.x and p1.y == p3.y:
        # Special case for p1 == p3
        return point_to_point_distance(p1, p2)

    if p1.x == p3.x:
        # Special case for slope infinity
        d = p2.x - p1.x
    elif p1.y == p3.y:
        # Special case for slope 0
        d = p2.y - p1.y
    else:
        # Normal case
        m = (p3.y - p1.y) / (p3.x - p1.x)
        b = p1.y - m * p1.x
        d = (p2.y - m * p2.x - b) / sqrt(m * m + 1)

    return fabs(d)


def find_neighborhood(points, p):
    """
    Finds neighboring, non-removed points for the point `p` in the point list
    `points`.
    """
    if not points:
        return (None, None)

    points_len = len(points)
    left_point = right_point = None

    # Find closest neighbor to the left
    i = p.i - 1
    while True:
        if i < 0:
            break
        point = points[i]
        if point.r is False:
            left_point = point
            break
        i -= 1

    # Find closest neighbor to the right
    i = p.i + 1
    while True:
        if i >= points_len:
            break
        point = points[i]
        if point.r is False:
            right_point = point
            break
        i += 1

    return (left_point, right_point)


def root_mean_square(l):
    """
    Returns the root mean square of values in a list `l`.
    """
    try:
        return sqrt(sum([i * i for i in l]) / len(l))
    except ZeroDivisionError:
        return None


def root_mean_square_error(points, start, end):
    """
    Returns the root mean square deviation for the exclusive range
    (neighborhood) of points specified by the given start and end points.
    """
    if not points:
        return None

    # Flip range if start.i > end.i
    if start.i > end.i:
        start, end = end, start

    # Get deviations for all points inside of the neighborhood's range
    ds = [
        point_to_line_distance(start, points[i], end)
        for i in range(start.i + 1, end.i)
    ]

    return root_mean_square(ds)


def max_error(points, start, end):
    """
    Returns the maximum deviation for the exclusive range (neighborhood) of
    points specified by the given start and end points.
    """
    if not points:
        return None

    # Flip range if start.i > end.i
    if start.i > end.i:
        start, end = end, start

    # Get deviations for all points inside of the neighborhood's range
    ds = [
        point_to_line_distance(start, points[i], end)
        for i in range(start.i + 1, end.i)
    ]

    return max(ds) if ds else None


def waringo_henrich_smooth(points, d_lim, max_steps=None):
    """
    Smooths a piecewise linear path described by a list of 2D points to within
    the specified maximum deviation `d_lim`.  The value `max_steps` may be
    optionally specified to limit the number of iterations when the algorithm
    is run.
    """
    points_len = len(points)

    def set_deviation(p):
        # Don't set deviation if p is None or if p is an end point
        if not p or p.i == 0 or p.i == points_len - 1:
            return

        left_point, right_point = find_neighborhood(points, p)
        p.d = max_error(points, left_point, right_point)

    # Copy and annotate points
    points = [PointWrapper(p.x, p.y, i) for i, p in enumerate(points)]

    removable = points[1:-1]

    for p in removable:
        set_deviation(p)

    steps = 0
    while True:
        if max_steps and steps >= max_steps:
            break

        remaining = [p for p in removable if p.r is False]

        if not remaining:
            break

        smallest = min(remaining, key=lambda p: p.d)

        if smallest.d < d_lim:
            smallest.r = True
            left_point, right_point = find_neighborhood(points, smallest)
            set_deviation(left_point)
            set_deviation(right_point)
        else:
            break

        steps += 1

    return [Point(p.x, p.y) for p in points if p.r is False]
