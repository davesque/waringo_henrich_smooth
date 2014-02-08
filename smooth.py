from __future__ import division

from math import (
    fabs,
    pow,
    sqrt,
)


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
