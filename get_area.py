import math
from typing import List


class Point:
    __slots__ = ('x', 'y')

    def __init__(self, x, y):
        self.x = x
        self.y = y


def get_area_of_polygon(points: List[Point]):
    area = 0
    if len(points) < 3:
        raise Exception("error")

    p1 = points[0]
    for i in range(1, len(points) - 1):
        p2 = points[1]
        p3 = points[2]

        vecp1p2 = Point(p2.x - p1.x, p2.y - p1.y)
        vecp2p3 = Point(p3.x - p2.x, p3.y - p2.y)

        vecMult = vecp1p2.x * vecp2p3.y - vecp1p2.y * vecp2p3.x
        sign = 0
        if vecMult > 0:
            sign = 1
        elif vecMult < 0:
            sign = -1

        triArea = get_area_of_triangle(p1, p2, p3) * sign
        area += triArea
    return abs(area)


def get_area_of_triangle(p1, p2, p3):
    p1p2 = get_line_length(p1, p2)
    p2p3 = get_line_length(p2, p3)
    p3p1 = get_line_length(p3, p1)
    s = (p1p2 + p2p3 + p3p1) / 2
    area = s * (s - p1p2) * (s - p2p3) * (s - p3p1)
    area = math.sqrt(area)
    return area


def get_line_length(p1, p2):
    length = math.pow((p1.x - p2.x), 2) + math.pow((p1.y - p2.y), 2)
    length = math.sqrt(length)
    return length


def x_y_to_points(x: List, y: List) -> List[Point]:
    points = []
    for index in range(len(x)):
        points.append(Point(x[index], y[index]))
    return points


def main():
    points = []
    x = [1, 0, 0, 1]
    y = [0, 0, 1, 1]
    points = x_y_to_points(x, y)

    area = get_area_of_polygon(points)
    print(area)
    assert math.ceil(area) == 1


if __name__ == '__main__':
    main()
