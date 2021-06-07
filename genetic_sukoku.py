import random
import math
import numpy

import itertools


class SudokuPool():

    """

    On instance of this class has multiple SudokeElements in it.
    It has a built-in genetic algorithm. It doesn't perform as good as expected (it gets better very slowly).

    Once instantiated a class you can run the genetic algorithm using run_genetic_algorithm


    """

    def __init__(self, number_of_elements, dimension=None, fixed_elements=None, mutation_rate=None):
        """
        Params:
        number_of_elements: number of pool elements/ population size
        dimension: square dimension of each sudoku puzzle
        fixed_elements: the elements that are already filled in the puzzle. It's a dictionary with tuple keys and integer values
        mutation_rate: float number representing the rate of mutation

        """
        self.dimension = dimension or 9
        self.mutation_rate = mutation_rate or 0.05
        self.fixed_elements = fixed_elements or dict()
        self.sudoku_squares = [
            SudokuElement(self.dimension, fixed_elements, mutation_rate)
            for i in range(number_of_elements)
        ]

    def selection(self):
        copied_squares = random.choices(
            self.sudoku_squares,
            weights=[i.fitness_function() for i in self.sudoku_squares],
            k=len(self.sudoku_squares)
        )
        self.sudoku_squares = [
            SudokuElement(self.dimension, self.fixed_elements, self.mutation_rate, i.square)
            for i in copied_squares
        ]

    def crossover(self):
        """

        If the number of elements is not even we will lose some elements

        """
        random.shuffle(self.sudoku_squares)
        pairs = list(zip(self.sudoku_squares[::2], self.sudoku_squares[1::2]))
        for elem_a, elem_b in pairs:
            elem_a.crossover(elem_b)

    def run_genetic_algorithm(self, iterations=1000):
        max_el = max(self.sudoku_squares, key=lambda x: x.fitness_function())
        for iteration in range(iterations):
            self.selection()
            if iteration < iterations - 1:
                self.crossover()
                for sud in self.sudoku_squares:
                    sud.mutation()
        max_el = max(self.sudoku_squares, key=lambda x: x.fitness_function())
        return max_el


class SudokuElement():
    """

    Represents a Sudoku n*n square.
    It has some fixed elements.
    It's mutable and is automatically filled with random values.


    """

    def __init__(self, dimension=None, fixed_elements=None, mutation_rate=None, square=None):
        """
        Params:
        dimension: square dimension of each sudoku puzzle
        fixed_elements: the elements that are already filled in the puzzle. It's a dictionary with tuple keys and integer values
        mutation_rate: float number representing the rate of mutation
        square: optional numpy.ndarray object, the initial square won't be generated randomly

        """
        # TODO: check what are the acceptable dimensions for a sudoku square
        # For our tests every number whose square root is an integer is accepted
        # If the dimension is not given we use the standar 9*9 square
        self.mutation_rate = mutation_rate or 0.05
        self.dimension = dimension or 9
        self.fixed_elements = fixed_elements or dict()
        self.coordinates = [
            (i, j) for i in range(self.dimension)
            for j in range(self.dimension) if (i, j) not in self.fixed_elements
        ]
        self.square = square.copy() if square is not None else self._get_initial_square()

        # It's a dictionary that groups all the coordinates of the square array/matrix.
        # We will get:
        # one element of the dictionary with all the rows (list of rows)
        # one element of the dictionary with all the columns (list of columns)
        # one element of the dictionary with all the square groups
        # It's mainly used for the last element as the rest would be easy to get anyways
        self.coordinates_mapping = self._get_coordinates_mapping()

    def _get_coordinates_mapping(self):
        coordinates = numpy.indices((self.dimension, self.dimension))
        x_axis_coordinates = list(map(tuple, coordinates.transpose().reshape((self.dimension ** 2, 2))))
        y_axis_coordinates = [
            (j, i) for i, j in x_axis_coordinates
        ]

        x_axis_coordinates = list(
            map(
                lambda x: list(map(tuple, x)),
                numpy.array_split(x_axis_coordinates, self.dimension)
            )
        )
        y_axis_coordinates = list(
            map(
                lambda x: list(map(tuple, x)),
                numpy.array_split(y_axis_coordinates, self.dimension)
            )
        )
        z_axis_coordinates = []
        """
        The next lines are like blackmagic, but they are easier to understand than one would expect
        First of all the `coordinates` is an array whose shape is (2, dimension, dimension),
        so it's a 3D array but it's not easy to visualize.
        Transposing this array/matrix will return an array with a (dimension, dimension, 2) shape,
        like a 2D list of coordinates.
        We have then to, more or less cleverly, split the arrays multiple times to get the squares of each
        sudoku puzzle

        partial_split = functools.partial(numpy.split, indices_or_sections=int(math.sqrt(self.dimension)), axis=1)
        m = list(map(partial_split, numpy.split(coordinates.transpose(), int(math.sqrt(self.dimension)))))
        """

        for i in numpy.split(coordinates.transpose(), math.sqrt(self.dimension)):
            for j in numpy.split(i, math.sqrt(self.dimension), 1):
                z_axis_coordinates.append(list(map(tuple, j.reshape((self.dimension, 2)))))

        return {
            'x': x_axis_coordinates,
            'y': y_axis_coordinates,
            'z': z_axis_coordinates
        }

    def set_square(self, new_square):
        self.square = new_square

    def perfect_points(self):
        return 3 * (self.dimension**4)

    def is_perfect(self):
        return int(self.fitness_function()) == 1

    def _get_initial_square(self):
        """
        We will initially fill the square with the
        fixed elements and some random ones.
        """
        remaining_values = [i for i in range(1, self.dimension + 1) for j in range(self.dimension)]
        starting_square = numpy.zeros((self.dimension, self.dimension), dtype=numpy.int8)
        for index, value in self.fixed_elements.items():
            if value in remaining_values:
                remaining_values.remove(value)
            starting_square[index] = value

        # excluding the coordinates given by the initial values,
        # fill the rest of the square with random values
        for coordinate in self.coordinates:
            value_set = set(starting_square[coordinate[0]]).union(set(starting_square[:, coordinate[1]]))
            value = random.choice(list(set(remaining_values) - value_set - {0}) or remaining_values or [1])
            if value in remaining_values:
                remaining_values.remove(value)
            starting_square[coordinate] = value

        return starting_square

    def backtrack_solve(self):
        good_square = self._get_nearly_perfect_square()
        return self._recursive_backtrack(good_square)

    def _find_empty_spot(self, square=None):
        x, y = numpy.where(square == 0)
        # the following is numpy suggested way to check if an array is empty
        if x.size > 0 and y.size > 0:
            return next(zip(x, y))
        return False

    def _get_used_values(self, coord, square):
        used_set = set()
        for axis in self.coordinates_mapping.values():
            coords_to_check = next(filter(lambda x: coord in x, axis), [])
            axis_set = set(map(lambda x: square[x], coords_to_check))
            used_set = used_set.union(axis_set)
        return used_set - {0}

    def _recursive_backtrack(self, square=None):
        empty_spot = self._find_empty_spot(square)
        if not empty_spot:
            return square
        used_values = self._get_used_values(empty_spot, square)
        possible_values = set(range(1, self.dimension + 1))
        value_set = list(possible_values - used_values)
        for value in value_set:
            square[empty_spot] = value
            recursive_result = self._recursive_backtrack(square)
            if isinstance(recursive_result, numpy.ndarray):
                return square
            square[empty_spot] = 0
        return False

    def _get_nearly_perfect_square(self):
        """
        Simple heuristic to try to fill the most right answer from scratch.

        Worst case scenario (computationally speaking): now fixed elements given --> we will loop through all

        Returns an array partially filled with the right values for the fixed elements of the instance
        of this class.


        """
        remaining_values = [i for i in range(1, self.dimension + 1) for j in range(self.dimension)]
        perfect_square = numpy.zeros((self.dimension, self.dimension), dtype=numpy.int8)

        unchecked_coordinates = self.coordinates.copy()
        possible_values = set(range(1, self.dimension + 1))
        for index, value in self.fixed_elements.items():
            if value in remaining_values:
                remaining_values.remove(value)
            perfect_square[index] = value
        for coord in itertools.chain(unchecked_coordinates * self.dimension):
            used_values = self._get_used_values(coord, perfect_square)
            value_set = possible_values - used_values
            if len(value_set) == 1:
                value = list(value_set)[0]
                perfect_square[coord] = value
                remaining_values.remove(value)
                unchecked_coordinates.remove(coord)
        return perfect_square

    def mutation(self):
        coordinates = [
            (i, j) for i in range(self.dimension)
            for j in range(self.dimension) if (i, j) not in self.fixed_elements
        ]
        for coordinate in coordinates:
            if random.random() < self.mutation_rate:
                self.square[coordinate] = random.randint(1, self.dimension)

    def crossover(self, other_element):
        split_axis = random.choice([0, 1])
        index = random.randint(0, self.dimension)
        elem_a1, elem_a2 = numpy.split(self.square, [index], split_axis)
        elem_b1, elem_b2 = numpy.split(other_element.square, [index], split_axis)
        self.set_square(numpy.concatenate([elem_a1, elem_b2], axis=split_axis))
        other_element.set_square(numpy.concatenate([elem_b1, elem_a2], axis=split_axis))

    def fitness_function(self):
        def count_elements(x):
            return len(set(x))
        total_x = sum(numpy.apply_along_axis(count_elements, 0, self.square))
        total_y = sum(numpy.apply_along_axis(count_elements, 1, self.square))
        total_z = 0
        dimension_sqrt = int(math.sqrt(self.dimension))
        for i in numpy.split(self.square, dimension_sqrt):
            for k in numpy.split(i, dimension_sqrt, 1):
                total_z += count_elements(k.ravel())
        total = (total_x**2) + (total_y**2) + (total_z ** 2)
        return total / self.perfect_points()


if __name__ == '__main__':
    random.seed(42)
    """
    Possible solution for following 4x4 sudoku:
    [4 3 2 1]
    [1 2 4 3]
    [2 1 3 4]
    [3 4 1 2]

    For testing purposes it's possible to comment/uncomment
    multiple entries to test the functioning of the Sudoku Solver

    """
    fixed_elements_4 = {
        (0, 0): 4,
        # (0, 1): 3,
        # (0, 2): 2,
        # (0, 3): 1,
        # (1, 0): 1,
        # (1, 1): 2,
        (1, 2): 4,
        # (1, 3): 3,
        (2, 0): 2,
        # (2, 1): 1,
        # (2, 2): 3,
        (2, 3): 4,
        # (3, 0): 3,
        # (3, 1): 4,
        (3, 2): 1,
        (3, 3): 2,
    }

    """

    One possible result of the following sudoku is:
    [7 4 9 8 6 1 5 2 3]
    [3 1 6 2 5 4 7 8 9]
    [8 2 5 7 3 9 4 6 1]
    [1 9 7 5 8 6 2 3 4]
    [5 8 4 9 2 3 6 1 7]
    [6 3 2 4 1 7 9 5 8]
    [2 7 3 6 9 8 1 4 5]
    [4 5 8 1 7 2 3 9 6]
    [9 6 1 3 4 5 8 7 2]

    """

    fixed_elements_9 = {
        (0, 2): 9,
        (0, 4): 6,
        (1, 5): 4,
        (1, 7): 8,
        (1, 8): 9,
        (2, 0): 8,
        (2, 1): 2,
        (2, 6): 4,
        (2, 8): 1,
        (3, 3): 5,
        (3, 5): 6,
        (3, 6): 2,
        (4, 1): 8,
        (4, 2): 4,
        (4, 3): 9,
        (4, 5): 3,
        (4, 6): 6,
        (4, 7): 1,
        (4, 8): 7,
        (5, 0): 6,
        (5, 1): 3,
        (5, 2): 2,
        (5, 3): 4,
        (5, 5): 7,
        (5, 7): 5,
        (5, 8): 8,
        (6, 5): 8,
        (6, 6): 1,
        (7, 1): 5,
        (7, 2): 8,
        (7, 4): 7,
        (7, 5): 2,
        (8, 0): 9,
        (8, 1): 6,
        (8, 4): 4,
        (8, 7): 7,
    }
    fixed_elements_16 = {(0, i): i + 1 for i in range(16)}

    # sud = SudokuPool(30, dimension=9, fixed_elements=fixed_elements_)
    # best = sud.run_genetic_algorithm(1000)
    # print(best.square)
    # print(best.fitness_function())
    s = SudokuElement(9, fixed_elements=fixed_elements_9)
    print(s._get_nearly_perfect_square())
    print(s.backtrack_solve())
    s = SudokuElement(4, fixed_elements=fixed_elements_4)
    print(s._get_nearly_perfect_square())
    print(s.backtrack_solve())
    s = SudokuElement(16, fixed_elements=fixed_elements_16)
    print(s._get_nearly_perfect_square())
    print(s.backtrack_solve())
