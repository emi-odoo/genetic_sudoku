# genetic_sudoku

Simple project on creating a `genetic algorithm` to solve Sudoku Puzzles.

The main class is `SudokuElement`, which represents a single Sudoku Puzzle (a n by n square). 

The second class is `SudokuPool`, which represents a collections of `SudokuElements` and has some helper methods to run the genetic algorithm.

## Genetic Algorithm

The genetic algorithm is pretty simple and standard.

* `fitness_function`: is a function that returns a float value representing how a particular puzzle is close to a `perfect square`.
* `selection`: a random new population is selected, using their `fitness` as the weight for the selection (more fitted `==` more possibilities to be chosen).
* `crossover`: each new element of the population is paired with another one and some of their `rows` or `columns` are swapped.
* `mutation`: using a parameter passed during the instantiation of the `SudokuElement` object, each element of the `n`x`n` square can be randomly changed to another random value.

The `Genetic Algorithm` is working nicely with 4 by 4 squares, while it struggles with 9 by 9.

## Backtracking Algorithm

This type of problem is best tackled using backtracking algorithms (or similar), that's why I implemented a simple backtracking algorithm to find a solution to a given Sudoku Puzzle.

## Conclusion

This is a very simple implementation and I might work in the future to add some features to it. It can be helpful if someone wants to work on somethings similar or is trying to find something to do in their free time :)
