import numpy as np


def print_line(a):
    assert a.size == 9, 'Array must have nine entries.'
    print('|', a[0], a[1], a[2], '|', a[3], a[4], a[5], '|', a[6], a[7], a[8], '|')
    return None


class SudokuSolver:
    def __init__(self):
        self.field_sudoku = np.zeros((9, 9), dtype=np.int)
        self.solutions = []

    def set_sudoku_field(self, field_input):
        assert field_input.shape == (9, 9), 'Sudoku field has to be of shape (9, 9).'
        self.field_sudoku = field_input
        if self.is_field_possible():
            return True
        else:
            print('Got a sudoku field, which has no solution:')
            self.show_field()
            return False

    def show_field(self):
        print(25 * '-')
        for box in range(0, 3):
            for row in range(0, 3):
                print_line(self.field_sudoku[box * 3 + row])
            print(25 * '-')
        print('\n')
        return None

    def check_possibility_entry(self, x, y, number):
        for row in range(0, 9):
            if self.field_sudoku[row, x] == number:
                return False
        for column in range(0, 9):
            if self.field_sudoku[y, column] == number:
                return False
        box_col = x // 3
        box_row = y // 3
        for i in range(3 * box_col, 3 * box_col + 3):
            for j in range(3 * box_row, 3 * box_row + 3):
                if self.field_sudoku[j, i] == number:
                    return False
        return True

    def is_field_possible(self):
        for col in range(0, 9):
            for row in range(0, 9):
                if self.field_sudoku[row, col] == 0:
                    continue
                else:
                    if np.sum(self.field_sudoku[row, :] == self.field_sudoku[row, col]) > 1:
                        print('Error found in row {}'.format(row + 1))
                        return False
                    elif np.sum(self.field_sudoku[:, col] == self.field_sudoku[row, col]) > 1:
                        print('Error found in column {}'.format(col + 1))
                        return False
                    else:
                        box_col = col // 3
                        box_row = row // 3
                        box = self.field_sudoku[3 * box_row:3 * box_row + 3, 3 * box_col:3 * box_col + 3]
                        if np.sum(box == self.field_sudoku[row, col]) > 1:
                            print('Error found in box ({}, {})'.format(box_row + 1, box_col + 1))
                            return False
        return True

    def solve_for_all_solutions(self):
        for x in range(0, 9):
            for y in range(0, 9):
                if self.field_sudoku[y, x] == 0:
                    for n in range(1, 10):
                        if self.check_possibility_entry(x, y, n):
                            self.field_sudoku[y, x] = n
                            self.solve_for_all_solutions()
                            self.field_sudoku[y, x] = 0
                    return None
        if np.min(self.field_sudoku) > 0:
            self.solutions.append(self.field_sudoku.copy())
