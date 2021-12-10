from SudokuSolver import SudokuSolver
from SudokuReader import SudokuReader

if __name__ == '__main__':
    path_img = 'Sudoku_Images/Sudoku1.jpeg'
    path_clf = 'NumberClassifier_v1'
    reader = SudokuReader(path_img=path_img, path_clf=path_clf, debug=False)
    if reader.get_sudoku_field_from_image():
        input_field = reader.sudoku_field
        solver = SudokuSolver()
        if solver.set_sudoku_field(input_field):
            solver.solve_for_all_solutions()
            print('Found {} solution(s)'.format(len(solver.solutions)))
            for i in range(0, len(solver.solutions)):
                reader.show_solution_on_sudoku(solver.solutions[i])
