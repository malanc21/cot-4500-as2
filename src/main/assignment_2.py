import numpy as np
from numpy.linalg import inv
from decimal import Decimal


np.set_printoptions(precision=7, suppress=True, linewidth=100)


def nevilles_method(x_points, y_points, value):
    num_of_points = len(x_points)
    matrix = np.zeros((num_of_points, num_of_points))

    # fill in y values
    for counter, i in enumerate(matrix):
        i[0] = Decimal(y_points[counter])

    # i = row, j = column
    for i in range(1, num_of_points):
        for j in range(1, i + 1):
            # Q_1,1 = (1/(x_1-x_0)) * [(x-x_0)*Q_1,0 - (x-x_1)*Q_0,0]
            first_multiplication = (value - Decimal(x_points[i-j])) * Decimal(matrix[i][j-1])
            second_multiplication = (value - Decimal(x_points[i])) * Decimal(matrix[i-1][j-1])

            denominator = Decimal(x_points[i] - x_points[i-j])

            coefficient = (first_multiplication - second_multiplication) / denominator

            matrix[i][j] = coefficient

    print(str(matrix[2][2]) + "\n")


def divided_difference(x_points, y_points):
    # set up the matrix
    size = len(x_points)
    matrix = np.zeros((size, size))

    # fill the matrix
    for index, row in enumerate(matrix):
        row[0] = y_points[index]

    # populate the matrix
    for i in range(1, size):
        for j in range(1, i+1):
            # difference = (left - upper left) / (span of x-values)
            numerator = Decimal(matrix[i][j-1] - matrix[i-1][j-1])
            denominator = Decimal(x_points[i] - x_points[i-j])

            operation = Decimal(numerator / denominator)

            matrix[i][j] = operation

    return matrix


def approximations_for_degrees(x_points, value, matrix):
    print("[" + str(matrix[1, 1]) + ", " + str(matrix[2, 2]) + ", " + str(matrix[3, 3]) + "]" + "\n")

    value = float(value)

    # p_1 = f(x_0) + a_1*(x - x_0)
    p_1 = matrix[0][0] + matrix[1][1]*(value - x_points[0])
    p_2 = p_1 + matrix[2][2]*(value-x_points[0])*(value-x_points[1])
    p_3 = p_2 + matrix[3][3]*(value-x_points[0])*(value-x_points[1])*(value-x_points[2])

    print(str(p_3) + "\n")


def hermite_interpolation(x_points, y_points, slopes):
    num_of_points = len(x_points)
    matrix = np.zeros((2*num_of_points, 2*num_of_points))

    # populate x values
    for x in range(0, num_of_points):
        matrix[2*x][0] = x_points[x]
        matrix[2*x+1][0] = x_points[x]

    # populate y values
    for y in range(0, num_of_points):
        matrix[2*y][1] = y_points[y]
        matrix[2*y+1][1] = y_points[y]

    # populate with derivates
    for z in range(0, num_of_points):
        matrix[2 * z + 1][2] = slopes[z]

    filled_matrix = hermite_divided_diff(matrix)

    print(str(filled_matrix) + "\n")


def hermite_divided_diff(matrix):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i+2):
            # skip if cell is already filled
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue

            # difference = (left - upper left) / (span of x-values)
            # numerator = left - diagonal left cell
            numerator = Decimal(matrix[i][j-1] - matrix[i-1][j-1])

            denominator = Decimal(matrix[i][0]) - Decimal(matrix[i-(j-1)][0])

            operation = numerator / denominator

            matrix[i][j] = operation

    return matrix


def cubic_spline(x_points, y_points):
    # Matrix A
    size = len(x_points)
    matrix_a = np.zeros((size, size))

    # fill diagonal cells through middle of matrix
    for i in range(0, size):
        for j in range(i, size):
            if j == 0 or j == size - 1:
                matrix_a[i][j] = 1
            else:
                # h_i = x_i+1 - x_i
                h_i_minus_1 = x_points[i] - x_points[i - 1]
                h_i = x_points[i + 1] - x_points[i]
                matrix_a[i][j] = 2 * (h_i + h_i_minus_1)

            i = i+1

            if i == (size-1):
                break

    # fill diagonal to the left of (below) middle diagonal
    for i in range(1, size-2):
        for j in range(i-1, size-1):
            if j == size - 2:
                break
            else:
                matrix_a[i][j] = x_points[(j + 1)] - x_points[j]
                i = i+1

    # fill diagonal to the right of (above) middle diagonal
    for i in range(1, size-1):
        for j in range(i+1, size):
            if j == size:
                break
            else:
                matrix_a[i][j] = x_points[i+1] - x_points[i]
                i = i+1

    print(str(matrix_a) + "\n")

    # Vector b
    vector_b = np.zeros(size)

    for i in range(1, size-1):
        # vector_b[1] = (3/h_1)*(a_2-a_1) - (3/h_0)(a_1-a_0)
        # always get 0 for 1st and last entry of b vector
        first_term = (3/(x_points[i+1] - x_points[i])) * (y_points[i+1]-y_points[i])
        second_term = (3/(x_points[i] - x_points[i-1])) * (y_points[i]-y_points[i-1])
        vector_b[i] = first_term - second_term

    print(str(vector_b) + "\n")

    # Vector x
    # A*x=b so x=A^-1*b
    matrix_a_inv = inv(matrix_a)
    vector_x = np.dot(matrix_a_inv, vector_b)

    print(str(vector_x) + "\n")


def main():
    # QUESTION 1 / Neville's method
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    interpolating_value = Decimal(3.7)

    nevilles_method(x_points, y_points, interpolating_value)

    # QUESTION 2 AND 3/ Newton's forward method AND Approximate f(x)
    x_points = [7.2, 7.4, 7.5, 7.6]
    y_points = [23.5492, 25.3913, 26.8224, 27.4589]
    approximation_value = Decimal(7.3)

    matrix = divided_difference(x_points, y_points)
    approximations_for_degrees(x_points, approximation_value, matrix)

    # QUESTION 4 / Hermite polynomial approximation matrix
    # point setup
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]

    hermite_interpolation(x_points, y_points, slopes)

    # QUESTION 5 / Cubic spline interpolation
    x_points = [2, 5, 8, 10]
    y_points = [3, 5, 7, 9]

    cubic_spline(x_points, y_points)


if __name__ == '__main__':
    main()
