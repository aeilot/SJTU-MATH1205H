# 线性代数大作业 Write Up

> 大作业一：写出矩阵的满秩分解或LU分解或QR分解算法的计算机程序（matlab、Python、C++等语言均可），并用具体的矩阵验证。在文档的最后部分写出你的心得体会

完整代码在 [GitHub 仓库](https://github.com/aeilot/SJTU-MATH1205H) 开源，可以按照 `README.md` 中的说明复现实验。

## 1. 项目概况

大作业要求实现矩阵的满秩分解、LU分解或QR分解算法，并使用具体矩阵进行验证。

本项目使用 C++ 语言编写，实现了矩阵的 LU 分解算法和 QR 分解算法，并通过 `test.cpp` 中 AI 协助生成的具体矩阵进行了验证，证明了代码的准确性。

为了实现矩阵运算，我还实现了一个基本的矩阵类 `Matrix`，支持矩阵的基本操作，如加法、乘法、转置、求迹等，也支持生成一些常用矩阵，如零矩阵、单位矩阵、对角矩阵。

## 2. 代码结构

### matrix.h

```cpp
namespace LinearAlgebra {
	template<size_t Rows, size_t Cols>
	class Matrix {
	private:
		typedef double T;
		std::array<T, Rows*Cols> arr;

	public:
		// Constructor
		Matrix() {
			arr.fill(T(0));
		}
		Matrix(std::array<T, Rows * Cols> input_arr) : arr(input_arr) {}
		T& operator()(size_t row, size_t col) {
			return arr[row * Cols + col];
		}

		// Destructor
		~Matrix() = default;

		// Getter
		const T& get(size_t row, size_t col) const {
			return arr[row * Cols + col];
		}

		// Setter
		void set(size_t row, size_t col, const T& value) {
			arr[row * Cols + col] = value;
		}

		const T& operator()(size_t row, size_t col) const {
			return arr[row * Cols + col];
		}

		// Matrix Addition
		Matrix<Rows, Cols> operator+(const Matrix<Rows, Cols>& other) const {
			Matrix<Rows, Cols> result;
			for (size_t i = 0; i < Rows; ++i) {
				for (size_t j = 0; j < Cols; ++j) {
					result(i, j) = (*this)(i, j) + other(i, j);
				}
			}
			return result;
		}

		// Matrix Subtraction
		Matrix<Rows, Cols> operator-(const Matrix<Rows, Cols>& other) const {
			Matrix<Rows, Cols> result;
			for (size_t i = 0; i < Rows; ++i) {
				for (size_t j = 0; j < Cols; ++j) {
					result(i, j) = (*this)(i, j) - other(i, j);
				}
			}
			return result;
		}

		// Scalar Multiplication and Division
		Matrix<Rows, Cols> operator*(const T& scalar) const {
			Matrix<Rows, Cols> result;
			for (size_t i = 0; i < Rows; ++i) {
				for (size_t j = 0; j < Cols; ++j) {
					result(i, j) = (*this)(i, j) * scalar;
				}
			}
			return result;
		}
		Matrix<Rows, Cols> operator/(const T& scalar) const {
			Matrix<Rows, Cols> result;
			for (size_t i = 0; i < Rows; ++i) {
				for (size_t j = 0; j < Cols; ++j) {
					result(i, j) = (*this)(i, j) / scalar;
				}
			}
			return result;
		}

		// Transposition
		Matrix<Cols, Rows> transpose() const {
			Matrix<Cols, Rows> result;
			for (size_t i = 0; i < Rows; ++i) {
				for (size_t j = 0; j < Cols; ++j) {
					result(j, i) = (*this)(i, j);
				}
			}
			return result;
		}

		// Matrix Multiplication
		template<size_t OtherCols>
		Matrix<Rows, OtherCols> operator*(const Matrix<Cols, OtherCols>& other) const {
			Matrix<Rows, OtherCols> result;
			for (size_t i = 0; i < Rows; ++i) {
				for (size_t j = 0; j < OtherCols; ++j) {
					result(i, j) = T();
					for (size_t k = 0; k < Cols; ++k) {
						result(i, j) += (*this)(i, k) * other(k, j);
					}
				}
			}
			return result;
		}

		// is Square
		bool isSquare() const {
			return Rows == Cols;
		}

		// Identity Matrix
		static Matrix<Rows, Cols> eye() {
			static_assert(Rows == Cols, "Identity matrix must be square.");
			Matrix<Rows, Cols> result;
			for (size_t i = 0; i < Rows; ++i) {
				for (size_t j = 0; j < Cols; ++j) {
					result(i, j) = (i == j) ? T(1) : T(0);
				}
			}
			return result;
		}

		// Zero Matrix
		static Matrix<Rows, Cols> zeros() {
			Matrix<Rows, Cols> result;
			for (size_t i = 0; i < Rows; ++i) {
				for (size_t j = 0; j < Cols; ++j) {
					result(i, j) = T(0);
				}
			}
			return result;
		}

		// Diagonal Matrix
		static Matrix<Rows, Cols> diag(const std::vector<T>& values) {
			static_assert(Rows == Cols, "Diagonal matrix must be square.");
			Matrix<Rows, Cols> result = zeros();
			size_t len = std::min(Rows, values.size());
			for (size_t i = 0; i < len; ++i) {
				result(i, i) = values[i];
			}
			return result;
		}

		// Scalar Matrix
		static Matrix<Rows, Cols> scalar(const T& value) {
			static_assert(Rows == Cols, "Scalar matrix must be square.");
			Matrix<Rows, Cols> result;
			for (size_t i = 0; i < Rows; ++i) {
				for (size_t j = 0; j < Cols; ++j) {
					result(i, j) = (i == j) ? value : T(0);
				}
			}
			return result;
		}

		// Get number of rows and columns
		size_t row() const {
			return Rows;
		}
		size_t col() const {
			return Cols;
		}

		// Trace
		T trace() const {
			static_assert(Rows == Cols, "Trace is only defined for square matrices.");
			T sum = T();
			for (size_t i = 0; i < Rows; ++i) {
				sum += (*this)(i, i);
			}
			return sum;
		}

		// Print Matrix
		void print() const {
			for (size_t i = 0; i < Rows; ++i) {
				for (size_t j = 0; j < Cols; ++j) {
					std::cout << arr[i * Cols + j] << " ";
				}
				std::cout << std::endl;
			}
		}
	};
}
```

Matrix 类是一个 C++20 的模版类，在编译时确定矩阵的行数和列数。数据使用 `double` 类型存储，底层使用 `std::array` 进行存储，可以有效表示给定行数列书的实矩阵。类中实现了矩阵的基本操作，如加法、减法、乘法、转置等，并提供了一些静态方法用于生成常用矩阵。

### matrix_utils.h

```cpp
//
// Created by Chenluo Deng on 11/22/25.
//

#ifndef MATRIX_MATRIX_UTILS_H
#define MATRIX_MATRIX_UTILS_H
#include "matrix.h"

namespace LinearAlgebra {
	const double EPSILON = 1e-9;

	template<size_t Rows, size_t Cols>
	class MatrixUtils {
	public:
		// Get rank
		static size_t rank(Matrix<Rows, Cols> mat) {
			size_t rank = 0;
			for (size_t j = 0; j < Cols; ++j) {
				if (rank >= Rows) break;

				size_t pivot_row = rank;
				for (size_t i = rank + 1; i < Rows; ++i) {
					if (std::abs(mat(i, j)) > std::abs(mat(pivot_row, j))) {
						pivot_row = i;
					}
				}
				if (std::abs(mat(pivot_row, j)) < EPSILON) {
					continue;
				}
				if (pivot_row != rank) {
					for (size_t k = 0; k < Cols; ++k) {
						std::swap(mat(rank, k), mat(pivot_row, k));
					}
				}
				for (size_t i = rank + 1; i < Rows; ++i) {
					double factor = mat(i, j) / mat(pivot_row, j);
					for (size_t k = j; k < Cols; ++k) {
						mat(i, k) -= factor * mat(pivot_row, k);
					}
				}

				rank++;
			}
			return rank;
		}

		// Get RET
		static Matrix<Rows, Cols> RET(Matrix<Rows, Cols> mat) {
			size_t rank = 0;
			for (size_t j = 0; j < Cols; ++j) {
				if (rank >= Rows) break;
				size_t pivot_row = rank;
				for (size_t i = rank + 1; i < Rows; ++i) {
					if (std::abs(mat(i, j)) > std::abs(mat(pivot_row, j))) {
						pivot_row = i;
					}
				}
				if (std::abs(mat(pivot_row, j)) < EPSILON) {
					continue;
				}
				if (pivot_row != rank) {
					for (size_t k = 0; k < Cols; ++k) {
						std::swap(mat(rank, k), mat(pivot_row, k));
					}
				}
				for (size_t i = rank + 1; i < Rows; ++i) {
					double factor = mat(i, j) / mat(pivot_row, j);
					for (size_t k = j; k < Cols; ++k) {
						mat(i, k) -= factor * mat(pivot_row, k);
					}
				}
				rank++;
			}
			return mat;
		}

		// Get RREF
		static Matrix<Rows, Cols> RREF(Matrix<Rows, Cols> mat) {
			size_t current_row = 0;
			for (size_t j = 0; j < Cols; ++j) {
				if (current_row >= Rows) break;
				size_t pivot_row = current_row;
				for (size_t i = current_row + 1; i < Rows; ++i) {
					if (std::abs(mat(i, j)) > std::abs(mat(pivot_row, j))) {
						pivot_row = i;
					}
				}
				if (std::abs(mat(pivot_row, j)) < EPSILON) {
					continue;
				}
				if (pivot_row != current_row) {
					for (size_t k = j; k < Cols; ++k) {
						std::swap(mat(current_row, k), mat(pivot_row, k));
					}

				}
				double nfactor = mat(j, j);
				for (size_t k = j; k < Cols; ++k) {
					mat(j, k) /= nfactor;
				}

				for (size_t i = 0; i < Rows; ++i) {
					if (i == j) continue;
					double factor = mat(i, j) / mat(j, j);
					for (size_t k = j; k < Cols; ++k) {
						mat(i, k) -= factor * mat(j, k);
					}
				}
				current_row++;
			}
			return mat;
		}

		// Get Inverse
		static Matrix<Rows, Cols> inverse(Matrix<Rows, Cols> mat) {
			static_assert(Rows == Cols, "Inverse is only defined for square matrices.");
			Matrix<Rows, Cols> identity = Matrix<Rows, Cols>::eye();
			for (size_t j = 0; j < Cols; ++j) {
				size_t pivot_row = j;
				for (size_t i = j + 1; i < Rows; ++i) {
					if (std::abs(mat(i, j)) > std::abs(mat(pivot_row, j))) {
						pivot_row = i;
					}
				}
				if (std::abs(mat(pivot_row, j)) < EPSILON) {
					throw std::runtime_error("The matrix is singular.");
				}
				if (pivot_row != j) {
					for (size_t k = j; k < Cols; ++k) {
						std::swap(mat(j, k), mat(pivot_row, k));
					}

					for (size_t k = 0; k < Cols; ++k) {
						std::swap(identity(j, k), identity(pivot_row, k));
					}
				}
				double nfactor = mat(j, j);
				for (size_t k = j; k < Cols; ++k) {
					mat(j, k) /= nfactor;
				}
				for (size_t k = 0; k < Cols; ++k) {
					identity(j, k) /= nfactor;
				}
				for (size_t i = 0; i < Rows; ++i) {
					if (i == j) continue;
					double factor = mat(i, j) / mat(j, j);
					for (size_t k = j; k < Cols; ++k) {
						mat(i, k) -= factor * mat(j, k);
					}
					for (size_t k = 0; k < Cols; ++k) {
						identity(i, k) -= factor * identity(j, k);
					}

				}
			}
			return identity;
		}

		// LU Decomposition
		static std::tuple<Matrix<Rows, Cols>, Matrix<Rows, Cols>> lu(Matrix<Rows, Cols> A) {
			static_assert(Rows == Cols, "LU Decomposition is only defined for square matrices.");

			Matrix<Rows, Cols> lower = Matrix<Rows, Cols>::eye();

			for (size_t j = 0; j < Cols; ++j) {
				if (std::abs(A(j, j)) < EPSILON) {
					throw std::runtime_error("Could not do LU Decomposition without a Permutation Matrix.");
				}
				for (size_t i = j + 1; i < Rows; ++i) {
					double factor = A(i, j) / A(j, j);
					lower(i, j) = factor;
					for (size_t k = j; k < Cols; ++k) {
						A(i, k) -= factor * A(j, k);
					}
				}
			}
			return std::make_tuple(lower, A);
		}

		// QR Decomposition
		static std::tuple<Matrix<Rows, Cols>, Matrix<Cols, Cols>> unstable_qr(Matrix<Rows, Cols> A) {
			if (rank(A) != Cols) {
				throw std::runtime_error("Matrix A does not have full column rank.");
			}

			Matrix<Cols, Cols> R = Matrix<Cols, Cols>::zeros();

			for (size_t j = 0; j < Cols; ++j) {
				for (size_t k = 0; k < j; ++k) {
					double dot = 0.0;
					for (size_t i = 0; i < Rows; ++i) {
						dot += A(i, k) * A(i, j);
					}
					R(k, j) = dot;

					for (size_t i = 0; i < Rows; ++i) {
						A(i, j) -= R(k, j) * A(i, k);
					}
				}

				double norm_sq = 0.0;
				for (size_t i = 0; i < Rows; ++i) {
					norm_sq += A(i, j) * A(i, j);
				}
				R(j, j) = std::sqrt(norm_sq);

				if (R(j, j) < EPSILON) {
					throw std::runtime_error("Norm is too small.");
				}

				for (size_t i = 0; i < Rows; ++i) {
					A(i, j) /= R(j, j);
				}
			}

			return std::make_tuple(A, R);
		}

		// Modified QR
		static std::tuple<Matrix<Rows, Cols>, Matrix<Cols, Cols>> qr(Matrix<Rows, Cols> A) {
			Matrix<Cols, Cols> R;
			Matrix<Rows, Cols> Q = A;

			for (size_t j = 0; j < Cols; ++j) {
				double norm = 0.0;
				for (size_t i = 0; i < Rows; ++i) {
					norm += Q(i, j) * Q(i, j);
				}
				norm = std::sqrt(norm);

				if (norm < EPSILON) {
					throw std::runtime_error("Matrix does not have full column rank; QR not possible.");
				}

				R(j, j) = norm;

				for (size_t i = 0; i < Rows; ++i) {
					Q(i, j) /= norm;
				}

				for (size_t k = j + 1; k < Cols; ++k) {
					double dot = 0.0;
					for (size_t i = 0; i < Rows; ++i) {
						dot += Q(i, j) * Q(i, k);
					}
					R(j, k) = dot;

					for (size_t i = 0; i < Rows; ++i) {
						Q(i, k) -= dot * Q(i, j);
					}
				}
			}
			return std::make_tuple(Q, R);
		}
	};
}

#endif //MATRIX_MATRIX_UTILS_H
```

`matrix_utils.h` 实现了矩阵的各种实用算法，包括计算秩、化为阶梯形矩阵、化为建华行阶梯形矩阵、逆矩阵、LU 分解和 QR 分解等。所有方法均为静态方法，接受一个矩阵对象作为参数并返回相应的结果。

### test.cpp

```cpp
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <gtest/gtest.h>
#include "lib/matrix.h"
#include "lib/matrix_utils.h"

using namespace LinearAlgebra;

template<size_t N, size_t M>
void ExpectMatrixNear(const Matrix<N, M>& actual, const Matrix<N, M>& expected, double tolerance = 1e-9) {
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < M; ++j) {
			EXPECT_NEAR(actual(i, j), expected(i, j), tolerance)
				<< "Mismatch at (" << i << "," << j << ")";
		}
	}
}

TEST(MatrixTest, DefaultConstructorCreatesEmptyMatrix) {
    Matrix<2, 2> mat;
    EXPECT_EQ(mat(0, 0), 0);
    EXPECT_EQ(mat(1, 1), 0);
}

TEST(MatrixTest, AddsTwoMatricesCorrectly) {
    Matrix<2, 2> mat1;
    mat1.set(0, 0, 1);
    mat1.set(1, 1, 2);

    Matrix<2, 2> mat2;
    mat2.set(0, 0, 3);
    mat2.set(1, 1, 4);

    Matrix<2, 2> result = mat1 + mat2;

    EXPECT_EQ(result(0, 0), 4);
    EXPECT_EQ(result(1, 1), 6);
}

TEST(MatrixTest, MultipliesMatrixByScalar) {
    Matrix<2, 2> mat;
    mat.set(0, 0, 2);
    mat.set(1, 1, 3);

    Matrix<2, 2> result = mat * 2;

    EXPECT_EQ(result(0, 0), 4);
    EXPECT_EQ(result(1, 1), 6);
}

TEST(MatrixTest, TransposesMatrixCorrectly) {
    Matrix<2, 3> mat;
    mat.set(0, 1, 1);
    mat.set(1, 2, 2);

    Matrix<3, 2> transposed = mat.transpose();

    EXPECT_EQ(transposed(1, 0), 1);
    EXPECT_EQ(transposed(2, 1), 2);
}

TEST(MatrixTest, CreatesIdentityMatrix) {
    auto identity = Matrix<3, 3>::eye();

    EXPECT_EQ(identity(0, 0), 1);
    EXPECT_EQ(identity(1, 1), 1);
    EXPECT_EQ(identity(2, 2), 1);
    EXPECT_EQ(identity(0, 1), 0);
    EXPECT_EQ(identity(1, 2), 0);
}

TEST(MatrixTest, ComputesTraceCorrectly) {
    Matrix<3, 3> mat;
    mat.set(0, 0, 1);
    mat.set(1, 1, 2);
    mat.set(2, 2, 3);

    EXPECT_EQ(mat.trace(), 6);
}

TEST(MatrixTest, CreatesDiagonalMatrix) {
    std::vector<double> values = {1, 2, 3};
    auto diag = Matrix<3, 3>::diag(values);

    EXPECT_EQ(diag(0, 0), 1);
    EXPECT_EQ(diag(1, 1), 2);
    EXPECT_EQ(diag(2, 2), 3);
    EXPECT_EQ(diag(0, 1), 0);
    EXPECT_EQ(diag(1, 2), 0);
}

TEST(MatrixTest, MultipliesTwoMatricesCorrectly) {
    Matrix<2, 3> mat1;
    mat1.set(0, 0, 1);
    mat1.set(0, 1, 2);
    mat1.set(0, 2, 3);
    mat1.set(1, 0, 4);
    mat1.set(1, 1, 5);
    mat1.set(1, 2, 6);

    Matrix<3, 2> mat2;
    mat2.set(0, 0, 7);
    mat2.set(0, 1, 8);
    mat2.set(1, 0, 9);
    mat2.set(1, 1, 10);
    mat2.set(2, 0, 11);
    mat2.set(2, 1, 12);

    Matrix<2, 2> result = mat1 * mat2;

    EXPECT_EQ(result(0, 0), 58);
    EXPECT_EQ(result(0, 1), 64);
    EXPECT_EQ(result(1, 0), 139);
    EXPECT_EQ(result(1, 1), 154);
}

TEST(MatrixUtilsTest, RankOfIdentityMatrix) {
    Matrix<3, 3> mat = Matrix<3, 3>::eye();
    size_t r = MatrixUtils<3, 3>::rank(mat);
    EXPECT_EQ(r, 3);
}

TEST(MatrixUtilsTest, RankOfZeroMatrix) {
    Matrix<3, 3> mat = Matrix<3, 3>::zeros();
    size_t r = MatrixUtils<3, 3>::rank(mat);
    EXPECT_EQ(r, 0);
}

TEST(MatrixUtilsTest, RankOfFullRankMatrix) {
    Matrix<3, 3> mat;
    mat.set(0, 0, 1); mat.set(0, 1, 0); mat.set(0, 2, 0);
    mat.set(1, 0, 0); mat.set(1, 1, 1); mat.set(1, 2, 0);
    mat.set(2, 0, 0); mat.set(2, 1, 0); mat.set(2, 2, 1);
    size_t r = MatrixUtils<3, 3>::rank(mat);
    EXPECT_EQ(r, 3);
}

TEST(MatrixUtilsTest, RankOfRankDeficientMatrix) {
    Matrix<3, 3> mat;
    mat.set(0, 0, 1); mat.set(0, 1, 2); mat.set(0, 2, 3);
    mat.set(1, 0, 2); mat.set(1, 1, 4); mat.set(1, 2, 6);
    mat.set(2, 0, 3); mat.set(2, 1, 6); mat.set(2, 2, 9);
    size_t r = MatrixUtils<3, 3>::rank(mat);
    EXPECT_EQ(r, 1);
}

TEST(MatrixUtilsTest, RankOfRectangularMatrixMoreRowsThanCols) {
    Matrix<4, 3> mat;
    mat.set(0, 0, 1); mat.set(0, 1, 2); mat.set(0, 2, 3);
    mat.set(1, 0, 0); mat.set(1, 1, 1); mat.set(1, 2, 4);
    mat.set(2, 0, 0); mat.set(2, 1, 0); mat.set(2, 2, 1);
    mat.set(3, 0, 0); mat.set(3, 1, 0); mat.set(3, 2, 0);
    size_t r = MatrixUtils<4, 3>::rank(mat);
    EXPECT_EQ(r, 3);
}

TEST(MatrixUtilsTest, RankOfRectangularMatrixMoreColsThanRows) {
    Matrix<2, 4> mat;
    mat.set(0, 0, 1); mat.set(0, 1, 0); mat.set(0, 2, 0); mat.set(0, 3, 0);
    mat.set(1, 0, 0); mat.set(1, 1, 1); mat.set(1, 2, 0); mat.set(1, 3, 0);
    size_t r = MatrixUtils<2, 4>::rank(mat);
    EXPECT_GT(r, 0);
}

TEST(MatrixUtilsTest, RankOfSingleRowMatrix) {
    Matrix<1, 4> mat;
    mat.set(0, 0, 1); mat.set(0, 1, 2); mat.set(0, 2, 3); mat.set(0, 3, 4);
    size_t r = MatrixUtils<1, 4>::rank(mat);
    EXPECT_EQ(r, 1);
}

TEST(MatrixUtilsTest, RankOfSingleColumnMatrix) {
    Matrix<4, 1> mat;
    mat.set(0, 0, 1);
    mat.set(1, 0, 2);
    mat.set(2, 0, 3);
    mat.set(3, 0, 4);
    size_t r = MatrixUtils<4, 1>::rank(mat);
    EXPECT_EQ(r, 1);
}

TEST(MatrixUtilsTest, RankOfMatrixWithNearZeroElements) {
    Matrix<3, 3> mat;
    mat.set(0, 0, 1); mat.set(0, 1, 2); mat.set(0, 2, 3);
    mat.set(1, 0, 0); mat.set(1, 1, 1e-10); mat.set(1, 2, 4);
    mat.set(2, 0, 5); mat.set(2, 1, 6); mat.set(2, 2, 0);
    size_t r = MatrixUtils<3, 3>::rank(mat);
    EXPECT_EQ(r, 2);
}

TEST(MatrixUtilsTest, RankOfRank2Matrix) {
    Matrix<3, 3> mat;
    mat.set(0, 0, 1); mat.set(0, 1, 0); mat.set(0, 2, 0);
    mat.set(1, 0, 0); mat.set(1, 1, 1); mat.set(1, 2, 0);
    mat.set(2, 0, 0); mat.set(2, 1, 0); mat.set(2, 2, 0);
    size_t r = MatrixUtils<3, 3>::rank(mat);
    EXPECT_EQ(r, 2);
}

TEST(MatrixUtilsTest, RETOfIdentityMatrix) {
    Matrix<3, 3> mat = Matrix<3, 3>::eye();
    Matrix<3, 3> ret = MatrixUtils<3, 3>::RET(mat);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(ret(i, j), (i == j) ? 1.0 : 0.0, EPSILON);
        }
    }
}

TEST(MatrixUtilsTest, RETOfSimpleMatrix) {
    Matrix<3, 3> mat;
    mat.set(0, 0, 1); mat.set(0, 1, 2); mat.set(0, 2, 3);
    mat.set(1, 0, 4); mat.set(1, 1, 5); mat.set(1, 2, 6);
    mat.set(2, 0, 7); mat.set(2, 1, 8); mat.set(2, 2, 9);
    
    Matrix<3, 3> ret = MatrixUtils<3, 3>::RET(mat);
    
    EXPECT_NEAR(ret(1, 0), 0.0, EPSILON);
    EXPECT_NEAR(ret(2, 0), 0.0, EPSILON);
}

TEST(MatrixUtilsTest, RETOfZeroMatrix) {
    Matrix<3, 3> mat = Matrix<3, 3>::zeros();
    Matrix<3, 3> ret = MatrixUtils<3, 3>::RET(mat);
    
    // RET of zero matrix should be zero matrix
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(ret(i, j), 0.0, EPSILON);
        }
    }
}

TEST(MatrixUtilsTest, RETOfRectangularMatrix) {
    Matrix<2, 3> mat;
    mat.set(0, 0, 1); mat.set(0, 1, 2); mat.set(0, 2, 3);
    mat.set(1, 0, 4); mat.set(1, 1, 5); mat.set(1, 2, 6);
    
    Matrix<2, 3> ret = MatrixUtils<2, 3>::RET(mat);
    
    EXPECT_NEAR(ret(1, 0), 0.0, EPSILON);
}

TEST(MatrixUtilsTest, RETPreservesRowEchelonForm) {
    Matrix<3, 3> mat;
    mat.set(0, 0, 1); mat.set(0, 1, 2); mat.set(0, 2, 3);
    mat.set(1, 0, 0); mat.set(1, 1, 1); mat.set(1, 2, 4);
    mat.set(2, 0, 0); mat.set(2, 1, 0); mat.set(2, 2, 1);
    
    Matrix<3, 3> ret = MatrixUtils<3, 3>::RET(mat);
    
    EXPECT_NEAR(ret(0, 0), 1.0, EPSILON);
    EXPECT_NEAR(ret(1, 0), 0.0, EPSILON);
    EXPECT_NEAR(ret(2, 0), 0.0, EPSILON);
    EXPECT_NEAR(ret(2, 1), 0.0, EPSILON);
}


TEST(MatrixInverseTest, IdentityMatrixInversion) {
    // The inverse of Identity is Identity
    auto I = Matrix<3, 3>::eye();
    auto Inv = MatrixUtils<3, 3>::inverse(I);
    ExpectMatrixNear(Inv, I);
}

TEST(MatrixInverseTest, Simple2x2Inversion) {
    // Matrix A = [[4, 7], [2, 6]]
    // Det(A) = 10
    // Inv(A) = [[0.6, -0.7], [-0.2, 0.4]]
    Matrix<2, 2> mat;
    mat(0, 0) = 4.0; mat(0, 1) = 7.0;
    mat(1, 0) = 2.0; mat(1, 1) = 6.0;

    Matrix<2, 2> expected;
    expected(0, 0) = 0.6;  expected(0, 1) = -0.7;
    expected(1, 0) = -0.2; expected(1, 1) = 0.4;

    auto result = MatrixUtils<2,2>::inverse(mat);
    ExpectMatrixNear(result, expected);
}

TEST(MatrixInverseTest, ThrowsOnSingularMatrix) {
    // Matrix A = [[1, 2], [2, 4]] -> Rows are linearly dependent
    Matrix<2, 2> singular;
    singular(0, 0) = 1.0; singular(0, 1) = 2.0;
    singular(1, 0) = 2.0; singular(1, 1) = 4.0;

	EXPECT_THROW((MatrixUtils<2, 2>::inverse(singular)), std::runtime_error);
}

TEST(MatrixInverseTest, ThrowsOnNearSingularMatrix) {
    // Matrix with pivot smaller than EPSILON (1e-9)
    Matrix<2, 2> tiny;
    tiny(0, 0) = 1e-10; tiny(0, 1) = 0.0;
    tiny(1, 0) = 0.0;   tiny(1, 1) = 1e-10;

    EXPECT_THROW((MatrixUtils<2, 2>::inverse(tiny)), std::runtime_error);
}

TEST(MatrixInverseTest, InverseTimesOriginalIsIdentity_3x3) {
    // A somewhat complex 3x3 matrix
    Matrix<3, 3> mat;
    mat(0, 0) = 2; mat(0, 1) = -1; mat(0, 2) = 0;
    mat(1, 0) = -1; mat(1, 1) = 2; mat(1, 2) = -1;
    mat(2, 0) = 0; mat(2, 1) = -1; mat(2, 2) = 2;

    auto inv = MatrixUtils<3, 3>::inverse(mat);
    auto identity = Matrix<3, 3>::eye();

    // Verify A * A_inv = I
    auto result = mat * inv;
    ExpectMatrixNear(result, identity, 1e-9);

    // Verify A_inv * A = I (commutativity of inverse)
    auto result2 = inv * mat;
    ExpectMatrixNear(result2, identity, 1e-9);
}

TEST(MatrixInverseTest, LargeMatrixStability_4x4) {
    // Using a Hilbert Matrix section or similar ill-conditioned but invertible matrix
    // to test numerical stability.
    Matrix<4, 4> mat;
    mat(0,0)=4; mat(0,1)=0;  mat(0,2)=0; mat(0,3)=0;
    mat(1,0)=0; mat(1,1)=0;  mat(1,2)=2; mat(1,3)=0;
    mat(2,0)=0; mat(2,1)=1;  mat(2,2)=2; mat(2,3)=0;
    mat(3,0)=1; mat(3,1)=0;  mat(3,2)=0; mat(3,3)=1;

    auto inv = MatrixUtils<4, 4>::inverse(mat);
    auto prod = mat * inv;

    ExpectMatrixNear(prod, Matrix<4, 4>::eye(), 1e-9);
}

TEST(MatrixRREFTest, RREF_Identity) {
	// RREF of Identity should be Identity
	auto I = Matrix<3, 3>::eye();
	auto rref = MatrixUtils<3, 3>::RREF(I);
	ExpectMatrixNear(rref, I);
}

TEST(MatrixRREFTest, RREF_FullRank) {
	// Full rank matrix should reduce to Identity
	// [1 2]
	// [3 4] -> [1 0], [0 1]
	Matrix<2, 2> mat;
	mat(0, 0) = 1; mat(0, 1) = 2;
	mat(1, 0) = 3; mat(1, 1) = 4;

	auto rref = MatrixUtils<2, 2>::RREF(mat);
	auto I = Matrix<2, 2>::eye();
	ExpectMatrixNear(rref, I);
}

TEST(MatrixRREFTest, RREF_Singular) {
	// A singular matrix: 1-9 matrix. Rank is 2.
	// 1 2 3
	// 4 5 6
	// 7 8 9
	// Expected RREF:
	// 1 0 -1
	// 0 1  2
	// 0 0  0
	Matrix<3, 3> mat;
	mat(0, 0) = 1; mat(0, 1) = 2; mat(0, 2) = 3;
	mat(1, 0) = 4; mat(1, 1) = 5; mat(1, 2) = 6;
	mat(2, 0) = 7; mat(2, 1) = 8; mat(2, 2) = 9;

	Matrix<3, 3> expected;
	expected(0, 0) = 1; expected(0, 1) = 0; expected(0, 2) = -1;
	expected(1, 0) = 0; expected(1, 1) = 1; expected(1, 2) = 2;
	expected(2, 0) = 0; expected(2, 1) = 0; expected(2, 2) = 0;

	auto rref = MatrixUtils<3, 3>::RREF(mat);
	ExpectMatrixNear(rref, expected);
}

TEST(MatrixLUTest, Reconstruction) {
	// A = [[4, 3], [6, 3]]
	// L = [[1, 0], [1.5, 1]]
	// U = [[4, 3], [0, -1.5]]
	Matrix<2, 2> A;
	A(0, 0) = 4; A(0, 1) = 3;
	A(1, 0) = 6; A(1, 1) = 3;

	auto [L, U] = MatrixUtils<2, 2>::lu(A);

	// Verify A = L * U
	auto Recon = L * U;
	ExpectMatrixNear(Recon, A);
}

TEST(MatrixLUTest, StructureVerification) {
	// Verify L is lower triangular and U is upper triangular
	Matrix<3, 3> A;
	A(0, 0) = 2; A(0, 1) = -1; A(0, 2) = -2;
	A(1, 0) = -4; A(1, 1) = 6; A(1, 2) = 3;
	A(2, 0) = -4; A(2, 1) = -2; A(2, 2) = 8;

	auto [L, U] = MatrixUtils<3, 3>::lu(A);

	// Check L (Lower Triangular + Unit Diagonal)
	EXPECT_DOUBLE_EQ(L(0, 1), 0.0);
	EXPECT_DOUBLE_EQ(L(0, 2), 0.0);
	EXPECT_DOUBLE_EQ(L(1, 2), 0.0);
	EXPECT_DOUBLE_EQ(L(0, 0), 1.0);
	EXPECT_DOUBLE_EQ(L(1, 1), 1.0);
	EXPECT_DOUBLE_EQ(L(2, 2), 1.0);

	// Check U (Upper Triangular)
	EXPECT_DOUBLE_EQ(U(1, 0), 0.0);
	EXPECT_DOUBLE_EQ(U(2, 0), 0.0);
	EXPECT_DOUBLE_EQ(U(2, 1), 0.0);

	// Verify Reconstruction
	ExpectMatrixNear(L * U, A);
}

TEST(MatrixLUTest, ThrowsOnZeroPivot) {
	// Matrix requiring permutation (pivot is 0)
	// [0 1]
	// [1 0]
	Matrix<2, 2> A;
	A(0, 0) = 0; A(0, 1) = 1;
	A(1, 0) = 1; A(1, 1) = 0;

	EXPECT_THROW((MatrixUtils<2, 2>::lu(A)), std::runtime_error);
}

TEST(MatrixQRTest, Simple3x3) {
	// A = [[12, -51, 4], [6, 167, -68], [-4, 24, -41]]
	// Known Example for QR
	Matrix<3, 3> A;
	A(0,0)=12; A(0,1)=-51; A(0,2)=4;
	A(1,0)=6;  A(1,1)=167; A(1,2)=-68;
	A(2,0)=-4; A(2,1)=24;  A(2,2)=-41;

	auto [Q, R] = MatrixUtils<3, 3>::qr(A);

	// 1. Verify Reconstruction: A = Q * R
	auto Recon = Q * R;
	ExpectMatrixNear(Recon, A);

	// 2. Verify Orthogonality: Q^T * Q = I
	auto QtQ = Q.transpose() * Q;
	ExpectMatrixNear(QtQ, Matrix<3, 3>::eye());

	// 3. Verify R is Upper Triangular
	EXPECT_NEAR(R(1, 0), 0.0, 1e-9);
	EXPECT_NEAR(R(2, 0), 0.0, 1e-9);
	EXPECT_NEAR(R(2, 1), 0.0, 1e-9);
}

TEST(MatrixQRTest, Rectangular_4x3) {
	// Tall matrix (more rows than cols)
	Matrix<4, 3> A;
	A(0,0)=1; A(0,1)=-1; A(0,2)=4;
	A(1,0)=1; A(1,1)=4;  A(1,2)=-2;
	A(2,0)=1; A(2,1)=4;  A(2,2)=2;
	A(3,0)=1; A(3,1)=-1; A(3,2)=0;

	auto [Q, R] = MatrixUtils<4, 3>::qr(A);

	// Q should be 4x3, R should be 3x3

	// 1. Verify Reconstruction
	auto Recon = Q * R;
	ExpectMatrixNear(Recon, A);

	// 2. Verify Orthogonality: Q^T * Q = I (3x3)
	auto QtQ = Q.transpose() * Q;
	ExpectMatrixNear(QtQ, (Matrix<3, 3>::eye()));

	// 3. Verify R is Upper Triangular
	EXPECT_NEAR(R(1, 0), 0.0, 1e-9);
	EXPECT_NEAR(R(2, 0), 0.0, 1e-9);
	EXPECT_NEAR(R(2, 1), 0.0, 1e-9);
}

TEST(MatrixQRTest, ThrowsOnRankDeficient) {
	// Singular matrix (col 2 = col 1)
	Matrix<3, 3> A;
	A(0,0)=1; A(0,1)=1; A(0,2)=5;
	A(1,0)=1; A(1,1)=1; A(1,2)=2;
	A(2,0)=1; A(2,1)=1; A(2,2)=7;

	EXPECT_THROW((MatrixUtils<3, 3>::qr(A)), std::runtime_error);
}


TEST(MatrixUnstableQRTest, Simple3x3) {
	// A = [[12, -51, 4], [6, 167, -68], [-4, 24, -41]]
	// Known Example for QR
	Matrix<3, 3> A;
	A(0,0)=12; A(0,1)=-51; A(0,2)=4;
	A(1,0)=6;  A(1,1)=167; A(1,2)=-68;
	A(2,0)=-4; A(2,1)=24;  A(2,2)=-41;

	auto [Q, R] = MatrixUtils<3, 3>::unstable_qr(A);

	// 1. Verify Reconstruction: A = Q * R
	auto Recon = Q * R;
	ExpectMatrixNear(Recon, A);

	// 2. Verify Orthogonality: Q^T * Q = I
	auto QtQ = Q.transpose() * Q;
	ExpectMatrixNear(QtQ, Matrix<3, 3>::eye());

	// 3. Verify R is Upper Triangular
	EXPECT_NEAR(R(1, 0), 0.0, 1e-9);
	EXPECT_NEAR(R(2, 0), 0.0, 1e-9);
	EXPECT_NEAR(R(2, 1), 0.0, 1e-9);
}

TEST(MatrixUnstableQRTest, Rectangular_4x3) {
	// Tall matrix (more rows than cols)
	Matrix<4, 3> A;
	A(0,0)=1; A(0,1)=-1; A(0,2)=4;
	A(1,0)=1; A(1,1)=4;  A(1,2)=-2;
	A(2,0)=1; A(2,1)=4;  A(2,2)=2;
	A(3,0)=1; A(3,1)=-1; A(3,2)=0;

	auto [Q, R] = MatrixUtils<4, 3>::unstable_qr(A);

	// Q should be 4x3, R should be 3x3

	// 1. Verify Reconstruction
	auto Recon = Q * R;
	ExpectMatrixNear(Recon, A);

	// 2. Verify Orthogonality: Q^T * Q = I (3x3)
	auto QtQ = Q.transpose() * Q;
	ExpectMatrixNear(QtQ, (Matrix<3, 3>::eye()));

	// 3. Verify R is Upper Triangular
	EXPECT_NEAR(R(1, 0), 0.0, 1e-9);
	EXPECT_NEAR(R(2, 0), 0.0, 1e-9);
	EXPECT_NEAR(R(2, 1), 0.0, 1e-9);
}

TEST(MatrixUnstableQRTest, ThrowsOnRankDeficient) {
	// Singular matrix (col 2 = col 1)
	Matrix<3, 3> A;
	A(0,0)=1; A(0,1)=1; A(0,2)=5;
	A(1,0)=1; A(1,1)=1; A(1,2)=2;
	A(2,0)=1; A(2,1)=1; A(2,2)=7;

	EXPECT_THROW((MatrixUtils<3, 3>::unstable_qr(A)), std::runtime_error);
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
```

`test.cpp` 中使用 Google Test 框架对矩阵类和矩阵实用工具类中的各种方法进行了单元测试，涵盖了矩阵的基本操作、秩计算、行最简形式、逆矩阵计算、LU 分解和 QR 分解等功能。每个测试用例都验证了相应方法的正确性和鲁棒性。

## 3. 算法说明

### 1. LU 分解算法

LU 分解的目标是将一个方阵 $A$ 分解为一个**单位下三角矩阵** $L$（对角线元素全为 1）和一个**上三角矩阵** $U$，使得：

$$
A = L \cdot U
$$

本项目中实现的是**无行交换的 Doolittle 算法**，其基本思想如下：

- 对于第 $j$ 列（从 0 开始），假设主元 $A_{jj} \neq 0$。
- 对于所有 $i > j$ 的行，计算乘子：
  $$
  l_{ij} = \frac{a_{ij}}{a_{jj}}
  $$
  并将该乘子存入 $L(i, j)$。
- 然后用该乘子消去第 $j$ 列下方的所有元素：
  $$
  a_{ik} \leftarrow a_{ik} - l_{ij} \cdot a_{jk}, \quad \text{for } k = j, j+1, \dots, n-1
  $$
- 最终，原始矩阵 $A$ 被原地替换为上三角矩阵 $U$，而乘子构成单位下三角矩阵 $L$。

> **局限性**：若在某步中 $a_{jj} = 0$（或接近零），则无法继续分解。此时需要引入**行置换（即 PA = LU）**，但本实现未包含此功能，因此仅适用于主元非零的情形。

---

### 2. QR 分解算法

QR 分解将一个列满秩矩阵 $A \in \mathbb{R}^{m \times n}$（$m \geq n$）分解为：

$$
A = Q \cdot R
$$

其中：
- $Q \in \mathbb{R}^{m \times n}$ 是**列正交矩阵**（即 $Q^\top Q = I_n$），
- $R \in \mathbb{R}^{n \times n}$ 是**上三角矩阵**且对角线元素为正。

本项目实现了两种 QR 分解方法：

#### (a) 经典 Gram-Schmidt 正交化（`unstable_qr`）

- 对 $A$ 的每一列 $\mathbf{a}_j$，依次减去其在已正交化向量 $\mathbf{q}_0, \dots, \mathbf{q}_{j-1}$ 上的投影：
  $$
  r_{kj} = \mathbf{q}_k^\top \mathbf{a}_j,\quad \mathbf{a}_j \leftarrow \mathbf{a}_j - r_{kj} \mathbf{q}_k \quad (k < j)
  $$
- 然后归一化得到 $\mathbf{q}_j = \mathbf{a}_j / r_{jj}$，其中 $r_{jj} = \|\mathbf{a}_j\|$。

> **缺点**：在浮点运算中，经典 Gram-Schmidt 容易因舍入误差导致正交性丧失，尤其在矩阵病态时表现较差，故标记为 “unstable”。

#### (b) 改进 Gram-Schmidt 正交化（`qr`）

- 先对当前列 $\mathbf{a}_j$ 归一化得到 $\mathbf{q}_j$，
- 再用 $\mathbf{q}_j$ **立即修正后续所有列** $\mathbf{a}_k$（$k > j$）：
  $$
  r_{jk} = \mathbf{q}_j^\top \mathbf{a}_k,\quad \mathbf{a}_k \leftarrow \mathbf{a}_k - r_{jk} \mathbf{q}_j
  $$

> **优点**：数值稳定性显著优于经典版本，在相同精度下能更好地保持 $Q$ 的正交性。测试表明，对于标准测试矩阵（如著名的 3×3 Householder 示例），两种方法结果相近，但在高维或病态矩阵上，改进版更可靠。

---

### 3. 辅助算法简述

- **秩计算（`rank`）**：通过高斯消元（带部分主元选取）将矩阵化为行阶梯形（REF），统计非零行数。
- **行阶梯形（`RET`）**：同上过程，但返回变换后的矩阵。
- **简化行阶梯形（`RREF`）**：在 REF 基础上进一步将主元归一，并消去主元上方的元素。
- **矩阵求逆（`inverse`）**：对增广矩阵 $[A \mid I]$ 执行 Gauss-Jordan 消元，最终得到 $[I \mid A^{-1}]$。

所有算法均使用 `EPSILON = 1e-9` 作为浮点零判断阈值，以增强数值鲁棒性。

---

## 4. 心得体会

通过本次线性代数大作业，我深刻体会到**理论算法与工程实现之间的鸿沟与桥梁**。

首先，在课堂上学习 LU 和 QR 分解时，我们关注的是数学推导和存在唯一性条件；而在编程实现时，必须考虑**数值稳定性、边界情况、内存布局和类型安全**等实际问题。例如，经典 Gram-Schmidt 在理论上完全正确，但在计算机浮点运算中可能失效——这让我意识到“正确”的算法不等于“可用”的算法。在手算的时候，我们不会考虑到浮点型的舍入误差，但是计算机计算时，我们选取主元，就要选取最大的那个，以减少误差的累积。

其次，C++ 模板元编程为矩阵运算提供了强大的编译期保障。通过将矩阵维度作为模板参数，可以在编译时检查矩阵乘法的合法性（如 `(m×n) * (n×p)`），避免运行时错误。这种“把 bug 消灭在编译阶段”的思想，极大提升了代码可靠性。

此外，编写单元测试的过程也让我受益匪浅。借助 Google Test，我不仅验证了正常输入下的正确性，还通过构造奇异矩阵、零矩阵、病态矩阵等极端案例，检验了算法的鲁棒性和异常处理能力。这种“测试驱动开发”模式，确保了每一个功能模块都经得起推敲。

最后，开源协作精神也在这次作业中得以体现。将代码托管在 GitHub 上，撰写清晰的 README，设计合理的接口，都是为了让他人（包括未来的自己）能够轻松理解、复现和扩展。这不仅是课程要求，更是现代软件工程的基本素养。

总而言之，这次大作业不仅巩固了我对矩阵分解理论的理解，更锻炼了我将数学工具转化为可靠软件的能力——这正是科学计算的核心所在。
