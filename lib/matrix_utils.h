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
			for (size_t j = 0; j < Cols; ++j) {
				size_t pivot_row = j;
				for (size_t i = j + 1; i < Rows; ++i) {
					if (std::abs(mat(i, j)) > std::abs(mat(pivot_row, j))) {
						pivot_row = i;
					}
				}
				if (std::abs(mat(pivot_row, j)) < EPSILON) {
					continue;
				}
				if (pivot_row != j) {
					for (size_t k = j; k < Cols; ++k) {
						std::swap(mat(j, k), mat(pivot_row, k));
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
	};
}

#endif //MATRIX_MATRIX_UTILS_H