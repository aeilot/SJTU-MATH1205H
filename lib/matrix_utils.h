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
			}
			return mat;
		}

	};
}

#endif //MATRIX_MATRIX_UTILS_H