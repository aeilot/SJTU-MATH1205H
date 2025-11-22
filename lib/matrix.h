//
// Created by Chenluo Deng on 11/22/25.
//

#ifndef MATRIX_MATRIX_H
#define MATRIX_MATRIX_H
#include <iostream>
#include <vector>

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
		Matrix(std::array<T, Cols> arr) : arr(arr) {}
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
					std::cout << arr[i * Cols + j];
				}
				std::cout << std::endl;
			}
		}
	};
}

#endif //MATRIX_MATRIX_H