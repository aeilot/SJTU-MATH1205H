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