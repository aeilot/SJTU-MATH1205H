#include <iostream>
#include <gtest/gtest.h>
#include "lib/matrix.h"
#include "lib/matrix_utils.h"

using namespace LinearAlgebra;

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

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}