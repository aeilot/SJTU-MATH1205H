#include <iostream>
#include <gtest/gtest.h>
#include "lib/matrix.h"

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

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}