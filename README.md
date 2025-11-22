# Matrix

A header-only C++20 linear algebra library providing compile-time matrix operations with comprehensive testing.

## Features

- **Template-based matrices** with compile-time dimensions
- **Basic operations**: addition, subtraction, scalar multiplication/division
- **Matrix operations**: multiplication, transpose, trace
- **Advanced operations**:
  - Matrix rank calculation
  - Row Echelon Form (REF)
  - Reduced Row Echelon Form (RREF)
  - Matrix inversion (Gauss-Jordan elimination)
  - LU decomposition
  - QR decomposition (Gram-Schmidt)
- **Factory methods**: identity, diagonal, and zero matrices
- **Type-safe** with compile-time dimension checking
- **Comprehensive test suite** using Google Test

## Requirements

- C++20 or higher
- CMake 4.0 or higher
- Google Test (automatically fetched by CMake)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Matrix
```

2. Build the project:
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

3. Run tests:
```bash
./Matrix
```

## Usage

### Basic Matrix Operations

```cpp
#include "lib/matrix.h"
#include "lib/matrix_utils.h"

using namespace LinearAlgebra;

// Create a 3x3 matrix
Matrix<3, 3> mat;
mat.set(0, 0, 1.0);
mat.set(1, 1, 2.0);
mat.set(2, 2, 3.0);

// Create identity matrix
auto identity = Matrix<3, 3>::eye();

// Create diagonal matrix
std::vector<double> values = {1, 2, 3};
auto diag = Matrix<3, 3>::diag(values);

// Create zero matrix
auto zeros = Matrix<3, 3>::zeros();
```

### Matrix Arithmetic

```cpp
Matrix<2, 2> mat1, mat2;
// ... initialize matrices ...

// Addition
auto sum = mat1 + mat2;

// Subtraction
auto diff = mat1 - mat2;

// Scalar multiplication
auto scaled = mat1 * 2.0;

// Matrix multiplication
Matrix<2, 3> A;
Matrix<3, 2> B;
// ... initialize ...
auto product = A * B;  // Result is 2x2

// Transpose
auto transposed = mat1.transpose();

// Trace
double tr = mat1.trace();
```

### Advanced Operations

```cpp
// Calculate rank
size_t r = MatrixUtils<3, 3>::rank(mat);

// Row Echelon Form
auto ref = MatrixUtils<3, 3>::RET(mat);

// Reduced Row Echelon Form
auto rref = MatrixUtils<3, 3>::RREF(mat);

// Matrix inversion
auto inv = MatrixUtils<3, 3>::inverse(mat);

// LU decomposition
auto [L, U] = MatrixUtils<3, 3>::lu(mat);

// QR decomposition (Modified Gram-Schmidt)
auto [Q, R] = MatrixUtils<3, 3>::qr(mat);

// QR decomposition (Classical Gram-Schmidt, less stable)
auto [Q2, R2] = MatrixUtils<3, 3>::unstable_qr(mat);
```

## Project Structure

```
Matrix/
├── CMakeLists.txt       # Build configuration
├── LICENSE              # MIT License
├── lib/
│   ├── matrix.h         # Core Matrix class
│   └── matrix_utils.h   # Matrix utilities and algorithms
└── test.cpp             # Comprehensive test suite
```

## Testing

The project includes extensive unit tests covering:
- Basic matrix operations
- Matrix multiplication and transposition
- Rank calculation for various matrix types
- Row Echelon Form transformations
- Matrix inversion edge cases
- LU decomposition correctness
- QR decomposition correctness
- RREF computation

Run all tests:
```bash
cd build
./Matrix
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Louis Deng (Chenluo Deng)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
