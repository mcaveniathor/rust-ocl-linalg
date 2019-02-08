# rust-ocl-linalg [![Build Status](https://travis-ci.org/mcaveniathor/rust-ocl-linalg.svg?branch=master)](https://travis-ci.org/mcaveniathor/rust-ocl-linalg)
To test: `cargo test --verbose`
## Features
* Scalar trait for guaranteed safe operation with built-in or custom datatypes
* Memory-safe, generic Matrix<T: Scalar> datatype
..* Create from 1-d or 2-d Vec<T: Scalar>, a submatrix of a Matrix struct, or one of several parameterized functions
..* Powerful methods and utilities which prove useful in linear algebra and vector calculus, including determinant calculator and row-echelon simplifier
