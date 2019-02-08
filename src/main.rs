extern crate rust_ocl_linalg;
use rust_ocl_linalg::matrix::Matrix;
//use rust_ocl_linalg::gpu;
fn main() {
    let mut m: Matrix<f32> = Matrix::new_from_flat_vec(3, 3, vec![2.0, 4.0, 1.0, 6.0, 27.0, 0.0, 9.0, 12.0, 13.0]);
    m.row_echelon();
    println!("{}", m);
}