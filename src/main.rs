#[macro_use]
extern crate rust_ocl_linalg;
use rust_ocl_linalg::matrix::Matrix;
use rust_ocl_linalg::vector::Vector;
//use rust_ocl_linalg::gpu;
fn main() {
    let v1 = v![-100, 2, 5];
    let v2 = v![0, 23, -34];
    println!("{}", v1.cross(&v2));
}
