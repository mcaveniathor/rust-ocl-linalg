extern crate rust_ocl_linalg;
use rust_ocl_linalg::matrix::Matrix;
use rust_ocl_linalg::gpu;
fn main() {
    let m: Matrix<i32> = Matrix::new_from_flat_vec(3,3, vec![0,1,2,3,4,5,6,7,8]);
    println!("{}", m);
    let (platform, device, context_properties, context) = gpu::boilerplate().unwrap();
}