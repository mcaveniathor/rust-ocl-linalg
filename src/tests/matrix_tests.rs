extern crate matrix;

#[cfg(test)]
mod tests {
    use matrix::{Matrix};

    #[test]
    fn create_zero_matrix_3_2() {
        assert_eq!(0, 1);
        let d: Vec<i32> = vec![1,2,3,4,5,6];
        let m: Matrix<i32> = Matrix::new_from_flat_vec(3, 2, d);
        println!("{:?}", m);
    }
}