#[macro_export]
/// A macro to create a matrix with its dimensions as a tuple and values in a flat, comma separated list bounded by brackets
/// ex. mat!((3,2), [0,1,2,3,4,5]);
macro_rules! mat {
        ( ( $m:expr, $n:expr ), [ $ ( $x:expr ) , * ] ) => {
            {
                let mut v = Vec::new();
                let mut count = 0;
                $(
                    v.push($x);
                    count += 1;
                )*
                assert_eq!($m * $n, count);
                Matrix::new_from_flat_vec($m, $n, v)
            };
        }
}

/// A macro to create an mx1 matrix (ie a column vector) without specifying its dimensions
/// ex. col_vec!([0,1,2,3]);
#[macro_export]
macro_rules! col_vec {
        ( [ $ ( $x:expr ) , * ] ) => {
            {
                let mut v = Vec::new();
                $ (
                    v.push($x);
                ) *
                Matrix::new_from_flat_vec(v.len(), 1, v)
            };
        }
}

#[macro_export]
/// A macro to create a row vector (as a matrix) without specifyin its dimensions
macro_rules! row_vec {
        ( [$ ($x: expr ), * ] ) => {
            {
                let mut v = Vec::new();
                $ (
                    v.push($x);
                ) *
                Matrix::new_from_flat_vec(1,v.len(),v)
            }
        };
}

/// A macro which creates a vector object (as opposed to an mx1 or 1xn matrix) with the given values.
#[macro_export]
macro_rules! v {
        [ $ ( $x: expr ), *] => {
            {
                let mut tmp = Vec::new();
                $ (
                    tmp.push($x);
                ) *
                Vector::new(tmp)
            }
        };
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;
    use crate::vector::Vector;

    #[test]
    fn test_mat_macro() {
        let m = mat!((3, 2), [0, 1, 2, 3, 4, 5]);
        assert_eq!(Matrix::new_from_flat_vec(3, 2, vec![0, 1, 2, 3, 4, 5]), m);
    }

    #[test]
    fn test_matrix_col_vec_macro() {
        let m = col_vec!([0, 2, 4]);
        assert_eq!(Matrix::new_from_flat_vec(3, 1, vec![0, 2, 4]), m);
    }

    #[test]
    fn test_matrix_row_vec_macro() {
        let m = row_vec!([0, 1]);
        assert_eq!(Matrix::new_from_flat_vec(1, 2, vec![0, 1]), m);
    }

    #[test]
    fn test_v_macro() {
        let c = v![3, 4, 5];
        assert_eq!(c, Vector::new(vec![3, 4, 5]));
    }

}
