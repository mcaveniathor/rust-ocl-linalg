use std::fmt::Debug;
use std::fmt::Display;
use std::ops;
use num::traits::Signed;

/// a set of simple traits for what values can be used inside matrices. From<bool> is used as it applies to all number types and is needed to cast 0
pub trait Scalar: PartialOrd + Sized + Display + Debug + Copy + PartialEq + From<i8> + ops::Sub<Output = Self> + ops::Add<Output=Self> + ops::Mul<Output = Self> + Signed{}
impl<T: Sized + Debug + Display + Copy + PartialOrd + PartialEq + From<i8> + ops::Mul<Output = Self> + ops::Sub<Output = Self> + ops::Add<Output=Self> + Signed> Scalar for T {}

