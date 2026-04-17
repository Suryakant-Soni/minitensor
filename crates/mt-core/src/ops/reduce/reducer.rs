use std::cmp::PartialOrd;
use std::ops::Add;

pub(crate) trait Reducer<T> {
    fn identity() -> T;
    fn combine(acc: T, elem: T) -> T;
}

struct SumReducer;

impl<T> Reducer<T> for SumReducer
where
    T: Add<Output = T> + Copy + Default,
{
    fn identity() -> T {
        return T::default();
}
    fn combine(acc: T, element: T) -> T {
        acc + element
    }
}

struct MaxReducer;

impl<T> Reducer<T> for MaxReducer
where
    T: Add<Output = T> + Copy + PartialOrd + Default,
{
    fn identity() -> T {
        return T::default();
    }
    fn combine(acc: T, element: T) -> T {
        if acc > element { acc } else { element }
    }
}
