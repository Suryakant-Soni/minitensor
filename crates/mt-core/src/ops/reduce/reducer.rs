
pub(crate) trait Reducer {
    fn identity() -> f32;
    fn combine(acc: f32, elem: f32) -> f32;
}

pub struct SumReducer;

impl Reducer for SumReducer
{
    fn identity() -> f32 {
        return f32::default();
}
    fn combine(acc: f32, element: f32) -> f32 {
        acc + element
    }
}

pub struct MaxReducer;

impl Reducer for MaxReducer
{
    fn identity() -> f32 {
        return f32::default();
    }
    fn combine(acc: f32, element: f32) -> f32 {
        if acc > element { acc } else { element }
    }
}
