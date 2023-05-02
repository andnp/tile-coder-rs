use numpy::{array, IntoPyArray};
use numpy::ndarray::Array1;
use numpy::{PyArray1,PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod utils {
    pub fn clip(a: f64, lo: f64, hi: f64) -> f64 {
        let t = f64::max(a, lo);
        f64::min(t, hi)
    }

    pub fn wrap(a: f64, hi: f64) -> f64 {
        a % hi
    }
}

fn test(arr: PyReadonlyArray1<f64>, x: f64) -> Array1<f64> {
    let x = array![x];
    let arr = arr.as_array();
    &arr + x
}

mod tc {
    use numpy::{PyReadonlyArray1, Ix1, ndarray::Array1};

    fn get_axis_cell(x: f64, tiles: usize) -> usize {
        let t = tiles as f64;
        let i = f64::floor(x * t) as usize;
        i.clamp(0, tiles - 1)
    }

    fn get_tiling_index(dims: usize, tiles_per_dim: usize, pos: Array1<f64>) -> usize {
        let mut ind = 0;

        let total_tiles = tiles_per_dim.pow(dims as u32);
        for d in 0..dims {
            let x = *pos.get(Ix1(d)).expect("Index out-of-bounds for numpy array");
            let axis = get_axis_cell(x, tiles_per_dim);
            let already_seen = tiles_per_dim.pow(d as u32);
            ind += axis * already_seen;
        }

        ind.clamp(0, total_tiles - 1)
    }

    pub fn get_tc_indices(dims: usize, tiles: usize, tilings: usize, offsets: PyReadonlyArray1<f64>, pos: PyReadonlyArray1<f64>) -> Array1<usize> {
        let total_tiles = tiles.pow(dims as u32);
        let mut index = Array1::zeros(tilings);

        let pos = pos.as_array();

        for ntl in 0..tilings {
            let off = *offsets.get(Ix1(ntl)).expect("Index out-of-bounds for offsets");
            let arr = &pos + off;
            let ind = get_tiling_index(dims, tiles, arr);
            index[ntl] = ind + total_tiles * ntl;
        }

        index
    }
}



/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn tile_coder(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    #[pyfn(m)]
    #[pyo3(name="get_tc_indices")]
    fn test_py<'py>(py: Python<'py>, dims: usize, tiles: usize, tilings: usize, offsets: PyReadonlyArray1<f64>, pos: PyReadonlyArray1<f64>) -> &'py PyArray1<usize> {
        let res = tc::get_tc_indices(dims, tiles, tilings, offsets, pos);
        res.into_pyarray(py)
    }

    Ok(())
}
