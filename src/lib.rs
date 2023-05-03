use std::str::FromStr;

use numpy::{IntoPyArray, PyReadonlyArray2};
use numpy::{PyArray1,PyReadonlyArray1};
use pyo3::prelude::*;

mod tc;

enum BoundStrat {
    Clip,
    Wrap,
}

impl FromStr for BoundStrat {
    type Err = ();

    fn from_str(input: &str) -> Result<BoundStrat, Self::Err> {
        match input {
            "clip" => Ok(BoundStrat::Clip),
            "wrap" => Ok(BoundStrat::Wrap),
            _ => Err(()),
        }
    }
}

fn bound(x: ) {

}

/// A Python module implemented in Rust.
#[pymodule]
fn tile_coder(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name="get_tc_indices")]
    fn test_py<'py>(
        py: Python<'py>,
        dims: u32,
        tiles: u32,
        tilings: u32,
        offsets: PyReadonlyArray2<f64>,
        pos: PyReadonlyArray1<f64>,
    ) -> &'py PyArray1<u32> {
        let res = tc::get_tc_indices(dims, tiles, tilings, offsets, pos);
        res.into_pyarray(py)
    }

    Ok(())
}
