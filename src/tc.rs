use numpy::{Ix1, ndarray::{Array1, s, ArrayView2, ArrayView1}};

fn get_axis_cell(x: f64, tiles: u32) -> u32 {
    let t = tiles as f64;
    let i = f64::floor(x * t) as u32;
    i.clamp(0, tiles - 1)
}

fn get_tiling_index(dims: u32, tiles_per_tiling: u32, tiles_per_dim: ArrayView1<u32>, pos: Array1<f64>) -> u32 {
    let mut ind = 0;
    let mut already_seen: u32 = 1;

    for d in 0..dims {
        let idx = Ix1(d as usize);
        let x = *pos.get(idx).expect("Index out-of-bounds for numpy array");
        let tiles = *tiles_per_dim.get(idx).expect("Index out-of-bounds");

        let axis = get_axis_cell(x, tiles);
        ind += axis * already_seen;
        already_seen *= tiles;
    }

    ind.clamp(0, tiles_per_tiling - 1)
}

pub fn get_tc_indices(dims: u32, tiles: ArrayView1<u32>, tilings: u32, offsets: ArrayView2<f64>, pos: Array1<f64>) -> Array1<u32> {
    let tiles_per_tiling = tiles.product();
    let mut index = Array1::zeros(tilings as usize);

    for ntl in 0..tilings {
        let off = offsets.slice(s![ntl as usize, ..]);
        let arr = &pos + &off;
        let ind = get_tiling_index(dims, tiles_per_tiling, tiles, arr);
        index[ntl as usize] = ind + tiles_per_tiling * ntl;
    }

    index
}

// ----------------
// -- Unit tests --
// ----------------
#[cfg(test)]
mod tests {
    use numpy::ndarray::Array1;

    #[test]
    fn get_axis_cell() {
        // check endpoints
        let res = super::get_axis_cell(0.0, 8);
        assert_eq!(res, 0);

        let res = super::get_axis_cell(1.0, 8);
        assert_eq!(res, 7);

        // check out-of-bounds
        let res = super::get_axis_cell(-0.01, 8);
        assert_eq!(res, 0);

        let res = super::get_axis_cell(1.03, 8);
        assert_eq!(res, 7);

        // check middle
        let res = super::get_axis_cell(0.124, 8);
        assert_eq!(res, 0);
        let res = super::get_axis_cell(0.126, 8);
        assert_eq!(res, 1);
        let res = super::get_axis_cell(0.249, 8);
        assert_eq!(res, 1);
        let res = super::get_axis_cell(0.26, 8);
        assert_eq!(res, 2);
    }

    #[test]
    fn get_tiling_index() {
        let arr = Array1::from_iter([0.1]);
        let tiles: Array1<u32> = Array1::from_iter([8]);
        let res = super::get_tiling_index(1, 8, tiles.view(), arr);
        assert_eq!(res, 0);

        let arr = Array1::from_iter([0.1, 0.1]);
        let tiles: Array1<u32> = Array1::from_iter([8, 8]);
        let res = super::get_tiling_index(2, 64, tiles.view(), arr);
        assert_eq!(res, 0);

        let arr = Array1::from_iter([0.126, 0.1]);
        let tiles: Array1<u32> = Array1::from_iter([8, 8]);
        let res = super::get_tiling_index(2, 64, tiles.view(), arr);
        assert_eq!(res, 1);

        let arr = Array1::from_iter([0.126, 0.126]);
        let tiles: Array1<u32> = Array1::from_iter([8, 8]);
        let res = super::get_tiling_index(2, 64, tiles.view(), arr);
        assert_eq!(res, 9);

        let arr = Array1::from_iter([1.0, 1.0]);
        let tiles: Array1<u32> = Array1::from_iter([8, 8]);
        let res = super::get_tiling_index(2, 64, tiles.view(), arr);
        assert_eq!(res, 63);

        let arr = Array1::from_iter([0.21, 1.0]);
        let tiles: Array1<u32> = Array1::from_iter([5, 8]);
        let res = super::get_tiling_index(2, 40, tiles.view(), arr);
        assert_eq!(res, 36);
    }
}
