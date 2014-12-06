extern crate rgsl;

use std::os;
use std::io::BufferedReader;
use std::io::File;
use std::num::Float;
use std::vec;
use std::collections::HashMap;
use std::uint;
use std::rc::Rc;
use rgsl::types::vector::VectorF64;
use rgsl::types::matrix::MatrixF64;
use rgsl::linear_algebra::SV_decomp;
use rgsl::blas::level2;
use rgsl::blas::level3;
use rgsl::cblas::Transpose;

fn build_vector(data: &vec::Vec<f64>) -> VectorF64 {
    let mut result = VectorF64::from_slice(data.as_slice()).unwrap();
    result.add_constant(-result.min());
    result.scale(1.0f64 / result.max());
    return result
}

fn main() {
    let args = os::args();
    assert_eq!(args.len(), 2);
    let path = Path::new(args[1].clone());
    let mut file = BufferedReader::new(File::open(&path));

    let dim = 3;

    let mut x1_vec = vec::Vec::new();
    let mut x2_vec = vec::Vec::new();
    let mut y_vec = vec::Vec::new();
    for line in file.lines() {
        let line_str = line.unwrap();
        let elems = line_str.split_str(",").collect::<Vec<&str>>();
        assert_eq!(elems.len(), 3);

        let x1_elem = from_str::<f64>(elems[0]).unwrap();
        let x2_elem = from_str::<f64>(elems[1]).unwrap();
        let y_elem = from_str::<f64>(elems[2].trim()).unwrap();
        x1_vec.push(x1_elem);
        x2_vec.push(x2_elem);
        y_vec.push(y_elem);
    }

    let n = x1_vec.len() as u64;

    let mut x0 = VectorF64::new(n).unwrap();
    x0.set_all(1.0f64);
    let x1 = build_vector(&x1_vec);
    let x2 = build_vector(&x2_vec);
    let y = VectorF64::from_slice(y_vec.as_slice()).unwrap();
    let mut f = MatrixF64::new(n, dim).unwrap();
    f.set_col(0, &x0);
    f.set_col(1, &x1);
    f.set_col(2, &x2);
    let f_orig = f.clone().unwrap();
    let mut u = MatrixF64::new(dim, dim).unwrap();
    let mut d_vec = VectorF64::new(dim).unwrap();
    let mut work = VectorF64::new(dim).unwrap();
    let res = SV_decomp(&f, &u, &d_vec, &work);
    let mut d = MatrixF64::new(dim, dim).unwrap();
    for i in range(0, dim) {
        d.set(i, i, 1.0f64 / d_vec.get(i));
    }
    let mut ud = MatrixF64::new(dim, dim).unwrap();
    level3::dgemm(Transpose::NoTrans, Transpose::NoTrans, 1.0f64, &u, &d, 1.0f64, &mut ud);
    let mut fplus = MatrixF64::new(dim, n).unwrap();
    level3::dgemm(Transpose::NoTrans, Transpose::Trans, 1.0f64, &ud, &f, 1.0f64, &mut fplus);
    let mut alpha = VectorF64::new(dim).unwrap();
    level2::dgemv(Transpose::NoTrans, 1.0f64, &fplus, &y, 1.0f64, &mut alpha);
    let mut y_predict = VectorF64::new(n).unwrap();
    level2::dgemv(Transpose::NoTrans, 1.0f64, &f_orig, &alpha, 1.0f64, &mut y_predict);
    println!("y_predict = \n{}", y_predict);
    let mut y_diff = y.clone().unwrap();
    y_diff.sub(&y_predict);
    for i in range(0, n) {
        println!("{} {} -> {} {}%", y.get(i), y_predict.get(i), y_diff.get(i), 100.0f64 * y_diff.get(i) / y.get(i));
    }
}
