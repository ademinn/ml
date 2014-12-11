extern crate rgsl;
extern crate common;

use std::os;
use std::rand::{task_rng, Rng};
use std::io::BufferedReader;
use std::io::File;
use std::num::Float;
use std::vec;
use rgsl::types::vector::VectorF64;
use rgsl::types::matrix::MatrixF64;
use rgsl::blas::level2;
use rgsl::cblas::Transpose;

use common::gradient_descent;

fn build_matrix(data: &[(f64, f64, i32)]) -> (MatrixF64, VectorF64) {
    let n = data.len() as u64;

    let x0 = VectorF64::new(n).unwrap();
    x0.set_all(1.0f64);
    let x1 = VectorF64::from_slice(data.iter().map(|&(x1_e, _, _)| x1_e).collect::<vec::Vec<f64>>().as_slice()).unwrap();
    let x2 = VectorF64::from_slice(data.iter().map(|&(_, x2_e, _)| x2_e).collect::<vec::Vec<f64>>().as_slice()).unwrap();
    let x3 = x1.clone().unwrap();
    x3.mul(&x1);
    let x4 = x1.clone().unwrap();
    x4.mul(&x2);
    let x5 = x2.clone().unwrap();
    x5.mul(&x2);
    let y = VectorF64::from_slice(data.iter().map(|&(_, _, y_e)| if y_e == 1i32 { 1.0f64 } else { -1.0f64 }).collect::<vec::Vec<f64>>().as_slice()).unwrap();
    let f = MatrixF64::new(n, 6).unwrap();
    f.set_col(0, &x0);
    f.set_col(1, &x1);
    f.set_col(2, &x2);
    f.set_col(3, &x3);
    f.set_col(4, &x4);
    f.set_col(5, &x5);
    return (f, y)
}

fn main() {
    let args = os::args();
    assert_eq!(args.len(), 2);
    let path = Path::new(args[1].clone());
    let mut file = BufferedReader::new(File::open(&path));

    let dim = 6;
    let lambda = 0.001f64;
    let eps = 0.0001f64;

    let mut data_vec = vec::Vec::new();
    for line in file.lines() {
        let line_str = line.unwrap();
        let elems = line_str.split_str(",").collect::<Vec<&str>>();
        assert_eq!(elems.len(), 3);

        let x1_elem = from_str::<f64>(elems[0]).unwrap();
        let x2_elem = from_str::<f64>(elems[1]).unwrap();
        let y_elem = from_str::<i32>(elems[2].trim()).unwrap();
        data_vec.push((x1_elem, x2_elem, y_elem));
    }

    let mut data_slice = data_vec.as_mut_slice();

    let mut rng = task_rng();
    rng.shuffle(data_slice);

    let teach_len = (data_slice.len() * 4 / 5) as u64;
    let validate_len = (data_slice.len() as u64) - teach_len;
    
    let (teach, validate) = data_slice.split_at(teach_len as uint);
    let (f, y) = build_matrix(teach);

    let (f_validate, y_validate) = build_matrix(validate);

    let func = |v: &VectorF64| -> f64 {
        let mut res_v = VectorF64::new(teach_len).unwrap();
        level2::dgemv(Transpose::NoTrans, 1.0f64, &f, v, 1.0f64, &mut res_v);
        res_v.mul(&y);
        res_v.scale(-1.0f64);
        let mut res = 0.0f64;
        for i in range(0, res_v.len()) {
            res += (1.0f64 + res_v.get(i).exp()).ln();
        }
        return res
    };
    let grad = |v: &VectorF64| -> VectorF64 {
        let mut res_v = VectorF64::new(teach_len).unwrap();
        level2::dgemv(Transpose::NoTrans, 1.0f64, &f, v, 1.0f64, &mut res_v);
        res_v.mul(&y);
        res_v.scale(-1.0f64);
        let mut res = VectorF64::new(dim).unwrap();
        for i in range(0, res_v.len()) {
            let res_exp = res_v.get(i).exp();
            let scale = -1.0f64 * y.get(i) * res_exp / (1.0f64 + res_exp);
            for j in range(0, dim) {
                let res_old = res.get(j);
                res.set(j, res_old + scale * f.get(i, j));
            }
        }
        return res
    };

    let alpha = gradient_descent(&VectorF64::new(dim).unwrap(), lambda, func, grad, eps);
    println!("alpha = {}\n", alpha);
    let mut y_predict = VectorF64::new(validate_len).unwrap();
    level2::dgemv(Transpose::NoTrans, 1.0f64, &f_validate, &alpha, 1.0f64, &mut y_predict);
    let y_diff = y_validate.clone().unwrap();
    y_diff.mul(&y_predict);
    let mut success_count = 0u;
    for i in range(0, validate_len) {
        if y_diff.get(i) > 0.0f64 {
            success_count += 1u;
        }
    }
    println!("ok: {}, total: {}", success_count, validate_len);
}
