extern crate rgsl;
extern crate common;

use std::os;
use std::io::BufferedReader;
use std::io::File;
use std::num::Float;
use std::vec;
use rgsl::types::vector::VectorF64;
use rgsl::types::matrix::MatrixF64;
use rgsl::blas::level2;
use rgsl::cblas::Transpose;

use common::gradient_descent;

fn main() {
    let args = os::args();
    assert_eq!(args.len(), 2);
    let path = Path::new(args[1].clone());
    let mut file = BufferedReader::new(File::open(&path));

    let dim = 6;
    let lambda = 0.001f64;
    let eps = 0.0001f64;

    let mut x1_vec = vec::Vec::new();
    let mut x2_vec = vec::Vec::new();
    let mut y_vec: vec::Vec<i32> = vec::Vec::new();
    for line in file.lines() {
        let line_str = line.unwrap();
        let elems = line_str.split_str(",").collect::<Vec<&str>>();
        assert_eq!(elems.len(), 3);

        let x1_elem = from_str::<f64>(elems[0]).unwrap();
        let x2_elem = from_str::<f64>(elems[1]).unwrap();
        let y_elem = from_str::<i32>(elems[2].trim()).unwrap();
        x1_vec.push(x1_elem);
        x2_vec.push(x2_elem);
        y_vec.push(y_elem);
    }

    let n = x1_vec.len() as u64;

    let x0 = VectorF64::new(n).unwrap();
    x0.set_all(1.0f64);
    let x1 = VectorF64::from_slice(x1_vec.as_slice()).unwrap();
    let x2 = VectorF64::from_slice(x2_vec.as_slice()).unwrap();
    let x3 = x1.clone().unwrap();
    x3.mul(&x1);
    let x4 = x1.clone().unwrap();
    x4.mul(&x2);
    let x5 = x2.clone().unwrap();
    x5.mul(&x2);
    let y = VectorF64::from_slice(y_vec.iter().map(|&x| if x == 1i32 { 1.0f64 } else { -1.0f64 }).collect::<vec::Vec<f64>>().as_slice()).unwrap();
    let f = MatrixF64::new(n, dim).unwrap();
    f.set_col(0, &x0);
    f.set_col(1, &x1);
    f.set_col(2, &x2);
    f.set_col(3, &x3);
    f.set_col(4, &x4);
    f.set_col(5, &x5);
    let f_orig = f.clone().unwrap();
    let func = |v: &VectorF64| -> f64 {
        let mut res_v = VectorF64::new(n).unwrap();
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
        let mut res_v = VectorF64::new(n).unwrap();
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
    let mut y_predict = VectorF64::new(n).unwrap();
    level2::dgemv(Transpose::NoTrans, 1.0f64, &f_orig, &alpha, 1.0f64, &mut y_predict);
    let y_diff = y.clone().unwrap();
    y_diff.sub(&y_predict);
    for i in range(0, n) {
        println!("{} {} -> {} {}%", y.get(i), y_predict.get(i), y_diff.get(i), 100.0f64 * y_diff.get(i) / y.get(i));
    }
}
