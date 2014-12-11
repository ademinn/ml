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
use rgsl::blas::level1;
use rgsl::blas::level2;
use rgsl::cblas::Transpose;

use common::gradient_descent;

fn build_vector(data: &vec::Vec<f64>) -> VectorF64 {
    let result = VectorF64::from_slice(data.as_slice()).unwrap();
    result.add_constant(-result.min());
    result.scale(1.0f64 / result.max());
    return result
}

fn sum_vec(v: &VectorF64) -> f64 {
    let mut result = 0.0f64;
    for i in range(0, v.len()) {
        result += v.get(i);
    }
    return result
}

fn build_matrix(data: &[(f64, f64, f64)]) -> (MatrixF64, VectorF64) {
    let x0 = VectorF64::new(data.len() as u64).unwrap();
    x0.set_all(1.0f64);
    let x1 = build_vector(&data.iter().map(|&(x1_e, _, _)| x1_e).collect::<vec::Vec<f64>>());
    let x2 = build_vector(&data.iter().map(|&(_, x2_e, _)| x2_e).collect::<vec::Vec<f64>>());
    let y = VectorF64::from_slice(data.iter().map(|&(_, _, y_e)| y_e).collect::<vec::Vec<f64>>().as_slice()).unwrap();
    let f = MatrixF64::new(data.len() as u64, 3).unwrap();
    f.set_col(0, &x0);
    f.set_col(1, &x1);
    f.set_col(2, &x2);
    return (f, y)
}

fn main() {
    let args = os::args();
    assert_eq!(args.len(), 2);
    let path = Path::new(args[1].clone());
    let mut file = BufferedReader::new(File::open(&path));

    let dim = 3;
    let lambda = 0.01f64;
    let eps = 0.000001f64;

    let mut data_vec = vec::Vec::new();
    for line in file.lines() {
        let line_str = line.unwrap();
        let elems = line_str.split_str(",").collect::<Vec<&str>>();
        assert_eq!(elems.len(), 3);

        let x1_elem = from_str::<f64>(elems[0]).unwrap();
        let x2_elem = from_str::<f64>(elems[1]).unwrap();
        let y_elem = from_str::<f64>(elems[2].trim()).unwrap();
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
        let mut res = y.clone().unwrap();
        level2::dgemv(Transpose::NoTrans, 1.0f64, &f, v, -1.0f64, &mut res);
        return level1::dnrm2(&res).powi(2i32)
    };
    let grad = |v: &VectorF64| -> VectorF64 {
        let mut tmp = y.clone().unwrap();
        let mut res = VectorF64::new(dim).unwrap();
        level2::dgemv(Transpose::NoTrans, 1.0f64, &f, v, -1.0f64, &mut tmp);
        for i in range(0, dim) {
            let (f_col, _) = f.get_col(i).unwrap();
            f_col.mul(&tmp);
            res.set(i, sum_vec(&f_col));
        }
        return res
    };

    let alpha = gradient_descent(&VectorF64::new(dim).unwrap(), lambda, func, grad, eps);
    println!("alpha = {}\n", alpha);
    let mut y_predict = VectorF64::new(validate_len).unwrap();
    level2::dgemv(Transpose::NoTrans, 1.0f64, &f_validate, &alpha, 1.0f64, &mut y_predict);
    let y_diff = y_validate.clone().unwrap();
    y_diff.sub(&y_predict);
    let sigma = level1::dnrm2(&y_diff) / (validate_len as f64).sqrt();
    println!("sigma = {}\n", sigma);
    for i in range(0, validate_len) {
        println!("{} {} -> {}%", y_validate.get(i), y_predict.get(i), 100.0f64 * y_diff.get(i).abs() / y_validate.get(i));
    }
}
