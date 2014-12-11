extern crate rgsl;
use rgsl::types::vector::VectorF64;
use std::num::Float;


pub fn gradient_descent(x0: &VectorF64, lambda: f64, func: |&VectorF64| -> f64, grad: |&VectorF64| -> VectorF64, eps: f64) -> VectorF64 {
    let mut x1 = x0.clone().unwrap();
    loop {
        let x2 = grad(&x1);
        x2.scale(-lambda);
        x2.add(&x1);
        if (func(&x2) - func(&x1)).abs() < eps {
            return x2
        } else {
            x1 = x2;
        }
    }
}
