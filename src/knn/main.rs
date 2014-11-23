use std::num::Float;
use std::vec;
use std::collections::HashMap;
use std::uint;

struct Point {
    x: f64,
    y: f64,
}

fn distance(p1: &Point, p2: &Point) -> f64 {
    ((p1.x - p2.x).powi(2i32) + (p1.y - p2.y).powi(2i32)).sqrt()
}

fn solve(points: &vec::Vec<(Point, i32)>, point: &Point, dist: |&Point, &Point| -> f64, k: uint) -> i32 {
    let mut sorted = vec::Vec::new();
    for &v in points.iter() {
        sorted.push(v);
    }
    sorted.sort_by(|&(p1, _), &(p2, _)| -> Ordering {
        match dist(&p1, point).partial_cmp(&dist(&p2, point)) {
            Some(x) => x,
            None => panic!("NaN value"),
        }
    });
    let mut counter: HashMap<i32, uint> = HashMap::new();
    for &(_, cluster) in sorted.iter().take(k) {
        let v = counter.remove(&cluster);
        match v {
            Some(x) => { counter.insert(cluster, x + 1u); }
            None => { counter.insert(cluster, 1u); }
        }
    }
    let mut result = None;
    let mut prev_max = uint::MIN;
    for (cluster, count) in counter.iter() {
        if prev_max < *count {
            result = Some(*cluster);
            prev_max = *count;
        }
    }
    match result {
        Some(cluster) => return cluster,
        None => panic!("No clusters found"),
    }
}


fn main() {
    let mut points = vec::Vec::new();
    for i in range(0i, 10i) {
        let p = (Point {x: i as f64, y: 0.0f64}, 0i32);
        points.push(p);
    }
    println!("{}", solve(&points, &Point{x: 10.0f64, y: 0.0f64}, distance, 1));
}
