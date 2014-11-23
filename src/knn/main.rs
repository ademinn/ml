use std::num::Float;
use std::vec;

struct Point {
    x: f64,
    y: f64,
}

fn distance(p1: &Point, p2: &Point) -> f64 {
    ((p1.x - p2.x).powi(2i32) + (p1.y - p2.y).powi(2i32)).sqrt()
}

fn solve(points: &vec::Vec<Box<(Point, i32)>>, point: &Point, dist: |&Point, &Point| -> f64) -> int {
    let mut sorted = vec::Vec::new();
    for v in points.iter() {
        sorted.push(v);
    }
    sorted.sort_by(|&& box (p1, _), && box (p2, _)| -> Ordering {
        match dist(&p1, point).partial_cmp(&dist(&p2, point)) {
            Some(x) => x,
            None => panic!("NaN value"),
        }
    });
    for && box (p, _) in sorted.iter() {
        println!("{}", p.x);
    }
    return 1
}


fn main() {
    let mut points = vec::Vec::new();
    for i in range(0i, 10i) {
        let p = box () (Point {x: i as f64, y: 0.0f64}, 0i32);
        points.push(p);
    }
    solve(&points, &Point{x: 10.0f64, y: 0.0f64}, distance);
}
