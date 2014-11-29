use std::os;
use std::io::BufferedReader;
use std::io::File;
use std::num::Float;
use std::vec;
use std::collections::HashMap;
use std::uint;
use std::rc::Rc;

#[deriving(Show)]
struct Point {
    x: f64,
    y: f64,
}

#[deriving(Show)]
struct ClusteredPoint {
    point: Point,
    cluster: i32,
}

fn distance(p1: &Point, p2: &Point) -> f64 {
    ((p1.x - p2.x).powi(2i32) + (p1.y - p2.y).powi(2i32)).sqrt()
}

fn solve(points: &vec::Vec<Rc<ClusteredPoint>>, point: &Point, dist: |&Point, &Point| -> f64, k: uint) -> i32 {
    let mut sorted = points.clone();
    sorted.sort_by(|ref p1, ref p2| -> Ordering {
        match dist(&p1.point, point).partial_cmp(&dist(&p2.point, point)) {
            Some(x) => x,
            None => panic!("NaN value"),
        }
    });
    let mut counter: HashMap<i32, uint> = HashMap::new();
    for ref cluster in sorted.iter().map(|ref p| p.cluster).take(k) {
        let v = counter.remove(cluster);
        match v {
            Some(x) => { counter.insert(*cluster, x + 1u); }
            None => { counter.insert(*cluster, 1u); }
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


fn leave_one_out(points: &vec::Vec<Rc<ClusteredPoint>>, k: uint) -> uint {
    let mut mut_points = points.clone();
    let mut success_count = 0u;
    for i in range(0u, mut_points.len()) {
        let p = mut_points.remove(i).unwrap();
        if solve(&mut_points, &p.point, distance, k) == p.cluster {
            success_count += 1;
        }
        mut_points.insert(i, p);
    }
    return success_count
}


fn main() {
    let args = os::args();
    assert_eq!(args.len(), 2);
    let path = Path::new(args[1].clone());
    let mut file = BufferedReader::new(File::open(&path));

    let mut points = vec::Vec::new();
    for line in file.lines() {
        let line_str = line.unwrap();
        let elems = line_str.split_str(",").collect::<Vec<&str>>();
        assert_eq!(elems.len(), 3);

        let x = from_str::<f64>(elems[0]).unwrap();
        let y = from_str::<f64>(elems[1]).unwrap();
        let cluster = from_str::<i32>(elems[2].trim()).unwrap();
        points.push(Rc::new(ClusteredPoint {point: Point {x: x, y: y}, cluster: cluster}));
    }

    println!("{} points", points.len());

    for k in range(1u, points.len()) {
        let success_count = leave_one_out(&points, k);
        println!("k = {}: {}", k, success_count);
    }

}
