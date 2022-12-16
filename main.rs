use std::path::Path;
use csv::StringRecord;
extern crate csv;
extern crate rustc_serialize;

/// Structure for holding data point's Clustercall to clusters
#[derive(Clone, Debug)]
pub struct Clustercall<'a> {
    data_point: &'a DataPoint,
    cluster_ind: usize,
}

#[derive(Clone, Debug, RustcDecodable)]
pub struct DataPoint {
    pub x: f64,
    pub y: f64,
}

impl From<StringRecord> for DataPoint {
    fn from(record: StringRecord) -> Self {
        let x = record[0].parse().unwrap();
        let y = record[1].parse().unwrap();
        DataPoint { x, y }
    }
}

pub fn read_file(file_path: &str) -> Vec<DataPoint> {
    let mut data = StringRecord::from(vec![]);
    let mut reader = csv::Reader::from_path(file_path).unwrap();
    for data_point in reader.records() {
        let data_point = data_point.unwrap();
        data.push(data_point).into();
    }
    data
}


pub fn edistance_squared(point_a: &DataPoint, point_b: &DataPoint) -> f64 {
   (point_b.x - point_a.x)*(point_b.x - point_a.x) + (point_b.y - point_a.y)*(point_b.y - point_a.y)
}


pub fn get_index_of_min_val(floats: &Vec<f64>) -> () {
    floats.iter().enumerate().fold(0,| min_ind, (ind, &val) |if val == f64::min(floats[min_ind], val) { 
        ind 
    } else { min_ind });
}

/// Assign points to clusters
fn expectation<'a>(data: &'a Vec<DataPoint>, cluster_centroids: &Vec<DataPoint>) -> Vec<Clustercall<'a>> {
    let mut Clustercall: Vec<(Clustercall)> = vec![];
    for point in data {
        let mut distance: Vec<f64> = vec![];
        for cluster in cluster_centroids {
            distance.push(edistance_squared(&point, cluster));
        }
        Clustercall.push(Clustercall{data_point: point, cluster_ind: get_index_of_min_val(&distance)});
    }
    Clustercall
}

pub fn Clustercall_number(Clustercall: &Vec<Clustercall>, cluster_ind: usize) -> usize {
    let points_in_cluster = points_in_cluster(Clustercall, cluster_ind);
    points_in_cluster.len()
}

pub fn points_in_cluster<'a>(Clustercall: &'a Vec<Clustercall>, cluster_ind: usize) -> Vec<Clustercall<'a>> {
    let mut points_in_cluster = Clustercall.clone();
    points_in_cluster.retain(|&Clustercall{data_point: _, cluster_ind: a}| a == cluster_ind);
    points_in_cluster
}

pub fn sum_assigned_values(Clustercall: &Vec<Clustercall>, cluster_ind: usize) -> DataPoint {
    let points_in_cluster = points_in_cluster(Clustercall, cluster_ind);
    let (mut x_tot, mut y_tot) = (0.0_f64, 0.0_f64);
    for point in points_in_cluster {
        x_tot += point.data_point.x;
        y_tot += point.data_point.y;
    }
    DataPoint{x: x_tot, y: y_tot}
}

/// Update cluster centres
fn maximisation(cluster_centroids: &mut Vec<DataPoint>, Clustercall: &Vec<(Clustercall)>) {
    for i in 0..cluster_centroids.len() {
        let num_points = count_Clustercall(&Clustercall, i);
        let sum_points = sum_assigned_values(&Clustercall, i);
        cluster_centroids[i] = DataPoint{
            x: sum_points.x/num_points as f64,
            y: sum_points.y/num_points as f64};
    }
}

pub fn get_error_metric(cluster_centroids: &Vec<DataPoint>, Clustercall: &Vec<Clustercall>) -> f64 {
        let mut error = 0.0;
        for i in 0..Clustercall.len() {
            let cluster_ind = Clustercall[i].cluster_ind;
            error += edistance_squared(Clustercall[i].data_point,
                                                &cluster_centroids[cluster_ind]);
        }
        error
}

pub fn kmeans_one_iteration<'a>(cluster_centroids: &mut Vec<DataPoint>, data: &'a Vec<DataPoint>) -> Vec<Clustercall<'a>> {
    let Clustercall = expectation(data, cluster_centroids);
    maximisation(cluster_centroids, &Clustercall);
    Clustercall
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    //test suare eucledian distance 
    fn test1() {
        let origin = DataPoint{x: 0.0, y: 0.0};
        let point = DataPoint{x: 1.0, y: 1.0};
        let expected = 2.0;
        let actual = edistance_squared(&origin, &point);
        assert_eq!(expected, actual)
    }
}

extern crate kmeans;

use kmeans::*;

fn main() {
    let data = read_file("BES_age&vote1.csv");
    let mut cluster_centroids = vec![DataPoint{x: 2.0, y: 50.0},
                                     DataPoint{x: 7.0, y: 100.0}];
    let (mut error, mut prev_error) = (0.0, -1.0);
    let mut Clustercall: Vec<Clustercall>;
    while error != prev_error {
        prev_error = error;
        Clustercall = kmeans_one_iteration(&mut cluster_centroids, &data);
        error = get_error_metric(&cluster_centroids, &Clustercall);
        println!("{}", error);
    }
}
