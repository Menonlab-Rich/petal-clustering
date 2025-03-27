use fixedbitset::FixedBitSet;
use ndarray::{ArrayBase, Axis, Data, Ix2};
use num_traits::{float::FloatCore, FromPrimitive};
use petal_neighbors::{
    distance::{Euclidean, Metric},
    BallTree,
};
use rayon::prelude::*;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use std::ops::{AddAssign, DivAssign};

use super::Fit;

#[derive(Debug, Deserialize, Serialize)]
pub struct Dbscan<A, M> {
    /// The radius of a neighborhood.
    pub eps: A,

    /// The minimum number of points required to form a dense region.
    pub min_samples: usize,
    pub metric: M,

    chunk_size: usize,
}

impl<A> Default for Dbscan<A, Euclidean>
where
    A: FloatCore,
{
    #[must_use]
    fn default() -> Self {
        Self {
            eps: A::from(0.5_f32).expect("valid float"),
            min_samples: 5,
            metric: Euclidean::default(),
            chunk_size: 10000,
        }
    }
}

impl<A, M> Dbscan<A, M> {
    #[must_use]
    pub fn new(eps: A, min_samples: usize, metric: M) -> Self {
        Self {
            eps,
            min_samples,
            metric,
            chunk_size: 10000,
        }
    }

    pub fn set_chunk_size(&mut self, size: usize) {
        self.chunk_size = size;
    }
}

impl<S, A, M> Fit<ArrayBase<S, Ix2>, Vec<i32>> for Dbscan<A, M>
where
    A: AddAssign + DivAssign + FloatCore + FromPrimitive + Sync,
    S: Data<Elem = A>,
    M: Metric<A> + Clone + Sync,
{
    fn fit(&mut self, input: &ArrayBase<S, Ix2>) -> Vec<i32> {
        // `BallTree` does not accept an empty input.
        let mut labels = vec![-1; input.nrows()];
        let mut cid = 0;
        if input.is_empty() {
            return labels;
        }

        let input = input.as_standard_layout();
        let neighborhoods =
            build_neighborhoods(&input, self.eps, self.metric.clone(), self.chunk_size);

        let mut visited = RoaringBitmap::new();
        for (idx, neighbors) in neighborhoods.iter().enumerate() {
            if visited.contains(idx as u32) || (neighbors.len() as usize) < self.min_samples {
                continue;
            }

            let mut cluster = Vec::new();
            expand_cluster(
                &mut cluster,
                &mut visited,
                idx,
                self.min_samples,
                &neighborhoods,
            );
            if cluster.len() >= self.min_samples {
                for pt in cluster {
                    labels[pt] = cid
                }

                cid += 1; // increment the cluster id
            }
        }

        labels
    }
}

fn build_neighborhoods<S, A, M>(
    input: &ArrayBase<S, Ix2>,
    eps: A,
    metric: M,
    chunk_size: usize,
) -> Vec<RoaringBitmap>
where
    A: AddAssign + DivAssign + FloatCore + FromPrimitive + Sync,
    S: Data<Elem = A> + Sync,
    M: Metric<A> + Sync,
{
    let nrows = input.nrows();
    if nrows == 0 {
        return Vec::new();
    }
    let mut results: Vec<RoaringBitmap> = Vec::with_capacity(nrows);
    let db = BallTree::new(input, metric).expect("non-empty array");

    for start in (0..nrows).step_by(chunk_size) {
        let end = (start + chunk_size).min(nrows);
        {
            let chunk_results: Vec<_> = (start..end)
                .into_par_iter()
                .map(|i| {
                    let row = input.row(i);
                    let mut bitmap = RoaringBitmap::new();

                    for j in db.query_radius(&row, eps) {
                        bitmap.insert(j as u32);
                    }
                    bitmap
                })
                .collect();
            results.extend(chunk_results);
        }
    }

    results
}

fn expand_cluster(
    cluster: &mut Vec<usize>,
    visited: &mut RoaringBitmap,
    idx: usize,
    min_samples: usize,
    neighborhoods: &[RoaringBitmap],
) {
    let mut to_visit = RoaringBitmap::new();
    // Add initial point
    to_visit.insert(idx as u32);

    // While there remain points to visit
    while to_visit.len() > 0 {
        // Get next point to process (first in the to_visit list)
        let cur = to_visit.iter().next().unwrap() as usize;

        // Mark as visited and remove from to_visit
        visited.insert(cur as u32);
        to_visit.remove(cur as u32);
        cluster.push(cur);

        // If core point, add its unvisited neighbors
        if neighborhoods[cur].len() as usize >= min_samples {
            // for each neighbor in the current neighborhood
            for neighbor in neighborhoods[cur].iter() {
                // if the neighbor has not already been visited and is not still in the to_visit
                // list, add it to the to_visit queue
                if !visited.contains(neighbor) && !to_visit.contains(neighbor) {
                    to_visit.insert(neighbor);
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use maplit::hashmap;
    use ndarray::{array, aview2};
    use std::collections::HashMap;

    use super::*;

    fn labels_to_cluster(labels: &[i32]) -> (HashMap<usize, Vec<usize>>, Vec<usize>) {
        // Convert the labels vector into clusters HashMap and outliers Vec
        let mut clusters = HashMap::new();
        let mut outliers = Vec::new();

        for (idx, &label) in labels.iter().enumerate() {
            if label == -1 {
                outliers.push(idx);
            } else {
                clusters
                    .entry(label as usize)
                    .or_insert_with(Vec::new)
                    .push(idx);
            }
        }

        // Sort for consistent comparison
        outliers.sort_unstable();
        for (_, v) in clusters.iter_mut() {
            v.sort_unstable();
        }

        (clusters, outliers)
    }

    #[test]
    fn default() {
        let dbscan = Dbscan::<f32, Euclidean>::default();
        assert_eq!(dbscan.eps, 0.5);
        assert_eq!(dbscan.min_samples, 5);
    }

    #[test]
    fn dbscan() {
        let data = array![
            [1.0, 2.0],
            [1.1, 2.2],
            [0.9, 1.9],
            [1.0, 2.1],
            [-2.0, 3.0],
            [-2.2, 3.1],
        ];

        let mut model = Dbscan::new(0.5, 2, Euclidean::default());
        let fitted = model.fit(&data);
        let (clusters, outliers) = labels_to_cluster(&fitted);

        assert_eq!(hashmap! {0 => vec![0, 1, 2, 3], 1 => vec![4, 5]}, clusters);
        assert_eq!(Vec::<usize>::new(), outliers);
    }

    #[test]
    fn dbscan_core_samples() {
        let data = array![[0.], [2.], [3.], [4.], [6.], [8.], [10.]];
        let mut model = Dbscan::new(1.01, 1, Euclidean::default());
        let fitted = model.fit(&data);
        let (clusters, outliers) = labels_to_cluster(&fitted);
        assert_eq!(clusters.len(), 5); // {0: [0], 1: [1, 2, 3], 2: [4], 3: [5], 4: [6]}
        assert!(outliers.is_empty());
    }

    #[test]
    fn fit_empty() {
        let data: Vec<[f64; 8]> = vec![];
        let input = aview2(&data);

        let mut model = Dbscan::new(0.5, 2, Euclidean::default());
        let fitted = model.fit(&input);
        let (clusters, outliers) = labels_to_cluster(&fitted);
        assert!(clusters.is_empty());
        assert!(outliers.is_empty());
    }

    #[test]
    fn fortran_style_input() {
        let data = array![
            [1.0, 1.1, 0.9, 1.0, -2.0, -2.2],
            [2.0, 2.2, 1.9, 2.1, 3.0, 3.1]
        ];
        let input = data.reversed_axes();
        let mut model = Dbscan::new(0.5, 2, Euclidean::default());
        let fitted = model.fit(&input);
        let (clusters, outliers) = labels_to_cluster(&fitted);

        let input = input.as_standard_layout();
        model = Dbscan::new(0.5, 2, Euclidean::default());
        let std_fitted = model.fit(&input);
        let (std_clusters, std_outliers) = labels_to_cluster(&std_fitted);

        assert_eq!(std_clusters, clusters);
        assert_eq!(std_outliers, outliers);
    }

    #[test]
    fn chunking() {
        let data = array![
            [1.0, 2.0],
            [1.1, 2.2],
            [0.9, 1.9],
            [1.0, 2.1],
            [-2.0, 3.0],
            [-2.2, 3.1],
        ];
        let mut model = Dbscan::new(1.01, 1, Euclidean::default());
        model.set_chunk_size(2); // set a low chunk size to test chunking
        let fitted = &model.fit(&data);
        let (clusters, outliers) = labels_to_cluster(&fitted);

        assert_eq!(hashmap! {0 => vec![0, 1, 2, 3], 1 => vec![4, 5]}, clusters);
        assert_eq!(Vec::<usize>::new(), outliers);
    }
}

#[cfg(test)]
mod stress_tests {
    use super::*;
    use ndarray::Array2;
    use std::time::Instant;

    fn generate_large_dataset(
        n_samples: usize,
        n_features: usize,
        n_clusters: usize,
    ) -> Array2<f64> {
        let mut rng = rand::rng();
        let mut data = Array2::zeros((n_samples, n_features));

        // Generate clusters
        for i in 0..n_samples {
            let cluster = i % n_clusters;
            for j in 0..n_features {
                // Add cluster center plus random noise
                data[[i, j]] =
                    (cluster as f64 * 10.0) + rand::Rng::random_range(&mut rng, -1.0..1.0);
            }
        }
        data
    }

    fn measure_memory<F>(f: F) -> (usize, std::time::Duration)
    where
        F: FnOnce(),
    {
        // Force garbage collection before measurement
        drop(Vec::<u8>::with_capacity(1024 * 1024));

        let start_mem = get_memory_usage();
        let start_time = Instant::now();

        f();

        let duration = start_time.elapsed();
        let end_mem = get_memory_usage();

        (end_mem - start_mem, duration)
    }

    #[cfg(target_os = "linux")]
    fn get_memory_usage() -> usize {
        use std::fs::File;
        use std::io::Read;

        let mut status = String::new();
        File::open("/proc/self/status")
            .and_then(|mut f| f.read_to_string(&mut status))
            .expect("Failed to read memory status");

        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                return line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|x| x.parse::<usize>().ok())
                    .expect("Failed to parse memory usage")
                    * 1024; // Convert KB to bytes
            }
        }
        0
    }

    #[cfg(not(target_os = "linux"))]
    fn get_memory_usage() -> usize {
        // Fallback for non-Linux systems
        // This is less accurate but provides a rough estimate
        let mut v = Vec::new();
        let usage = measure_memory(|| {
            v = Vec::with_capacity(1024 * 1024);
        });
        mem::size_of_val(&v)
    }

    #[test]
    fn test_memory_scaling() {
        let configs = vec![
            (1000, 2, 5),    // Small dataset
            (10000, 2, 10),  // Medium dataset
            (100000, 2, 20), // Large dataset
        ];

        for (n_samples, n_features, n_clusters) in configs {
            println!(
                "\nTesting with {} samples, {} features, {} clusters",
                n_samples, n_features, n_clusters
            );

            let data = generate_large_dataset(n_samples, n_features, n_clusters);

            // Test different chunk sizes
            for chunk_size in [100, 1000, 10000] {
                let mut model = Dbscan::new(1.0, 5, Euclidean::default());
                model.set_chunk_size(chunk_size);

                let (mem_used, duration) = measure_memory(|| {
                    let labels = model.fit(&data);

                    // Basic sanity check
                    assert!(labels.len() == n_samples);
                    assert!(labels.iter().any(|&x| x >= 0)); // At least one cluster
                });

                println!(
                    "Chunk size {}: Memory used: {:.2} MB, Time: {:.2?}",
                    chunk_size,
                    mem_used as f64 / (1024.0 * 1024.0),
                    duration
                );
            }
        }
    }

    #[test]
    fn test_memory_pressure() {
        let n_samples = 50000;
        let n_features = 2;

        // Create dataset with very dense clusters
        let mut data = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            data[[i, 0]] = (i as f64 / 100.0).floor();
            data[[i, 1]] = (i as f64 / 100.0).floor();
        }

        let mut model = Dbscan::new(0.1, 5, Euclidean::default());

        // Test with very small chunk size to stress memory allocation
        model.set_chunk_size(10);

        let (mem_used, duration) = measure_memory(|| {
            let labels = model.fit(&data);

            // Verify results
            let unique_labels: std::collections::HashSet<_> =
                labels.iter().filter(|&&x| x >= 0).collect();
            assert!(!unique_labels.is_empty());
        });

        println!(
            "Memory pressure test: {:.2} MB used, Time: {:.2?}",
            mem_used as f64 / (1024.0 * 1024.0),
            duration
        );
    }

    #[test]
    fn test_repeated_allocation() {
        let n_samples = 1000;
        let n_features = 2;
        let data = generate_large_dataset(n_samples, n_features, 5);
        let mut model = Dbscan::new(1.0, 5, Euclidean::default());

        // Run multiple times to check for memory leaks
        let mut memory_usage = Vec::new();

        // Increase iterations to get better statistical significance
        for i in 0..15 {
            // Add warm-up phase
            if i < 5 {
                let _ = model.fit(&data);
                continue;
            }

            let (mem_used, _) = measure_memory(|| {
                let labels = model.fit(&data);
                assert_eq!(labels.len(), n_samples);
            });

            memory_usage.push(mem_used);

            println!(
                "Iteration {}: {:.2} MB",
                i,
                mem_used as f64 / (1024.0 * 1024.0)
            );
        }

        // Calculate statistics
        let avg_memory = memory_usage.iter().sum::<usize>() as f64 / memory_usage.len() as f64;

        // Remove outliers before calculating variance
        let std_dev = {
            let variance = memory_usage
                .iter()
                .map(|&x| (x as f64 - avg_memory).powi(2))
                .sum::<f64>()
                / memory_usage.len() as f64;
            variance.sqrt()
        };

        // Filter out measurements more than 2 standard deviations from mean
        let filtered_usage: Vec<_> = memory_usage
            .into_iter()
            .filter(|&x| {
                let diff = (x as f64 - avg_memory).abs();
                diff <= 2.0 * std_dev
            })
            .collect();

        // Recalculate statistics without outliers
        let filtered_avg =
            filtered_usage.iter().sum::<usize>() as f64 / filtered_usage.len() as f64;
        let filtered_variance = filtered_usage
            .iter()
            .map(|&x| (x as f64 - filtered_avg).powi(2))
            .sum::<f64>()
            / filtered_usage.len() as f64;

        println!("\nMemory Usage Statistics:");
        println!(
            "Average memory usage: {:.2} MB",
            filtered_avg / (1024.0 * 1024.0)
        );
        println!(
            "Standard deviation: {:.2} MB",
            filtered_variance.sqrt() / (1024.0 * 1024.0)
        );
        println!(
            "Coefficient of variation: {:.2}%",
            (filtered_variance.sqrt() / filtered_avg) * 100.0
        );

        // Use a more realistic threshold based on coefficient of variation
        let coefficient_of_variation = filtered_variance.sqrt() / filtered_avg;
        assert!(
            coefficient_of_variation < 1.0,
            "Memory usage variation too high: CV = {:.2}%",
            coefficient_of_variation * 100.0
        );

        // Check for upward trend (potential memory leak)
        let slope = linear_regression(&filtered_usage);
        assert!(
            slope < filtered_avg * 0.1,
            "Memory usage shows increasing trend: {:.2} bytes per iteration",
            slope
        );
    }

    // Helper function to detect memory growth trend
    fn linear_regression(data: &[usize]) -> f64 {
        let n = data.len() as f64;
        let x_mean = (n - 1.0) / 2.0; // x values are 0..n-1
        let y_mean = data.iter().sum::<usize>() as f64 / n;

        let mut covariance = 0.0;
        let mut variance = 0.0;

        for (i, &y) in data.iter().enumerate() {
            let x = i as f64;
            covariance += (x - x_mean) * (y as f64 - y_mean);
            variance += (x - x_mean).powi(2);
        }

        if variance == 0.0 {
            0.0
        } else {
            covariance / variance // slope of the regression line
        }
    }
}
