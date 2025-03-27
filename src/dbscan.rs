use fixedbitset::FixedBitSet;
use ndarray::{ArrayBase, Axis, Data, Ix2};
use num_traits::{float::FloatCore, FromPrimitive};
use petal_neighbors::{
    distance::{Euclidean, Metric},
    BallTree,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::ops::{AddAssign, DivAssign};

use super::Fit;

#[derive(Debug, Deserialize, Serialize)]
pub struct Dbscan<A, M> {
    /// The radius of a neighborhood.
    pub eps: A,

    /// The minimum number of points required to form a dense region.
    pub min_samples: usize,
    pub metric: M,
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
        }
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
        let neighborhoods = build_neighborhoods(&input, self.eps, self.metric.clone());

        let mut visited = FixedBitSet::with_capacity(input.nrows());
        for (idx, neighbors) in neighborhoods.iter().enumerate() {
            if visited[idx] || neighbors.len() < self.min_samples {
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

fn build_neighborhoods<S, A, M>(input: &ArrayBase<S, Ix2>, eps: A, metric: M) -> Vec<FixedBitSet>
where
    A: AddAssign + DivAssign + FloatCore + FromPrimitive + Sync,
    S: Data<Elem = A>,
    M: Metric<A> + Sync,
{
    let nrows = input.nrows();
    if nrows == 0 {
        return Vec::new();
    }
    let mut results: Vec<FixedBitSet> = Vec::with_capacity(nrows);
    let db = BallTree::new(input, metric).expect("non-empty array");

    for chunk in input.axis_chunks_iter(Axis(0), 10000) {
        let rows: Vec<_> = chunk.rows().into_iter().collect();
        let chunk_results: Vec<FixedBitSet> = rows
            .into_par_iter()
            .map(|p| {
                let mut bitset = FixedBitSet::with_capacity(nrows);
                for i in db.query_radius(&p, eps) {
                    bitset.set(i, true);
                }
                bitset
            })
            .collect();
        results.extend(chunk_results);
    }

    results
}

fn expand_cluster(
    cluster: &mut Vec<usize>,
    visited: &mut FixedBitSet,
    idx: usize,
    min_samples: usize,
    neighborhoods: &[FixedBitSet],
) {
    let mut to_visit = FixedBitSet::with_capacity(visited.len());
    to_visit.set(idx, true); // Start with initial index

    while to_visit.count_ones(..) > 0 {
        // Get next point to process (any set bit)
        let cur = to_visit.ones().next().unwrap();

        // Mark as visited and remove from to_visit
        visited.set(cur, true);
        to_visit.set(cur, false);
        cluster.push(cur);

        // If core point, add its unvisited neighbors
        if neighborhoods[cur].count_ones(..) >= min_samples {
            // Create temporary bitset of new neighbors
            let mut new_neighbors = neighborhoods[cur].clone();
            // Remove already visited points
            new_neighbors.difference_with(visited);
            // Remove points already in to_visit to avoid duplicates
            new_neighbors.difference_with(&to_visit);
            // Union with to_visit to add new points to check
            to_visit.union_with(&new_neighbors);
        }
    }
}

#[cfg(test)]
mod test {
    use maplit::hashmap;
    use ndarray::{array, aview2};

    use super::*;

    fn labels_to_cluster(labels: &Vec<i32>) -> (HashMap<usize, Vec<usize>>, Vec<usize>) {
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
}
