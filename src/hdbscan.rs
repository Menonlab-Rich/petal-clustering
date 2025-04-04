use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::mem::MaybeUninit;
use std::ops::{AddAssign, Div, DivAssign, Sub};

use ndarray::{Array1, ArrayBase, ArrayView1, ArrayView2, Data, Ix2};
use num_traits::{float::FloatCore, FromPrimitive};
use petal_neighbors::distance::{Euclidean, Metric};
use petal_neighbors::BallTree;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use succinct::{BitVecMut, BitVector};

use super::Fit;

/// HDBSCAN (hierarchical density-based spatial clustering of applications with noise)
/// clustering algorithm.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use petal_neighbors::distance::Euclidean;
/// use petal_clustering::{HDbscan, Fit};
///
/// let points = array![
///             [1.0, 2.0],
///             [1.1, 2.2],
///             [0.9, 1.9],
///             [1.0, 2.1],
///             [-2.0, 3.0],
///             [-2.2, 3.1],
///         ];
/// let mut hdbscan = HDbscan {
///    alpha: 1.,
///    min_samples: 2,
///    min_cluster_size: 2,
///    metric: Euclidean::default(),
///    boruvka: false,
/// };
/// let (clusters, outliers, _outlier_scores) = hdbscan.fit(&points);
/// assert_eq!(clusters.len(), 2);   // two clusters found
///
/// assert_eq!(
///     outliers.len(),
///     points.nrows() - clusters.values().fold(0, |acc, v| acc + v.len()));
/// ```
#[derive(Debug, Deserialize, Serialize)]
pub struct HDbscan<A, M> {
    /// The radius of a neighborhood.
    pub alpha: A,

    /// The minimum number of points required to form a dense region.
    pub min_samples: usize,
    pub min_cluster_size: usize,
    pub metric: M,
    pub boruvka: bool,
}

impl<A> Default for HDbscan<A, Euclidean>
where
    A: FloatCore,
{
    #[must_use]
    fn default() -> Self {
        Self {
            alpha: A::one(),
            min_samples: 15,
            min_cluster_size: 15,
            metric: Euclidean::default(),
            boruvka: true,
        }
    }
}

impl<S, A, M> Fit<ArrayBase<S, Ix2>, (HashMap<usize, Vec<usize>>, Vec<usize>, Vec<A>)>
    for HDbscan<A, M>
where
    A: AddAssign + DivAssign + FloatCore + FromPrimitive + Sync + Send,
    S: Data<Elem = A>,
    M: Metric<A> + Clone + Sync + Send,
{
    fn fit(
        &mut self,
        input: &ArrayBase<S, Ix2>,
    ) -> (HashMap<usize, Vec<usize>>, Vec<usize>, Vec<A>) {
        if input.is_empty() {
            return (HashMap::new(), Vec::new(), Vec::new());
        }
        let input = input.as_standard_layout();
        let db = BallTree::new(input.view(), self.metric.clone()).expect("non-empty array");

        let (mut mst, _offset) = if self.boruvka {
            let boruvka = Boruvka::new(db, self.min_samples);
            boruvka.min_spanning_tree().into_raw_vec_and_offset()
        } else {
            let core_distances = Array1::from_vec(
                input
                    .rows()
                    .into_iter()
                    .map(|r| {
                        db.query(&r, self.min_samples)
                            .1
                            .last()
                            .copied()
                            .expect("at least one point should be returned")
                    })
                    .collect(),
            );
            mst_linkage(
                input.view(),
                &self.metric,
                core_distances.view(),
                self.alpha,
            )
            .into_raw_vec_and_offset()
        };

        mst.sort_unstable_by(|a, b| a.2.partial_cmp(&(b.2)).expect("invalid distance"));
        let sorted_mst = Array1::from_vec(mst);
        let labeled = label(sorted_mst);
        let condensed = condense_mst(labeled.view(), self.min_cluster_size);
        let outlier_scores = glosh(&condensed, self.min_cluster_size);
        let (clusters, outliers) = find_clusters(&Array1::from_vec(condensed).view());
        (clusters, outliers, outlier_scores)
    }
}

fn mst_linkage<A: FloatCore>(
    input: ArrayView2<A>,
    metric: &dyn Metric<A>,
    core_distances: ArrayView1<A>,
    alpha: A,
) -> Array1<(usize, usize, A)> {
    let nrows = input.nrows();

    assert_eq!(
        nrows,
        core_distances.len(),
        "dimensions of distance_metric and core_distances should match"
    );

    if nrows == 0 {
        // If there are no input points, return an empty MST.
        return Array1::from_vec(vec![]);
    }

    if nrows == 1 {
        // If there is only one input point, return a single edge with zero distance.
        return Array1::from_vec(vec![(0, 0, A::zero())]);
    }

    let mut mst = Array1::<(usize, usize, A)>::uninit(nrows - 1);
    let mut in_tree: Vec<bool> = vec![false; nrows];
    let mut cur = 0;
    // edge uv: shortest_edges[v] = (mreachability_as_||uv||, u)
    // shortest as in shortest edges to v among  all nodes currently in tree
    let mut shortest_edges: Vec<(A, usize)> = vec![(A::max_value(), 1); nrows];

    for i in 0..nrows - 1 {
        // Add `cur` to tree
        in_tree[cur] = true;
        let core_cur = core_distances[cur];

        // next edge to add to tree
        let mut source: usize = 0;
        let mut next: usize = 0;
        let mut distance = A::max_value();

        for j in 0..nrows {
            if in_tree[j] {
                // skip if j is already in the tree
                continue;
            }

            let right = shortest_edges[j];
            let mut left = (metric.distance(&input.row(cur), &input.row(j)), cur);

            if alpha != A::from(1).expect("conversion failure") {
                left.0 = left.0 / alpha;
            } // convert distance matrix to `distance_metric / alpha` ?

            let core_j = core_distances[j];

            // right < MReachability_cur_j
            if (right.0 < core_cur || right.0 < left.0 || right.0 < core_j) && right.0 < distance {
                next = j;
                distance = right.0;
                source = right.1;
            }

            let tmp = if core_j > core_cur { core_j } else { core_cur };
            if tmp > left.0 {
                left.0 = tmp;
            }

            if left.0 < right.0 {
                shortest_edges[j] = left;
                if left.0 < distance {
                    distance = left.0;
                    source = left.1;
                    next = j;
                }
            } else if right.0 < distance {
                distance = right.0;
                source = right.1;
                next = j;
            }
        }

        mst[i] = MaybeUninit::new((source, next, distance)); // check MaybeUninit usage!
        cur = next;
    }

    unsafe { mst.assume_init() }
}

fn label<A: FloatCore>(mst: Array1<(usize, usize, A)>) -> Array1<(usize, usize, A, usize)> {
    let n = mst.len() + 1;
    let mut uf = UnionFind::new(n);
    mst.into_iter()
        .map(|(mut a, mut b, delta)| {
            a = uf.fast_find(a);
            b = uf.fast_find(b);
            (a, b, delta, uf.union(a, b))
        })
        .collect()
}

fn condense_mst<A: FloatCore + Div>(
    mst: ArrayView1<(usize, usize, A, usize)>,
    min_cluster_size: usize,
) -> Vec<(usize, usize, A, usize)> {
    let root = mst.len() * 2;
    let n = mst.len() + 1;

    let mut relabel = Array1::<usize>::uninit(root + 1);
    relabel[root] = MaybeUninit::new(n);
    let mut next_label = n + 1;
    let mut ignore = vec![false; root + 1];
    let mut result = Vec::new();

    let bsf = bfs_mst(mst, root);
    for node in bsf {
        if node < n {
            continue;
        }
        if ignore[node] {
            continue;
        }
        let info = mst[node - n];
        let lambda = if info.2 > A::zero() {
            A::one() / info.2
        } else {
            A::max_value()
        };
        let left = info.0;
        let left_count = if left < n { 1 } else { mst[left - n].3 };

        let right = info.1;
        let right_count = if right < n { 1 } else { mst[right - n].3 };

        match (
            left_count >= min_cluster_size,
            right_count >= min_cluster_size,
        ) {
            (true, true) => {
                relabel[left] = MaybeUninit::new(next_label);
                result.push((
                    unsafe { relabel[node].assume_init() },
                    next_label,
                    lambda,
                    left_count,
                ));
                next_label += 1;

                relabel[right] = MaybeUninit::new(next_label);
                result.push((
                    unsafe { relabel[node].assume_init() },
                    next_label,
                    lambda,
                    right_count,
                ));
                next_label += 1;
            }
            (true, false) => {
                relabel[left] = relabel[node];
                for child in bfs_mst(mst, right) {
                    if child < n {
                        result.push((unsafe { relabel[node].assume_init() }, child, lambda, 1));
                    }
                    ignore[child] = true;
                }
            }
            (false, true) => {
                relabel[right] = relabel[node];
                for child in bfs_mst(mst, left) {
                    if child < n {
                        result.push((unsafe { relabel[node].assume_init() }, child, lambda, 1));
                    }
                    ignore[child] = true;
                }
            }
            (false, false) => {
                for child in bfs_mst(mst, node).into_iter().skip(1) {
                    if child < n {
                        result.push((unsafe { relabel[node].assume_init() }, child, lambda, 1));
                    }
                    ignore[child] = true;
                }
            }
        }
    }
    result
}

fn get_stability<A: FloatCore + FromPrimitive + AddAssign + Sub>(
    condensed_tree: &ArrayView1<(usize, usize, A, usize)>,
) -> HashMap<usize, A> {
    let mut births: HashMap<_, _> = condensed_tree.iter().fold(HashMap::new(), |mut births, v| {
        let entry = births.entry(v.1).or_insert(v.2);
        if *entry > v.2 {
            *entry = v.2;
        }
        births
    });

    let min_parent = condensed_tree
        .iter()
        .min_by_key(|v| v.0)
        .expect("couldn't find the smallest cluster")
        .0;

    let entry = births.entry(min_parent).or_insert_with(A::zero);
    *entry = A::zero();

    condensed_tree.iter().fold(
        HashMap::new(),
        |mut stability, (parent, _child, lambda, size)| {
            let entry = stability.entry(*parent).or_insert_with(A::zero);
            let birth = births.get(parent).expect("invalid child node.");
            let Some(size) = A::from_usize(*size) else {
                panic!("invalid size");
            };
            *entry += (*lambda - *birth) * size;
            stability
        },
    )
}

fn find_clusters<A: FloatCore + FromPrimitive + AddAssign + Sub>(
    condensed_tree: &ArrayView1<(usize, usize, A, usize)>,
) -> (HashMap<usize, Vec<usize>>, Vec<usize>) {
    let mut stability = get_stability(condensed_tree);
    let mut nodes: Vec<_> = stability.keys().copied().collect();
    nodes.sort_unstable();
    nodes.reverse();
    nodes.remove(nodes.len() - 1);

    let tree: Vec<_> = condensed_tree
        .iter()
        .filter_map(|(p, c, _, s)| if *s > 1 { Some((*p, *c)) } else { None })
        .collect();

    let mut clusters: HashSet<_> = stability.keys().copied().collect();
    for node in nodes {
        let subtree_stability = tree.iter().fold(A::zero(), |acc, (p, c)| {
            if *p == node {
                acc + *stability.get(c).expect("corrupted stability dictionary")
            } else {
                acc
            }
        });

        stability.entry(node).and_modify(|v| {
            if *v < subtree_stability {
                clusters.remove(&node);
                *v = subtree_stability;
            } else {
                let bfs = bfs_tree(&tree, node);
                for child in bfs {
                    if child != node {
                        clusters.remove(&child);
                    }
                }
            }
        });
    }

    let mut clusters: Vec<_> = clusters.into_iter().collect();
    clusters.sort_unstable();
    let clusters: HashMap<_, _> = clusters
        .into_iter()
        .enumerate()
        .map(|(id, c)| (c, id))
        .collect();
    let max_parent = condensed_tree
        .iter()
        .max_by_key(|v| v.0)
        .expect("no maximum parent available")
        .0;
    let min_parent = condensed_tree
        .iter()
        .min_by_key(|v| v.0)
        .expect("no minimum parent available")
        .0;

    let mut uf = TreeUnionFind::new(max_parent + 1);
    for (parent, child, _, _) in condensed_tree {
        if !clusters.contains_key(child) {
            uf.union(*parent, *child);
        }
    }

    let mut res_clusters: HashMap<_, Vec<_>> = HashMap::new();
    let mut outliers = vec![];
    for n in 0..min_parent {
        let cluster = uf.find(n);
        if cluster > min_parent {
            let c = res_clusters.entry(cluster).or_default();
            c.push(n);
        } else {
            outliers.push(n);
        }
    }
    (res_clusters, outliers)
}

// GLOSH: Global-Local Outlier Score from Hierarchies
// Reference: https://dl.acm.org/doi/10.1145/2733381
//
// Given the following hierarchy (min_cluster_size = 3),
//               Root
//              /    \
//             A     ...
// eps_x ->   / \
//           x   A
//              / \
//             y   A
//                /|\   <- eps_A: A is still a cluster w.r.t. min_cluster_size at this level
//               a b c
//
// To compute the outlier score of point x, we need:
//    - eps_x: eps that x joins to cluster A (A is the first cluster that x joins to)
//    - eps_A: lowest eps that A or any of A's child clusters survives w.r.t. min_cluster_size.
// Then, the outlier score of x is defined as:
//    score(x) = 1 - eps_A / eps_x
//
// Since we are working with density lambda values (where lambda = 1/eps):
//    lambda_x = 1 / eps_x
//    lambda_A = 1 / eps_A
//    score(x) = 1 - lambda_x / lambda_A
fn glosh<A: FloatCore>(
    condensed_mst: &[(usize, usize, A, usize)],
    min_cluster_size: usize,
) -> Vec<A> {
    let deaths = max_lambdas(condensed_mst, min_cluster_size);

    // min_parent gives the number of events in the hierarchy
    let num_events = condensed_mst
        .iter()
        .map(|(parent, _, _, _)| *parent)
        .min()
        .map_or(0, |min_parent| min_parent);

    let mut scores = vec![A::zero(); num_events];
    for (parent, child, lambda, _) in condensed_mst {
        if *child >= num_events {
            continue;
        }
        let lambda_max = deaths[*parent];
        if lambda_max == A::zero() {
            scores[*child] = A::zero();
        } else {
            scores[*child] = (lambda_max - *lambda) / lambda_max;
        }
    }
    scores
}

// Return the maximum lambda value (min eps) for each cluster C such that
// the cluster or any of its child clusters has at least min_cluster_size points.
fn max_lambdas<A: FloatCore>(
    condensed_mst: &[(usize, usize, A, usize)],
    min_cluster_size: usize,
) -> Vec<A> {
    let largest_cluster_id = condensed_mst
        .iter()
        .map(|(parent, child, _, _)| parent.max(child))
        .max()
        .expect("empty condensed_mst");

    // bottom-up traverse the hierarchy to keep track of the maximum lambda values
    // (same with the reverse order iteration on the condensed_mst)
    let mut parent_sizes: Vec<usize> = vec![0; largest_cluster_id + 1];
    let mut deaths_arr: Vec<A> = vec![A::zero(); largest_cluster_id + 1];
    for (parent, child, lambda, child_size) in condensed_mst.iter().rev() {
        parent_sizes[*parent] += *child_size;
        if parent_sizes[*parent] >= min_cluster_size {
            deaths_arr[*parent] = deaths_arr[*parent].max(*lambda);
        }
        if *child_size >= min_cluster_size {
            deaths_arr[*parent] = deaths_arr[*parent].max(deaths_arr[*child]);
        }
    }
    deaths_arr
}

fn bfs_tree(tree: &[(usize, usize)], root: usize) -> Vec<usize> {
    let mut result = vec![];
    let mut to_process = HashSet::new();
    to_process.insert(root);
    while !to_process.is_empty() {
        result.extend(to_process.iter());
        to_process = tree
            .iter()
            .filter_map(|(p, c)| {
                if to_process.contains(p) {
                    Some(*c)
                } else {
                    None
                }
            })
            .collect::<HashSet<_>>();
    }
    result
}

fn bfs_mst<A: FloatCore>(mst: ArrayView1<(usize, usize, A, usize)>, start: usize) -> Vec<usize> {
    let n = mst.len() + 1;

    let mut to_process = vec![start];
    let mut result = vec![];

    while !to_process.is_empty() {
        result.extend_from_slice(to_process.as_slice());
        to_process = to_process
            .into_iter()
            .filter_map(|x| {
                if x >= n {
                    Some(vec![mst[x - n].0, mst[x - n].1].into_iter())
                } else {
                    None
                }
            })
            .flatten()
            .collect();
    }
    result
}

#[allow(dead_code)]
#[derive(Debug)]
struct TreeUnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    is_component: BitVector<u64>,
}

#[allow(dead_code)]
impl TreeUnionFind {
    fn new(n: usize) -> Self {
        let parent = (0..n).collect();
        let size = vec![0; n];
        let is_component = BitVector::with_fill(
            u64::try_from(n).expect("fail to build a large enough bit vector"),
            true,
        );
        Self {
            parent,
            size,
            is_component,
        }
    }

    fn find(&mut self, x: usize) -> usize {
        assert!(x < self.parent.len());
        if x != self.parent[x] {
            self.parent[x] = self.find(self.parent[x]);
            self.is_component.set_bit(
                u64::try_from(x).expect("fail to convert usize to u64"),
                false,
            );
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let xx = self.find(x);
        let yy = self.find(y);

        match self.size[xx].cmp(&self.size[yy]) {
            Ordering::Greater => self.parent[yy] = xx,
            Ordering::Equal => {
                self.parent[yy] = xx;
                self.size[xx] += 1;
            }
            Ordering::Less => self.parent[xx] = yy,
        }
    }

    fn components(&self) -> Vec<usize> {
        self.is_component
            .iter()
            .enumerate()
            .filter_map(|(idx, v)| if v { Some(idx) } else { None })
            .collect()
    }

    fn num_components(&self) -> usize {
        self.is_component.iter().filter(|b| *b).count()
    }
}

struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    next_label: usize,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        let parent = (0..2 * n).collect();
        let size = vec![1]
            .into_iter()
            .cycle()
            .take(n)
            .chain(vec![0].into_iter().cycle().take(n - 1))
            .collect();
        Self {
            parent,
            size,
            next_label: n,
        }
    }

    fn union(&mut self, m: usize, n: usize) -> usize {
        self.parent[m] = self.next_label;
        self.parent[n] = self.next_label;
        let res = self.size[m] + self.size[n];
        self.size[self.next_label] = res;
        self.next_label += 1;
        res
    }

    fn fast_find(&mut self, mut n: usize) -> usize {
        let mut root = n;
        while self.parent[n] != n {
            n = self.parent[n];
        }
        while self.parent[root] != n {
            let tmp = self.parent[root];
            self.parent[root] = n;
            root = tmp;
        }
        n
    }
}

#[allow(dead_code)]
struct Boruvka<'a, A, M>
where
    A: FloatCore,
    M: Metric<A>,
{
    db: BallTree<'a, A, M>,
    min_samples: usize,
    candidates: Candidates<A>,
    components: Components,
    core_distances: Array1<A>,
    bounds: Vec<A>,
    mst: Vec<(usize, usize, A)>,
}

#[allow(dead_code)]
impl<'a, A, M> Boruvka<'a, A, M>
where
    A: FloatCore + AddAssign + DivAssign + FromPrimitive + Sync + Send,
    M: Metric<A> + Sync + Send,
{
    fn new(db: BallTree<'a, A, M>, min_samples: usize) -> Self {
        let mut candidates = Candidates::new(db.points.nrows());
        let components = Components::new(db.nodes.len(), db.points.nrows());
        let bounds = vec![A::max_value(); db.nodes.len()];
        let core_distances = compute_core_distances(&db, min_samples, &mut candidates);
        let mst = Vec::with_capacity(db.points.nrows() - 1);
        Boruvka {
            db,
            min_samples,
            candidates,
            components,
            core_distances,
            bounds,
            mst,
        }
    }

    fn min_spanning_tree(mut self) -> Array1<(usize, usize, A)> {
        let mut num_components = self.update_components();

        while num_components > 1 {
            self.traversal(0, 0);
            num_components = self.update_components();
        }
        Array1::from_vec(self.mst)
    }

    fn update_components(&mut self) -> usize {
        let components = self.components.get_current();
        for i in components {
            let Some((src, sink, dist)) = self.candidates.get(i) else {
                continue;
            };

            if self.components.add(src, sink).is_none() {
                self.candidates.reset(i);
                continue;
            }

            self.candidates.distances[i] = A::max_value();

            self.mst.push((src, sink, dist));

            if self.mst.len() == self.db.num_points() - 1 {
                return self.components.len();
            }
        }
        self.components.update_points();
        for n in (0..self.db.num_nodes()).rev() {
            match self.db.children_of(n) {
                None => {
                    let mut points = self
                        .db
                        .points_of(n)
                        .iter()
                        .map(|i| self.components.point[*i]);
                    let pivot = points.next().expect("empty node");
                    if points.all(|c| c == pivot) {
                        self.components.node[n] =
                            u32::try_from(pivot).expect("overflow components");
                    }
                }
                Some((left, right)) => {
                    if self.components.node[left] == self.components.node[right]
                        && self.components.node[left] != u32::MAX
                    {
                        self.components.node[n] = self.components.node[left];
                    }
                }
            }
        }
        self.reset_bounds();
        self.components.len()
    }

    fn traversal(&mut self, query: usize, reference: usize) {
        // prune min{||query - ref||} >= bound_query
        let node_dist = self.db.node_distance_lower_bound(query, reference);
        if node_dist >= self.bounds[query] {
            return;
        }
        // prune when query and ref are in the same component
        if self.components.node[query] == self.components.node[reference]
            && self.components.node[query] != u32::MAX
        {
            return;
        }

        let query_children = self.db.children_of(query);
        let ref_children = self.db.children_of(reference);
        match (
            query_children,
            ref_children,
            self.db.compare_nodes(query, reference),
        ) {
            (None, None, _) => {
                let mut lower = A::max_value();
                let mut upper = A::zero();
                for &i in self.db.points_of(query) {
                    let c1 = self.components.point[i];
                    // mreach(i, j) >= core_i > candidate[c1]
                    // i.e. current best candidate for component c1 => prune
                    if self.core_distances[i] > self.candidates.distances[c1] {
                        continue;
                    }
                    for &j in self.db.points_of(reference) {
                        let c2 = self.components.point[j];
                        // mreach(i, j) >= core_j > candidate[c1] => prune
                        // i, j in the same component => prune
                        if self.core_distances[j] > self.candidates.distances[c1] || c1 == c2 {
                            continue;
                        }

                        let mut mreach = self
                            .db
                            .metric
                            .distance(&self.db.points.row(i), &self.db.points.row(j));
                        if self.core_distances[j] > mreach {
                            mreach = self.core_distances[j];
                        }
                        if self.core_distances[i] > mreach {
                            mreach = self.core_distances[i];
                        }

                        if mreach < self.candidates.distances[c1] {
                            self.candidates.update(c1, (i, j, mreach));
                        }
                    }
                    if self.candidates.distances[c1] < lower {
                        lower = self.candidates.distances[c1];
                    }
                    if self.candidates.distances[c1] > upper {
                        upper = self.candidates.distances[c1];
                    }
                }

                let radius = self.db.radius_of(query);
                let mut bound = lower + radius + radius;
                if bound > upper {
                    bound = upper;
                }
                if bound < self.bounds[query] {
                    self.bounds[query] = bound;
                    let mut cur = query;
                    while cur > 0 {
                        let p = (cur - 1) / 2;
                        let new_bound = self.bound(p);
                        if new_bound >= self.bounds[p] {
                            break;
                        }
                        self.bounds[p] = new_bound;
                        cur = p;
                    }
                }
            }
            (None, Some((left, right)), _)
            | (_, Some((left, right)), Some(std::cmp::Ordering::Less)) => {
                let left_bound = self.db.node_distance_lower_bound(query, left);
                let right_bound = self.db.node_distance_lower_bound(query, right);

                if left_bound < right_bound {
                    self.traversal(query, left);
                    self.traversal(query, right);
                } else {
                    self.traversal(query, right);
                    self.traversal(query, left);
                }
            }
            (Some((left, right)), _, _) => {
                let left_bound = self.db.node_distance_lower_bound(reference, left);
                let right_bound = self.db.node_distance_lower_bound(reference, right);
                if left_bound < right_bound {
                    self.traversal(left, reference);
                    self.traversal(right, reference);
                } else {
                    self.traversal(right, reference);
                    self.traversal(left, reference);
                }
            }
        }
    }

    fn reset_bounds(&mut self) {
        self.bounds.iter_mut().for_each(|v| *v = A::max_value());
    }

    #[inline]
    fn lower_bound(&self, node: usize, parent: usize) -> A {
        let diff = self.db.radius_of(parent) - self.db.radius_of(node);
        self.bounds[node] + diff + diff
    }

    #[inline]
    fn bound(&self, parent: usize) -> A {
        let left = 2 * parent + 1;
        let right = left + 1;

        let upper = if self.bounds[left] > self.bounds[right] {
            self.bounds[left]
        } else {
            self.bounds[right]
        };

        let lower_left = self.lower_bound(left, parent);
        let lower_right = self.lower_bound(right, parent);
        let lower = if lower_left > lower_right {
            lower_right
        } else {
            lower_left
        };

        if lower > A::zero() && lower < upper {
            lower
        } else {
            upper
        }
    }
}

// core_distances: distance of center to min_samples' closest point (including the center).
fn compute_core_distances<A, M>(
    db: &BallTree<A, M>,
    min_samples: usize,
    candidates: &mut Candidates<A>,
) -> Array1<A>
where
    A: AddAssign + DivAssign + FromPrimitive + FloatCore + Sync + Send,
    M: Metric<A> + Sync + Send,
{
    let mut knn_indices = vec![0; db.points.nrows() * min_samples];
    let mut core_distances = vec![A::zero(); db.points.nrows()];
    let rows: Vec<(usize, (&mut [usize], &mut A))> = knn_indices
        .chunks_mut(min_samples)
        .zip(core_distances.iter_mut())
        .enumerate()
        .collect();
    rows.into_par_iter().for_each(|(i, (indices, dist))| {
        let row = db.points.row(i);
        let (idx, d) = db.query(&row, min_samples);
        indices.clone_from_slice(&idx);
        *dist = *d.last().expect("ball tree query failed");
    });

    knn_indices
        .chunks_exact(min_samples)
        .enumerate()
        .for_each(|(n, row)| {
            for val in row.iter().skip(1).rev() {
                if core_distances[*val] <= core_distances[n] {
                    candidates.update(n, (n, *val, core_distances[n]));
                }
            }
        });

    Array1::from_vec(core_distances)
}

#[allow(dead_code)]
struct Candidates<A> {
    points: Vec<u32>,
    neighbors: Vec<u32>,
    distances: Vec<A>,
}

#[allow(dead_code)]
impl<A: FloatCore> Candidates<A> {
    fn new(n: usize) -> Self {
        // define max_value as NULL
        let neighbors = vec![u32::MAX; n];
        // define max_value as NULL
        let points = vec![u32::MAX; n];
        // define max_value as infinite far
        let distances = vec![A::max_value(); n];
        Self {
            points,
            neighbors,
            distances,
        }
    }

    fn get(&self, i: usize) -> Option<(usize, usize, A)> {
        if self.is_undefined(i) {
            None
        } else {
            Some((
                usize::try_from(self.points[i]).expect("fail to convert points"),
                usize::try_from(self.neighbors[i]).expect("fail to convert neighbor"),
                self.distances[i],
            ))
        }
    }

    fn update(&mut self, i: usize, val: (usize, usize, A)) {
        self.distances[i] = val.2;
        self.points[i] = u32::try_from(val.0).expect("candidate index overflow");
        self.neighbors[i] = u32::try_from(val.1).expect("candidate index overflow");
    }

    fn reset(&mut self, i: usize) {
        self.points[i] = u32::MAX;
        self.neighbors[i] = u32::MAX;
        self.distances[i] = A::max_value();
    }

    fn is_undefined(&self, i: usize) -> bool {
        self.points[i] == u32::MAX || self.neighbors[i] == u32::MAX
    }
}

#[allow(dead_code)]
struct Components {
    point: Vec<usize>,
    node: Vec<u32>,
    uf: TreeUnionFind,
}

#[allow(dead_code)]
impl Components {
    fn new(m: usize, n: usize) -> Self {
        // each point started as its own component.
        let point = (0..n).collect();
        // the component of the node is concluded when
        // all the enclosed points are in the same component
        let node = vec![u32::MAX; m];
        let uf = TreeUnionFind::new(n);
        Self { point, node, uf }
    }

    fn add(&mut self, src: usize, sink: usize) -> Option<()> {
        let current_src = self.uf.find(src);
        let current_sink = self.uf.find(sink);
        if current_src == current_sink {
            return None;
        }
        self.uf.union(current_src, current_sink);
        Some(())
    }

    fn update_points(&mut self) {
        for i in 0..self.point.len() {
            self.point[i] = self.uf.find(i);
        }
    }

    fn get_current(&self) -> Vec<usize> {
        self.uf.components()
    }

    fn len(&self) -> usize {
        self.uf.num_components()
    }
}

mod test {
    #[test]
    fn hdbscan32() {
        use ndarray::{array, Array2};
        use petal_neighbors::distance::Euclidean;

        use crate::Fit;

        let data: Array2<f32> = array![
            [1.0, 2.0],
            [1.1, 2.2],
            [0.9, 1.9],
            [1.0, 2.1],
            [-2.0, 3.0],
            [-2.2, 3.1],
        ];
        let mut hdbscan = super::HDbscan {
            alpha: 1.,
            min_samples: 2,
            min_cluster_size: 2,
            metric: Euclidean::default(),
            boruvka: false,
        };
        let (clusters, outliers, _) = hdbscan.fit(&data);
        assert_eq!(clusters.len(), 2);
        assert_eq!(
            outliers.len(),
            data.nrows() - clusters.values().fold(0, |acc, v| acc + v.len())
        );
    }

    #[test]
    fn hdbscan64() {
        use ndarray::{array, Array2};
        use petal_neighbors::distance::Euclidean;

        use crate::Fit;

        let data: Array2<f64> = array![
            [1.0, 2.0],
            [1.1, 2.2],
            [0.9, 1.9],
            [1.0, 2.1],
            [-2.0, 3.0],
            [-2.2, 3.1],
        ];
        let mut hdbscan = super::HDbscan {
            alpha: 1.,
            min_samples: 2,
            min_cluster_size: 2,
            metric: Euclidean::default(),
            boruvka: false,
        };
        let (clusters, outliers, _) = hdbscan.fit(&data);
        assert_eq!(clusters.len(), 2);
        assert_eq!(
            outliers.len(),
            data.nrows() - clusters.values().fold(0, |acc, v| acc + v.len())
        );
    }

    #[test]
    fn mst_linkage() {
        use ndarray::{arr1, arr2};
        use petal_neighbors::distance::Euclidean;
        //  0, 1, 2, 3, 4, 5, 6
        // {A, B, C, D, E, F, G}
        // {AB = 7, AD = 5,
        //  BC = 8, BD = 9, BE = 7,
        //  CB = 8, CE = 5,
        //  DB = 9, DE = 15, DF = 6,
        //  EF = 8, EG = 9
        //  FG = 11}
        let input = arr2(&[
            [0., 0.],
            [7., 0.],
            [15., 0.],
            [0., -5.],
            [15., -5.],
            [7., -7.],
            [15., -14.],
        ]);
        let core_distances = arr1(&[5., 7., 5., 5., 5., 6., 9.]);
        let mst = super::mst_linkage(
            input.view(),
            &Euclidean::default(),
            core_distances.view(),
            1.,
        );
        let answer = arr1(&[
            (0, 3, 5.),
            (0, 1, 7.),
            (1, 5, 7.),
            (1, 2, 8.),
            (2, 4, 5.),
            (4, 6, 9.),
        ]);
        assert_eq!(mst, answer);
    }

    #[test]
    fn boruvka() {
        use ndarray::{arr1, arr2};
        use petal_neighbors::{distance::Euclidean, BallTree};

        let input = arr2(&[
            [0., 0.],
            [7., 0.],
            [15., 0.],
            [0., -5.],
            [15., -5.],
            [7., -7.],
            [15., -14.],
        ]);

        let db = BallTree::new(input, Euclidean::default()).unwrap();
        let boruvka = super::Boruvka::new(db, 2);
        let mst = boruvka.min_spanning_tree();

        let answer = arr1(&[
            (0, 3, 5.0),
            (1, 0, 7.0),
            (2, 4, 5.0),
            (5, 1, 7.0),
            (6, 4, 9.0),
            (1, 2, 8.0),
        ]);
        assert_eq!(answer, mst);
    }

    #[test]
    fn outlier_scores() {
        use ndarray::array;
        use petal_neighbors::distance::Euclidean;

        use crate::Fit;

        let data = array![
            // cluster1:
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 2.0],
            // cluster2:
            [4.0, 1.0],
            [4.0, 2.0],
            [5.0, 1.0],
            [5.0, 2.0],
            // cluster3:
            [9.0, 1.0],
            [9.0, 2.0],
            [10.0, 1.0],
            [10.0, 2.0],
            [11.0, 1.0],
            [11.0, 2.0],
            // outlier1:
            [2.0, 5.0],
            // outlier2:
            [10.0, 8.0],
        ];
        let mut hdbscan = super::HDbscan {
            alpha: 1.,
            min_samples: 4,
            min_cluster_size: 4,
            metric: Euclidean::default(),
            boruvka: false,
        };
        let (_, _, outlier_scores) = hdbscan.fit(&data);

        // The first 14 data objects immediately form their clusters at eps = √2
        // The outlier scores of these objects are all 0:
        //      glosh(x) = 1 - √2 / √2 = 0
        for i in 0..14 {
            assert_eq!(outlier_scores[i], 0.0);
        }

        // Outlier1 joins the cluster C = {cluster1 ∪ cluster2} at:
        //      eps_outlier1 = √13
        // The lowest eps that C or any of its child clusters survives w.r.t. min_cluster_size = 4 is:
        //      eps_C = √2 (due to cluster1 or cluster2)
        // Then the outlier score of outlier1 is:
        //      glosh(outlier1) =  1 - √2 / √13 = 0.60776772972
        assert_eq!(outlier_scores[14], 1.0 - 2.0_f64.sqrt() / 13.0_f64.sqrt());

        // Outlier2 joins the root cluster at at eps = √37
        // The lowest eps that the root cluster survives w.r.t. min_cluster_size = 4 is:
        //      eps_root = √2
        // Then the outlier score of outlier2 is:
        //      glosh(outlier2) =  1 - √2 / √37 = 0.76750472251
        assert_eq!(outlier_scores[15], 1.0 - 2.0_f64.sqrt() / 37.0_f64.sqrt());
    }

    #[test]
    fn tree_union_find() {
        use succinct::{BitVecMut, BitVector};

        let parent = vec![0, 0, 1, 2, 4];
        let size = vec![0; 5];
        let is_component = BitVector::with_fill(5, true);
        let mut uf = super::TreeUnionFind {
            parent,
            size,
            is_component,
        };
        assert_eq!(0, uf.find(3));
        assert_eq!(vec![0, 0, 0, 0, 4], uf.parent);
        uf.union(4, 0);
        assert_eq!(vec![4, 0, 0, 0, 4], uf.parent);
        assert_eq!(vec![0, 0, 0, 0, 1], uf.size);
        let mut bv = BitVector::with_fill(5, false);
        bv.set_bit(0, true);
        bv.set_bit(4, true);
        assert_eq!(bv, uf.is_component);
        assert_eq!(vec![0, 4], uf.components());

        uf = super::TreeUnionFind::new(3);
        assert_eq!((0..3).collect::<Vec<_>>(), uf.parent);
        assert_eq!(vec![0; 3], uf.size);
    }

    #[test]
    fn union_find() {
        let mut uf = super::UnionFind::new(7);
        let pairs = vec![(0, 3), (4, 2), (3, 5), (0, 1), (1, 4), (4, 6)];
        let uf_res: Vec<_> = pairs
            .into_iter()
            .map(|(l, r)| {
                let ll = uf.fast_find(l);
                let rr = uf.fast_find(r);
                (ll, rr, uf.union(ll, rr))
            })
            .collect();
        assert_eq!(
            uf_res,
            vec![
                (0, 3, 2),
                (4, 2, 2),
                (7, 5, 3),
                (9, 1, 4),
                (10, 8, 6),
                (11, 6, 7)
            ]
        )
    }

    #[test]
    fn label() {
        use ndarray::arr1;
        let mst = arr1(&[
            (0, 3, 5.),
            (4, 2, 5.),
            (3, 5, 6.),
            (0, 1, 7.),
            (1, 4, 7.),
            (4, 6, 9.),
        ]);
        let labeled_mst = super::label(mst);
        assert_eq!(
            labeled_mst,
            arr1(&[
                (0, 3, 5., 2),
                (4, 2, 5., 2),
                (7, 5, 6., 3),
                (9, 1, 7., 4),
                (10, 8, 7., 6),
                (11, 6, 9., 7)
            ])
        );
    }

    #[test]
    fn bfs_mst() {
        use ndarray::arr1;
        let mst = arr1(&[
            (0, 3, 5., 2),
            (4, 2, 5., 2),
            (7, 5, 6., 3),
            (9, 1, 7., 4),
            (10, 8, 7., 6),
            (11, 6, 9., 7),
        ]);
        let root = mst.len() * 2;
        let bfs = super::bfs_mst(mst.view(), root);
        assert_eq!(bfs, [12, 11, 6, 10, 8, 9, 1, 4, 2, 7, 5, 0, 3]);

        let bfs = super::bfs_mst(mst.view(), 11);
        assert_eq!(bfs, vec![11, 10, 8, 9, 1, 4, 2, 7, 5, 0, 3]);

        let bfs = super::bfs_mst(mst.view(), 8);
        assert_eq!(bfs, vec![8, 4, 2]);
    }

    #[test]
    fn condense_mst() {
        use ndarray::arr1;

        let mst = arr1(&[
            (0, 3, 5., 2),
            (4, 2, 5., 2),
            (7, 5, 6., 3),
            (9, 1, 7., 4),
            (10, 8, 7., 6),
            (11, 6, 9., 7),
        ]);

        let condensed_mst = super::condense_mst(mst.view(), 3);
        assert_eq!(
            condensed_mst,
            vec![
                (7, 6, 1. / 9., 1),
                (7, 4, 1. / 7., 1),
                (7, 2, 1. / 7., 1),
                (7, 1, 1. / 7., 1),
                (7, 5, 1. / 6., 1),
                (7, 0, 1. / 6., 1),
                (7, 3, 1. / 6., 1)
            ],
        );
    }

    #[test]
    fn get_stability() {
        use std::collections::HashMap;

        use ndarray::arr1;

        let condensed = arr1(&[
            (7, 6, 1. / 9., 1),
            (7, 4, 1. / 7., 1),
            (7, 2, 1. / 7., 1),
            (7, 1, 1. / 7., 1),
            (7, 5, 1. / 6., 1),
            (7, 0, 1. / 6., 1),
            (7, 3, 1. / 6., 1),
        ]);
        let stability_map = super::get_stability(&condensed.view());
        let mut answer = HashMap::new();
        answer.insert(7, 1. / 9. + 3. / 7. + 3. / 6.);
        assert_eq!(stability_map, answer);
    }
}
