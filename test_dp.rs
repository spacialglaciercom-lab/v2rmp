use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Copy, Clone, PartialEq)]
struct State { cost: f64, position: usize }
impl Eq for State {}
impl Ord for State { fn cmp(&self, other: &Self) -> Ordering { other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal) } }
impl PartialOrd for State { fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) } }

fn main() {
    let num_odd = 4;
    let dist_matrix = vec![vec![0.0, 1.0, 2.0, 3.0], vec![1.0, 0.0, 4.0, 5.0], vec![2.0, 4.0, 0.0, 6.0], vec![3.0, 5.0, 6.0, 0.0]];
    let mut memo = vec![f64::MAX; 1 << num_odd];
    let mut parent = vec![usize::MAX; 1 << num_odd];
    memo[0] = 0.0;
    
    for mask in 0..(1 << num_odd) {
        if memo[mask] == f64::MAX { continue; }
        let mut i = 0;
        while i < num_odd {
            if (mask & (1 << i)) == 0 { break; }
            i += 1;
        }
        if i == num_odd { continue; }
        
        for j in (i + 1)..num_odd {
            if (mask & (1 << j)) == 0 {
                let next_mask = mask | (1 << i) | (1 << j);
                let new_cost = memo[mask] + dist_matrix[i][j];
                if new_cost < memo[next_mask] {
                    memo[next_mask] = new_cost;
                    parent[next_mask] = mask;
                }
            }
        }
    }
    
    let mut pairs = Vec::new();
    let mut curr = (1 << num_odd) - 1;
    while curr > 0 {
        let prev = parent[curr];
        if prev == usize::MAX { break; }
        let diff = curr ^ prev;
        let mut u = usize::MAX;
        let mut v = usize::MAX;
        for i in 0..num_odd {
            if (diff & (1 << i)) != 0 {
                if u == usize::MAX { u = i; } else { v = i; }
            }
        }
        pairs.push((u, v));
        curr = prev;
    }
    println!("{:?}", pairs);
}
