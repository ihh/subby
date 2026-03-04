/// Tree structure and utilities.

/// Validate that every non-leaf node has exactly 2 children.
pub fn validate_binary_tree(parent_index: &[i32]) {
    let r = parent_index.len();
    let mut child_count = vec![0u32; r];
    for n in 1..r {
        child_count[parent_index[n] as usize] += 1;
    }
    for n in 0..r {
        assert!(
            child_count[n] == 0 || child_count[n] == 2,
            "Node {} has {} children; expected 0 or 2",
            n,
            child_count[n]
        );
    }
}

/// Compute left_child, right_child, sibling arrays.
/// Returns (left_child, right_child, sibling), each of length R.
/// -1 indicates no child or no sibling.
pub fn children_of(parent_index: &[i32]) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let r = parent_index.len();
    let mut left_child = vec![-1i32; r];
    let mut right_child = vec![-1i32; r];
    let mut sibling = vec![-1i32; r];

    for n in 1..r {
        let p = parent_index[n] as usize;
        if left_child[p] == -1 {
            left_child[p] = n as i32;
        } else {
            right_child[p] = n as i32;
        }
    }

    for n in 1..r {
        let p = parent_index[n] as usize;
        if left_child[p] == n as i32 {
            sibling[n] = right_child[p];
        } else {
            sibling[n] = left_child[p];
        }
    }

    (left_child, right_child, sibling)
}
