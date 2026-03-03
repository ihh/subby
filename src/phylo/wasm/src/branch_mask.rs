/// Branch masking via Steiner tree identification.

/// Compute branch mask: identify active branches per column.
/// Returns (R*C) flat bool as u8 (0 or 1).
pub fn compute_branch_mask(
    alignment: &[i32],
    parent_index: &[i32],
    a: usize,
    r: usize,
    c: usize,
) -> Vec<u8> {
    // Determine leaves
    let mut child_count = vec![0u32; r];
    for n in 1..r {
        child_count[parent_index[n] as usize] += 1;
    }
    let is_leaf: Vec<bool> = child_count.iter().map(|&cc| cc == 0).collect();

    // Ungapped leaf classification
    let mut is_ungapped_leaf = vec![false; r * c];
    for row in 0..r {
        for col in 0..c {
            let tok = alignment[row * c + col];
            is_ungapped_leaf[row * c + col] = is_leaf[row] && tok >= 0 && tok <= a as i32;
        }
    }

    // Upward: propagate "has ungapped descendant"
    let mut has_ungapped = is_ungapped_leaf.clone();
    for n in (1..r).rev() {
        let p = parent_index[n] as usize;
        for col in 0..c {
            if has_ungapped[n * c + col] {
                has_ungapped[p * c + col] = true;
            }
        }
    }

    // Count ungapped children
    let mut ungapped_child_count = vec![0u32; r * c];
    for n in 1..r {
        let p = parent_index[n] as usize;
        for col in 0..c {
            if has_ungapped[n * c + col] {
                ungapped_child_count[p * c + col] += 1;
            }
        }
    }

    // Steiner nodes
    let mut is_steiner = vec![false; r * c];
    for row in 0..r {
        for col in 0..c {
            is_steiner[row * c + col] =
                is_ungapped_leaf[row * c + col] || ungapped_child_count[row * c + col] >= 2;
        }
    }

    // Preorder propagation
    for n in 1..r {
        let p = parent_index[n] as usize;
        for col in 0..c {
            if is_steiner[p * c + col] && has_ungapped[n * c + col] {
                is_steiner[n * c + col] = true;
            }
        }
    }

    // Branch mask
    let mut mask = vec![0u8; r * c];
    for n in 1..r {
        let p = parent_index[n] as usize;
        for col in 0..c {
            mask[n * c + col] =
                if is_steiner[n * c + col] && is_steiner[p * c + col] { 1 } else { 0 };
        }
    }

    mask
}
