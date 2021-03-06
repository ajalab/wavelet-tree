use std::fmt;

use fid::{BitVector, FID};
use num_traits::Num;
use std::ops::{BitOr, Shl};

pub struct WaveletMatrix<T> {
    rows: Vec<BitVector>,
    size: u64,
    len: u64,
    partitions: Vec<u64>,
    _t: std::marker::PhantomData<T>,
}

impl<T> WaveletMatrix<T>
where
    T: Into<u64> + Copy + Clone + Num + BitOr<T, Output = T> + Shl<u64, Output = T>,
{
    pub fn new_with_size<K: AsRef<[T]>>(text: K, size: u64) -> Self {
        let mut rows: Vec<BitVector> = vec![];
        let mut zeros: Vec<T> = text.as_ref().to_vec();
        let mut ones: Vec<T> = Vec::new();
        let mut partitions: Vec<u64> = Vec::new();
        for r in 0..size {
            let mut bv = BitVector::new();
            let mut new_zeros: Vec<T> = Vec::new();
            let mut new_ones: Vec<T> = Vec::new();
            for arr in &[zeros, ones] {
                for &c in arr {
                    let b = c.into();
                    let bit = (b >> (size - r - 1)) & 1 > 0;
                    if bit {
                        new_ones.push(c);
                    } else {
                        new_zeros.push(c);
                    }
                    bv.push(bit);
                }
            }
            zeros = new_zeros;
            ones = new_ones;
            rows.push(bv);
            partitions.push(zeros.len() as u64);
        }
        WaveletMatrix {
            rows: rows,
            size: size,
            len: text.as_ref().len() as u64,
            partitions: partitions,
            _t: std::marker::PhantomData::<T>,
        }
    }

    pub fn new<K: AsRef<[T]>>(text: K) -> Self {
        Self::new_with_size(text, std::mem::size_of::<T>() as u64 * 8)
    }

    pub fn access(&self, k: u64) -> T {
        let mut i = k;
        let mut n = T::zero();
        for (r, bv) in self.rows.iter().enumerate() {
            let b = bv.get(i);
            if b {
                i = self.partitions[r] + bv.rank1(i);
                n = n | (T::one() << (self.size - (r as u64) - 1));
            } else {
                i = bv.rank0(i);
            }
        }
        n
    }

    pub fn rank(&self, c: T, k: u64) -> u64 {
        let n = c.into();
        let mut s = 0u64;
        let mut e = if k < self.len { k } else { self.len };
        for (r, bv) in self.rows.iter().enumerate() {
            let b = (n >> (self.size - (r as u64) - 1)) & 1 > 0;
            s = bv.rank(b, s);
            e = bv.rank(b, e);
            if b {
                let z = self.partitions[r];
                s = s + z;
                e = e + z;
            }
        }
        e - s
    }

    pub fn select(&self, c: T, k: u64) -> u64 {
        let n = c.into();
        let mut s = 0u64;
        for (r, bv) in self.rows.iter().enumerate() {
            let b = (n >> (self.size - (r as u64) - 1)) & 1 > 0;
            s = bv.rank(b, s);
            if b {
                let z = self.partitions[r];
                s = s + z;
            }
        }
        let mut e = s + k;
        for (r, bv) in self.rows.iter().enumerate().rev() {
            let b = (n >> (self.size - (r as u64) - 1)) & 1 > 0;
            if b {
                let z = self.partitions[r];
                e = bv.select1(e - z);
            } else {
                e = bv.select0(e);
            }
        }
        e
    }

    pub fn len(&self) -> u64 {
        self.len
    }
}

impl<T: fmt::Debug> fmt::Debug for WaveletMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let len = self.rows[0].len();
        writeln!(f, "WaveletMatrix {{")?;
        for bv in &self.rows {
            write!(f, "  ")?;
            for i in 0..len {
                write!(f, "{}", if bv.get(i) { "1" } else { "0" })?;
            }
            writeln!(f, "")?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rank_small() {
        let numbers = &[4u8, 7, 6, 5, 3, 2, 1, 0, 1, 4, 1, 7];
        let size = 3;
        let wm = WaveletMatrix::new_with_size(numbers, size);
        assert_eq!(wm.len, numbers.len() as u64);
        for i in 0..(1 << size) {
            let mut r = 0;
            for (k, &n) in numbers.iter().enumerate() {
                assert!(
                    wm.rank(i as u8, k as u64) == r,
                    "wm.rank({}, {}) == {}",
                    i,
                    k,
                    r
                );
                if n == i {
                    r = r + 1;
                }
            }
        }
    }

    #[test]
    fn access_small() {
        let numbers = &[4u8, 7, 6, 5, 3, 2, 1, 0, 1, 4, 1, 7];
        let size = 3;
        let wm = WaveletMatrix::new_with_size(numbers, size);
        assert_eq!(wm.len, numbers.len() as u64);
        for (i, &n) in numbers.iter().enumerate() {
            assert!(wm.access(i as u64) == n, "wm.access({}) == {}", i, n);
        }
    }

    #[test]
    fn select_small() {
        let numbers = &[4u8, 7, 6, 5, 3, 2, 1, 0, 1, 4, 1, 7];
        let size = 3;
        let wm = WaveletMatrix::new_with_size(numbers, size);

        let mut ans: Vec<Vec<u64>> = vec![vec![]; 1 << size];
        for (i, &n) in numbers.iter().enumerate() {
            ans[n as usize].push(i as u64);
        }

        for (c, a) in ans.iter().enumerate() {
            for (k, &i) in a.iter().enumerate() {
                assert!(
                    wm.select(c as u8, k as u64) == i,
                    "wm.select({}, {}) == {}",
                    c,
                    k,
                    i
                );
            }
        }
    }

    #[test]
    fn empty() {
        let empty_vec: Vec<u8> = vec![];
        let wm = WaveletMatrix::new(&empty_vec);
        assert_eq!(wm.len, 0);
        assert_eq!(wm.rank(0u8, 0), 0);
        assert_eq!(wm.rank(0u8, 10), 0);
        assert_eq!(wm.rank(1u8, 0), 0);
        assert_eq!(wm.rank(1u8, 10), 0);
    }
}
