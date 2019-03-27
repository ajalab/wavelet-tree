use fid::{BitVector, FID};
use std::fmt;

use fid::{BitVector, FID};
use try_from::TryFrom;

pub struct WaveletMatrix {
    rows: Vec<BitVector>,
    size: usize,
    len: usize,
}

impl WaveletMatrix {
    pub fn new_with_size<T: Into<usize> + Copy + Clone, K: AsRef<[T]>>(
        text: K,
        size: usize,
    ) -> Self {
        let mut rows: Vec<BitVector> = vec![];
        let mut zeros: Vec<T> = text.as_ref().to_vec();
        let mut ones: Vec<T> = Vec::new();
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
        }
        WaveletMatrix {
            rows: rows,
            size: size,
            len: text.as_ref().len(),
        }
    }

    pub fn new<T: Into<usize> + Copy + Clone, K: AsRef<[T]>>(text: K) -> Self {
        Self::new_with_size(text, std::mem::size_of::<T>() * 8)
    }

    pub fn access<T: TryFrom<usize> + Copy + Clone>(self: &Self, k: usize) -> T
    where
        <T as try_from::TryFrom<usize>>::Err: std::fmt::Debug,
    {
        let mut i = k as u64;
        let mut n = 0usize;
        for (r, bv) in self.rows.iter().enumerate() {
            let b = bv.get(i);
            if b {
                i = bv.rank0(self.len as u64) + bv.rank1(i);
                n |= 1 << (self.size - r - 1);
            } else {
                i = bv.rank0(i);
            }
        }
        TryFrom::try_from(n).unwrap()
    }

    pub fn rank<T: Into<usize> + Copy + Clone>(self: &Self, c: T, k: usize) -> usize {
        let n = c.into();
        let mut s = 0u64;
        let mut e = if k < self.len { k } else { self.len } as u64;
        for (r, bv) in self.rows.iter().enumerate() {
            let b = (n >> (self.size - r - 1)) & 1 > 0;
            s = bv.rank(b, s);
            e = bv.rank(b, e);
            if b {
                let l = bv.rank0(bv.len());
                s = s + l;
                e = e + l;
            }
        }
        (e - s) as usize
    }
}

impl fmt::Debug for WaveletMatrix {
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
    fn it_works() {
        let wm = WaveletMatrix::new("abc");
        wm.rank(b'a', 3);
    }

    #[test]
    fn numbers() {
        let numbers = &[4u8, 7, 6, 5, 3, 2, 1, 0, 1, 4, 1, 7];
        let size = 3;
        let wm = WaveletMatrix::new_with_size(numbers, size);
        assert_eq!(wm.len, numbers.len());
        for i in 0..(1 << size) {
            let mut r = 0;
            for (k, &n) in numbers.iter().enumerate() {
                assert!(wm.rank(i as u8, k) == r, "wm.rank({}, {}) == {}", i, k, r);
                if n == i {
                    r = r + 1;
                }
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
