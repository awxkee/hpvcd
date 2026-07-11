/*
 * // Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//! # Safety contract for the `Shared` variant
//! Multiple `Shared` planes may alias one backing buffer **only** when the
//! wavefront guarantees the live `&mut` regions are disjoint at every instant.
//! The 2-CTB diagonal lag provides exactly that: at any moment, row `r` writes
//! CTB column `c` while row `r-1` is at least at column `c+2`, so their touched
//! pixels/grid cells never overlap. Reads of the row above (intra neighbors)
//! land in cells row `r-1` finished ≥2 CTBs ago, i.e. no longer being written.
//! Holding the pointer as `Send` is sound under this discipline; violating the
//! lag would be UB.

use std::ops::{Deref, DerefMut};

pub(crate) enum Plane<T> {
    Owned(Vec<T>),
    /// `(ptr, len)` aliasing a buffer owned by a `SharedPicture`. The pointer is
    /// valid for `len` elements for the lifetime of the wavefront decode.
    Shared(*mut T, usize),
}

// SAFETY: sending a `Shared` plane to a worker thread is sound because the
// wavefront's lag discipline keeps every thread's live `&mut` region disjoint
// (see module docs). `T: Send` ensures the elements themselves are movable.
unsafe impl<T: Send> Send for Plane<T> {}

impl<T> Plane<T> {
    #[inline]
    pub(crate) fn owned(v: Vec<T>) -> Self {
        Plane::Owned(v)
    }

    /// Construct a shared aliasing view. SAFETY: caller guarantees `ptr` is valid
    /// for `len` `T`s and that concurrent access stays disjoint per the lag
    /// discipline.
    #[inline]
    pub(crate) unsafe fn shared(ptr: *mut T, len: usize) -> Self {
        Plane::Shared(ptr, len)
    }

    /// Take the owned `Vec`, replacing `self` with an empty owned plane. Panics
    /// if the plane is a `Shared` view (which must never happen after the
    /// wavefront reclaims ownership before the serial filter stages).
    #[inline]
    pub(crate) fn take_vec(&mut self) -> Vec<T> {
        match std::mem::replace(self, Plane::Owned(Vec::new())) {
            Plane::Owned(v) => v,
            Plane::Shared(..) => panic!("take_vec on a Shared plane"),
        }
    }
}

impl<T: Clone> Plane<T> {
    /// Clone the plane's contents into a fresh `Vec` (used by SAO for the
    /// untouched-source snapshot). Works for both variants.
    #[inline]
    pub(crate) fn to_vec_clone(&self) -> Vec<T> {
        self.deref().to_vec()
    }
}

impl<T> Deref for Plane<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        match self {
            Plane::Owned(v) => v.as_slice(),
            // SAFETY: valid-for-`len` per the construction contract.
            Plane::Shared(p, len) => unsafe { std::slice::from_raw_parts(*p, *len) },
        }
    }
}

impl<T> DerefMut for Plane<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            Plane::Owned(v) => v.as_mut_slice(),
            // SAFETY: valid-for-`len`; disjointness upheld by the lag discipline.
            Plane::Shared(p, len) => unsafe { std::slice::from_raw_parts_mut(*p, *len) },
        }
    }
}

impl<T> Default for Plane<T> {
    fn default() -> Self {
        Plane::Owned(Vec::new())
    }
}
