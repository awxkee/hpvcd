/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
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

use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut, Range};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

pub(crate) struct DisjointMut<T> {
    inner: UnsafeCell<Vec<T>>,
    #[cfg(debug_assertions)]
    borrows: Mutex<Vec<Range<usize>>>,
}

unsafe impl<T: Send> Sync for DisjointMut<T> {}
unsafe impl<T: Send> Send for DisjointMut<T> {}

impl<T> DisjointMut<T> {
    pub(crate) fn new(v: Vec<T>) -> Self {
        DisjointMut {
            inner: UnsafeCell::new(v),
            #[cfg(debug_assertions)]
            borrows: Mutex::new(Vec::new()),
        }
    }

    /// Number of elements in the underlying buffer.
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        // SAFETY: reading the length does not alias any element storage.
        unsafe { (*self.inner.get()).len() }
    }

    /// Borrow `range` as `&mut [T]`.
    pub(crate) fn slice_mut(&self, range: Range<usize>) -> DisjointMutGuard<'_, T> {
        // SAFETY: bounds are validated here; the disjointness of `range` against
        // other live borrows is the caller's contract (checked below in debug).
        let vec = unsafe { &mut *self.inner.get() };
        assert!(
            range.end <= vec.len() && range.start <= range.end,
            "DisjointMut::slice_mut range {range:?} out of bounds (len {})",
            vec.len()
        );

        #[cfg(debug_assertions)]
        {
            // Determine overlap while holding the lock, but drop the guard
            // *before* asserting so a panic never poisons the mutex (which would
            // in turn make the guard's `Drop` panic during unwinding).
            let conflict = {
                let mut live = self.borrows.lock().unwrap_or_else(|p| p.into_inner());
                let clash = live
                    .iter()
                    .find(|other| range.start < other.end && other.start < range.end)
                    .cloned();
                if clash.is_none() {
                    live.push(range.clone());
                }
                clash
            };
            assert!(
                conflict.is_none(),
                "DisjointMut: overlapping borrow {range:?} vs live {:?}",
                conflict.unwrap()
            );
        }

        let ptr = vec.as_mut_ptr();
        // SAFETY: `range` is in bounds and (by contract / debug check) disjoint
        // from every other live borrow, so this `&mut` uniquely owns its region.
        let slice = unsafe {
            std::slice::from_raw_parts_mut(ptr.add(range.start), range.end - range.start)
        };
        DisjointMutGuard {
            slice,
            #[cfg(debug_assertions)]
            parent: self,
            #[cfg(debug_assertions)]
            range,
        }
    }

    /// Consume the wrapper and return the underlying buffer.
    pub(crate) fn into_inner(self) -> Vec<T> {
        self.inner.into_inner()
    }
}

/// RAII guard for a borrowed region of a [`DisjointMut`].
pub(crate) struct DisjointMutGuard<'a, T> {
    slice: &'a mut [T],
    #[cfg(debug_assertions)]
    parent: &'a DisjointMut<T>,
    #[cfg(debug_assertions)]
    range: Range<usize>,
}

impl<T> Deref for DisjointMutGuard<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.slice
    }
}

impl<T> DerefMut for DisjointMutGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.slice
    }
}

#[cfg(debug_assertions)]
impl<T> Drop for DisjointMutGuard<'_, T> {
    fn drop(&mut self) {
        // Recover from poisoning: even if another borrow panicked, we must not
        // panic again here (that would abort the process during unwinding).
        let mut live = self
            .parent
            .borrows
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if let Some(pos) = live.iter().position(|r| *r == self.range) {
            live.swap_remove(pos);
        }
    }
}

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Deque {
    jobs: Mutex<VecDeque<Job>>,
}

impl Deque {
    fn new() -> Self {
        Deque {
            jobs: Mutex::new(VecDeque::new()),
        }
    }
    fn push(&self, job: Job) {
        self.jobs.lock().unwrap().push_back(job);
    }
    fn pop(&self) -> Option<Job> {
        self.jobs.lock().unwrap().pop_back()
    }
    fn steal(&self) -> Option<Job> {
        self.jobs.lock().unwrap().pop_front()
    }
}

/// Shared state every worker sees.
struct Shared {
    deques: Vec<Deque>,
    /// Overflow / external submissions land here and are drained by any worker.
    injector: Mutex<VecDeque<Job>>,
    /// Number of jobs not yet finished across all queues, for `wait_idle`.
    pending: AtomicUsize,
    /// Signalled when new work arrives or a job completes.
    cvar: Condvar,
    /// Paired with `cvar`; guards the "there might be work / progress" signal.
    lock: Mutex<()>,
    shutdown: AtomicBool,
}

impl Shared {
    /// Try to obtain one job: own deque first, then steal round-robin, then the
    /// injector. `me` is the calling worker's index (or `deques.len()` for the
    /// external submitter, which owns no deque).
    fn find_job(&self, me: usize) -> Option<Job> {
        if let Some(d) = self.deques.get(me)
            && let Some(j) = d.pop()
        {
            return Some(j);
        }
        let n = self.deques.len();
        for off in 1..=n {
            let victim = (me + off) % n;
            if victim == me {
                continue;
            }
            if let Some(j) = self.deques[victim].steal() {
                return Some(j);
            }
        }
        self.injector.lock().unwrap().pop_front()
    }

    fn notify_work(&self) {
        let _g = self.lock.lock().unwrap();
        self.cvar.notify_all();
    }

    fn finish_one(&self) {
        // Release ordering pairs with the Acquire load in `wait_idle`.
        if self.pending.fetch_sub(1, Ordering::Release) == 1 {
            let _g = self.lock.lock().unwrap();
            self.cvar.notify_all();
        }
    }
}

/// A persistent work-stealing thread pool.
pub(crate) struct ThreadPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    round_robin: AtomicUsize,
}

impl ThreadPool {
    /// Build a pool with `threads` workers (clamped to at least 1).
    pub(crate) fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let mut deques = Vec::with_capacity(threads);
        for _ in 0..threads {
            deques.push(Deque::new());
        }
        let shared = Arc::new(Shared {
            deques,
            injector: Mutex::new(VecDeque::new()),
            pending: AtomicUsize::new(0),
            cvar: Condvar::new(),
            lock: Mutex::new(()),
            shutdown: AtomicBool::new(false),
        });

        let mut workers = Vec::with_capacity(threads);
        for id in 0..threads {
            let shared = Arc::clone(&shared);
            let handle = thread::Builder::new()
                .name(format!("hpvcd-worker-{}", id))
                .spawn(move || worker_loop(shared, id))
                .expect("spawn worker thread");
            workers.push(handle);
        }

        ThreadPool {
            shared,
            workers,
            round_robin: AtomicUsize::new(0),
        }
    }

    /// Build a pool sized to the machine's available parallelism.
    pub(crate) fn with_available_parallelism() -> Self {
        let n = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        ThreadPool::new(n)
    }

    /// Number of worker threads.
    pub(crate) fn threads(&self) -> usize {
        self.workers.len()
    }

    /// Submit a `'static` job onto the least-recently-fed worker's deque.
    fn submit(&self, job: Job) {
        self.shared.pending.fetch_add(1, Ordering::Relaxed);
        let n = self.shared.deques.len();
        let idx = self.round_robin.fetch_add(1, Ordering::Relaxed) % n;
        self.shared.deques[idx].push(job);
        self.shared.notify_work();
    }

    /// Run `f` with a [`Scope`] that can spawn jobs borrowing stack data. All
    /// spawned jobs are guaranteed to finish before this returns. The calling
    /// thread also helps drain work while waiting, so it never blocks idly.
    pub(crate) fn scope<'scope, F, R>(&'scope self, f: F) -> R
    where
        F: FnOnce(&Scope<'scope>) -> R,
    {
        let scope = Scope {
            pool: self,
            outstanding: AtomicUsize::new(0),
            done: Condvar::new(),
            done_lock: Mutex::new(()),
        };
        let result = f(&scope);
        scope.wait();
        result
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        self.shared.notify_work();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn worker_loop(shared: Arc<Shared>, id: usize) {
    loop {
        if let Some(job) = shared.find_job(id) {
            job();
            shared.finish_one();
            continue;
        }
        if shared.shutdown.load(Ordering::SeqCst) {
            // Drain anything that raced in before shutting down.
            if let Some(job) = shared.find_job(id) {
                job();
                shared.finish_one();
                continue;
            }
            break;
        }
        // Nothing to do: park until notified.
        let guard = shared.lock.lock().unwrap();
        if shared.shutdown.load(Ordering::SeqCst) {
            break;
        }
        // Re-check for work before sleeping to avoid a lost-wakeup race.
        let has_work = shared.pending.load(Ordering::Acquire) > 0;
        if has_work {
            drop(guard);
            continue;
        }
        let _unused = shared
            .cvar
            .wait_timeout(guard, std::time::Duration::from_millis(1))
            .unwrap();
    }
}

/// A scope tied to a [`ThreadPool`]. Jobs spawned via [`Scope::spawn`] may
/// borrow data living at least as long as `'scope`; [`ThreadPool::scope`]
/// guarantees they all complete before the borrow ends.
pub(crate) struct Scope<'scope> {
    pool: &'scope ThreadPool,
    outstanding: AtomicUsize,
    done: Condvar,
    done_lock: Mutex<()>,
}

impl<'scope> Scope<'scope> {
    /// Spawn a job that may borrow `'scope` data.
    pub(crate) fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'scope,
    {
        self.outstanding.fetch_add(1, Ordering::SeqCst);

        // The scope out-lives every job (we join in `wait` before returning),
        // so widening the job lifetime to 'static for storage on the shared
        // queues is sound. `scope_ptr` lets the job signal completion.
        let scope_ptr: *const Scope<'scope> = self;
        let scope_addr = scope_ptr as usize;

        let job: Box<dyn FnOnce() + Send + 'scope> = Box::new(move || {
            f();
            // SAFETY: `wait` has not returned yet (it waits for `outstanding`
            // to hit zero, which this decrement drives), so `*scope_ptr` is
            // still a live, valid `Scope`.
            let scope = unsafe { &*(scope_addr as *const Scope<'scope>) };
            if scope.outstanding.fetch_sub(1, Ordering::SeqCst) == 1 {
                let _g = scope.done_lock.lock().unwrap();
                scope.done.notify_all();
            }
        });

        // SAFETY: transmute the job's lifetime to 'static for queue storage.
        // Soundness is upheld by `Scope::wait`, which blocks until every job
        // has run, so no job outlives the borrowed `'scope` data.
        let job: Job = unsafe {
            std::mem::transmute::<
                Box<dyn FnOnce() + Send + 'scope>,
                Box<dyn FnOnce() + Send + 'static>,
            >(job)
        };
        self.pool.submit(job);
    }

    /// Block until all spawned jobs finish. The calling thread helps by draining
    /// jobs itself (using the external-submitter index) so it is never idle
    /// while work remains.
    fn wait(&self) {
        let external = self.pool.shared.deques.len();
        loop {
            if self.outstanding.load(Ordering::SeqCst) == 0 {
                return;
            }
            if let Some(job) = self.pool.shared.find_job(external) {
                job();
                self.pool.shared.finish_one();
                continue;
            }
            // No stealable job right now, but some are still running on workers.
            let guard = self.done_lock.lock().unwrap();
            if self.outstanding.load(Ordering::SeqCst) == 0 {
                return;
            }
            let _unused = self
                .done
                .wait_timeout(guard, std::time::Duration::from_millis(1))
                .unwrap();
        }
    }
}

/// Run `body(i)` for every `i` in `0..count`, distributing indices across the
/// pool and joining before returning. `body` may borrow stack data.
pub(crate) fn parallel_for<F>(pool: &ThreadPool, count: usize, body: F)
where
    F: Fn(usize) + Send + Sync,
{
    if count == 0 {
        return;
    }
    if count == 1 || pool.threads() == 1 {
        for i in 0..count {
            body(i);
        }
        return;
    }
    let body = &body;
    pool.scope(|s| {
        for i in 0..count {
            s.spawn(move || body(i));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disjoint_regions_write_independently() {
        let dm = DisjointMut::new(vec![0u32; 8]);
        {
            let mut a = dm.slice_mut(0..4);
            let mut b = dm.slice_mut(4..8);
            for (i, x) in a.iter_mut().enumerate() {
                *x = i as u32;
            }
            for (i, x) in b.iter_mut().enumerate() {
                *x = 100 + i as u32;
            }
        }
        assert_eq!(dm.into_inner(), vec![0, 1, 2, 3, 100, 101, 102, 103]);
    }

    #[test]
    #[should_panic(expected = "overlapping borrow")]
    #[cfg(debug_assertions)]
    fn overlapping_borrow_panics() {
        let dm = DisjointMut::new(vec![0u8; 8]);
        let _a = dm.slice_mut(0..5);
        let _b = dm.slice_mut(3..8); // overlaps [3,5)
    }

    #[test]
    fn pool_parallel_for_disjoint_write() {
        let pool = ThreadPool::new(4);
        let dm = DisjointMut::new(vec![0usize; 16]);
        parallel_for(&pool, 4, |tile| {
            let mut region = dm.slice_mut(tile * 4..tile * 4 + 4);
            for (i, x) in region.iter_mut().enumerate() {
                *x = tile * 4 + i;
            }
        });
        let out = dm.into_inner();
        assert_eq!(out, (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn pool_reused_across_scopes() {
        let pool = ThreadPool::new(3);
        let sum = AtomicUsize::new(0);
        for _ in 0..5 {
            parallel_for(&pool, 10, |i| {
                sum.fetch_add(i, Ordering::Relaxed);
            });
        }
        assert_eq!(sum.load(Ordering::Relaxed), 45 * 5);
    }
}
