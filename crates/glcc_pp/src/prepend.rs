//! Rescan queue for the preprocessor: an iterator
//! with `prepend_extend`, so substituted tokens go back to the front of
//! the stream for rescanning.

use std::{collections::VecDeque, fmt::Debug};

#[derive(Debug)]
pub struct PrependingPeekableIterator<I: Iterator + Debug> {
    queue: VecDeque<I::Item>,
    inner: I,
}

impl<I: Iterator + Debug> Iterator for PrependingPeekableIterator<I>
where
    I::Item: Debug,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.queue.pop_front() {
            return Some(item);
        } else {
            self.inner.next()
        }
    }
}

impl<I: Iterator + Debug> PrependingPeekableIterator<I>
where
    I::Item: Debug,
{
    pub fn new(i: I) -> Self {
        Self { queue: VecDeque::new(), inner: i }
    }

    pub fn prepend_extend(&mut self, iter: impl Iterator<Item = I::Item>) {
        // Preserve order: first item ends up front.
        let mut items: Vec<I::Item> = iter.collect();
        for item in items.drain(..).rev() {
            self.queue.push_front(item);
        }
    }

    /// Items still buffered (queue plus untouched inner input).
    pub fn remaining(&self) -> usize
    where
        I: ExactSizeIterator,
    {
        self.queue.len() + self.inner.len()
    }
}
