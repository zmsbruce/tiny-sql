use std::collections::{btree_map, BTreeMap};

use super::Storage;
use crate::Result;

#[derive(Default)]
pub struct MemoryStorage {
    map: BTreeMap<Vec<u8>, Vec<u8>>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Storage for MemoryStorage {
    type Iterator<'a> = MemoryStorageIterator<'a>;

    fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        self.map.insert(key.to_vec(), value.to_vec());
        Ok(())
    }

    fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        Ok(self.map.get(key).cloned())
    }

    fn delete(&mut self, key: &[u8]) -> Result<()> {
        self.map.remove(key);
        Ok(())
    }

    fn scan<R>(&mut self, range: R) -> Self::Iterator<'_>
    where
        R: std::ops::RangeBounds<Vec<u8>>,
    {
        MemoryStorageIterator {
            inner: self.map.range(range),
        }
    }
}

pub struct MemoryStorageIterator<'a> {
    inner: btree_map::Range<'a, Vec<u8>, Vec<u8>>,
}

impl Iterator for MemoryStorageIterator<'_> {
    type Item = Result<(Vec<u8>, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| Ok((k.clone(), v.clone())))
    }
}

impl DoubleEndedIterator for MemoryStorageIterator<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner
            .next_back()
            .map(|(k, v)| Ok((k.clone(), v.clone())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_storage() {
        let mut storage = MemoryStorage::new();

        storage.put(b"key1", b"value1").unwrap();
        storage.put(b"key2", b"value2").unwrap();
        storage.put(b"key3", b"value3").unwrap();

        assert_eq!(storage.get(b"key1").unwrap().unwrap(), b"value1");
        assert_eq!(storage.get(b"key2").unwrap().unwrap(), b"value2");
        assert_eq!(storage.get(b"key3").unwrap().unwrap(), b"value3");

        storage.delete(b"key2").unwrap();
        assert_eq!(storage.get(b"key2").unwrap(), None);

        let mut iter = storage.scan(b"key1".to_vec()..=b"key3".to_vec());
        assert_eq!(
            iter.next().unwrap().unwrap(),
            (b"key1".to_vec(), b"value1".to_vec())
        );
        assert_eq!(
            iter.next().unwrap().unwrap(),
            (b"key3".to_vec(), b"value3".to_vec())
        );
        assert!(iter.next().is_none());
    }
}
