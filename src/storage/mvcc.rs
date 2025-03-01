use std::{
    collections::{BTreeMap, HashSet},
    ops::Add,
    sync::{Arc, Mutex, MutexGuard},
};

use serde::{Deserialize, Serialize};

use super::Storage;
use crate::{
    Error::{self, InternalError, WriteConflict},
    Result,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Version(u64);

impl Version {
    pub fn encode(&self) -> Result<Vec<u8>> {
        bincode::serialize(&self).map_err(|e| e.into())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).map_err(|e| e.into())
    }

    pub fn max() -> Self {
        Self(u64::MAX)
    }

    pub fn min() -> Self {
        Self(0)
    }
}

impl Add<u64> for Version {
    type Output = Self;

    fn add(self, rhs: u64) -> Self::Output {
        Self(self.0.checked_add(rhs).expect("version overflow"))
    }
}

impl From<u64> for Version {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

type Key = Vec<u8>;

/// MVCC 存储引擎的 key
///
/// - `NextVersion`: 下一个版本号
/// - `TxnActive`: 活跃事务
/// - `TxnWrite`: 事务写入记录，用于回滚事务
/// - `Version`: 版本记录，用于事务的可见性判断
#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum MvccKey {
    NextVersion,
    TxnActive(Version),
    TxnWrite(Version, Key),
    Version(Key, Version),
}

impl MvccKey {
    /// 编码 key
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut bytes = bincode::serialize(&self).map_err(Error::from)?;
        // 由于 bincode 的编码方式，需要对 Version 的编码进行特殊处理以适应前缀扫描
        //
        // bincode 对枚举的编码方式为：[索引, 数据]
        // 对于索引部分，如果 MvccKey 和 MvccKeyPrefix 的内容顺序一致，索引部分就完全相同
        //
        // 对于数据部分，NextVersion 和 TxnActive 的前缀数据部分为空，不影响前缀扫描
        // 对于 TxnWrite，前缀数据部分是 Version (u64)，也不影响前缀扫描
        //
        // 对于 Version，前缀数据部分是 Key (Vec<u8>)
        // bincode 对 Vec 的编码方式是：[长度, 数据]，长度是 u64 类型，占 8 字节
        // 由于 Version 前缀中 Key 长度和 Version 的不一定相等，所以会导致前缀扫描失败
        // 比如，`MvccKey::Version("key".to_vec(), 42)` 的编码为：[3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 107, 101, 121, 82, 191, 1, 0, 0, 0, 0, 0]
        // 而 `MvccKeyPrefix::Version("ke".to_vec())` 的编码为：[3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 107, 101]
        //
        // 我们将长度去除，只保留数据，上面的例子的编码就变为：
        // `MvccKey::Version("key".to_vec(), 42)`：[3, 0, 0, 0, 107, 101, 121, 82, 191, 1, 0, 0, 0, 0, 0]
        // `MvccKeyPrefix::Version("ke".to_vec())`：[3, 0, 0, 0, 107, 101]
        if let MvccKey::Version(_, _) = self {
            bytes.drain(4..12); // 前 4 个字节是 Version 枚举对应的索引编码
        }

        Ok(bytes)
    }

    /// 解码 key
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        // 由于编码时对 Version 进行了特殊处理，解码时也需要进行特殊处理
        //
        // 如果前缀是 Version，则需要在前面加上长度
        // 长度为编码后的长度 - 4（前 4 个字节是 Version 枚举对应的索引编码）- 8（Version u64 的版本号的长度）
        let mut bytes = bytes.to_vec();
        if bytes.len() > 4 && bytes[0..4] == [3, 0, 0, 0] {
            let len = (bytes.len() - 4 - 8) as u64;
            bytes.splice(4..4, len.to_le_bytes().iter().copied());
        }
        bincode::deserialize(&bytes).map_err(Error::from)
    }
}

/// MVCC 存储引擎的 key 前缀，用于扫描一个范围使用
#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum MvccKeyPrefix {
    NextVersion,
    TxnActive,
    TxnWrite(Version),
    Version(Key),
}

impl MvccKeyPrefix {
    /// 编码 key 前缀
    pub fn encode(&self) -> Result<Vec<u8>> {
        // 需要和编码 MvccKey 相同的处理方式
        // 具体参考 MvccKey 的 encode 方法
        let mut bytes = bincode::serialize(&self).map_err(Error::from)?;
        if let MvccKeyPrefix::Version(_) = self {
            bytes.drain(4..12);
        }

        Ok(bytes)
    }
}

/// MVCC 存储引擎
pub struct Mvcc<S: Storage> {
    storage: Arc<Mutex<S>>,
}

impl<S: Storage> Mvcc<S> {
    /// 创建一个新的 MVCC 存储引擎
    pub fn new(storage: S) -> Self {
        Self {
            storage: Arc::new(Mutex::new(storage)),
        }
    }

    /// 开启一个新事务
    pub fn start_txn(&self) -> Result<MvccTxn<S>> {
        MvccTxn::begin(self.storage.clone())
    }
}

/// MVCC 事务
pub struct MvccTxn<S: Storage> {
    storage: Arc<Mutex<S>>,
    version: Version,
    active_versions: HashSet<Version>,
}

impl<S: Storage> MvccTxn<S> {
    /// 开启一个新事务
    pub fn begin(s: Arc<Mutex<S>>) -> Result<Self> {
        // 获取当前存储引擎的锁
        let mut storage = s.lock()?;

        // 获取下一个版本号，如果不存在则从 1 开始
        let version = if let Some(value) = storage.get(&MvccKey::NextVersion.encode()?)? {
            Version::decode(&value)?
        } else {
            Version(1)
        };

        // 将下一个版本号加 1，写入存储引擎
        storage.put(&MvccKey::NextVersion.encode()?, &(version + 1).encode()?)?;

        // 扫描所有活跃事务
        let active_versions = Self::scan_active_txn(&mut storage)?;

        // 将新事务加入活跃事务列表
        // 在扫描之后加入，否则会将自己加入活跃事务列表从而导致自己不可见
        storage.put(&MvccKey::TxnActive(version).encode()?, &[])?;

        Ok(Self {
            storage: s.clone(),
            version,
            active_versions,
        })
    }

    /// 查找所有活跃事务
    fn scan_active_txn(storage: &mut MutexGuard<S>) -> Result<HashSet<Version>> {
        let mut active_versions = HashSet::new();

        // 扫描前缀为 TxnActive 的 key
        let mut iter = storage.scan_prefix(&MvccKeyPrefix::TxnActive.encode()?);
        while let Some((key, _)) = iter.next().transpose()? {
            // 解码 key，获取事务版本，并加入活跃事务列表
            if let MvccKey::TxnActive(version) = MvccKey::decode(&key)? {
                active_versions.insert(version);
            } else {
                return Err(InternalError(format!(
                    "unexpected key {} when scanning active transactions",
                    String::from_utf8_lossy(key.as_slice())
                )));
            }
        }
        Ok(active_versions)
    }

    /// 版本是否可见
    ///
    /// 版本可见的条件是：
    ///
    /// - 版本小于等于当前版本；
    /// - 版本不在活跃事务列表中。
    #[inline]
    fn is_version_visible(&self, version: Version) -> bool {
        version <= self.version && !self.active_versions.contains(&version)
    }

    /// 更新/删除数据的内置函数
    ///
    /// - 如果 `value` 为 `None`，则删除 `key` 对应的数据
    /// - 否则更新 `key` 对应的数据
    fn write_inner(&self, key: &[u8], value: Option<Vec<u8>>) -> Result<()> {
        // 获取当前存储引擎的锁
        let mut storage = self.storage.lock()?;

        // 活跃事务和大于当前版本的事务都不可见
        // 取活跃事务的最小值到可能存在的版本最大值，构成一个范围，其中会包括所有不可见的事务
        let begin = self
            .active_versions
            .iter()
            .min()
            .copied()
            .unwrap_or(self.version + 1);
        let begin_key = MvccKey::Version(key.to_vec(), begin).encode()?;
        let end_key = MvccKey::Version(key.to_vec(), Version::max()).encode()?;

        // 检查是否有不可见的版本写入了 key
        // 首先根据活跃事务和大于当前版本的事务的范围，找到最后一个可能不可见的事务
        // 如果这个事务不可见，则说明有不可见的事务写入了 key，返回写冲突
        //
        // 为什么只需检查最后一个可能不可见的版本即可：
        // 若最后版本不可见：直接判定存在写冲突，无需检查更早的版本，因为该版本是当前事务可能冲突的最高版本。
        // 若最后版本可见：所有更早的版本要么已被提交（可见），要么会发生写冲突。
        if let Some((key, _)) = storage.scan(begin_key..=end_key).last().transpose()? {
            if let MvccKey::Version(_, version) = MvccKey::decode(&key)? {
                if !self.is_version_visible(version) {
                    return Err(WriteConflict);
                }
            } else {
                return Err(InternalError(format!(
                    "unexpected key {} when scanning versions",
                    String::from_utf8_lossy(key.as_slice())
                )));
            }
        }

        // 记录新版本写入了哪些 key，用于回滚事务
        storage.put(
            &MvccKey::TxnWrite(self.version, key.to_vec()).encode()?,
            &[],
        )?;

        // 如果 value 不为 None，则写入新的数据，否则删除数据
        storage.put(
            &MvccKey::Version(key.to_vec(), self.version).encode()?,
            &bincode::serialize(&value)?,
        )?;

        Ok(())
    }

    /// 更新 `key` 对应的值
    #[inline]
    pub fn set(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.write_inner(key, Some(value.to_vec()))
    }

    /// 删除 `key` 对应的值
    #[inline]
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        self.write_inner(key, None)
    }

    /// 获取 `key` 对应的值
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        // 获取当前存储引擎的锁
        let mut storage = self.storage.lock()?;

        // 设置范围为 0 到当前版本，因为大于当前版本的事务一定不可见
        let begin = MvccKey::Version(key.to_vec(), Version::min()).encode()?;
        let end = MvccKey::Version(key.to_vec(), self.version).encode()?;

        // 从范围中找到最新的可见版本
        let mut iter = storage.scan(begin..=end).rev(); // 新版本在后面
        while let Some((key, value)) = iter.next().transpose()? {
            if let MvccKey::Version(_, version) = MvccKey::decode(&key)? {
                // 判断是否可见，此处指的是不在活跃事务中，因为范围已经排除了大于当前版本的事务
                if self.is_version_visible(version) {
                    // 存储的数据为 Option<Vec<u8>>，Option 为 None 表示删除，需要解析
                    return Ok(bincode::deserialize(&value)?);
                }
            } else {
                return Err(InternalError(format!(
                    "unexpected key {} when scanning versions",
                    String::from_utf8_lossy(key.as_slice())
                )));
            }
        }

        // 没有找到可见版本，返回 None
        Ok(None)
    }

    /// 扫描 `prefix` 开头的所有可见的事务记录
    pub fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Key, Vec<u8>)>> {
        // 获取当前存储引擎的锁
        let mut storage = self.storage.lock()?;

        let prefix = MvccKeyPrefix::Version(prefix.to_vec()).encode()?;

        let mut result = BTreeMap::new();
        let mut iter = storage.scan_prefix(&prefix);
        while let Some((key, value)) = iter.next().transpose()? {
            match MvccKey::decode(&key)? {
                // 如果版本可见，则返回 key-value，之后的过滤中被保留
                // 如果版本可见但 value 为 None，表示删除，返回 None，并且删除前面的版本中已经存在的 key-value
                MvccKey::Version(k, version) => {
                    if !self.is_version_visible(version) {
                        continue;
                    }
                    let value: Option<Vec<u8>> = bincode::deserialize(&value)?;
                    if let Some(value) = &value {
                        result.insert(k, value.clone());
                    } else {
                        result.remove(&k);
                    }
                }
                // 如果解析不是 Version，则返回错误
                _ => {
                    return Err(InternalError(format!(
                        "unexpected key {} when scanning versions",
                        String::from_utf8_lossy(&key)
                    )))?
                }
            }
        }

        Ok(result.into_iter().collect())
    }

    /// 提交事务
    ///
    /// 对于提交事务，实际上是让这个事务的修改对后续新开启的事务是可见的。
    /// 因此，只需要将当前事务对应的所有 TxnWrite 记录，以及当前事务在活跃事务列表中的记录删除即可。
    pub fn commit(&self) -> Result<()> {
        // 获取当前存储引擎的锁
        let mut storage = self.storage.lock()?;

        // 找到当前事务对应的所有 TxnWrite 记录
        let txn_keys = storage
            .scan_prefix(&MvccKeyPrefix::TxnWrite(self.version).encode()?)
            .map(|item| {
                let (key, _) = item?;
                if let MvccKey::TxnWrite(_, key) = MvccKey::decode(&key)? {
                    Ok(key)
                } else {
                    Err(InternalError(format!(
                        "unexpected key {} when scanning txn writes",
                        String::from_utf8_lossy(&key)
                    )))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // 将当前事务对应的所有 TxnWrite 记录从存储引擎中删除
        for key in txn_keys {
            storage.delete(&key)?;
        }

        // 将当前事务从活跃事务列表中移除
        storage.delete(&MvccKey::TxnActive(self.version).encode()?)?;

        Ok(())
    }

    /// 回滚事务
    pub fn rollback(&self) -> Result<()> {
        // 获取当前存储引擎的锁
        let mut storage = self.storage.lock()?;

        // 找到当前事务对应的所有 TxnWrite 记录，并转换为 Version 记录
        // 之后将 TxnWrite 记录和 Version 记录都添加到删除列表中
        let txn_keys = storage
            .scan_prefix(&MvccKeyPrefix::TxnWrite(self.version).encode()?)
            .map(|item| {
                let (tx_write_key, _) = item?;
                if let MvccKey::TxnWrite(_, raw_version_key) = MvccKey::decode(&tx_write_key)? {
                    let version_key = MvccKey::Version(raw_version_key, self.version).encode()?;
                    Ok((tx_write_key, version_key))
                } else {
                    Err(InternalError(format!(
                        "unexpected key {} when scanning txn writes",
                        String::from_utf8_lossy(&tx_write_key)
                    )))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // 将当前事务对应的所有 TxnWrite 记录和 Version 记录从存储引擎中删除
        for (tx_write_key, version_key) in txn_keys {
            storage.delete(&tx_write_key)?;
            storage.delete(&version_key)?;
        }

        // 将当前事务从活跃事务列表中移除
        storage.delete(&MvccKey::TxnActive(self.version).encode()?)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        storage::{disk::DiskStorage, memory::MemoryStorage},
        Result,
    };

    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_mvcckey() -> Result<()> {
        let key_1 = MvccKey::NextVersion;
        let encoded_1 = key_1.encode()?;
        let decoded_1 = MvccKey::decode(&encoded_1)?;
        assert_eq!(key_1, decoded_1);

        let key_2 = MvccKey::TxnActive(1.into());
        let encoded_2 = key_2.encode()?;
        let decoded_2 = MvccKey::decode(&encoded_2)?;
        assert_eq!(key_2, decoded_2);

        let key_3 = MvccKey::TxnWrite(1.into(), b"key".to_vec());
        let encoded_3 = key_3.encode()?;
        let decoded_3 = MvccKey::decode(&encoded_3)?;
        assert_eq!(key_3, decoded_3);

        let key_4 = MvccKey::Version(b"key".to_vec(), 1.into());
        let encoded_4 = key_4.encode()?;
        let decoded_4 = MvccKey::decode(&encoded_4)?;
        assert_eq!(key_4, decoded_4);

        assert_ne!(encoded_1, encoded_2);
        assert_ne!(encoded_1, encoded_3);
        assert_ne!(encoded_1, encoded_4);
        assert_ne!(encoded_2, encoded_3);
        assert_ne!(encoded_2, encoded_4);
        assert_ne!(encoded_3, encoded_4);

        Ok(())
    }

    #[test]
    fn test_mvcckey_prefix() -> Result<()> {
        let key_prefix_1 = MvccKeyPrefix::TxnActive;
        let encoded_prefix_1 = key_prefix_1.encode()?;

        let key_1 = MvccKey::TxnActive(114514.into());
        let encoded_1 = key_1.encode()?;
        assert!(encoded_1.starts_with(&encoded_prefix_1));

        let key_prefix_2 = MvccKeyPrefix::Version(b"ke".to_vec());
        let encoded_prefix_2 = key_prefix_2.encode()?;

        let key_2 = MvccKey::Version(b"key".to_vec(), 114514.into());
        let encoded_2 = key_2.encode()?;

        assert!(encoded_2.starts_with(&encoded_prefix_2));
        assert!(!encoded_2.starts_with(&encoded_prefix_1));

        Ok(())
    }

    macro_rules! test_all_storage {
        ($code:expr) => {
            let file = NamedTempFile::new().unwrap();
            let storage = DiskStorage::new(file.path()).unwrap();
            $code(&Mvcc::new(storage))?;

            let storage = MemoryStorage::new();
            $code(&Mvcc::new(storage))?;
        };
    }

    #[test]
    fn test_read() -> Result<()> {
        test_all_storage!(|mvcc: &Mvcc<_>| -> Result<()> {
            let tx0 = mvcc.start_txn()?;
            tx0.set(b"key1", b"val1")?;
            tx0.set(b"key2", b"val2")?;
            tx0.set(b"key2", b"val3")?;
            tx0.set(b"key3", b"val4")?;
            tx0.delete(b"key3")?;
            tx0.commit()?;

            let tx1 = mvcc.start_txn()?;
            assert_eq!(tx1.get(b"key1")?, Some(b"val1".to_vec()));
            assert_eq!(tx1.get(b"key2")?, Some(b"val3".to_vec()));
            assert_eq!(tx1.get(b"key3")?, None);

            Ok(())
        });

        Ok(())
    }

    #[test]
    fn test_isolation() -> Result<()> {
        test_all_storage!(|mvcc: &Mvcc<_>| -> Result<()> {
            let tx_1 = mvcc.start_txn()?;
            tx_1.set(b"key1", b"val1")?;
            tx_1.set(b"key2", b"val2")?;
            tx_1.set(b"key2", b"val3")?;
            tx_1.set(b"key3", b"val4")?;
            tx_1.commit()?;

            let tx_2 = mvcc.start_txn()?;
            tx_2.set(b"key1", b"val2")?;

            let tx_3 = mvcc.start_txn()?;

            let tx_4 = mvcc.start_txn()?;
            tx_4.set(b"key2", b"val4")?;
            tx_4.delete(b"key3")?;
            tx_4.commit()?;

            assert_eq!(tx_3.get(b"key1")?, Some(b"val1".to_vec()));
            assert_eq!(tx_3.get(b"key2")?, Some(b"val3".to_vec()));
            assert_eq!(tx_3.get(b"key3")?, Some(b"val4".to_vec()));

            Ok(())
        });

        Ok(())
    }

    #[test]
    fn test_write() -> Result<()> {
        test_all_storage!(|mvcc: &Mvcc<_>| -> Result<()> {
            let tx_1 = mvcc.start_txn()?;
            tx_1.set(b"key1", b"val1")?;
            tx_1.set(b"key2", b"val2")?;
            tx_1.set(b"key2", b"val3")?;
            tx_1.set(b"key3", b"val4")?;
            tx_1.set(b"key4", b"val5")?;
            tx_1.commit()?;

            let tx_2 = mvcc.start_txn()?;
            let tx_3 = mvcc.start_txn()?;

            tx_2.set(b"key1", b"val1-1")?;
            tx_2.set(b"key2", b"val3-1")?;
            tx_2.set(b"key2", b"val3-2")?;

            tx_3.set(b"key3", b"val4-1")?;
            tx_3.set(b"key4", b"val5-1")?;

            tx_2.commit()?;
            tx_3.commit()?;

            let tx_4 = mvcc.start_txn()?;
            assert_eq!(tx_4.get(b"key1")?, Some(b"val1-1".to_vec()));
            assert_eq!(tx_4.get(b"key2")?, Some(b"val3-2".to_vec()));
            assert_eq!(tx_4.get(b"key3")?, Some(b"val4-1".to_vec()));
            assert_eq!(tx_4.get(b"key4")?, Some(b"val5-1".to_vec()));

            Ok(())
        });

        Ok(())
    }

    #[test]
    fn test_write_conflict() -> Result<()> {
        test_all_storage!(|mvcc: &Mvcc<_>| -> Result<()> {
            let tx_1 = mvcc.start_txn()?;
            tx_1.set(b"key1", b"val1")?;
            tx_1.set(b"key2", b"val2")?;
            tx_1.set(b"key2", b"val3")?;
            tx_1.set(b"key3", b"val4")?;
            tx_1.set(b"key4", b"val5")?;
            tx_1.commit()?;

            let tx_2 = mvcc.start_txn()?;
            let tx_3 = mvcc.start_txn()?;

            tx_2.set(b"key1", b"val1-1")?;
            tx_2.set(b"key1", b"val1-2")?;

            assert_eq!(tx_3.set(b"key1", b"val1-3"), Err(WriteConflict));

            let tx_4 = mvcc.start_txn()?;
            tx_4.set(b"key5", b"val6")?;
            tx_4.commit()?;

            assert_eq!(tx_1.set(b"key5", b"val6-1"), Err(WriteConflict));

            Ok(())
        });

        Ok(())
    }

    #[test]
    fn test_scan_prefix() -> Result<()> {
        test_all_storage!(|mvcc: &Mvcc<_>| -> Result<()> {
            let tx_1 = mvcc.start_txn()?;
            tx_1.set(b"aabb", b"val1")?;
            tx_1.set(b"abcc", b"val2")?;
            tx_1.set(b"bbaa", b"val3")?;
            tx_1.set(b"acca", b"val4")?;
            tx_1.set(b"aaca", b"val5")?;
            tx_1.set(b"bcca", b"val6")?;
            tx_1.commit()?;

            let tx_2 = mvcc.start_txn()?;
            assert_eq!(
                tx_2.scan_prefix(b"aa")?,
                vec![
                    (b"aabb".to_vec(), b"val1".to_vec()),
                    (b"aaca".to_vec(), b"val5".to_vec()),
                ]
            );

            let tx_3 = mvcc.start_txn()?;
            assert_eq!(
                tx_3.scan_prefix(b"a")?,
                vec![
                    (b"aabb".to_vec(), b"val1".to_vec()),
                    (b"aaca".to_vec(), b"val5".to_vec()),
                    (b"abcc".to_vec(), b"val2".to_vec()),
                    (b"acca".to_vec(), b"val4".to_vec()),
                ]
            );

            let tx_4 = mvcc.start_txn()?;
            assert_eq!(
                tx_4.scan_prefix(b"bc")?,
                vec![(b"bcca".to_vec(), b"val6".to_vec())]
            );

            Ok(())
        });

        Ok(())
    }

    #[test]
    fn test_delete() -> Result<()> {
        test_all_storage!(|mvcc: &Mvcc<_>| -> Result<()> {
            let tx_1 = mvcc.start_txn()?;
            tx_1.set(b"key1", b"val1")?;
            tx_1.set(b"key2", b"val2")?;
            tx_1.set(b"key3", b"val3")?;
            tx_1.delete(b"key2")?;
            tx_1.delete(b"key3")?;
            tx_1.set(b"key3", b"val3-1")?;
            assert_eq!(tx_1.get(b"key2")?, None);
            assert_eq!(tx_1.get(b"key3")?, Some(b"val3-1".to_vec()));
            tx_1.commit()?;

            let tx_2 = mvcc.start_txn()?;
            assert_eq!(tx_2.get(b"key2")?, None);
            assert_eq!(
                tx_2.scan_prefix(b"k")?,
                vec![
                    (b"key1".to_vec(), b"val1".to_vec()),
                    (b"key3".to_vec(), b"val3-1".to_vec())
                ]
            );

            Ok(())
        });

        Ok(())
    }

    #[test]
    fn test_dirty_read() -> Result<()> {
        test_all_storage!(|mvcc: &Mvcc<_>| -> Result<()> {
            let tx_1 = mvcc.start_txn()?;
            tx_1.set(b"key1", b"val1")?;
            tx_1.set(b"key2", b"val2")?;
            tx_1.set(b"key3", b"val3")?;
            tx_1.commit()?;

            let tx_2 = mvcc.start_txn()?;
            tx_2.set(b"key1", b"val1-1")?;
            assert_eq!(tx_2.get(b"key1")?, Some(b"val1-1".to_vec()));

            let tx_3 = mvcc.start_txn()?;
            assert_eq!(tx_3.get(b"key1")?, Some(b"val1".to_vec()));

            Ok(())
        });

        Ok(())
    }

    #[test]
    fn test_unrepeatable_read() -> Result<()> {
        test_all_storage!(|mvcc: &Mvcc<_>| -> Result<()> {
            let tx_1 = mvcc.start_txn()?;
            tx_1.set(b"key1", b"val1")?;
            tx_1.set(b"key2", b"val2")?;
            tx_1.set(b"key3", b"val3")?;
            tx_1.commit()?;

            let tx_2 = mvcc.start_txn()?;
            assert_eq!(tx_2.get(b"key1")?, Some(b"val1".to_vec()));

            let tx_3 = mvcc.start_txn()?;
            tx_3.set(b"key1", b"val1-1")?;
            assert_eq!(tx_3.get(b"key1")?, Some(b"val1-1".to_vec()));
            tx_3.commit()?;

            assert_eq!(tx_2.get(b"key1")?, Some(b"val1".to_vec()));

            Ok(())
        });

        Ok(())
    }

    #[test]
    fn test_phantom_read() -> Result<()> {
        test_all_storage!(|mvcc: &Mvcc<_>| -> Result<()> {
            let tx_1 = mvcc.start_txn()?;
            tx_1.set(b"key1", b"val1")?;
            tx_1.set(b"key2", b"val2")?;
            tx_1.set(b"key3", b"val3")?;
            tx_1.commit()?;

            let tx_2 = mvcc.start_txn()?;
            assert_eq!(
                tx_2.scan_prefix(b"key")?,
                vec![
                    (b"key1".to_vec(), b"val1".to_vec()),
                    (b"key2".to_vec(), b"val2".to_vec()),
                    (b"key3".to_vec(), b"val3".to_vec()),
                ]
            );

            let tx_3 = mvcc.start_txn()?;
            tx_3.delete(b"key1")?;
            assert_eq!(
                tx_3.scan_prefix(b"key")?,
                vec![
                    (b"key2".to_vec(), b"val2".to_vec()),
                    (b"key3".to_vec(), b"val3".to_vec()),
                ]
            );
            tx_3.commit()?;

            assert_eq!(
                tx_2.scan_prefix(b"key")?,
                vec![
                    (b"key1".to_vec(), b"val1".to_vec()),
                    (b"key2".to_vec(), b"val2".to_vec()),
                    (b"key3".to_vec(), b"val3".to_vec()),
                ]
            );

            Ok(())
        });

        Ok(())
    }

    #[test]
    fn test_rollback() -> Result<()> {
        test_all_storage!(|mvcc: &Mvcc<_>| -> Result<()> {
            let tx_1 = mvcc.start_txn()?;
            tx_1.set(b"key1", b"val1")?;
            tx_1.set(b"key2", b"val2")?;
            tx_1.set(b"key3", b"val3")?;
            tx_1.commit()?;

            let tx_2 = mvcc.start_txn()?;
            tx_2.set(b"key1", b"val1-1")?;
            tx_2.set(b"key2", b"val2-1")?;
            tx_2.set(b"key3", b"val3-1")?;
            tx_2.rollback()?;

            let tx_3 = mvcc.start_txn()?;
            assert_eq!(tx_3.get(b"key1")?, Some(b"val1".to_vec()));
            assert_eq!(tx_3.get(b"key2")?, Some(b"val2".to_vec()));
            assert_eq!(tx_3.get(b"key3")?, Some(b"val3".to_vec()));

            Ok(())
        });

        Ok(())
    }
}
