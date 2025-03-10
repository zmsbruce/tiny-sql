use fs4::fs_std::FileExt;

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    vec,
};

use super::Storage;
use crate::Result;

/// 基于 Bitcast 的磁盘存储，参考论文 [Bitcask: A Log-Structured Hash Table for Key/Value Data](https://riak.com/assets/bitcask-intro.pdf)。
///
/// 本实现的 `DiskStorage` 与论文中的 `Bitcask` 不同：
/// - `DiskStorage` 采用了 `BTreeMap` 作为内存中的 KeyDir 以支持范围扫描，而 `Bitcask` 使用了 `HashMap`；
/// - `DiskStorage` 将文件存储在一个文件中，而 `Bitcask` 将文件存储在多个文件中；
/// - `DiskStorage` 没有实现多线程的 `merge` 操作，而是提供了 `compact` 方法来压缩日志文件；
///
/// # 存储结构
///
/// - **磁盘上的 Log**：| key_len | val_len | key | value |；
/// - **磁盘上的 Hint**：| key_len | val_len | val_offset | key |；
/// - **内存中的 KeyDir**：`BTreeMap`：(key，(磁盘上 value 的索引, value 的长度))；
///
/// # 初始化
/// 1. 打开 log 文件，如果文件不存在则创建文件；
/// 2. 打开 hint 文件，如果文件不存在则创建文件；
/// 3. 如果存在 hint 文件，从 hint 文件中读取数据，构建 KeyDir；
/// 4. 如果不存在 hint 文件，则从 log 文件中读取数据，构建 KeyDir；
///
/// # 读取数据
/// 1. 根据 key 在内存中的 KeyDir 中查找 key 对应的 (offset, len)；
/// 2. 根据 offset 读取磁盘上的数据；
///
/// # 写入数据
/// 1. 将 key 和 value 写入 Log，并获取 offset 和 value 的长度；
/// 2. 将 key 和 (offset, len) 写入 Hint；
/// 3. 将 key 和 (offset, len) 写入内存中的 KeyDir；
///
/// # 删除数据
/// 1. 将 value 的长度设置最高位为 1，其余为 0，然后将 key 和 value 写入 Log；
/// 2. 将 value 的长度设置最高位为 1，其余为 0，将 key 和 (offset, len) 写入 Hint；
/// 3. 将 key 从内存中的 KeyDir 删除；
///
/// # 压缩数据
/// 1. 将 KeyDir 中的数据重新写入一个新的 Log 中，记录 value 的偏移；
/// 2. 将 KeyDir 中的 key 和 value 的偏移重新写入一个新的 Hint 中；
/// 3. 重新构建 KeyDir；
///
pub struct DiskStorage {
    keydir: BTreeMap<Vec<u8>, (u64, u64)>,
    log: File,
    hint: File,
    log_path: PathBuf,
}

impl DiskStorage {
    /// 创建一个新的 `DiskStorage` 实例。
    pub fn new<P>(filename: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let file_path = filename.as_ref().to_path_buf();
        // 如果文件所在的目录不存在，则创建目录
        // 如果目录为 None，则会在创建文件时报错
        if let Some(dir) = file_path.parent() {
            if !dir.exists() {
                std::fs::create_dir_all(dir)?;
            }
        }

        // 打开文件，如果文件不存在则创建文件，设置为追加写入
        let log_file = fs::OpenOptions::new()
            .read(true)
            .create(true)
            .append(true)
            .open(&file_path)?;

        // 打开 hint 文件，如果文件不存在则创建文件，设置为追加写入
        let hint_file = fs::OpenOptions::new()
            .read(true)
            .create(true)
            .append(true)
            .open(file_path.with_extension("hint"))?;

        // 尝试获取文件的独占锁
        log_file.try_lock_exclusive()?;
        hint_file.try_lock_exclusive()?;

        // 如果存在压缩后的文件，则删除压缩后的文件
        if file_path.with_extension("compact").exists() {
            fs::remove_file(file_path.with_extension("compact"))?;
        }

        let keydir = BTreeMap::new();

        let mut storage = DiskStorage {
            keydir,
            log: log_file,
            hint: hint_file,
            log_path: file_path,
        };
        storage.build_keydir()?; // 从磁盘上读取数据，构建 KeyDir

        Ok(storage)
    }

    /// 从 hint 文件中读取数据，构建 KeyDir。
    /// hint 文件中的 entry 格式为：| key_len | val_len | val_offset | key |。
    /// offset 和 len 分别表示 key 对应的数据在 log 文件中的偏移量和长度。
    /// 如果 key 对应的数据已经被删除，则 len 的最高位为 1。
    fn build_keydir_from_hint(&mut self) -> Result<()> {
        let file_sz = self.hint.metadata()?.len();
        let mut file_reader = BufReader::new(&self.hint);
        file_reader.seek(SeekFrom::Start(0))?; // 从文件头开始读取

        while file_reader.stream_position()? < file_sz {
            let mut len_buf = [0u8; u64::BITS as usize / 8];

            // 读取 key_len
            file_reader.read_exact(&mut len_buf)?;
            let key_len = u64::from_le_bytes(len_buf);

            // 读取 val_len
            file_reader.read_exact(&mut len_buf)?;
            let val_len = u64::from_le_bytes(len_buf);

            // 读取 val_offset
            file_reader.read_exact(&mut len_buf)?;
            let offset = u64::from_le_bytes(len_buf);

            // 读取 key
            let mut key = vec![0u8; key_len as usize];
            file_reader.read_exact(&mut key)?;

            if Self::is_marked_deleted(val_len) {
                // 如果 len 最高位为 1，则表示该 key 对应的数据已经被删除
                self.keydir.remove(&key);
            } else {
                // 将 key 和 (offset, len) 写入 KeyDir
                self.keydir.insert(key, (offset, val_len));
            }
        }

        Ok(())
    }

    /// 判断 value 是否已经被删除。
    /// 如果 value 的最高位为 1，则表示 value 已经被删除。
    #[inline]
    fn is_marked_deleted(val_len: u64) -> bool {
        val_len & (1 << (u64::BITS - 1)) != 0
    }

    /// 从磁盘中读取数据，构建 KeyDir。
    fn build_keydir(&mut self) -> Result<()> {
        // 清空 KeyDir
        self.keydir.clear();

        // 如果 hint 文件存在，则从 hint 文件中读取数据
        if let Ok(metadata) = self.hint.metadata() {
            if metadata.len() > 0 {
                return self.build_keydir_from_hint();
            }
        }

        let file_sz = self.log.metadata()?.len();
        let mut file_reader = BufReader::new(&self.log);
        file_reader.seek(SeekFrom::Start(0))?; // 从文件头开始读取

        while file_reader.stream_position()? < file_sz {
            let mut len_buf = [0u8; u64::BITS as usize / 8];

            // 读取 key_len
            file_reader.read_exact(&mut len_buf)?;
            let key_len = u64::from_le_bytes(len_buf);

            // 读取 val_len
            file_reader.read_exact(&mut len_buf)?;
            let val_len = u64::from_le_bytes(len_buf);

            // 读取 key
            let mut key = vec![0u8; key_len as usize];
            file_reader.read_exact(&mut key)?;

            if Self::is_marked_deleted(val_len) {
                // 如果 val_len 最高位为 1，则表示该 key 对应的数据已经被删除
                self.keydir.remove(&key);

                // 如果删除，value 的长度为 0，所以不需要读取 value
            } else {
                // 将 key 和 (offset, len) 写入 KeyDir
                self.keydir
                    .insert(key, (file_reader.stream_position()?, val_len));
                // 跳过 value
                file_reader.seek(SeekFrom::Current(val_len as i64))?;
            }
        }
        Ok(())
    }

    /// 压缩日志文件
    ///
    /// 将日志文件中的数据重新写入一个新的文件中，然后将新文件重命名为原文件，从而去除已经删除的数据。
    pub fn compact(&mut self) -> Result<()> {
        // 创建一个新的日志文件和一个写入器
        let new_log_path = self.log_path.with_extension("compact");
        let mut log_writer = BufWriter::new(File::create(&new_log_path)?);

        // 创建一个新的 hint 文件和一个写入器
        let new_hint_path = self.log_path.with_extension("hint.compact");
        let mut hint_writer = BufWriter::new(File::create(&new_hint_path)?);

        // 将数据重新写入新的日志文件
        // 由于 KeyDir 中没有删除的数据，所以只需要将 KeyDir 中的数据重新写入新的日志文件即可
        for (key, (offset, len)) in self.keydir.iter() {
            // 读取 value 对应的数据
            let value = Self::read_value_from_log(&mut self.log, *offset, *len)?;

            // 将 key 和 value 写入新的日志文件
            let val_offset = Self::write_log_data(&mut log_writer, key, &value)?;

            // 将 key 和 (offset, len) 写入新的 hint 文件
            Self::write_hint_data(&mut hint_writer, key, val_offset, value.len())?;
        }

        // 刷新缓冲区
        log_writer.flush()?;
        hint_writer.flush()?;

        // 重命名新日志文件为原日志文件
        fs::rename(&new_log_path, &self.log_path)?;
        fs::rename(&new_hint_path, self.log_path.with_extension("hint"))?;

        // 重新打开文件进行读取操作
        self.log = fs::OpenOptions::new()
            .read(true)
            .append(true)
            .open(&self.log_path)?;
        self.hint = fs::OpenOptions::new()
            .read(true)
            .append(true)
            .open(self.log_path.with_extension("hint"))?;
        self.log.try_lock_exclusive()?;
        self.hint.try_lock_exclusive()?;

        // 由于 compact 之后 keydir 中 offset 发生了变化，所以需要重新构建 keydir
        self.build_keydir()?;

        Ok(())
    }

    /// 将数据写入 log 文件。
    /// 返回 value 在文件中的偏移量。
    /// log 文件中的 entry 格式为：| key_len | val_len | key | value |。
    fn write_log_data<W>(writer: &mut BufWriter<W>, key: &[u8], value: &[u8]) -> Result<u64>
    where
        W: Write + Seek,
    {
        // 写入 key_len 和 val_len
        writer.write_all(&(key.len() as u64).to_le_bytes())?;
        writer.write_all(&(value.len() as u64).to_le_bytes())?;

        // 写入 key
        writer.write_all(key)?;

        let offset = writer.stream_position()?; // 记录 value 的偏移量，用于写入 hint 文件
                                                // 写入 value
        writer.write_all(value)?;

        Ok(offset)
    }

    /// 将删除的数据写入 log 文件。
    /// 删除的数据的 value 长度为 0，最高位为 1。不需要写入 value。
    fn write_deleted_log_data<W>(writer: &mut BufWriter<W>, key: &[u8]) -> Result<()>
    where
        W: Write,
    {
        // 写入 key_len 和 val_len
        writer.write_all(&(key.len() as u64).to_le_bytes())?;
        writer.write_all(&((1u64 << (u64::BITS - 1)).to_le_bytes()))?;

        // 写入 key
        writer.write_all(key)?;

        // 删除的数据的 value 长度为 0，所以不需要写入 value

        Ok(())
    }

    /// 将数据写入 hint 文件。
    /// hint 文件中的 entry 格式为：| key_len | val_len | val_offset | key |。
    fn write_hint_data<W>(
        writer: &mut BufWriter<W>,
        key: &[u8],
        val_offset: u64,
        val_len: usize,
    ) -> Result<()>
    where
        W: Write,
    {
        // 写入 key_len、val_len 和 val_offset
        writer.write_all(&(key.len() as u64).to_le_bytes())?;
        writer.write_all(&(val_len as u64).to_le_bytes())?;
        writer.write_all(&val_offset.to_le_bytes())?;

        // 写入 key
        writer.write_all(key)?;

        Ok(())
    }

    /// 将删除的数据写入 hint 文件。
    /// 删除的数据的 value 长度为 0，最高位为 1。value 的偏移量为 0。
    fn write_deleted_hint_data<W>(writer: &mut BufWriter<W>, key: &[u8]) -> Result<()>
    where
        W: Write,
    {
        // 写入 key_len、val_len 和 val_offset
        writer.write_all(&(key.len() as u64).to_le_bytes())?;
        writer.write_all(&(1u64 << (u64::BITS - 1)).to_le_bytes())?; // value 的长度为 0，最高位为 1
        writer.write_all(&(0u64).to_le_bytes())?; // value 的偏移量为 0
        writer.write_all(key)?;

        Ok(())
    }

    /// 从 log 文件中读取 value 数据。
    fn read_value_from_log(log: &mut File, offset: u64, len: u64) -> Result<Vec<u8>> {
        log.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; len as usize];
        log.read_exact(&mut buf)?;

        Ok(buf)
    }
}

pub struct DiskStorageIterator<'a> {
    inner: std::collections::btree_map::Range<'a, Vec<u8>, (u64, u64)>,
    file: &'a mut File,
}

impl Iterator for DiskStorageIterator<'_> {
    type Item = Result<(Vec<u8>, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, (offset, len))| {
            self.file.seek(SeekFrom::Start(*offset))?;
            let mut buf = vec![0u8; *len as usize];
            self.file.read_exact(&mut buf)?;

            Ok((k.clone(), buf))
        })
    }
}

impl DoubleEndedIterator for DiskStorageIterator<'_> {
    fn next_back(&mut self) -> Option<Result<(Vec<u8>, Vec<u8>)>> {
        self.inner.next_back().map(|(k, (offset, len))| {
            self.file.seek(SeekFrom::Start(*offset))?;
            let mut buf = vec![0u8; *len as usize];
            self.file.read_exact(&mut buf)?;

            Ok((k.clone(), buf))
        })
    }
}

impl Storage for DiskStorage {
    type Iterator<'a> = DiskStorageIterator<'a>;

    fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        self.log.seek(SeekFrom::End(0))?;

        // 写入 key 和 value 到 log 文件
        let mut log_writer = BufWriter::with_capacity(
            usize::BITS as usize / 8 * 2 + key.len() + value.len(),
            &self.log,
        );
        let offset = Self::write_log_data(&mut log_writer, key, value)?;
        log_writer.flush()?;

        // 写入 key 和 (offset, len) 到 hint 文件
        let mut hint_writer =
            BufWriter::with_capacity(usize::BITS as usize / 8 * 3 + key.len(), &self.hint);
        Self::write_hint_data(&mut hint_writer, key, offset, value.len())?;
        hint_writer.flush()?;

        // 将 key 和 (offset, len) 写入 KeyDir
        self.keydir
            .insert(key.to_vec(), (offset, value.len() as u64));

        Ok(())
    }

    fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if let Some((offset, len)) = self.keydir.get(key) {
            let value = Self::read_value_from_log(&mut self.log, *offset, *len)?;
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    fn delete(&mut self, key: &[u8]) -> Result<()> {
        if self.keydir.contains_key(key) {
            self.log.seek(SeekFrom::End(0))?;

            // 写入删除的数据到 log 文件
            let mut log_writer =
                BufWriter::with_capacity(u64::BITS as usize / 8 * 2 + key.len(), &self.log);
            Self::write_deleted_log_data(&mut log_writer, key)?;
            log_writer.flush()?;

            // 写入删除的数据到 hint 文件
            let mut hint_writer =
                BufWriter::with_capacity(u64::BITS as usize / 8 * 3 + key.len(), &self.hint);
            Self::write_deleted_hint_data(&mut hint_writer, key)?;
            hint_writer.flush()?;

            // 从 KeyDir 中删除 key
            self.keydir.remove(key);
        }
        Ok(())
    }

    fn scan<R>(&mut self, range: R) -> Self::Iterator<'_>
    where
        R: std::ops::RangeBounds<Vec<u8>>,
    {
        let inner = self.keydir.range(range);
        DiskStorageIterator {
            inner,
            file: &mut self.log,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_build_keydir() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key1value1\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key2value2\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key3value3\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x80key2").unwrap();

        let storage = DiskStorage::new(file.path()).unwrap();
        assert_eq!(storage.keydir.len(), 2);
        assert_eq!(*storage.keydir.get(&b"key1"[..]).unwrap(), (20, 6));
        assert!(!storage.keydir.contains_key(&b"key2"[..]));
        assert_eq!(*storage.keydir.get(&b"key3"[..]).unwrap(), (72, 6));
    }

    #[test]
    fn test_build_keydir_from_hint() {
        let dir = tempfile::tempdir().unwrap();
        let mut log_file = File::create(dir.path().join("log")).unwrap();
        log_file.write_all(b"\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key1value1\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key2value2\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key3value3\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x80key2").unwrap();

        let mut hint_file = File::create(dir.path().join("log.hint")).unwrap();
        hint_file.write_all(b"\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00key1\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00\x48\x00\x00\x00\x00\x00\x00\x00key3\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00key2").unwrap();

        let storage = DiskStorage::new(dir.path().join("log")).unwrap();
        assert_eq!(storage.keydir.len(), 2);
        assert_eq!(*storage.keydir.get(&b"key1"[..]).unwrap(), (20, 6));
        assert_eq!(*storage.keydir.get(&b"key3"[..]).unwrap(), (72, 6));
        assert!(!storage.keydir.contains_key(&b"key2"[..]));
    }

    #[test]
    fn test_reload_storage() {
        let file = NamedTempFile::new().unwrap();
        {
            let mut storage = DiskStorage::new(file.path()).unwrap();
            storage.put(b"key1", b"value1").unwrap();
            storage.put(b"key2", b"value2").unwrap();
            storage.put(b"key3", b"value3").unwrap();
            storage.delete(b"key2").unwrap();
        }
        {
            let mut storage = DiskStorage::new(file.path()).unwrap();
            assert_eq!(storage.keydir.len(), 2);
            assert_eq!(storage.get(b"key1").unwrap(), Some(b"value1".to_vec()));
            assert_eq!(storage.get(b"key2").unwrap(), None);
            assert_eq!(storage.get(b"key3").unwrap(), Some(b"value3".to_vec()));
        }
    }

    #[test]
    fn test_compact() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key1value1\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key2value2\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key3value3\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x80key2").unwrap();

        let mut storage = DiskStorage::new(file.path()).unwrap();
        storage.compact().unwrap();

        let mut buf = Vec::new();
        storage.log.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key1value1\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00key3value3");
    }
}
