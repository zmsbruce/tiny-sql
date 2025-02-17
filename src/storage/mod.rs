use std::ops::RangeBounds;

use crate::Result;

mod disk;
mod memory;

pub trait Storage {
    type Iterator<'a>: DoubleEndedIterator<Item = Result<(Vec<u8>, Vec<u8>)>>
    where
        Self: 'a;

    /// 获取指定 key 对应的 value
    fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>>;

    /// 将 key-value 存入数据库
    fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()>;

    /// 删除指定 key 对应的 value
    fn delete(&mut self, key: &[u8]) -> Result<()>;

    /// 返回一个迭代器，遍历指定范围内的 key-value
    ///
    /// # 注意
    /// 迭代器存活期间，禁止对存储进行写入或删除操作。
    fn scan<R>(&mut self, range: R) -> Self::Iterator<'_>
    where
        R: RangeBounds<Vec<u8>>;
}
