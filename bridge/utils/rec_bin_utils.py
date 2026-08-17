#!/usr/bin/env python3
"""
Rec Binary 文件读取工具

用于读取数据采集侧生成的 .bin 格式的录制数据文件。
.bin 文件是一种紧凑的二进制格式，相比 CSV 具有更小的文件体积和更快的读取速度。

文件格式:
  - 前 8 字节: header_offset (int64, 表头结束位置)
  - 接下来 8 字节: row_len (int64, 每行数据的字节长度)
  - 接下来 header_offset - 16 字节: CSV格式的表头行
  - 之后是数据行，每行 row_len 字节:
    - 前 36 字节: frame_uuid (ASCII字符串)
    - 之后每 8 字节为一个字段值:
      - 以 _int 结尾的字段: int64
      - 其他字段: float64 (double)

使用方法:
  from utils.rec_bin_utils import read_monitor_bin, read_bin_header, RecBinReader
  
  # 方式1: 迭代器方式读取
  for row in read_monitor_bin("path/to/file.bin"):
      print(row['frame_uuid'])
  
  # 方式2: 只读取表头
  header = read_bin_header("path/to/file.bin")
  
  # 方式3: 使用 Reader 类（支持更多控制）
  with RecBinReader("path/to/file.bin") as reader:
      print(reader.header)
      for row in reader:
          print(row)
"""

import csv
import struct
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union


# UUID 字段固定为 36 字节
UUID_SIZE = 36


def read_bin_header(path: Union[str, Path]) -> List[str]:
    """
    读取 .bin 文件的表头
    
    Args:
        path: .bin 文件路径
        
    Returns:
        表头字段名列表
    """
    with open(path, "rb") as f:
        header_offset = struct.unpack("<q", f.read(8))[0]
        _ = struct.unpack("<q", f.read(8))[0]  # row_len
        header_bytes = f.read(header_offset - 16)
        header_line = header_bytes.decode("utf-8").strip()
        header = next(csv.reader([header_line]))
        return header


def read_bin_metadata(path: Union[str, Path]) -> Dict[str, any]:
    """
    读取 .bin 文件的元数据
    
    Args:
        path: .bin 文件路径
        
    Returns:
        包含 header_offset, row_len, header, num_fields 的字典
    """
    with open(path, "rb") as f:
        header_offset = struct.unpack("<q", f.read(8))[0]
        row_len = struct.unpack("<q", f.read(8))[0]
        header_bytes = f.read(header_offset - 16)
        header_line = header_bytes.decode("utf-8").strip()
        header = next(csv.reader([header_line]))
        
        # 计算行数
        f.seek(0, 2)  # 移动到文件末尾
        file_size = f.tell()
        data_size = file_size - header_offset
        num_rows = data_size // row_len if row_len > 0 else 0
        
        return {
            "header_offset": header_offset,
            "row_len": row_len,
            "header": header,
            "num_fields": len(header),
            "num_rows": num_rows,
            "file_size": file_size,
        }


def read_monitor_bin(path: Union[str, Path]) -> Iterator[Dict[str, object]]:
    """
    迭代读取 .bin 文件的所有行
    
    Args:
        path: .bin 文件路径
        
    Yields:
        每行数据的字典，包含所有字段
    """
    with open(path, "rb") as f:
        header_offset = struct.unpack("<q", f.read(8))[0]
        row_len = struct.unpack("<q", f.read(8))[0]
        header_bytes = f.read(header_offset - 16)
        header_line = header_bytes.decode("utf-8").strip()
        header = next(csv.reader([header_line]))

        field_names = header[1:]  # skip frame_uuid (第一个字段)
        
        while True:
            row = f.read(row_len)
            if len(row) < row_len:
                break
                
            uuid_raw = row[:UUID_SIZE]
            frame_uuid = uuid_raw.decode("ascii", errors="ignore").rstrip("\0")
            
            values = {}
            offset = UUID_SIZE
            
            for name in field_names:
                chunk = row[offset:offset + 8]
                offset += 8
                if name.endswith("_int"):
                    values[name] = struct.unpack("<q", chunk)[0]
                else:
                    values[name] = struct.unpack("<d", chunk)[0]
                    
            values["frame_uuid"] = frame_uuid
            yield values


class RecBinReader:
    """
    Rec Binary 文件读取器
    
    支持上下文管理器和迭代器协议，提供更多的控制选项。
    
    Attributes:
        path: 文件路径
        header: 表头字段列表
        header_offset: 表头结束位置
        row_len: 每行字节长度
        num_rows: 总行数
    """
    
    def __init__(self, path: Union[str, Path]):
        """
        初始化读取器
        
        Args:
            path: .bin 文件路径
        """
        self.path = Path(path)
        self._file = None
        self._header: Optional[List[str]] = None
        self._header_offset: int = 0
        self._row_len: int = 0
        self._field_names: Optional[List[str]] = None
        self._num_rows: Optional[int] = None
        
    def open(self):
        """打开文件并读取表头"""
        if self._file is not None:
            return
            
        self._file = open(self.path, "rb")
        self._header_offset = struct.unpack("<q", self._file.read(8))[0]
        self._row_len = struct.unpack("<q", self._file.read(8))[0]
        
        header_bytes = self._file.read(self._header_offset - 16)
        header_line = header_bytes.decode("utf-8").strip()
        self._header = next(csv.reader([header_line]))
        self._field_names = self._header[1:]  # skip frame_uuid
        
        # 计算行数
        self._file.seek(0, 2)
        file_size = self._file.tell()
        data_size = file_size - self._header_offset
        self._num_rows = data_size // self._row_len if self._row_len > 0 else 0
        
        # 移动到数据开始位置
        self._file.seek(self._header_offset)
        
    def close(self):
        """关闭文件"""
        if self._file is not None:
            self._file.close()
            self._file = None
            
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
        
    @property
    def header(self) -> List[str]:
        """获取表头"""
        if self._header is None:
            self.open()
        return self._header
        
    @property
    def header_offset(self) -> int:
        """获取表头结束位置"""
        if self._file is None:
            self.open()
        return self._header_offset
        
    @property
    def row_len(self) -> int:
        """获取每行字节长度"""
        if self._file is None:
            self.open()
        return self._row_len
        
    @property
    def num_rows(self) -> int:
        """获取总行数"""
        if self._num_rows is None:
            self.open()
        return self._num_rows
        
    def seek_to_row(self, row_idx: int):
        """
        移动到指定行
        
        Args:
            row_idx: 行索引（从0开始）
        """
        if self._file is None:
            self.open()
        position = self._header_offset + row_idx * self._row_len
        self._file.seek(position)
        
    def read_row(self) -> Optional[Dict[str, object]]:
        """
        读取当前行
        
        Returns:
            行数据字典，如果到达文件末尾返回 None
        """
        if self._file is None:
            self.open()
            
        row = self._file.read(self._row_len)
        if len(row) < self._row_len:
            return None
            
        uuid_raw = row[:UUID_SIZE]
        frame_uuid = uuid_raw.decode("ascii", errors="ignore").rstrip("\0")
        
        values = {}
        offset = UUID_SIZE
        
        for name in self._field_names:
            chunk = row[offset:offset + 8]
            offset += 8
            if name.endswith("_int"):
                values[name] = struct.unpack("<q", chunk)[0]
            else:
                values[name] = struct.unpack("<d", chunk)[0]
                
        values["frame_uuid"] = frame_uuid
        return values
        
    def read_row_uuid_only(self) -> Optional[str]:
        """
        只读取当前行的 frame_uuid（用于快速匹配）
        
        Returns:
            frame_uuid 字符串，如果到达文件末尾返回 None
        """
        if self._file is None:
            self.open()
            
        uuid_raw = self._file.read(UUID_SIZE)
        if len(uuid_raw) < UUID_SIZE:
            return None
            
        # 跳过剩余字段
        self._file.read(self._row_len - UUID_SIZE)
        
        return uuid_raw.decode("ascii", errors="ignore").rstrip("\0")
        
    def __iter__(self):
        """迭代所有行"""
        if self._file is None:
            self.open()
        self._file.seek(self._header_offset)
        return self
        
    def __next__(self) -> Dict[str, object]:
        row = self.read_row()
        if row is None:
            raise StopIteration
        return row


def iter_bin_uuids(path: Union[str, Path]) -> Iterator[Tuple[int, str]]:
    """
    快速迭代 .bin 文件中的所有 frame_uuid
    
    只读取 UUID 字段，跳过其他数据，用于快速匹配。
    
    Args:
        path: .bin 文件路径
        
    Yields:
        (row_idx, frame_uuid) 元组
    """
    with open(path, "rb") as f:
        header_offset = struct.unpack("<q", f.read(8))[0]
        row_len = struct.unpack("<q", f.read(8))[0]
        
        # 跳过表头
        f.seek(header_offset)
        
        row_idx = 0
        while True:
            uuid_raw = f.read(UUID_SIZE)
            if len(uuid_raw) < UUID_SIZE:
                break
                
            frame_uuid = uuid_raw.decode("ascii", errors="ignore").rstrip("\0")
            
            # 跳过剩余字段
            remaining = row_len - UUID_SIZE
            f.read(remaining)
            
            if frame_uuid:  # 只返回非空 UUID
                yield row_idx, frame_uuid
                
            row_idx += 1


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python rec_bin_utils.py <path_to_bin>")
        print("\nOptions:")
        print("  --header    只打印表头")
        print("  --meta      打印元数据")
        print("  --rows N    打印前N行数据 (默认3)")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if "--header" in sys.argv:
        header = read_bin_header(path)
        print(f"表头字段数: {len(header)}")
        for i, h in enumerate(header):
            print(f"{i}: {h}")
    elif "--meta" in sys.argv:
        meta = read_bin_metadata(path)
        print(f"Header Offset: {meta['header_offset']}")
        print(f"Row Length: {meta['row_len']}")
        print(f"Num Fields: {meta['num_fields']}")
        print(f"Num Rows: {meta['num_rows']}")
        print(f"File Size: {meta['file_size']} bytes")
    else:
        n_rows = 3
        for i, arg in enumerate(sys.argv):
            if arg == "--rows" and i + 1 < len(sys.argv):
                n_rows = int(sys.argv[i + 1])
                break
                
        print(f"前 {n_rows} 行数据:")
        for i, row in enumerate(read_monitor_bin(path)):
            if i >= n_rows:
                break
            print(f"\n=== Row {i} ===")
            print(f"frame_uuid: {row['frame_uuid']}")
            # 只打印几个关键字段
            for key in ['Recording Status.frame_int', 'Recording Status.time']:
                if key in row:
                    print(f"{key}: {row[key]}")
