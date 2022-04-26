// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <atomic>  // IWYU pragma: export
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "arrow/buffer.h"
#include "arrow/result.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

// When slicing, we do not know the null count of the sliced range without
// doing some computation. To avoid doing this eagerly, we set the null count
// to -1 (any negative number will do). When Array::null_count is called the
// first time, the null count will be computed. See ARROW-33
constexpr int64_t kUnknownNullCount = -1;

// ----------------------------------------------------------------------
// Generic array data container

/// \class ArrayData
/// \brief Mutable container for generic Arrow array data
///
/// This data structure is a self-contained representation of the memory and
/// metadata inside an Arrow array data structure (called vectors in Java). The
/// classes arrow::Array and its subclasses provide strongly-typed accessors
/// with support for the visitor pattern and other affordances.
///
/// This class is designed for easy internal data manipulation, analytical data
/// processing, and data transport to and from IPC messages. For example, we
/// could cast from int64 to float64 like so:
///
/// Int64Array arr = GetMyData();
/// auto new_data = arr.data()->Copy();
/// new_data->type = arrow::float64();
/// DoubleArray double_arr(new_data);
///
/// This object is also useful in an analytics setting where memory may be
/// reused. For example, if we had a group of operations all returning doubles,
/// say:
///
/// Log(Sqrt(Expr(arr)))
///
/// Then the low-level implementations of each of these functions could have
/// the signatures
///
/// void Log(const ArrayData& values, ArrayData* out);
///
/// As another example a function may consume one or more memory buffers in an
/// input array and replace them with newly-allocated data, changing the output
/// data type as well.
struct ARROW_EXPORT ArrayDataBase {
  ArrayDataBase() = default;
protected:

  ArrayDataBase(int64_t length, int64_t null_count, int64_t offset)
    : length(length), null_count(null_count), offset(offset) {}

  public:
  // Access a buffer's data as a typed C pointer
  template <typename T>
  inline const T* GetValues(int i, int64_t absolute_offset) const {
    if (buffers[i]) {
      return reinterpret_cast<const T*>(buffers[i]->data()) + absolute_offset;
    } else {
      return NULLPTR;
    }
  }

  // Move constructor
  ArrayDataBase(ArrayDataBase&& other) noexcept
      : length(other.length),
        offset(other.offset),
        buffers(std::move(other.buffers)),
        child_data(std::move(other.child_data)),
        dictionary(std::move(other.dictionary)) {
    SetNullCount(other.null_count);
  }

  // Copy constructor
  ArrayDataBase(const ArrayDataBase& other) noexcept
      : length(other.length),
        offset(other.offset),
        buffers(other.buffers),
        child_data(other.child_data),
        dictionary(other.dictionary) {
    SetNullCount(other.null_count);
  }

  // Move assignment
  ArrayDataBase& operator=(ArrayDataBase&& other) {
    length = other.length;
    SetNullCount(other.null_count);
    offset = other.offset;
    buffers = std::move(other.buffers);
    child_data = std::move(other.child_data);
    dictionary = std::move(other.dictionary);
    return *this;
  }

  // Copy assignment
  ArrayDataBase& operator=(const ArrayDataBase& other) {
    length = other.length;
    SetNullCount(other.null_count);
    offset = other.offset;
    buffers = other.buffers;
    child_data = other.child_data;
    dictionary = other.dictionary;
    return *this;
  }


public:
  template <typename T>
  inline const T* GetValues(int i) const {
    return GetValues<T>(i, offset);
  }

  // Like GetValues, but returns NULLPTR instead of aborting if the underlying
  // buffer is not a CPU buffer.
  template <typename T>
  inline const T* GetValuesSafe(int i, int64_t absolute_offset) const {
    if (buffers[i] && buffers[i]->is_cpu()) {
      return reinterpret_cast<const T*>(buffers[i]->data()) + absolute_offset;
    } else {
      return NULLPTR;
    }
  }

  template <typename T>
  inline const T* GetValuesSafe(int i) const {
    return GetValuesSafe<T>(i, offset);
  }

  // Access a buffer's data as a typed C pointer
  template <typename T>
  inline T* GetMutableValues(int i, int64_t absolute_offset) {
    if (buffers[i]) {
      return reinterpret_cast<T*>(buffers[i]->mutable_data()) + absolute_offset;
    } else {
      return NULLPTR;
    }
  }

  template <typename T>
  inline T* GetMutableValues(int i) {
    return GetMutableValues<T>(i, offset);
  }

  void SetNullCount(int64_t v) { null_count.store(v); }

  /// \brief Return null count, or compute and set it if it's not known
  int64_t GetNullCount() const;

  bool MayHaveNulls() const {
    // If an ArrayData is slightly malformed it may have kUnknownNullCount set
    // but no buffer
    return null_count.load() != 0 && buffers[0] != NULLPTR;
  }

  virtual DataType &Type() const = 0;

  int64_t length = 0;
  mutable std::atomic<int64_t> null_count{0};
  // The logical start point into the physical buffers (in values, not bytes).
  // Note that, for child data, this must be *added* to the child data's own offset.
  int64_t offset = 0;
  std::vector<std::shared_ptr<Buffer>> buffers;
  std::vector<std::shared_ptr<ArrayData>> child_data;

  // The dictionary for this Array, if any. Only used for dictionary type
  std::shared_ptr<ArrayData> dictionary;
};

template<typename TypeReference>
struct ARROW_EXPORT TypedArrayData : public ArrayDataBase {
  //TypedArrayData() = default;
  using ArrayDataBase::ArrayDataBase;

  TypedArrayData(TypeReference type, int64_t length,
            int64_t null_count = kUnknownNullCount, int64_t offset = 0)
      : ArrayDataBase(length, null_count, offset), type(std::move(type)) {}

  TypedArrayData(TypeReference type, int64_t length,
            std::vector<std::shared_ptr<Buffer>> buffers,
            int64_t null_count = kUnknownNullCount, int64_t offset = 0)
      : TypedArrayData(std::move(type), length, null_count, offset) {
    this->buffers = std::move(buffers);
  }

  TypedArrayData(TypeReference type, int64_t length,
            std::vector<std::shared_ptr<Buffer>> buffers,
            std::vector<std::shared_ptr<ArrayData>> child_data,
            int64_t null_count = kUnknownNullCount, int64_t offset = 0)
      : TypedArrayData(std::move(type), length, null_count, offset) {
    this->buffers = std::move(buffers);
    this->child_data = std::move(child_data);
  }

  // Move constructor
  TypedArrayData(TypedArrayData&& other) noexcept
      : ArrayDataBase(std::forward<TypedArrayData&&>(other)),
        type(std::move(other.type)) {}

  // Copy constructor
  TypedArrayData(const TypedArrayData& other) noexcept
      : ArrayDataBase(other),
        type(other.type) {}

  // Move assignment
  TypedArrayData& operator=(TypedArrayData&& other) {
    type = std::move(other.type);
    *static_cast<ArrayDataBase *>(this) = std::forward<TypedArrayData&&>(other);
    return *this;
  }

  // Copy assignment
  TypedArrayData& operator=(const TypedArrayData& other) {
    type = other.type;
    *static_cast<ArrayDataBase *>(this) = other;
    return *this;
  }


  static std::shared_ptr<TypedArrayData<TypeReference>> Make(TypeReference type, int64_t length,
                                         std::vector<std::shared_ptr<Buffer>> buffers,
                                         int64_t null_count = kUnknownNullCount,
                                         int64_t offset = 0);

  static std::shared_ptr<TypedArrayData> Make(
      TypeReference type, int64_t length,
      std::vector<std::shared_ptr<Buffer>> buffers,
      std::vector<std::shared_ptr<ArrayData>> child_data,
      int64_t null_count = kUnknownNullCount, int64_t offset = 0);

  static std::shared_ptr<TypedArrayData> Make(
      TypeReference type, int64_t length,
      std::vector<std::shared_ptr<Buffer>> buffers,
      std::vector<std::shared_ptr<ArrayData>> child_data,
      std::shared_ptr<ArrayData> dictionary, int64_t null_count = kUnknownNullCount,
      int64_t offset = 0);

  static std::shared_ptr<TypedArrayData> Make(TypeReference type, int64_t length,
                                         int64_t null_count = kUnknownNullCount,
                                         int64_t offset = 0);



  std::shared_ptr<TypedArrayData> Copy() const { return std::make_shared<TypedArrayData>(*this); }

  /// \brief Construct a zero-copy slice of the data with the given offset and length
  std::shared_ptr<TypedArrayData> Slice(int64_t offset, int64_t length) const;

  /// \brief Input-checking variant of Slice
  ///
  /// An Invalid Status is returned if the requested slice falls out of bounds.
  /// Note that unlike Slice, `length` isn't clamped to the available buffer size.
  Result<std::shared_ptr<TypedArrayData>> SliceSafe(int64_t offset, int64_t length) const;

  virtual DataType &Type() const { return *type; };

  TypeReference type;
};

template struct TypedArrayData<std::shared_ptr<DataType>>;
template struct TypedArrayData<DataType *>;
using ArrayData = TypedArrayData<std::shared_ptr<DataType>>;
//using ExecArrayData = TypedArrayData<std::shared_ptr<DataType>>;

struct ExecArrayData : public TypedArrayData<DataType *> {
public:
  using TypedArrayData::TypedArrayData;
  explicit ExecArrayData(ArrayData &regular) {
    type = regular.type.get();
    *static_cast<ArrayDataBase *>(this) = regular;
  }

  std::shared_ptr<ExecArrayData> Slice(int64_t offset, int64_t length) const;

  std::shared_ptr<ArrayData> ToArrayData();
};

namespace internal {

/// Construct a zero-copy view of this ArrayData with the given type.
///
/// This method checks if the types are layout-compatible.
/// Nested types are traversed in depth-first order. Data buffers must have
/// the same item sizes, even though the logical types may be different.
/// An error is returned if the types are not layout-compatible.
ARROW_EXPORT
Result<std::shared_ptr<ArrayData>> GetArrayView(const std::shared_ptr<ArrayData>& data,
                                                const std::shared_ptr<DataType>& type);

}  // namespace internal
}  // namespace arrow
