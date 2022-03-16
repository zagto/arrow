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

#include "arrow/compute/exec/key_hash.h"

#include <memory.h>

#include <algorithm>
#include <cstdint>

#include "arrow/compute/exec/util.h"
#include "arrow/util/ubsan.h"

namespace arrow {
namespace compute {

inline uint32_t Hashing32::round(uint32_t acc, uint32_t input) {
  acc += input * PRIME32_2;
  acc = ROTL(acc, 13);
  acc *= PRIME32_1;
  return acc;
}

inline uint32_t Hashing32::combine_accumulators(uint32_t acc1, uint32_t acc2,
                                                uint32_t acc3, uint32_t acc4) {
  return ROTL(acc1, 1) + ROTL(acc2, 7) + ROTL(acc3, 12) + ROTL(acc4, 18);
}

inline void Hashing32::process_full_stripes(uint64_t num_stripes, const uint8_t* key,
                                            uint32_t* out_acc1, uint32_t* out_acc2,
                                            uint32_t* out_acc3, uint32_t* out_acc4) {
  uint32_t acc1, acc2, acc3, acc4;
  acc1 = static_cast<uint32_t>(
      (static_cast<uint64_t>(PRIME32_1) + static_cast<uint64_t>(PRIME32_2)) & 0xffffffff);
  acc2 = PRIME32_2;
  acc3 = 0;
  acc4 = static_cast<uint32_t>(-static_cast<int32_t>(PRIME32_1));

  for (int64_t istripe = 0; istripe < static_cast<int64_t>(num_stripes) - 1; ++istripe) {
    const uint8_t* stripe = key + istripe * 4 * sizeof(uint32_t);
    uint32_t stripe1 = util::SafeLoadAs<const uint32_t>(stripe);
    uint32_t stripe2 = util::SafeLoadAs<const uint32_t>(stripe + sizeof(uint32_t));
    uint32_t stripe3 = util::SafeLoadAs<const uint32_t>(stripe + 2 * sizeof(uint32_t));
    uint32_t stripe4 = util::SafeLoadAs<const uint32_t>(stripe + 3 * sizeof(uint32_t));
    acc1 = round(acc1, stripe1);
    acc2 = round(acc2, stripe2);
    acc3 = round(acc3, stripe3);
    acc4 = round(acc4, stripe4);
  }

  *out_acc1 = acc1;
  *out_acc2 = acc2;
  *out_acc3 = acc3;
  *out_acc4 = acc4;
}

inline void Hashing32::process_last_stripe(uint32_t mask1, uint32_t mask2, uint32_t mask3,
                                           uint32_t mask4, const uint8_t* last_stripe,
                                           uint32_t* acc1, uint32_t* acc2, uint32_t* acc3,
                                           uint32_t* acc4) {
  uint32_t stripe1 = util::SafeLoadAs<const uint32_t>(last_stripe);
  uint32_t stripe2 = util::SafeLoadAs<const uint32_t>(last_stripe + sizeof(uint32_t));
  uint32_t stripe3 = util::SafeLoadAs<const uint32_t>(last_stripe + 2 * sizeof(uint32_t));
  uint32_t stripe4 = util::SafeLoadAs<const uint32_t>(last_stripe + 3 * sizeof(uint32_t));
  stripe1 &= mask1;
  stripe2 &= mask2;
  stripe3 &= mask3;
  stripe4 &= mask4;
  *acc1 = round(*acc1, stripe1);
  *acc2 = round(*acc2, stripe2);
  *acc3 = round(*acc3, stripe3);
  *acc4 = round(*acc4, stripe4);
}

inline void Hashing32::stripe_mask(int i, uint32_t* mask1, uint32_t* mask2,
                                   uint32_t* mask3, uint32_t* mask4) {
  // Return a 16 byte mask (encoded as 4x 32-bit integers), where the first i
  // bytes are 0xff and the remaining ones are 0x00
  //

  ARROW_DCHECK(i >= 0 && i <= kStripeSize);

  static const uint32_t bytes[] = {~0U, ~0U, ~0U, ~0U, 0U, 0U, 0U, 0U};
  int offset = kStripeSize - i;
  const uint8_t* mask_base = reinterpret_cast<const uint8_t*>(bytes) + offset;
  *mask1 = util::SafeLoadAs<uint32_t>(mask_base);
  *mask2 = util::SafeLoadAs<uint32_t>(mask_base + sizeof(uint32_t));
  *mask3 = util::SafeLoadAs<uint32_t>(mask_base + 2 * sizeof(uint32_t));
  *mask4 = util::SafeLoadAs<uint32_t>(mask_base + 3 * sizeof(uint32_t));
}

template <bool T_COMBINE_HASHES>
void Hashing32::hash_fixedlen_imp(uint32_t num_rows, uint64_t length, const uint8_t* keys,
                                  uint32_t* hashes) {
  // Calculate the number of rows that skip the last 16 bytes
  //
  uint32_t num_rows_safe = num_rows;
  while (num_rows_safe > 0 && (num_rows - num_rows_safe) * length < kStripeSize) {
    --num_rows_safe;
  }

  // Compute masks for the last 16 byte stripe
  //
  uint64_t num_stripes = bit_util::CeilDiv(length, kStripeSize);
  uint32_t mask1, mask2, mask3, mask4;
  stripe_mask(((length - 1) & (kStripeSize - 1)) + 1, &mask1, &mask2, &mask3, &mask4);

  for (uint32_t i = 0; i < num_rows_safe; ++i) {
    const uint8_t* key = keys + static_cast<uint64_t>(i) * length;
    uint32_t acc1, acc2, acc3, acc4;
    process_full_stripes(num_stripes, key, &acc1, &acc2, &acc3, &acc4);
    process_last_stripe(mask1, mask2, mask3, mask4, key + (num_stripes - 1) * kStripeSize,
                        &acc1, &acc2, &acc3, &acc4);
    uint32_t acc = combine_accumulators(acc1, acc2, acc3, acc4);
    acc = avalanche(acc);

    if (T_COMBINE_HASHES) {
      hashes[i] = combine_hashes(hashes[i], acc);
    } else {
      hashes[i] = acc;
    }
  }

  uint32_t last_stripe_copy[4];
  for (uint32_t i = num_rows_safe; i < num_rows; ++i) {
    const uint8_t* key = keys + static_cast<uint64_t>(i) * length;
    uint32_t acc1, acc2, acc3, acc4;
    process_full_stripes(num_stripes, key, &acc1, &acc2, &acc3, &acc4);
    memcpy(last_stripe_copy, key + (num_stripes - 1) * kStripeSize,
           length - (num_stripes - 1) * kStripeSize);
    process_last_stripe(mask1, mask2, mask3, mask4,
                        reinterpret_cast<const uint8_t*>(last_stripe_copy), &acc1, &acc2,
                        &acc3, &acc4);
    uint32_t acc = combine_accumulators(acc1, acc2, acc3, acc4);
    acc = avalanche(acc);

    if (T_COMBINE_HASHES) {
      hashes[i] = combine_hashes(hashes[i], acc);
    } else {
      hashes[i] = acc;
    }
  }
}

template <typename T, bool T_COMBINE_HASHES>
void Hashing32::hash_varlen_imp(uint32_t num_rows, const T* offsets,
                                const uint8_t* concatenated_keys, uint32_t* hashes) {
  // Calculate the number of rows that skip the last 16 bytes
  //
  uint32_t num_rows_safe = num_rows;
  while (num_rows_safe > 0 && offsets[num_rows] - offsets[num_rows_safe] < kStripeSize) {
    --num_rows_safe;
  }

  for (uint32_t i = 0; i < num_rows_safe; ++i) {
    uint64_t length = offsets[i + 1] - offsets[i];

    // Compute masks for the last 16 byte stripe.
    // For an empty string set number of stripes to 1 but mask to all zeroes.
    //
    int is_non_empty = length == 0 ? 0 : 1;
    uint64_t num_stripes = bit_util::CeilDiv(length, kStripeSize) + (1 - is_non_empty);
    uint32_t mask1, mask2, mask3, mask4;
    stripe_mask(((length - is_non_empty) & (kStripeSize - 1)) + is_non_empty, &mask1,
                &mask2, &mask3, &mask4);

    const uint8_t* key = concatenated_keys + offsets[i];
    uint32_t acc1, acc2, acc3, acc4;
    process_full_stripes(num_stripes, key, &acc1, &acc2, &acc3, &acc4);
    if (num_stripes > 0) {
      process_last_stripe(mask1, mask2, mask3, mask4,
                          key + (num_stripes - 1) * kStripeSize, &acc1, &acc2, &acc3,
                          &acc4);
    }
    uint32_t acc = combine_accumulators(acc1, acc2, acc3, acc4);
    acc = avalanche(acc);

    if (T_COMBINE_HASHES) {
      hashes[i] = combine_hashes(hashes[i], acc);
    } else {
      hashes[i] = acc;
    }
  }

  uint32_t last_stripe_copy[4];
  for (uint32_t i = num_rows_safe; i < num_rows; ++i) {
    uint64_t length = offsets[i + 1] - offsets[i];

    // Compute masks for the last 16 byte stripe.
    // For an empty string set number of stripes to 1 but mask to all zeroes.
    //
    int is_non_empty = length == 0 ? 0 : 1;
    uint64_t num_stripes = bit_util::CeilDiv(length, kStripeSize) + (1 - is_non_empty);
    uint32_t mask1, mask2, mask3, mask4;
    stripe_mask(((length - is_non_empty) & (kStripeSize - 1)) + is_non_empty, &mask1,
                &mask2, &mask3, &mask4);

    const uint8_t* key = concatenated_keys + offsets[i];
    uint32_t acc1, acc2, acc3, acc4;
    process_full_stripes(num_stripes, key, &acc1, &acc2, &acc3, &acc4);
    if (length > 0) {
      memcpy(last_stripe_copy, key + (num_stripes - 1) * kStripeSize,
             length - (num_stripes - 1) * kStripeSize);
    }
    if (num_stripes > 0) {
      process_last_stripe(mask1, mask2, mask3, mask4,
                          reinterpret_cast<const uint8_t*>(last_stripe_copy), &acc1,
                          &acc2, &acc3, &acc4);
    }
    uint32_t acc = combine_accumulators(acc1, acc2, acc3, acc4);
    acc = avalanche(acc);

    if (T_COMBINE_HASHES) {
      hashes[i] = combine_hashes(hashes[i], acc);
    } else {
      hashes[i] = acc;
    }
  }
}

void Hashing32::hash_varlen(int64_t hardware_flags, bool combine_hashes,
                            uint32_t num_rows, const uint32_t* offsets,
                            const uint8_t* concatenated_keys, uint32_t* hashes,
                            uint32_t* hashes_temp_for_combine) {
  uint32_t num_processed = 0;
#if defined(ARROW_HAVE_AVX2)
  if (hardware_flags & arrow::internal::CpuInfo::AVX2) {
    num_processed = hash_varlen_avx2(combine_hashes, num_rows, offsets, concatenated_keys,
                                     hashes, hashes_temp_for_combine);
  }
#endif
  if (combine_hashes) {
    hash_varlen_imp<uint32_t, true>(num_rows - num_processed, offsets + num_processed,
                                    concatenated_keys, hashes + num_processed);
  } else {
    hash_varlen_imp<uint32_t, false>(num_rows - num_processed, offsets + num_processed,
                                     concatenated_keys, hashes + num_processed);
  }
}

void Hashing32::hash_varlen(int64_t hardware_flags, bool combine_hashes,
                            uint32_t num_rows, const uint64_t* offsets,
                            const uint8_t* concatenated_keys, uint32_t* hashes,
                            uint32_t* hashes_temp_for_combine) {
  uint32_t num_processed = 0;
#if defined(ARROW_HAVE_AVX2)
  if (hardware_flags & arrow::internal::CpuInfo::AVX2) {
    num_processed = hash_varlen_avx2(combine_hashes, num_rows, offsets, concatenated_keys,
                                     hashes, hashes_temp_for_combine);
  }
#endif
  if (combine_hashes) {
    hash_varlen_imp<uint64_t, true>(num_rows - num_processed, offsets + num_processed,
                                    concatenated_keys, hashes + num_processed);
  } else {
    hash_varlen_imp<uint64_t, false>(num_rows - num_processed, offsets + num_processed,
                                     concatenated_keys, hashes + num_processed);
  }
}

template <bool T_COMBINE_HASHES>
void Hashing32::hash_bit_imp(int64_t bit_offset, uint32_t num_keys, const uint8_t* keys,
                             uint32_t* hashes) {
  for (uint32_t i = 0; i < num_keys; ++i) {
    uint32_t bit = bit_util::GetBit(keys, bit_offset + i) ? 1 : 0;
    uint32_t hash = PRIME32_1 * (1 - bit) + PRIME32_2 * bit;

    if (T_COMBINE_HASHES) {
      hashes[i] = combine_hashes(hashes[i], hash);
    } else {
      hashes[i] = hash;
    }
  }
}

void Hashing32::hash_bit(bool combine_hashes, int64_t bit_offset, uint32_t num_keys,
                         const uint8_t* keys, uint32_t* hashes) {
  if (combine_hashes) {
    hash_bit_imp<true>(bit_offset, num_keys, keys, hashes);
  } else {
    hash_bit_imp<false>(bit_offset, num_keys, keys, hashes);
  }
}

template <bool T_COMBINE_HASHES, typename T>
void Hashing32::hash_int_imp(uint32_t num_keys, const T* keys, uint32_t* hashes) {
  constexpr uint64_t multiplier = 11400714785074694791ULL;
  for (uint32_t ikey = 0; ikey < num_keys; ++ikey) {
    uint64_t x = static_cast<uint64_t>(keys[ikey]);
    uint32_t hash = static_cast<uint32_t>(BYTESWAP(x * multiplier));

    if (T_COMBINE_HASHES) {
      hashes[ikey] = combine_hashes(hashes[ikey], hash);
    } else {
      hashes[ikey] = hash;
    }
  }
}

void Hashing32::hash_int(bool combine_hashes, uint32_t num_keys, uint64_t length_key,
                         const uint8_t* keys, uint32_t* hashes) {
  switch (length_key) {
    case sizeof(uint8_t):
      if (combine_hashes) {
        hash_int_imp<true, uint8_t>(num_keys, keys, hashes);
      } else {
        hash_int_imp<false, uint8_t>(num_keys, keys, hashes);
      }
      break;
    case sizeof(uint16_t):
      if (combine_hashes) {
        hash_int_imp<true, uint16_t>(num_keys, reinterpret_cast<const uint16_t*>(keys),
                                     hashes);
      } else {
        hash_int_imp<false, uint16_t>(num_keys, reinterpret_cast<const uint16_t*>(keys),
                                      hashes);
      }
      break;
    case sizeof(uint32_t):
      if (combine_hashes) {
        hash_int_imp<true, uint32_t>(num_keys, reinterpret_cast<const uint32_t*>(keys),
                                     hashes);
      } else {
        hash_int_imp<false, uint32_t>(num_keys, reinterpret_cast<const uint32_t*>(keys),
                                      hashes);
      }
      break;
    case sizeof(uint64_t):
      if (combine_hashes) {
        hash_int_imp<true, uint64_t>(num_keys, reinterpret_cast<const uint64_t*>(keys),
                                     hashes);
      } else {
        hash_int_imp<false, uint64_t>(num_keys, reinterpret_cast<const uint64_t*>(keys),
                                      hashes);
      }
      break;
    default:
      ARROW_DCHECK(false);
      break;
  }
}

void Hashing32::hash_fixed(int64_t hardware_flags, bool combine_hashes, uint32_t num_rows,
                           uint64_t length, const uint8_t* keys, uint32_t* hashes,
                           uint32_t* hashes_temp_for_combine) {
  if (ARROW_POPCOUNT64(length) == 1 && length <= sizeof(uint64_t)) {
    hash_int(combine_hashes, num_rows, length, keys, hashes);
    return;
  }

  uint32_t num_processed = 0;
#if defined(ARROW_HAVE_AVX2)
  if (hardware_flags & arrow::internal::CpuInfo::AVX2) {
    num_processed = hash_fixedlen_avx2(combine_hashes, num_rows, length, keys, hashes,
                                       hashes_temp_for_combine);
  }
#endif
  if (combine_hashes) {
    hash_fixedlen_imp<true>(num_rows - num_processed, length,
                            keys + length * num_processed, hashes + num_processed);
  } else {
    hash_fixedlen_imp<false>(num_rows - num_processed, length,
                             keys + length * num_processed, hashes + num_processed);
  }
}

void Hashing32::HashMultiColumn(const std::vector<KeyEncoder::KeyColumnArray>& cols,
                                KeyEncoder::KeyEncoderContext* ctx, uint32_t* hashes) {
  uint32_t num_rows = static_cast<uint32_t>(cols[0].length());

  constexpr uint32_t max_batch_size = util::MiniBatch::kMiniBatchLength;

  auto hash_temp_buf = util::TempVectorHolder<uint32_t>(ctx->stack, max_batch_size);
  uint32_t* hash_temp = hash_temp_buf.mutable_data();

  auto null_indices_buf = util::TempVectorHolder<uint16_t>(ctx->stack, max_batch_size);
  uint16_t* null_indices = null_indices_buf.mutable_data();
  int num_null_indices;

  auto null_hash_temp_buf = util::TempVectorHolder<uint32_t>(ctx->stack, max_batch_size);
  uint32_t* null_hash_temp = null_hash_temp_buf.mutable_data();

  for (uint32_t first_row = 0; first_row < num_rows;) {
    uint32_t batch_size_next = std::min(num_rows - first_row, max_batch_size);

    for (size_t icol = 0; icol < cols.size(); ++icol) {
      if (cols[icol].metadata().is_null_type) {
        if (icol == 0) {
          for (uint32_t i = 0; i < batch_size_next; ++i) {
            hashes[first_row + i] = 0;
          }
        } else {
          for (uint32_t i = 0; i < batch_size_next; ++i) {
            hashes[first_row + i] = combine_hashes(hashes[first_row + i], 0);
          }
        }
        continue;
      }

      // Get indices of null values within current minibatch
      if (cols[icol].data(0)) {
        util::bit_util::bits_to_indexes(
            0, ctx->hardware_flags, batch_size_next, cols[icol].data(0) + first_row / 8,
            &num_null_indices, null_indices, first_row % 8 + cols[icol].bit_offset(0));
        // Make a backup copy of hash for nulls if needed
        if (icol > 0) {
          for (int i = 0; i < num_null_indices; ++i) {
            null_hash_temp[i] = hashes[first_row + null_indices[i]];
          }
        }
      }

      if (cols[icol].metadata().is_fixed_length) {
        uint32_t col_width = cols[icol].metadata().fixed_length;
        if (col_width == 0) {
          hash_bit(icol > 0, cols[icol].bit_offset(1), batch_size_next,
                   cols[icol].data(1) + first_row / 8, hashes + first_row);
        } else {
          hash_fixed(ctx->hardware_flags, icol > 0, batch_size_next, col_width,
                     cols[icol].data(1) + first_row * col_width, hashes + first_row,
                     hash_temp);
        }
      } else {
        // TODO: add support for 64-bit offsets
        hash_varlen(ctx->hardware_flags, icol > 0, batch_size_next,
                    cols[icol].offsets() + first_row, cols[icol].data(2),
                    hashes + first_row, hash_temp);
      }

      // Zero hash for nulls
      if (cols[icol].data(0)) {
        if (icol == 0) {
          for (int i = 0; i < num_null_indices; ++i) {
            hashes[first_row + null_indices[i]] = 0;
          }
        } else {
          for (int i = 0; i < num_null_indices; ++i) {
            hashes[first_row + null_indices[i]] = combine_hashes(null_hash_temp[i], 0);
          }
        }
      }
    }

    first_row += batch_size_next;
  }
}

void Hashing32::HashBatch(const ExecBatch& key_batch, int start_row, int num_rows,
                          uint32_t* hashes,
                          std::vector<KeyEncoder::KeyColumnArray>& column_arrays,
                          int64_t hardware_flags, util::TempVectorStack* temp_stack) {
  ColumnArraysFromExecBatch(key_batch, start_row, num_rows, column_arrays);
  KeyEncoder::KeyEncoderContext ctx;
  ctx.hardware_flags = hardware_flags;
  ctx.stack = temp_stack;
  HashMultiColumn(column_arrays, &ctx, hashes);
}

}  // namespace compute
}  // namespace arrow
