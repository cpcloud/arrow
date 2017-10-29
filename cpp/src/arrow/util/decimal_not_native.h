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

#ifndef ARROW_DECIMAL_NOT_NATIVE_H
#define ARROW_DECIMAL_NOT_NATIVE_H

#include <array>
#include <cstdint>
#include <ostream>
#include <string>
#include <type_traits>

#include "arrow/status.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {

static constexpr uint64_t kIntMask = 0xFFFFFFFF;
static constexpr auto kCarryBit = static_cast<uint64_t>(1) << static_cast<uint64_t>(32);

/// Represents a signed 128-bit integer in two's complement.
/// Calculations wrap around and overflow is ignored.
///
/// For a discussion of the algorithms, look at Knuth's volume 2,
/// Semi-numerical Algorithms section 4.3.1.
///
/// Adapted from the Apache ORC C++ implementation
class ARROW_EXPORT Decimal128 {
 public:
  /// \brief Create an Decimal128 from the two's complement representation.
  constexpr Decimal128(int64_t high, uint64_t low) : high_bits_(high), low_bits_(low) {}

  /// \brief Empty constructor creates an Decimal128 with a value of 0.
  constexpr Decimal128() : Decimal128(0, 0) {}

  /// \brief Convert any integer value into an Decimal128.
  template <typename T,
            typename = typename std::enable_if<std::is_integral<T>::value, T>::type>
  constexpr Decimal128(T value)
      : Decimal128(static_cast<int64_t>(value) >= 0 ? 0 : -1,
                   static_cast<uint64_t>(value)) {}

  /// \brief Parse the number from a base 10 string representation.
  explicit Decimal128(const std::string& value);

  /// \brief Create an Decimal128 from an array of bytes. Bytes are assumed to be in
  /// little endian byte order.
  explicit Decimal128(const uint8_t* bytes);

  /// \brief Negate the current value
  Decimal128& Negate() {
    low_bits_ = ~low_bits_ + 1;
    high_bits_ = ~high_bits_;
    if (low_bits_ == 0) {
      ++high_bits_;
    }
    return *this;
  }

  /// \brief Add a number to this one. The result is truncated to 128 bits.
  Decimal128& operator+=(const Decimal128& right) {
    const uint64_t sum = low_bits_ + right.low_bits_;
    high_bits_ += right.high_bits_;
    if (sum < low_bits_) {
      ++high_bits_;
    }
    low_bits_ = sum;
    return *this;
  }

  /// \brief Subtract a number from this one. The result is truncated to 128 bits.
  Decimal128& operator-=(const Decimal128& right) {
    const uint64_t diff = low_bits_ - right.low_bits_;
    high_bits_ -= right.high_bits_;
    if (diff > low_bits_) {
      --high_bits_;
    }
    low_bits_ = diff;
    return *this;
  }

  /// \brief Multiply this number by another number. The result is truncated to 128 bits.
  Decimal128& operator*=(const Decimal128& right) {
    // Break the left and right numbers into 32 bit chunks
    // so that we can multiply them without overflow.
    const uint64_t L0 = static_cast<uint64_t>(high_bits_) >> 32;
    const uint64_t L1 = static_cast<uint64_t>(high_bits_) & kIntMask;
    const uint64_t L2 = low_bits_ >> 32;
    const uint64_t L3 = low_bits_ & kIntMask;

    const uint64_t R0 = static_cast<uint64_t>(right.high_bits_) >> 32;
    const uint64_t R1 = static_cast<uint64_t>(right.high_bits_) & kIntMask;
    const uint64_t R2 = right.low_bits_ >> 32;
    const uint64_t R3 = right.low_bits_ & kIntMask;

    uint64_t product = L3 * R3;
    low_bits_ = product & kIntMask;

    uint64_t sum = product >> 32;

    product = L2 * R3;
    sum += product;

    product = L3 * R2;
    sum += product;

    low_bits_ += sum << 32;

    high_bits_ = static_cast<int64_t>(sum < product ? kCarryBit : 0);
    if (sum < product) {
      high_bits_ += kCarryBit;
    }

    high_bits_ += static_cast<int64_t>(sum >> 32);
    high_bits_ += L1 * R3 + L2 * R2 + L3 * R1;
    high_bits_ += (L0 * R3 + L1 * R2 + L2 * R1 + L3 * R0) << 32;
    return *this;
  }

  /// \brief Cast the value to char. This is used when converting the value a string.
  explicit operator char() const {
    DCHECK(high_bits_ == 0 || high_bits_ == -1)
        << "Trying to cast an Decimal128 greater than the value range of a "
           "char. high_bits_ must be equal to 0 or -1, got: "
        << high_bits_;
    DCHECK_LE(low_bits_, std::numeric_limits<char>::max())
        << "low_bits_ too large for C type char, got: " << low_bits_;
    return static_cast<char>(low_bits_);
  }

  /// \brief Bitwise or between two Decimal128.
  Decimal128& operator|=(const Decimal128& right) {
    low_bits_ |= right.low_bits_;
    high_bits_ |= right.high_bits_;
    return *this;
  }

  /// \brief Bitwise and between two Decimal128.
  Decimal128& operator&=(const Decimal128& right) {
    low_bits_ &= right.low_bits_;
    high_bits_ &= right.high_bits_;
    return *this;
  }

  /// \brief Shift left by the given number of bits.
  Decimal128& operator<<=(uint32_t bits) {
    if (bits != 0) {
      if (bits < 64) {
        high_bits_ <<= bits;
        high_bits_ |= (low_bits_ >> (64 - bits));
        low_bits_ <<= bits;
      } else if (bits < 128) {
        high_bits_ = static_cast<int64_t>(low_bits_) << (bits - 64);
        low_bits_ = 0;
      } else {
        high_bits_ = 0;
        low_bits_ = 0;
      }
    }
    return *this;
  }

  /// \brief Shift right by the given number of bits. Negative values will
  Decimal128& operator>>=(uint32_t bits) {
    if (bits != 0) {
      if (bits < 64) {
        low_bits_ >>= bits;
        low_bits_ |= static_cast<uint64_t>(high_bits_ << (64 - bits));
        high_bits_ = static_cast<int64_t>(static_cast<uint64_t>(high_bits_) >> bits);
      } else if (bits < 128) {
        low_bits_ = static_cast<uint64_t>(high_bits_ >> (bits - 64));
        high_bits_ = static_cast<int64_t>(high_bits_ >= 0L ? 0L : -1L);
      } else {
        high_bits_ = static_cast<int64_t>(high_bits_ >= 0L ? 0L : -1L);
        low_bits_ = static_cast<uint64_t>(high_bits_);
      }
    }
    return *this;
  }

  /// Divide this number by right and return the result. This operation is
  /// not destructive.
  /// The answer rounds to zero. Signs work like:
  ///   21 /  5 ->  4,  1
  ///  -21 /  5 -> -4, -1
  ///   21 / -5 -> -4,  1
  ///  -21 / -5 ->  4, -1
  /// \param divisor the number to divide by
  /// \param remainder the remainder after the division
  Status Divide(const Decimal128& divisor, Decimal128* result,
                Decimal128* remainder) const;

  /// \brief In-place division.
  Decimal128& operator/=(const Decimal128& right) {
    Decimal128 remainder;
    Status s = Divide(right, this, &remainder);
  }

  Decimal128 operator%=(const Decimal128& right) {
    Decimal128 result;
    Status s = left.Divide(right, &result, this);
    DCHECK(s.ok());
    return *this;
  }

  /// \brief Get the high bits of the two's complement representation of the number.
  int64_t high_bits() const { return high_bits_; }

  /// \brief Get the low bits of the two's complement representation of the number.
  uint64_t low_bits() const { return low_bits_; }

  /// \brief Return the raw bytes of the value in little-endian byte order.
  std::array<uint8_t, 16> ToBytes() const {
    const uint64_t raw[] = {BitUtil::ToLittleEndian(low_bits_),
                            BitUtil::ToLittleEndian(static_cast<uint64_t>(high_bits_))};
    const auto* raw_data = reinterpret_cast<const uint8_t*>(raw);
    std::array<uint8_t, 16> out{{0}};
    std::copy(raw_data, raw_data + out.size(), out.begin());
    return out;
  }

  /// \brief Convert the Decimal128 value to a base 10 decimal string with the given
  /// precision and scale.
  Status ToString(int precision, int scale, std::string* out) const;
  std::string ToString(int precision, int scale) const;

  /// \brief Convert a decimal string to an Decimal128 value, optionally including
  /// precision and scale if they're passed in and not null.
  static Status FromString(const std::string& s, Decimal128* out,
                           int* precision = NULLPTR, int* scale = NULLPTR);

 private:
#if ARROW_LITTLE_ENDIAN
  uint64_t low_bits_;
  int64_t high_bits_;
#else
  int64_t high_bits_;
  uint64_t low_bits_;
#endif
};

ARROW_EXPORT bool operator==(const Decimal128& left, const Decimal128& right) {
  return left.high_bits() == right.high_bits() && left.low_bits() == right.low_bits();
}

ARROW_EXPORT bool operator!=(const Decimal128& left, const Decimal128& right) {
  return !operator==(left, right);
}

ARROW_EXPORT bool operator<(const Decimal128& left, const Decimal128& right) {
  return left.high_bits() < right.high_bits() ||
         (left.high_bits() == right.high_bits() && left.low_bits() < right.low_bits());
}

ARROW_EXPORT bool operator<=(const Decimal128& left, const Decimal128& right) {
  return !operator>(left, right);
}

ARROW_EXPORT bool operator>(const Decimal128& left, const Decimal128& right) {
  return operator<(right, left);
}

ARROW_EXPORT bool operator>=(const Decimal128& left, const Decimal128& right) {
  return !operator<(left, right);
}

ARROW_EXPORT Decimal128 operator-(const Decimal128& operand);

ARROW_EXPORT Decimal128 operator~(const Decimal128& operand) {
  return Decimal128(~operand.high_bits(), ~operand.low_bits());
}

ARROW_EXPORT Decimal128 operator+(const Decimal128& left, const Decimal128& right);
ARROW_EXPORT Decimal128 operator-(const Decimal128& left, const Decimal128& right);
ARROW_EXPORT Decimal128 operator*(const Decimal128& left, const Decimal128& right);
ARROW_EXPORT Decimal128 operator/(const Decimal128& left, const Decimal128& right);
ARROW_EXPORT Decimal128 operator%(const Decimal128& left, const Decimal128& right);

}  // namespace arrow

#endif  //  ARROW_DECIMAL_NOT_NATIVE_H
