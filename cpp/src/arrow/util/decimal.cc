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

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)
#endif

#include "arrow/util/bit-util.h"
#include "arrow/util/decimal.h"
#include "arrow/util/logging.h"

namespace arrow {

Decimal128::Decimal128(const std::string& str) : Decimal128() {
  Status status(Decimal128::FromString(str, this));
  DCHECK(status.ok()) << status.message();
}

Decimal128::Decimal128(const uint8_t* bytes)
    : Decimal128(BitUtil::FromLittleEndian(reinterpret_cast<const int64_t*>(bytes)[1]),
                 BitUtil::FromLittleEndian(reinterpret_cast<const uint64_t*>(bytes)[0])) {
}

Status Decimal128::ToString(int precision, int scale, std::string* out) const {
  DCHECK_GE(precision, 0);
  DCHECK_LE(precision, 38);

  const bool is_negative = *this < 0;

  // Decimal values are sent to clients as strings so in the interest of speed the string
  // will be created without the using stringstream with the whole/fractional_part().
  size_t last_char_idx =
      precision + static_cast<size_t>(scale > 0)  // Add a space for decimal place
      + static_cast<size_t>(scale == precision)   // Add a space for leading 0
      + static_cast<size_t>(is_negative);         // Add a space for negative sign

  out->resize(last_char_idx, '0');
  std::string& str(*out);

  // Start filling in the values in reverse order by taking the last digit
  // of the value. Use a positive value and worry about the sign later. At this
  // point the last_char_idx points to the string terminator.
  Decimal128 remaining_value(*this);

  const auto first_digit_idx = static_cast<size_t>(is_negative);
  if (is_negative) {
    remaining_value.Negate();
  }

  if (scale > 0) {
    int remaining_scale = scale;
    do {
      str[--last_char_idx] =
          static_cast<char>(remaining_value % 10 + '0');  // Ascii offset
      remaining_value /= 10;
    } while (--remaining_scale > 0);
    str[--last_char_idx] = '.';
    DCHECK_GT(last_char_idx, first_digit_idx) << "Not enough space remaining";
  }

  do {
    str[--last_char_idx] = static_cast<char>(remaining_value % 10 + '0');  // Ascii offset
    remaining_value /= 10;
    if (remaining_value == 0) {
      // Trim any extra leading 0's.
      if (last_char_idx > first_digit_idx) {
        str.erase(0, last_char_idx - first_digit_idx);
      }

      break;
    }
    // For safety, enforce string length independent of remaining_value.
  } while (last_char_idx > first_digit_idx);

  if (remaining_value != 0) {
    return Status::Invalid(
        "Decimal128 value too big to be represented with precision 38");
  }

  if (is_negative) {
    str[0] = '-';
  }

  return Status::OK();
}

std::string Decimal128::ToString(int precision, int scale) const {
  std::string out;
  Status s = ToString(precision, scale, &out);
  DCHECK(s.ok()) << s.message();
  return out;
}

static constexpr auto kInt64DecimalDigits =
    static_cast<size_t>(std::numeric_limits<int64_t>::digits10);
static constexpr int64_t kPowersOfTen[kInt64DecimalDigits + 1] = {1LL,
                                                                  10LL,
                                                                  100LL,
                                                                  1000LL,
                                                                  10000LL,
                                                                  100000LL,
                                                                  1000000LL,
                                                                  10000000LL,
                                                                  100000000LL,
                                                                  1000000000LL,
                                                                  10000000000LL,
                                                                  100000000000LL,
                                                                  1000000000000LL,
                                                                  10000000000000LL,
                                                                  100000000000000LL,
                                                                  1000000000000000LL,
                                                                  10000000000000000LL,
                                                                  100000000000000000LL,
                                                                  1000000000000000000LL};

static void StringToInteger(const std::string& str, Decimal128* out) {
  using std::size_t;

  DCHECK_NE(out, nullptr) << "Decimal128 output variable cannot be nullptr";
  DCHECK_EQ(*out, 0)
      << "When converting a string to Decimal128 the initial output must be 0";

  const size_t length = str.length();

  DCHECK_GT(length, 0) << "length of parsed decimal string should be greater than 0";

  for (size_t posn = 0; posn < length;) {
    const size_t group = std::min(kInt64DecimalDigits, length - posn);
    const int64_t chunk = std::stoll(str.substr(posn, group));
    const int64_t multiple = kPowersOfTen[group];

    *out *= multiple;
    *out += chunk;

    posn += group;
  }
}

Status Decimal128::FromString(const std::string& s, Decimal128* out, int* precision,
                              int* scale) {
  // Implements this regex: "(\\+?|-?)((0*)(\\d*))(\\.(\\d+))?";
  if (s.empty()) {
    return Status::Invalid("Empty string cannot be converted to decimal");
  }

  std::string::const_iterator charp = s.cbegin();
  std::string::const_iterator end = s.cend();

  char first_char = *charp;
  bool is_negative = false;
  if (first_char == '+' || first_char == '-') {
    is_negative = first_char == '-';
    ++charp;
  }

  if (charp == end) {
    std::stringstream ss;
    ss << "Single character: '" << first_char << "' is not a valid decimal value";
    return Status::Invalid(ss.str());
  }

  std::string::const_iterator numeric_string_start = charp;

  DCHECK_LT(charp, end);

  // skip leading zeros
  charp = std::find_if_not(charp, end, [](char c) { return c == '0'; });

  // all zeros and no decimal point
  if (charp == end) {
    if (out != nullptr) {
      *out = 0;
    }

    // Not sure what other libraries assign precision to for this case (this case of
    // a string consisting only of one or more zeros)
    if (precision != nullptr) {
      *precision = static_cast<int>(charp - numeric_string_start);
    }

    if (scale != nullptr) {
      *scale = 0;
    }

    return Status::OK();
  }

  std::string::const_iterator whole_part_start = charp;

  charp = std::find_if_not(charp, end, [](char c) { return std::isdigit(c) != 0; });

  std::string::const_iterator whole_part_end = charp;
  std::string whole_part(whole_part_start, whole_part_end);

  if (charp != end && *charp == '.') {
    ++charp;

    if (charp == end) {
      return Status::Invalid(
          "Decimal point must be followed by at least one base ten digit. Reached the "
          "end of the string.");
    }

    if (std::isdigit(*charp) == 0) {
      std::stringstream ss;
      ss << "Decimal point must be followed by a base ten digit. Found '" << *charp
         << "'";
      return Status::Invalid(ss.str());
    }
  } else {
    if (charp != end) {
      std::stringstream ss;
      ss << "Expected base ten digit or decimal point but found '" << *charp
         << "' instead.";
      return Status::Invalid(ss.str());
    }
  }

  std::string::const_iterator fractional_part_start = charp;

  // The rest must be digits, because if we have a decimal point it must be followed by
  // digits
  if (charp != end) {
    charp = std::find_if_not(charp, end, [](char c) { return std::isdigit(c) != 0; });

    // The while loop has ended before the end of the string which means we've hit a
    // character that isn't a base ten digit
    if (charp != end) {
      std::stringstream ss;
      ss << "Found non base ten digit character '" << *charp
         << "' before the end of the string";
      return Status::Invalid(ss.str());
    }
  }

  std::string::const_iterator fractional_part_end = charp;
  std::string fractional_part(fractional_part_start, fractional_part_end);

  if (precision != nullptr) {
    *precision = static_cast<int>(whole_part.size() + fractional_part.size());
  }

  if (scale != nullptr) {
    *scale = static_cast<int>(fractional_part.size());
  }

  if (out != nullptr) {
    // zero out in case we've passed in a previously used value
    *out = 0;
    StringToInteger(whole_part + fractional_part, out);
    if (is_negative) {
      out->Negate();
    }
  }

  return Status::OK();
}

#ifndef __SIZEOF_INT128__
/// Expands the given value into an array of ints so that we can work on
/// it. The array will be converted to an absolute value and the wasNegative
/// flag will be set appropriately. The array will remove leading zeros from
/// the value.
/// \param array an array of length 4 to set with the value
/// \param was_negative a flag for whether the value was original negative
/// \result the output length of the array
static int64_t FillInArray(const Decimal128& value, uint32_t* array, bool& was_negative) {
  uint64_t high;
  uint64_t low;
  const int64_t highbits = value.high_bits();
  const uint64_t lowbits = value.low_bits();

  if (highbits < 0) {
    low = ~lowbits + 1;
    high = static_cast<uint64_t>(~highbits);
    if (low == 0) {
      ++high;
    }
    was_negative = true;
  } else {
    low = lowbits;
    high = static_cast<uint64_t>(highbits);
    was_negative = false;
  }

  if (high != 0) {
    if (high > std::numeric_limits<uint32_t>::max()) {
      array[0] = static_cast<uint32_t>(high >> 32);
      array[1] = static_cast<uint32_t>(high);
      array[2] = static_cast<uint32_t>(low >> 32);
      array[3] = static_cast<uint32_t>(low);
      return 4;
    }

    array[0] = static_cast<uint32_t>(high);
    array[1] = static_cast<uint32_t>(low >> 32);
    array[2] = static_cast<uint32_t>(low);
    return 3;
  }

  if (low >= std::numeric_limits<uint32_t>::max()) {
    array[0] = static_cast<uint32_t>(low >> 32);
    array[1] = static_cast<uint32_t>(low);
    return 2;
  }

  if (low == 0) {
    return 0;
  }

  array[0] = static_cast<uint32_t>(low);
  return 1;
}

/// \brief Find last set bit in a 32 bit integer. Bit 1 is the LSB and bit 32 is the MSB.
static int64_t FindLastSetBit(uint32_t value) {
#if defined(__clang__) || defined(__GNUC__)
  // Count leading zeros
  return __builtin_clz(value) + 1;
#elif defined(_MSC_VER)
  unsigned long index;                                         // NOLINT
  _BitScanReverse(&index, static_cast<unsigned long>(value));  // NOLINT
  return static_cast<int64_t>(index + 1UL);
#endif
}

/// Shift the number in the array left by bits positions.
/// \param array the number to shift, must have length elements
/// \param length the number of entries in the array
/// \param bits the number of bits to shift (0 <= bits < 32)
static void ShiftArrayLeft(uint32_t* array, int64_t length, int64_t bits) {
  if (length > 0 && bits != 0) {
    for (int64_t i = 0; i < length - 1; ++i) {
      array[i] = (array[i] << bits) | (array[i + 1] >> (32 - bits));
    }
    array[length - 1] <<= bits;
  }
}

/// Shift the number in the array right by bits positions.
/// \param array the number to shift, must have length elements
/// \param length the number of entries in the array
/// \param bits the number of bits to shift (0 <= bits < 32)
static void ShiftArrayRight(uint32_t* array, int64_t length, int64_t bits) {
  if (length > 0 && bits != 0) {
    for (int64_t i = length - 1; i > 0; --i) {
      array[i] = (array[i] >> bits) | (array[i - 1] << (32 - bits));
    }
    array[0] >>= bits;
  }
}

/// \brief Fix the signs of the result and remainder at the end of the division based on
/// the signs of the dividend and divisor.
static void FixDivisionSigns(Decimal128* result, Decimal128* remainder,
                             bool dividend_was_negative, bool divisor_was_negative) {
  if (dividend_was_negative != divisor_was_negative) {
    result->Negate();
  }

  if (dividend_was_negative) {
    remainder->Negate();
  }
}

/// \brief Build a Decimal128 from a list of ints.
static Status BuildFromArray(Decimal128* value, uint32_t* array, int64_t length) {
  switch (length) {
    case 0:
      *value = {static_cast<int64_t>(0)};
      break;
    case 1:
      *value = {static_cast<int64_t>(array[0])};
      break;
    case 2:
      *value = {static_cast<int64_t>(0),
                (static_cast<uint64_t>(array[0]) << 32) + array[1]};
      break;
    case 3:
      *value = {static_cast<int64_t>(array[0]),
                (static_cast<uint64_t>(array[1]) << 32) + array[2]};
      break;
    case 4:
      *value = {(static_cast<int64_t>(array[0]) << 32) + array[1],
                (static_cast<uint64_t>(array[2]) << 32) + array[3]};
      break;
    case 5:
      if (array[0] != 0) {
        return Status::Invalid("Can't build Decimal128 with 5 ints.");
      }
      *value = {(static_cast<int64_t>(array[1]) << 32) + array[2],
                (static_cast<uint64_t>(array[3]) << 32) + array[4]};
      break;
    default:
      return Status::Invalid("Unsupported length for building Decimal128");
  }

  return Status::OK();
}

/// \brief Do a division where the divisor fits into a single 32 bit value.
static Status SingleDivide(const uint32_t* dividend, int64_t dividend_length,
                           uint32_t divisor, Decimal128* remainder,
                           bool dividend_was_negative, bool divisor_was_negative,
                           Decimal128* result) {
  uint64_t r = 0;
  uint32_t result_array[5];
  for (int64_t j = 0; j < dividend_length; j++) {
    r <<= 32;
    r += dividend[j];
    result_array[j] = static_cast<uint32_t>(r / divisor);
    r %= divisor;
  }
  RETURN_NOT_OK(BuildFromArray(result, result_array, dividend_length));
  *remainder = static_cast<int64_t>(r);
  FixDivisionSigns(result, remainder, dividend_was_negative, divisor_was_negative);
  return Status::OK();
}

Status Decimal128::Divide(const Decimal128& divisor, Decimal128* result,
                          Decimal128* remainder) const {
  // Split the dividend and divisor into integer pieces so that we can
  // work on them.
  uint32_t dividend_array[5];
  uint32_t divisor_array[4];
  bool dividend_was_negative;
  bool divisor_was_negative;
  // leave an extra zero before the dividend
  dividend_array[0] = 0;
  int64_t dividend_length =
      FillInArray(*this, dividend_array + 1, dividend_was_negative) + 1;
  int64_t divisor_length = FillInArray(divisor, divisor_array, divisor_was_negative);

  // Handle some of the easy cases.
  if (dividend_length <= divisor_length) {
    *remainder = *this;
    *result = 0;
    return Status::OK();
  }

  if (divisor_length == 0) {
    return Status::Invalid("Division by 0 in Decimal128");
  }

  if (divisor_length == 1) {
    return SingleDivide(dividend_array, dividend_length, divisor_array[0], remainder,
                        dividend_was_negative, divisor_was_negative, result);
  }

  int64_t result_length = dividend_length - divisor_length;
  uint32_t result_array[4];

  // Normalize by shifting both by a multiple of 2 so that
  // the digit guessing is better. The requirement is that
  // divisor_array[0] is greater than 2**31.
  int64_t normalize_bits = 32 - FindLastSetBit(divisor_array[0]);
  ShiftArrayLeft(divisor_array, divisor_length, normalize_bits);
  ShiftArrayLeft(dividend_array, dividend_length, normalize_bits);

  // compute each digit in the result
  for (int64_t j = 0; j < result_length; ++j) {
    // Guess the next digit. At worst it is two too large
    uint32_t guess = std::numeric_limits<uint32_t>::max();
    auto high_dividend =
        static_cast<uint64_t>(dividend_array[j]) << 32 | dividend_array[j + 1];
    if (dividend_array[j] != divisor_array[0]) {
      guess = static_cast<uint32_t>(high_dividend / divisor_array[0]);
    }

    // catch all of the cases where guess is two too large and most of the
    // cases where it is one too large
    auto rhat = static_cast<uint32_t>(high_dividend -
                                      guess * static_cast<uint64_t>(divisor_array[0]));
    while (static_cast<uint64_t>(divisor_array[1]) * guess >
           (static_cast<uint64_t>(rhat) << 32) + dividend_array[j + 2]) {
      --guess;
      rhat += divisor_array[0];
      if (static_cast<uint64_t>(rhat) < divisor_array[0]) {
        break;
      }
    }

    // subtract off the guess * divisor from the dividend
    uint64_t mult = 0;
    for (int64_t i = divisor_length - 1; i >= 0; --i) {
      mult += static_cast<uint64_t>(guess) * divisor_array[i];
      uint32_t prev = dividend_array[j + i + 1];
      dividend_array[j + i + 1] -= static_cast<uint32_t>(mult);
      mult >>= 32;
      if (dividend_array[j + i + 1] > prev) {
        ++mult;
      }
    }
    uint32_t prev = dividend_array[j];
    dividend_array[j] -= static_cast<uint32_t>(mult);

    // if guess was too big, we add back divisor
    if (dividend_array[j] > prev) {
      --guess;

      uint32_t carry = 0;
      for (int64_t i = divisor_length - 1; i >= 0; --i) {
        uint64_t sum =
            static_cast<uint64_t>(divisor_array[i]) + dividend_array[j + i + 1] + carry;
        dividend_array[j + i + 1] = static_cast<uint32_t>(sum);
        carry = static_cast<uint32_t>(sum >> 32);
      }
      dividend_array[j] += carry;
    }

    result_array[j] = guess;
  }

  // denormalize the remainder
  ShiftArrayRight(dividend_array, dividend_length, normalize_bits);

  // return result and remainder
  RETURN_NOT_OK(BuildFromArray(result, result_array, result_length));
  RETURN_NOT_OK(BuildFromArray(remainder, dividend_array, dividend_length));
  FixDivisionSigns(result, remainder, dividend_was_negative, divisor_was_negative);
  return Status::OK();
}
#endif

Decimal128 operator-(const Decimal128& operand) {
  Decimal128 result(operand);
  return result.Negate();
}

Decimal128 operator+(const Decimal128& left, const Decimal128& right) {
  Decimal128 result(left);
  result += right;
  return result;
}

Decimal128 operator-(const Decimal128& left, const Decimal128& right) {
  Decimal128 result(left);
  result -= right;
  return result;
}

Decimal128 operator*(const Decimal128& left, const Decimal128& right) {
  Decimal128 result(left);
  result *= right;
  return result;
}

Decimal128 operator/(const Decimal128& left, const Decimal128& right) {
  Decimal128 result(left);
  result /= right;
  return result;
}

Decimal128 operator%(const Decimal128& left, const Decimal128& right) {
  Decimal128 result(left);
  result %= right;
  return result;
}

}  // namespace arrow
