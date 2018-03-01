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

#include <chrono>
#include <exception>
#include <boost/algorithm/string.hpp>

#include "arrow/python/pyarrow.h"

#include "arrow/array.h"
#include "arrow/builder.h"
#include "arrow/type.h"
#include "arrow/memory_pool.h"

#include "arrow/python/builtin_convert.h"

#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/numpy.h"
#include "pybind11/include/pybind11/chrono.h"
#include "pybind11/include/pybind11/stl.h"

#define THROW_NOT_OK(exception_type, status) \
  do {                                       \
    auto s = (status);                       \
    if (!s.ok()) {                           \
      throw exception_type(s.message());     \
    }                                        \
  } while (false)

namespace pyb = pybind11;


std::vector<std::string> split(const std::string& text, const std::string& sep) {
  std::vector<std::string> strs;
  boost::split(strs, text, boost::is_any_of(sep));
  return strs;
}

std::string indent(const std::string& text, size_t spaces) {
  if (spaces == 0) {
    return text;
  }
  std::string block(4, ' ');
  std::vector<std::string> buf;

  for (const auto& line : split(text, "\n")) {
    buf.emplace_back(block + line);
  }
  return boost::join(buf, "\n");
}


class Value {
 public:
  explicit Value(std::shared_ptr<arrow::Array> array, int64_t i) :
      array_(std::move(array)), i_(i) {}
  virtual pyb::object as_py() const = 0;
  virtual ~Value() = default;
 protected:
  std::shared_ptr<arrow::Array> array_;
  int64_t i_;
 private:
  ARROW_DISALLOW_COPY_AND_ASSIGN(Value);
};


class BooleanValue : public Value {
 public:
  using Value::Value;
  pyb::object as_py() const override {
    return pyb::bool_(static_cast<const arrow::BooleanArray&>(*array_).Value(i_));
  }
};

template <typename ArrayType>
class IntegerValue : public Value {
 public:
  using Value::Value;
  pyb::object as_py() const override {
    return pyb::int_(static_cast<const ArrayType&>(*array_).Value(i_));
  }
};

using UInt8Value = IntegerValue<arrow::UInt8Array>;
using UInt16Value = IntegerValue<arrow::UInt16Array>;
using UInt32Value = IntegerValue<arrow::UInt32Array>;
using UInt64Value = IntegerValue<arrow::UInt64Array>;

using Int8Value = IntegerValue<arrow::Int8Array>;
using Int16Value = IntegerValue<arrow::Int16Array>;
using Int32Value = IntegerValue<arrow::Int32Array>;
using Int64Value = IntegerValue<arrow::Int64Array>;

class Date32Value : public Value {
 public:
  using Value::Value;
  pyb::object as_py() const override {
    const auto value = static_cast<const arrow::Date32Array&>(*array_).Value(i_) * 86400;
    return pyb::cast(std::chrono::system_clock::from_time_t(value)).attr("date")();
  }
};

class Date64Value : public Value {
 public:
  using Value::Value;
  pyb::object as_py() const override {
    const auto value = static_cast<const arrow::Date64Array&>(*array_).Value(i_);
    return pyb::cast(std::chrono::system_clock::from_time_t(value / 1000)).attr("date")();
  }
};

static const auto kEpoch = std::chrono::system_clock::from_time_t(0);

class Time32Value : public Value {
 public:
  using Value::Value;
  pyb::object as_py() const override {
    const auto value = static_cast<const arrow::Time32Array&>(*array_).Value(i_);
    const auto& type = static_cast<const arrow::Time32Type&>(*array_->type());
    pyb::object result;
    if (type.unit() == arrow::TimeUnit::SECOND) {
      result = pyb::cast(kEpoch + std::chrono::seconds(value));
    } else {
      result = pyb::cast(kEpoch + std::chrono::milliseconds(value));
    }
    return result.attr("time")();
  }
};

class Time64Value : public Value {
 public:
  using Value::Value;
  pyb::object as_py() const override {
    const auto value = static_cast<const arrow::Time64Array&>(*array_).Value(i_);
    const auto& type = static_cast<const arrow::Time64Type&>(*array_->type());
    pyb::object result;
    if (type.unit() == arrow::TimeUnit::MICRO) {
      result = pyb::cast(kEpoch + std::chrono::microseconds(value));
    } else {
      result = pyb::cast(kEpoch + std::chrono::nanoseconds(value));
    }
    return result.attr("time")();
  }
};

class TimestampValue : public Value {
 public:
  using Value::Value;
  pyb::object as_py() const override {
    using namespace pybind11::literals;
    const auto value = static_cast<const arrow::TimestampArray&>(*array_).Value(i_);
    const auto &type = static_cast<const arrow::TimestampType &>(*array_->type());
    const std::string &timezone = type.timezone();
    if (timezone.empty()) {
      switch (type.unit()) {
        case arrow::TimeUnit::SECOND:
          return Timestamp_(value, "unit"_a = "s");
        case arrow::TimeUnit::MILLI:
          return Timestamp_(value, "unit"_a = "ms");
        case arrow::TimeUnit::MICRO:
          return Timestamp_(value, "unit"_a = "us");
        case arrow::TimeUnit::NANO:
          return Timestamp_(value, "unit"_a = "ns");
        default:
          throw std::logic_error("invalid unit");
      }
    }
    switch (type.unit()) {
      case arrow::TimeUnit::SECOND:
        return Timestamp_(value, "unit"_a = "s", "tz"_a = timezone);
      case arrow::TimeUnit::MILLI:
        return Timestamp_(value, "unit"_a = "ms");
      case arrow::TimeUnit::MICRO:
        return Timestamp_(value, "unit"_a = "us");
      case arrow::TimeUnit::NANO:
        return Timestamp_(value, "unit"_a = "ns");
      default:
        throw std::logic_error("invalid unit");
    }
  }
 private:
  static pyb::object Timestamp_;
};

pyb::object TimestampValue::Timestamp_(pyb::module::import("pandas").attr("Timestamp"));

class HalfFloatValue : public Value {
 public:
  using Value::Value;
  pyb::object as_py() const override {
    const auto value = static_cast<const arrow::HalfFloatArray&>(*array_).Value(i_);
    return pyb::float_(pyb::module::import("numpy").attr("float16")(value));
  }
};

class FloatValue : public Value {
 public:
  using Value::Value;
  pyb::object as_py() const override {
    return pyb::float_(static_cast<const arrow::FloatArray&>(*array_).Value(i_));
  }
};

class DoubleValue : public Value {
 public:
  using Value::Value;
  pyb::object as_py() const override {
    return pyb::float_(static_cast<const arrow::DoubleArray&>(*array_).Value(i_));
  }
};

class DecimalValue : public Value {
 public:
  using Value::Value;
  pyb::object as_py() const override {
    return Decimal_(
        pyb::str(static_cast<const arrow::Decimal128Array&>(*array_).FormatValue(i_)));
  }
 private:
  static pyb::object Decimal_;
};

pyb::object DecimalValue::Decimal_(pyb::module::import("decimal").attr("Decimal"));

class StringValue : public Value {
 public:
  using Value::Value;
  pyb::object as_py() const override {
    return pyb::str(static_cast<const arrow::StringArray&>(*array_).GetString(i_));
  }
};

class BinaryValue : public Value {
 public:
  using Value::Value;
  pyb::object as_py() const override {
    return pyb::bytes(static_cast<const arrow::BinaryArray&>(*array_).GetString(i_));
  }
};


std::unique_ptr<Value> scalar_value(const std::shared_ptr<arrow::Array>& array, int64_t i) {
  switch (array->type_id()) {
    case arrow::Type::BOOL:
      return std::unique_ptr<Value>(new BooleanValue(array, i));
    case arrow::Type::UINT8:
      return std::unique_ptr<Value>(new UInt8Value(array, i));
    case arrow::Type::UINT16:
      return std::unique_ptr<Value>(new UInt16Value(array, i));
    case arrow::Type::UINT32:
      return std::unique_ptr<Value>(new UInt32Value(array, i));
    case arrow::Type::UINT64:
      return std::unique_ptr<Value>(new UInt64Value(array, i));
    case arrow::Type::INT8:
      return std::unique_ptr<Value>(new Int8Value(array, i));
    case arrow::Type::INT16:
      return std::unique_ptr<Value>(new Int16Value(array, i));
    case arrow::Type::INT32:
      return std::unique_ptr<Value>(new Int32Value(array, i));
    case arrow::Type::INT64:
      return std::unique_ptr<Value>(new Int64Value(array, i));
    case arrow::Type::DATE32:
      return std::unique_ptr<Value>(new Date32Value(array, i));
    case arrow::Type::DATE64:
      return std::unique_ptr<Value>(new Date64Value(array, i));
    case arrow::Type::TIME32:
      return std::unique_ptr<Value>(new Time32Value(array, i));
    case arrow::Type::TIME64:
      return std::unique_ptr<Value>(new Time64Value(array, i));
    case arrow::Type::TIMESTAMP:
      return std::unique_ptr<Value>(new TimestampValue(array, i));
    case arrow::Type::HALF_FLOAT:
      return std::unique_ptr<Value>(new HalfFloatValue(array, i));
    case arrow::Type::FLOAT:
      return std::unique_ptr<Value>(new FloatValue(array, i));
    case arrow::Type::DOUBLE:
      return std::unique_ptr<Value>(new DoubleValue(array, i));
    case arrow::Type::DECIMAL:
      return std::unique_ptr<Value>(new DecimalValue(array, i));
    case arrow::Type::STRING:
      return std::unique_ptr<Value>(new StringValue(array, i));
    case arrow::Type::BINARY:
      return std::unique_ptr<Value>(new BinaryValue(array, i));
    default:
      throw std::logic_error("invalid type");
  }
}


PYBIND11_MODULE(pyarrow2, m) {
  pyb::class_<arrow::Array, std::shared_ptr<arrow::Array>>(m, "Array")
      .def("is_null", &arrow::Array::IsNull)
      .def("length", &arrow::Array::length)
      .def("null_count", &arrow::Array::null_count)
      .def(pyb::init([](const std::vector<std::string>& sequence) {
        arrow::StringBuilder builder(arrow::utf8(), arrow::default_memory_pool());
        for (const auto& value : sequence) {
          THROW_NOT_OK(pyb::value_error, builder.Append(value));
        }
        std::shared_ptr<arrow::Array> out;
        THROW_NOT_OK(pyb::value_error, builder.Finish(&out));
        return out;
      }))
      .def("__getitem__", [](const std::shared_ptr<arrow::Array>& array, int64_t i) {
        return scalar_value(array, i)->as_py();
      });
}
