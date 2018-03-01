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

#include "arrow/python/pyarrow.h"

#include "arrow/array.h"
#include "arrow/builder.h"
#include "arrow/table.h"
#include "arrow/tensor.h"
#include "arrow/type.h"

#include "pybind11/include/pybind11/pybind11.h"

namespace arrow {
namespace py {

template <typename T>
PyObject* wrap(const std::shared_ptr<T>& obj) {
  return pybind11::cast(obj).ptr();
}

template <typename T>
Status unwrap(PyObject* obj, std::shared_ptr<T>* out) {
  auto result = pybind11::reinterpret_borrow<pybind11::object>(obj);
  try {
    *out = result.cast<std::shared_ptr<T>>();
  } catch (const std::exception& e) {
    return Status::Invalid(e.what());
  }
  return Status::OK();
}

template <typename T>
bool is_arrow_type(PyObject* obj) {
  auto result = pybind11::reinterpret_borrow<pybind11::object>(obj);
  try {
    result.cast<std::shared_ptr<T>>();
  } catch (const pybind11::cast_error&) {
    return false;
  }
  return true;
}

bool is_buffer(PyObject* buf) { return is_arrow_type<Buffer>(buf); }

Status unwrap_buffer(PyObject* buf, std::shared_ptr<Buffer>* out) {
  return unwrap(buf, out);
}

PyObject* wrap_buffer(const std::shared_ptr<Buffer>& buffer) { return wrap(buffer); }

bool is_data_type(PyObject* obj) { return is_arrow_type<DataType>(obj); }

Status unwrap_data_type(PyObject* object, std::shared_ptr<DataType>* out) {
  return unwrap(object, out);
}

PyObject* wrap_data_type(const std::shared_ptr<DataType>& type) { return wrap(type); }

bool is_field(PyObject* field) { return is_arrow_type<Field>(field); }

Status unwrap_field(PyObject* field, std::shared_ptr<Field>* out) {
  return unwrap(field, out);
}

PyObject* wrap_field(const std::shared_ptr<Field>& field) { return wrap(field); }

bool is_schema(PyObject* schema) { return is_arrow_type<Schema>(schema); }

Status unwrap_schema(PyObject* schema, std::shared_ptr<Schema>* out) {
  return unwrap(schema, out);
}

PyObject* wrap_schema(const std::shared_ptr<Schema>& schema) { return wrap(schema); }

bool is_array(PyObject* array) { return is_arrow_type<Array>(array); }

Status unwrap_array(PyObject* array, std::shared_ptr<Array>* out) {
  return unwrap(array, out);
}

PyObject* wrap_array(const std::shared_ptr<Array>& array) { return wrap(array); }

bool is_tensor(PyObject* tensor) { return is_arrow_type<Tensor>(tensor); }

Status unwrap_tensor(PyObject* tensor, std::shared_ptr<Tensor>* out) {
  return unwrap(tensor, out);
}

PyObject* wrap_tensor(const std::shared_ptr<Tensor>& tensor) { return wrap(tensor); }

bool is_column(PyObject* column) { return is_arrow_type<Column>(column); }

Status unwrap_column(PyObject* column, std::shared_ptr<Column>* out) {
  return unwrap(column, out);
}

PyObject* wrap_column(const std::shared_ptr<Column>& column) { return wrap(column); }

bool is_table(PyObject* table) { return is_arrow_type<Table>(table); }

Status unwrap_table(PyObject* table, std::shared_ptr<Table>* out) {
  return unwrap(table, out);
}

PyObject* wrap_table(const std::shared_ptr<Table>& table) { return wrap(table); }

bool is_record_batch(PyObject* batch) { return is_arrow_type<RecordBatch>(batch); }

Status unwrap_record_batch(PyObject* batch, std::shared_ptr<RecordBatch>* out) {
  return unwrap(batch, out);
}

PyObject* wrap_record_batch(const std::shared_ptr<RecordBatch>& batch) {
  return wrap(batch);
}

}  // namespace py
}  // namespace arrow
