# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import ast
import collections
import json
import re

import pandas.core.internals as _int
import numpy as np
import pandas as pd

import six

import pyarrow as pa
from pyarrow.compat import PY2, zip_longest, frombytes  # noqa


def infer_dtype(column):
    try:
        return pd.api.types.infer_dtype(column)
    except AttributeError:
        return pd.lib.infer_dtype(column)


_logical_type_map = {}


def get_logical_type_map():
    global _logical_type_map

    if not _logical_type_map:
        _logical_type_map.update({
            pa.lib.Type_NA: 'float64',  # NaNs
            pa.lib.Type_BOOL: 'bool',
            pa.lib.Type_INT8: 'int8',
            pa.lib.Type_INT16: 'int16',
            pa.lib.Type_INT32: 'int32',
            pa.lib.Type_INT64: 'int64',
            pa.lib.Type_UINT8: 'uint8',
            pa.lib.Type_UINT16: 'uint16',
            pa.lib.Type_UINT32: 'uint32',
            pa.lib.Type_UINT64: 'uint64',
            pa.lib.Type_HALF_FLOAT: 'float16',
            pa.lib.Type_FLOAT: 'float32',
            pa.lib.Type_DOUBLE: 'float64',
            pa.lib.Type_DATE32: 'date',
            pa.lib.Type_DATE64: 'date',
            pa.lib.Type_TIME32: 'time',
            pa.lib.Type_TIME64: 'time',
            pa.lib.Type_BINARY: 'bytes',
            pa.lib.Type_FIXED_SIZE_BINARY: 'bytes',
            pa.lib.Type_STRING: 'unicode',
        })
    return _logical_type_map


def get_logical_type(arrow_type):
    logical_type_map = get_logical_type_map()

    try:
        return logical_type_map[arrow_type.id]
    except KeyError:
        if isinstance(arrow_type, pa.lib.DictionaryType):
            return 'categorical'
        elif isinstance(arrow_type, pa.lib.ListType):
            return 'list[{}]'.format(get_logical_type(arrow_type.value_type))
        elif isinstance(arrow_type, pa.lib.TimestampType):
            return 'datetimetz' if arrow_type.tz is not None else 'datetime'
        elif isinstance(arrow_type, pa.lib.Decimal128Type):
            return 'decimal'
        raise NotImplementedError(str(arrow_type))


_numpy_logical_type_map = {
    np.bool_: 'bool',
    np.int8: 'int8',
    np.int16: 'int16',
    np.int32: 'int32',
    np.int64: 'int64',
    np.uint8: 'uint8',
    np.uint16: 'uint16',
    np.uint32: 'uint32',
    np.uint64: 'uint64',
    np.float32: 'float32',
    np.float64: 'float64',
    'datetime64[D]': 'date',
    np.str_: 'unicode',
    np.bytes_: 'bytes',
}


def get_logical_type_from_numpy(pandas_collection):
    try:
        return _numpy_logical_type_map[pandas_collection.dtype.type]
    except KeyError:
        if hasattr(pandas_collection.dtype, 'tz'):
            return 'datetimetz'
        result = infer_dtype(pandas_collection)

        if result == 'string':
            return 'bytes' if PY2 else 'unicode'
        return result


def get_extension_dtype_info(column):
    dtype = column.dtype
    if str(dtype) == 'category':
        cats = getattr(column, 'cat', column)
        assert cats is not None
        metadata = {
            'num_categories': len(cats.categories),
            'ordered': cats.ordered,
        }
        physical_dtype = str(cats.codes.dtype)
    elif hasattr(dtype, 'tz'):
        metadata = {'timezone': str(dtype.tz)}
        physical_dtype = 'datetime64[ns]'
    else:
        metadata = None
        physical_dtype = str(dtype)
    return physical_dtype, metadata


def get_column_metadata(column, name, arrow_type, field_name):
    """Construct the metadata for a given column

    Parameters
    ----------
    column : pandas.Series or pandas.Index
    name : str
    arrow_type : pyarrow.DataType
    field_name : str
        Equivalent to `name` when `column` is a `Series`, otherwise if `column`
        is a pandas Index then `field_name` will not be the same as `name`.
        This is the name of the field in the arrow Table's schema.

    Returns
    -------
    dict
    """
    logical_type = get_logical_type(arrow_type)

    string_dtype, extra_metadata = get_extension_dtype_info(column)
    if logical_type == 'decimal':
        extra_metadata = {
            'precision': arrow_type.precision,
            'scale': arrow_type.scale,
        }
        string_dtype = 'object'

    if name is not None and not isinstance(name, six.string_types):
        raise TypeError(
            'Column name must be a string. Got column {} of type {}'.format(
                name, type(name).__name__
            )
        )

    if not isinstance(field_name, six.string_types):
        field_name = frombytes(field_name)

    return {
        'name': name,
        'field_name': field_name,
        'pandas_type': logical_type,
        'numpy_type': string_dtype,
        'metadata': extra_metadata,
    }


def construct_metadata(df, column_names, index_levels, index_column_names,
                       preserve_index, types):
    """Returns a dictionary containing enough metadata to reconstruct a pandas
    DataFrame as an Arrow Table, including index columns.

    Parameters
    ----------
    df : pandas.DataFrame
    index_levels : List[pd.Index]
    presere_index : bool
    types : List[pyarrow.DataType]

    Returns
    -------
    dict
    """
    df_types = types[:-len(index_levels)]
    index_types = types[-len(index_levels):]

    column_metadata = [
        get_column_metadata(
            df[col_name],
            name=sanitized_name,
            arrow_type=arrow_type,
            field_name=sanitized_name
        ) for col_name, sanitized_name, arrow_type in zip(
            df.columns, column_names, df_types
        )
    ]

    if preserve_index:
        index_column_metadata = [
            get_column_metadata(
                level,
                name=level.name,
                arrow_type=arrow_type,
                field_name=field_name,
            ) for i, (level, arrow_type, field_name) in enumerate(
                zip(index_levels, index_types, index_column_names)
            )
        ]

        column_indexes = []

        for level in getattr(df.columns, 'levels', [df.columns]):
            string_dtype, extra_metadata = get_extension_dtype_info(level)

            pandas_type = get_logical_type_from_numpy(level)
            if pandas_type == 'unicode':
                assert not extra_metadata
                extra_metadata = {'encoding': 'UTF-8'}

            column_index = {
                'name': level.name,
                'field_name': level.name,
                'pandas_type': pandas_type,
                'numpy_type': string_dtype,
                'metadata': extra_metadata,
            }
            column_indexes.append(column_index)
    else:
        index_column_names = index_column_metadata = column_indexes = []

    return {
        b'pandas': json.dumps({
            'index_columns': index_column_names,
            'column_indexes': column_indexes,
            'columns': column_metadata + index_column_metadata,
            'pandas_version': pd.__version__
        }).encode('utf8')
    }


def _column_name_to_strings(name):
    """Convert a column name (or level) to either a string or a recursive
    collection of strings.

    Parameters
    ----------
    name : str or tuple

    Returns
    -------
    value : str or tuple

    Examples
    --------
    >>> name = 'foo'
    >>> _column_name_to_strings(name)
    'foo'
    >>> name = ('foo', 'bar')
    >>> _column_name_to_strings(name)
    ('foo', 'bar')
    >>> import pandas as pd
    >>> name = (1, pd.Timestamp('2017-02-01 00:00:00'))
    >>> _column_name_to_strings(name)
    ('1', '2017-02-01 00:00:00')
    """
    if isinstance(name, six.string_types):
        return name
    elif isinstance(name, tuple):
        return tuple(map(_column_name_to_strings, name))
    elif isinstance(name, collections.Sequence):
        raise TypeError("Unsupported type for MultiIndex level")
    elif name is None:
        return None
    return str(name)


def _index_level_name(index, i, column_names):
    """Return the name of an index level or a default name if `index.name` is
    None or is already a column name.

    Parameters
    ----------
    index : pandas.Index
    i : int

    Returns
    -------
    name : str
    """
    if index.name is not None and index.name not in column_names:
        return index.name
    else:
        return '__index_level_{:d}__'.format(i)


def dataframe_to_arrays(df, schema, preserve_index, nthreads=1):
    column_names = []
    index_columns = []
    index_column_names = []
    type = None

    if preserve_index:
        n = len(getattr(df.index, 'levels', [df.index]))
        index_columns.extend(df.index.get_level_values(i) for i in range(n))

    columns_to_convert = []
    convert_types = []

    if not df.columns.is_unique:
        raise ValueError(
            'Duplicate column names found: {}'.format(list(df.columns))
        )

    for name in df.columns:
        col = df[name]
        if not isinstance(name, six.string_types):
            name = _column_name_to_strings(name)
            if name is not None:
                name = frombytes(name)

        if schema is not None:
            field = schema.field_by_name(name)
            type = getattr(field, "type", None)

        columns_to_convert.append(col)
        convert_types.append(type)
        column_names.append(name)

    for i, column in enumerate(index_columns):
        columns_to_convert.append(column)
        convert_types.append(None)
        name = _index_level_name(column, i, column_names)
        index_column_names.append(name)

    # NOTE(wesm): If nthreads=None, then we use a heuristic to decide whether
    # using a thread pool is worth it. Currently the heuristic is whether the
    # nrows > 100 * ncols.
    if nthreads is None:
        nrows, ncols = len(df), len(df.columns)
        if nrows > ncols * 100:
            nthreads = pa.cpu_count()
        else:
            nthreads = 1

    def convert_column(col, ty):
        return pa.array(col, from_pandas=True, type=ty)

    if nthreads == 1:
        arrays = [convert_column(c, t)
                  for c, t in zip(columns_to_convert,
                                  convert_types)]
    else:
        from concurrent import futures
        with futures.ThreadPoolExecutor(nthreads) as executor:
            arrays = list(executor.map(convert_column,
                                       columns_to_convert,
                                       convert_types))

    types = [x.type for x in arrays]

    metadata = construct_metadata(
        df, column_names, index_columns, index_column_names, preserve_index,
        types
    )
    names = column_names + index_column_names
    return names, arrays, metadata


def get_datetimetz_type(values, dtype, type_):
    from pyarrow.compat import DatetimeTZDtype

    if values.dtype.type != np.datetime64:
        return values, type_

    if isinstance(dtype, DatetimeTZDtype) and type_ is None:
        # If no user type passed, construct a tz-aware timestamp type
        tz = dtype.tz
        unit = dtype.unit
        type_ = pa.timestamp(unit, tz)
    elif type_ is None:
        # Trust the NumPy dtype
        type_ = pa.from_numpy_dtype(values.dtype)

    return values, type_

# ----------------------------------------------------------------------
# Converting pandas.DataFrame to a dict containing only NumPy arrays or other
# objects friendly to pyarrow.serialize


def dataframe_to_serialized_dict(frame):
    block_manager = frame._data

    blocks = []
    axes = [ax for ax in block_manager.axes]

    for block in block_manager.blocks:
        values = block.values
        block_data = {}

        if isinstance(block, _int.DatetimeTZBlock):
            block_data['timezone'] = values.tz.zone
            values = values.values
        elif isinstance(block, _int.CategoricalBlock):
            block_data.update(dictionary=values.categories,
                              ordered=values.ordered)
            values = values.codes

        block_data.update(
            placement=block.mgr_locs.as_array,
            block=values
        )
        blocks.append(block_data)

    return {
        'blocks': blocks,
        'axes': axes
    }


def serialized_dict_to_dataframe(data):
    reconstructed_blocks = [_reconstruct_block(block)
                            for block in data['blocks']]

    block_mgr = _int.BlockManager(reconstructed_blocks, data['axes'])
    return pd.DataFrame(block_mgr)


def _reconstruct_block(item):
    # Construct the individual blocks converting dictionary types to pandas
    # categorical types and Timestamps-with-timezones types to the proper
    # pandas Blocks

    block_arr = item['block']
    placement = item['placement']
    if 'dictionary' in item:
        cat = pd.Categorical.from_codes(block_arr,
                                        categories=item['dictionary'],
                                        ordered=item['ordered'])
        block = _int.make_block(cat, placement=placement,
                                klass=_int.CategoricalBlock)
    elif 'timezone' in item:
        dtype = _make_datetimetz(item['timezone'])
        block = _int.make_block(block_arr, placement=placement,
                                klass=_int.DatetimeTZBlock,
                                dtype=dtype)
    else:
        block = _int.make_block(block_arr, placement=placement)

    return block


def _make_datetimetz(tz):
    from pyarrow.compat import DatetimeTZDtype
    return DatetimeTZDtype('ns', tz=tz)


# ----------------------------------------------------------------------
# Converting pyarrow.Table efficiently to pandas.DataFrame


def table_to_blockmanager(options, table, memory_pool, nthreads=1,
                          categoricals=None):
    from pyarrow.compat import DatetimeTZDtype

    index_columns = []
    columns = []
    column_indexes = []
    index_arrays = []
    index_names = []
    schema = table.schema
    row_count = table.num_rows
    metadata = schema.metadata

    has_pandas_metadata = metadata is not None and b'pandas' in metadata

    if has_pandas_metadata:
        pandas_metadata = json.loads(metadata[b'pandas'].decode('utf8'))
        index_columns = pandas_metadata['index_columns']
        columns = pandas_metadata['columns']
        column_indexes = pandas_metadata.get('column_indexes', [])
        table = _add_any_metadata(table, pandas_metadata)

    block_table = table

    index_columns_set = frozenset(index_columns)

    # 0. 'field_name' is the name of the column in the arrow Table
    # 1. 'name' is the user-facing name of the column, that is, it came from
    #    pandas
    # 2. 'field_name' and 'name' differ for index columns
    # 3. We fall back on c['name'] for backwards compatibility
    logical_index_names = [
        c['name'] for c in columns
        if c.get('field_name', c['name']) in index_columns_set
    ]

    # There must be the same number of field names and physical names
    # (fields in the arrow Table)
    assert len(logical_index_names) == len(index_columns_set)

    # It can never be the case in a released version of pyarrow that
    # c['name'] is None *and* 'field_name' is not a key in the column metadata,
    # because the change to allow c['name'] to be None and the change to add
    # 'field_name' are in the same release (0.8.0)
    assert all(
        (c['name'] is None and 'field_name' in c) or c['name'] is not None
        for c in columns
    )

    # Build up a list of index columns and names while removing those columns
    # from the original table
    for raw_name, logical_name in zip(index_columns, logical_index_names):
        i = schema.get_field_index(raw_name)
        if i != -1:
            col = table.column(i)
            col_pandas = col.to_pandas()
            values = col_pandas.values
            if hasattr(values, 'flags') and not values.flags.writeable:
                # ARROW-1054: in pandas 0.19.2, factorize will reject
                # non-writeable arrays when calling MultiIndex.from_arrays
                values = values.copy()

            if isinstance(col_pandas.dtype, DatetimeTZDtype):
                index_array = (pd.Series(values).dt.tz_localize('utc')
                               .dt.tz_convert(col_pandas.dtype.tz))
            else:
                index_array = pd.Series(values, dtype=col_pandas.dtype)
            index_arrays.append(index_array)
            index_names.append(
                _backwards_compatible_index_name(raw_name, logical_name)
            )
            block_table = block_table.remove_column(
                block_table.schema.get_field_index(raw_name)
            )

    blocks = _table_to_blocks(options, block_table, nthreads, memory_pool)

    # Construct the row index
    if len(index_arrays) > 1:
        index = pd.MultiIndex.from_arrays(index_arrays, names=index_names)
    elif len(index_arrays) == 1:
        index = pd.Index(index_arrays[0], name=index_names[0])
    else:
        index = pd.RangeIndex(row_count)

    column_strings = [x.name for x in block_table.itercolumns()]
    if columns:
        columns_name_dict = {}
        for c in columns:
            column_name = c['name']
            if not isinstance(column_name, six.text_type):
                column_name = frombytes(column_name)

            columns_name_dict[c.get('field_name', column_name)] = c['name']

        columns_values = [
            columns_name_dict.get(name, name) for name in column_strings
        ]
    else:
        columns_values = column_strings

    # If we're passed multiple column indexes then evaluate with
    # ast.literal_eval, since the column index values show up as a list of
    # tuples
    to_pair = ast.literal_eval if len(column_indexes) > 1 else lambda x: (x,)

    # Create the column index

    # Construct the base index
    if not columns_values:
        columns = pd.Index(columns_values)
    else:
        columns = pd.MultiIndex.from_tuples(
            list(map(to_pair, columns_values)),
            names=[col_index['name'] for col_index in column_indexes] or None,
        )

    # if we're reconstructing the index
    if has_pandas_metadata:
        columns = _reconstruct_columns_from_metadata(columns, column_indexes)

    # ARROW-1751: flatten a single level column MultiIndex for pandas 0.21.0
    columns = _flatten_single_level_multiindex(columns)

    axes = [columns, index]
    return _int.BlockManager(blocks, axes)


def _backwards_compatible_index_name(raw_name, logical_name):
    # Part of table_to_blockmanager
    pattern = r'^__index_level_\d+__$'
    if raw_name == logical_name and re.match(pattern, raw_name) is not None:
        return None
    else:
        return logical_name


def _reconstruct_columns_from_metadata(columns, column_indexes):
    # Part of table_to_blockmanager

    # Get levels and labels, and provide sane defaults if the index has a
    # single level to avoid if/else spaghetti.
    levels = getattr(columns, 'levels', None) or [columns]
    labels = getattr(columns, 'labels', None) or [
        pd.RangeIndex(len(level)) for level in levels
    ]

    # Convert each level to the dtype provided in the metadata
    levels_dtypes = [
        (level, col_index.get('numpy_type', level.dtype))
        for level, col_index in zip_longest(
            levels, column_indexes, fillvalue={}
        )
    ]
    new_levels = [
        _level if _level.dtype == _dtype else _level.astype(_dtype)
        for _level, _dtype in levels_dtypes
    ]

    return pd.MultiIndex(
        levels=new_levels,
        labels=labels,
        names=columns.names
    )


def _table_to_blocks(options, block_table, nthreads, memory_pool):
    # Part of table_to_blockmanager

    # Convert an arrow table to Block from the internal pandas API
    result = pa.lib.table_to_blocks(options, block_table, nthreads,
                                    memory_pool)

    # Defined above
    return [_reconstruct_block(item) for item in result]


def _flatten_single_level_multiindex(index):
    if isinstance(index, pd.MultiIndex) and index.nlevels == 1:
        levels, = index.levels
        labels, = index.labels

        # Cheaply check that we do not somehow have duplicate column names
        if not index.is_unique:
            raise ValueError('Found non-unique column index')

        return pd.Index([levels[_label] if _label != -1 else None
                         for _label in labels],
                        name=index.names[0])
    return index


def _add_any_metadata(table, pandas_metadata):
    modified_columns = {}

    schema = table.schema

    index_columns = pandas_metadata['index_columns']
    n_index_levels = len(index_columns)
    n_columns = len(pandas_metadata['columns']) - n_index_levels

    # Add time zones
    for i, col_meta in enumerate(pandas_metadata['columns']):

        raw_name = col_meta.get('field_name')
        if not raw_name:
            # deal with metadata written with arrow < 0.8
            raw_name = col_meta['name']
            if i >= n_columns:
                # index columns
                raw_name = index_columns[i - n_columns]
            if raw_name is None:
                raw_name = 'None'

        idx = schema.get_field_index(raw_name)
        if idx != -1:
            if col_meta['pandas_type'] == 'datetimetz':
                col = table[idx]
                converted = col.to_pandas()
                tz = col_meta['metadata']['timezone']
                tz_aware_type = pa.timestamp('ns', tz=tz)
                with_metadata = pa.Array.from_pandas(converted.values,
                                                     type=tz_aware_type)

                field = pa.field(schema[idx].name, tz_aware_type)
                modified_columns[idx] = pa.Column.from_array(field,
                                                             with_metadata)

    if len(modified_columns) > 0:
        columns = []
        for i in range(len(table.schema)):
            if i in modified_columns:
                columns.append(modified_columns[i])
            else:
                columns.append(table[i])
        return pa.Table.from_arrays(columns)
    else:
        return table
