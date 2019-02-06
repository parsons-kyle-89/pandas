import numpy as np
import pandas._libs.window as libwindow
import pytest
import numpy.testing as nt


@pytest.fixture
def mock_fixed_window_indexer():
    values = np.arange(10)
    left_off = -2
    right_off = 2
    minp = 2
    left_closed = 0
    right_closed = 0
    index = None
    floor = None
    return libwindow.MockFixedWindowIndexer(values, left_off, right_off,
                                            minp, left_closed, right_closed,
                                            index, floor)


def test_mock_fixed_window_indexer_get_data(mock_fixed_window_indexer):
    _, _, N, win, left_off, right_off, minp, is_variable = \
        mock_fixed_window_indexer.get_data()
    assert N == 10
    assert win == 4
    assert left_off == -2
    assert right_off == 2
    assert minp == 2
    assert is_variable == 0


@pytest.fixture
def fixed_window_indexer():
    values = np.arange(10)
    left_off = -2
    right_off = 2
    minp = 2
    left_closed = 0
    right_closed = 0
    index = None
    floor = None
    return libwindow.FixedWindowIndexer(values, left_off, right_off,
                                        minp, left_closed, right_closed,
                                        index, floor)


def test_fixed_window_indexer_metadata(fixed_window_indexer):
    _, _, N, win, left_off, right_off, minp, is_variable = \
        fixed_window_indexer.get_data()
    assert N == 10
    assert win == 4
    assert left_off == -2
    assert right_off == 2
    assert minp == 2
    assert is_variable == 0


def test_fixed_window_indexer_start(fixed_window_indexer):
    start, _, _, _, _, _, _, _ = fixed_window_indexer.get_data()
    expected_start = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    nt.assert_array_equal(start, expected_start)


def test_fixed_window_indexer_end(fixed_window_indexer):
    _, end, _, _, _, _, _, _ = fixed_window_indexer.get_data()
    expected_end = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10])
    nt.assert_array_equal(end, expected_end)


@pytest.mark.parametrize("left_off", range(-2, 0))
@pytest.mark.parametrize("right_off", range(1, 3))
def test_fixed_window_indexer_on_length_zero_array(left_off, right_off):
    fixed_window_indexer = \
        libwindow.FixedWindowIndexer(np.array([]), left_off, right_off,
                                     0, 0, 0, None, None)
    start, end, _, _, _, _, _, _ = fixed_window_indexer.get_data()
    assert len(start) == 0
    assert len(end) == 0


def variable_window_indexer(left_off, right_off, left_closed, right_closed):
    values = np.arange(10)
    minp = 1
    index = np.array([-2, -1, 1, 2, 4, 5, 7, 8, 10, 11])
    floor = None
    return libwindow.VariableWindowIndexer(values, left_off, right_off,
                                           minp, left_closed, right_closed,
                                           index, floor)


def test_varaible_window_indexer_metadata():
    indexer = variable_window_indexer(-2, 2, 0, 1)
    _, _, N, _, left_off, right_off, minp, is_variable = indexer.get_data()
    assert N == 10
    assert left_off == -2
    assert right_off == 2
    assert minp == 1
    assert is_variable == 1


def test_varaible_window_indexer_window():
    indexer = variable_window_indexer(-2, 2, 0, 1)
    _, _, _, win, _, _, _, _ = indexer.get_data()
    assert win == 3


def test_varaible_window_indexer_start():
    indexer = variable_window_indexer(-2, 2, 0, 1)
    start, _, _, _, _, _, _, _ = indexer.get_data()
    expected_start = np.array([0, 0, 2, 2, 4, 4, 6, 6, 8, 8])
    nt.assert_array_equal(start, expected_start)


def test_varaible_window_indexer_end():
    indexer = variable_window_indexer(-2, 2, 0, 1)
    _, end, _, _, _, _, _, _ = indexer.get_data()
    expected_end = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10])
    nt.assert_array_equal(end, expected_end)


def test_varaible_window_indexer_start_closed_left():
    indexer = variable_window_indexer(-2, 2, 1, 1)
    start, _, _, _, _, _, _, _ = indexer.get_data()
    expected_start = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    nt.assert_array_equal(start, expected_start)


def test_varaible_window_indexer_end_open_right():
    indexer = variable_window_indexer(-2, 2, 0, 0)
    _, end, _, _, _, _, _, _ = indexer.get_data()
    expected_end = np.array([2, 2, 4, 4, 6, 6, 8, 8, 10, 10])
    nt.assert_array_equal(end, expected_end)

ROLL_FUNCTION_PARAMETER_SETS = (
    {"values": range(10),
     "left_off": -2,
     "right_off": 2,
     "minp": 1,
     "index": [-2, -1, 1, 2, 4, 5, 7, 8, 10, 11],
     "closed": None},

    {"values": range(10),
     "left_off": -2,
     "right_off": 2,
     "minp": 1,
     "index": None,
     "closed": None}
)

ROLL_FUNCTION_PARAMETER_SET_IDS = (
    "basic index",
    "basic no index"
)

@pytest.mark.parametrize(
    ["params", "expected_data"],

    zip(ROLL_FUNCTION_PARAMETER_SETS,
        ([1.0, 3, 5, 9, 9, 15, 13, 21, 17, 17],
         [1.0, 3, 6, 10, 14, 18, 22, 26, 30, 24])),

    ids = ROLL_FUNCTION_PARAMETER_SET_IDS
)
def test_roll_sum(params, expected_data):
    values = np.array(params["values"], dtype=np.float64)
    left_off = params["left_off"]
    right_off = params["right_off"]
    minp = params["minp"]
    index = np.array(params["index"]) if params["index"] is not None else None
    closed = params["closed"]
    actual = libwindow.roll_sum(values, left_off, right_off, minp, index, closed)
    expected = np.array(expected_data)
    nt.assert_array_equal(actual, expected)


def roll_mean_with_fix_window(right, left, minp):
    values = np.array(list(range(5))) + 0.0
    left_off = left
    right_off = right
    minp = minp
    index = None
    closed = None
    return libwindow.roll_mean(values, left_off, right_off, minp, index, closed)


@pytest.mark.parametrize(
    "params, expected",

    [({"right": 1,
       "left": -0,
       "minp": 1},
      np.array([0.0, 1, 2, 3, 4])),
     ({"right": 2,
       "left": -1,
       "minp": 1},
      np.array([0.5, 1, 2, 3, 3.5])),
     ({"right": 2,
       "left": -2,
       "minp": 1},
      np.array([0.5, 1, 6.0 / 4, 2.5, 3])),
     ({"right": 2,
       "left": -1,
       "minp": 3},
      np.array([np.nan, 1.0, 2, 3, np.nan])),
     ({"right": 2,
       "left": -2,
       "minp": 3},
      np.array([np.nan, 1.0, 6.0 / 4, 2.5, 3]))]
)
def test_roll_mean_with_fix_window(params, expected):
    actual = roll_mean_with_fix_window(**params)
    np.testing.assert_array_almost_equal(actual, expected)


def roll_var_with_fix_window(right, left, minp):
    values = np.array(list(range(5))) + 0.0
    left_off = left
    right_off = right
    minp = minp
    index = None
    closed = None
    return libwindow.roll_var(values, left_off, right_off, minp, index, closed)


@pytest.mark.parametrize(
    "params, expected",

    [({"right": 1,
       "left": -0,
       "minp": 1},
      np.array([np.nan] * 5)),
     ({"right": 2,
       "left": -1,
       "minp": 1},
      np.array([0.5, 1, 1, 1, 0.5])),
     ({"right": 2,
       "left": -2,
       "minp": 1},
      np.array([0.5, 1, 5.0 / 3, 5.0 / 3, 1])),
     ({"right": 2,
       "left": -1,
       "minp": 3},
      np.array([np.nan, 1.0, 1, 1, np.nan])),
     ({"right": 2,
       "left": -2,
       "minp": 3},
      np.array([np.nan, 1.0, 5.0 / 3, 5.0 / 3, 1]))]
)
def test_roll_var_with_fix_window(params, expected):
    actual = roll_var_with_fix_window(**params)
    np.testing.assert_array_almost_equal(actual, expected)
