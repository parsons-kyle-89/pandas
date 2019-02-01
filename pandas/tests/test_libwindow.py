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
    expected_start = np.array([0.0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    nt.assert_array_equal(start, expected_start)


def test_fixed_window_indexer_end(fixed_window_indexer):
    _, end, _, _, _, _, _, _ = fixed_window_indexer.get_data()
    expected_end = np.array([2.0, 3, 4, 5, 6, 7, 8, 9, 10, 10])
    nt.assert_array_equal(end, expected_end)

@pytest.mark.parametrize("left_off", range(-3, 3))
@pytest.mark.parametrize("right_off", range(-3, 3))
def test_fixed_window_indexer_on_length_zero_array(left_off, right_off):
    fixed_window_indexer = \
        libwindow.FixedWindowIndexer(np.array([]), left_off, right_off,
                                     0, 0, 0, None, None)
    start, end, _, _, _, _, _, _ = fixed_window_indexer.get_data()
    assert len(start) == 0
    assert len(end) == 0


def roll_mean_with_fix_window(right, left, minp):
    values = np.array(list(range(5)))
    left_off = left
    right_off = right
    minp = minp
    index = None
    closed = None
    return libwindow.roll_mean(values, left_off, right_off, minp, index, closed)


@pytest.mark.parametrize(
    "params, expected",

    [({"right": 0,
       "left": -0,
       "minp": 1},
      np.array([0.0, 1, 2, 3, 4])),
     ({"right": 1,
       "left": -1,
       "minp": 1},
      np.array([0.5, 1, 2, 3, 3.5])),
     ({"right": 1,
       "left": -2,
       "minp": 1},
      np.array([0.5, 1, 6.0 / 4, 2.5, 3])),
     ({"right": 1,
       "left": -1,
       "minp": 3},
      np.array([np.nan, 1, 2, 3, np.nan])),
     ({"right": 1,
       "left": -2,
       "minp": 3},
      np.array([np.nan, 1, 6.0 / 4, 2.5, 3]))]
)
def test_roll_mean_with_fix_window(params, expected):
    actual = roll_mean_with_fix_window(**params)
    np.testing.assert_array_almost_equal(actual, expected)
