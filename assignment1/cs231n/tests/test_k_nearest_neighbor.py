from cs231n.classifiers import KNearestNeighbor
import numpy as np


mock_X_train = np.array(
    [
        (1, 3, 5, 8),
        (2, 6, 8, 10),
    ],
)

mock_y_train = np.array(
    [
        (1),
        (2),
    ],
)

mock_X_test = np.array(
    [
        (2, 4, 6, 9),
    ],
)

knn_instance = KNearestNeighbor()
knn_instance.train(mock_X_train, mock_y_train)

expected_dists = np.array(
    [
        [(2), (3)],
    ]
)


def test_compute_distance_two_loops():
    actual_dists = knn_instance.compute_distances_two_loops(mock_X_test)
    np.testing.assert_array_equal(actual_dists, expected_dists)


def test_compute_distance_one_loop():
    actual_dists = knn_instance.compute_distances_one_loop(mock_X_test)
    np.testing.assert_array_equal(actual_dists, expected_dists)
