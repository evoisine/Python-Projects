import unittest

class TestKMeans(unittest.TestCase):


    # ================================================================================ Test Labels
    def test_find_labels_00(self):
        from kmeans import find_labels
        import numpy as np
        pts = np.array(
            [
                [1,1],
                [2,2]
            ])
        centroids = np.array(
            [
                [0,0]
            ])
        labels = find_labels(pts, centroids)
        self.assertTrue(np.all(labels==[0,0]))


    def test_find_labels_01(self):
        from kmeans import find_labels
        import numpy as np
        pts = np.array(
            [
                [1,1],
                [4,4],
                [5,3]
            ])
        centroids = np.array(
            [
                [0,0],
                [4,2]
            ])
        labels = find_labels(pts, centroids)
        self.assertTrue(np.all(labels==[0,1,1]))


    def test_find_labels_02(self):
        from kmeans import find_labels
        import numpy as np
        pts = np.array(
            [
                [1,1,1],
                [4,4,4],
                [5,5,3],
                [0,0,1]
            ])
        centroids = np.array(
            [
                [0,0,0],
                [4,2,1],
                [1,1,0]
            ])
        labels = find_labels(pts, centroids)
        self.assertTrue(np.all(labels==[2,1,1,0]))


    # ================================================================================ Test Centroid Computation
    def test_recompute_centroids_00(self):
        from kmeans import recompute_centroids
        import numpy as np
        labels = np.array([
            0,
            1,
        ])
        pts = np.array([
            [0,0],
            [2,2]
        ])
        old_centroids = np.array([
            [0,1],
            [2,1]
        ])
        new_centroids = recompute_centroids(pts, old_centroids, labels)
        self.assertTrue(np.all(np.isclose(new_centroids[0,:], [0,0], atol=0.01)))
        self.assertTrue(np.all(np.isclose(new_centroids[1,:], [2,2], atol=0.01)))



    def test_recompute_centroids_01(self):
        from kmeans import recompute_centroids
        import numpy as np
        labels = np.array([
            0,
            1,
            1,
            0,
            1
        ])
        pts = np.array([
            [0,0],
            [1,3],
            [2,4],
            [0,1],
            [5,5]
        ])
        old_centroids = np.array([
            [1,1],
            [2,2]
        ])
        new_centroids = recompute_centroids(pts, old_centroids, labels)
        self.assertTrue(np.all(np.isclose(new_centroids[0,:], [0, 0.5], atol=0.01)))
        self.assertTrue(np.all(np.isclose(new_centroids[1,:], [2.66, 4], atol=0.01)))



    def test_recompute_centroids_02(self):
        from kmeans import recompute_centroids
        import numpy as np
        labels = np.array([
            0,
            1,
            1,
            0,
            1,
            2
        ])
        pts = np.array([
            [0,0,0],
            [1,3,2],
            [2,4,4],
            [0,1,1],
            [5,5,5],
            [10,10,11]
        ])
        old_centroids = np.array([
            [1,1,1],
            [3,3,3],
            [9,9,9]
        ])
        new_centroids = recompute_centroids(pts, old_centroids, labels)
        self.assertTrue(np.all(np.isclose(new_centroids[0,:], [0, 0.5, 0.5], atol=0.01)))
        self.assertTrue(np.all(np.isclose(new_centroids[1,:], [2.66, 4, 3.66], atol=0.01)))
        self.assertTrue(np.all(np.isclose(new_centroids[2,:], [10, 10, 11], atol=0.01)))





    # ================================================================================ Test SSE Calculation
    def test_get_sse_00(self):
        from kmeans import get_sse
        import numpy as np
        pts = np.array(
        [
            [1,1],
            [2,2]
        ])
        centroids = np.array(
        [
            [0,0]
        ])
        labels = np.array([0,0])
        sse = get_sse(pts, centroids, labels)
        self.assertAlmostEqual(sse, 10.0, 0.01)

    
    def test_get_sse_01(self):
        from kmeans import get_sse
        import numpy as np
        pts = np.array(
            [
                [1,1],
                [4,4],
                [5,3]
            ])
        centroids = np.array(
            [
                [0,0],
                [4,2]
            ])
        labels = np.array(
            [
                0,
                1,
                1
            ]
        )
        sse = get_sse(pts, centroids, labels)
        self.assertAlmostEqual(sse, 8.0, 0.01)


    def test_get_sse_02(self):
        from kmeans import get_sse
        import numpy as np
        pts = np.array(
            [
                [1,1,1],
                [4,4,4],
                [5,5,3],
                [0,0,1]
            ])
        centroids = np.array(
            [
                [0,0,0],
                [4,2,1],
                [1,1,0]
            ])
        labels = np.array(
            [
                2,
                1,
                1,
                0
            ]
        )
        sse = get_sse(pts, centroids, labels)
        self.assertAlmostEqual(sse, 29.0, 0.01)
        


    # ================================================================================ Test Entire K-Means
    def test_2dim_2cluster_00(self):
        """
        2-dimension, 2 cluster
        """
        from testing import generate_2D_blob, evaluate
        pts = generate_2D_blob((10,0.2), (11,0.2), 55, 0)
        pts = pts.append(generate_2D_blob((4,0.5), (8,0.3), 45, 1), ignore_index=True)
        pts = pts.sample(frac=1)
        pts.reset_index(inplace=True)
        err = evaluate(pts, 2)
        self.assertTrue(err <= 0.05)


    def test_2dim_2cluster_01(self):
        """
        2-dimension, 2 cluster
        """
        from testing import generate_2D_blob, evaluate
        pts = generate_2D_blob((3,0.6), (11,0.6), 45, 0)
        pts = pts.append(generate_2D_blob((20,0.5), (88,0.7), 55, 1), ignore_index=True)
        pts = pts.sample(frac=1)
        pts.reset_index(inplace=True)

        err = evaluate(pts, 2)
        self.assertTrue(err <= 0.05)


    def test_2dim_3cluster_00(self):
        """
        2-dimension, 3 cluster
        """
        from testing import generate_2D_blob, evaluate
        pts = generate_2D_blob((10,0.2), (12,0.2), 33, 0)
        pts = pts.append(generate_2D_blob((5,0.5), (4,0.5), 33, 1), ignore_index=True)
        pts = pts.append(generate_2D_blob((6,0.2), (8,0.3), 34, 2), ignore_index=True)
        pts = pts.sample(frac=1)
        pts.reset_index(inplace=True)

        err = evaluate(pts, 3)
        self.assertTrue(err <= 0.05)


    def test_3dim_2cluster_00(self):
        """
        3-dimension, 2 cluster
        """
        from testing import generate_3D_blob, evaluate
        pts = generate_3D_blob((1,0.2), (1,0.2), (1,0.2), 25, 0)
        pts = pts.append(generate_3D_blob((3,0.2), (3,0.2), (3,0.2), 25, 1), ignore_index=True)
        pts = pts.sample(frac=1)
        pts.reset_index(inplace=True)

        err = evaluate(pts, 2)
        self.assertTrue(err <= 0.05)


    def test_3dim_3cluster_00(self):
        """
        3-dimension, 3 cluster
        """
        from testing import generate_3D_blob, evaluate
        pts = generate_3D_blob((7,0.2), (7,0.2), (9,0.2), 33, 0)
        pts = pts.append(generate_3D_blob((40,0.2), (22,0.2), (7,0.2), 33, 1), ignore_index=True)
        pts = pts.append(generate_3D_blob((10,0.2), (12,0.2), (71,0.2), 34, 2), ignore_index=True)
        pts = pts.sample(frac=1)
        pts.reset_index(inplace=True)

        err = evaluate(pts, 3)
        self.assertTrue(err <= 0.05)


