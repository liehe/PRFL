import numpy as np


class ModelThresholding(object):
    def __init__(self, tau, iterations) -> None:
        self.tau = tau
        self.iterations = iterations

    def __call__(self, x, centers):
        K = len(centers)
        m = len(x)
        tau = self.tau

        update_count = 0
        for i in range(self.iterations):
            tau *= 0.9
            for k in range(K):
                update_count += sum(
                    1 if np.linalg.norm(centers[k] - x[j]) <= tau else 0
                    for j in range(m)
                )
                centers[k] = (
                    sum(
                        x[j] if np.linalg.norm(centers[k] - x[j]) <= tau else centers[k]
                        for j in range(m)
                    )
                    / m
                )

        # Find the assignment
        assignments = []
        for i in range(m):
            distances = np.array([np.linalg.norm(centers[k] - x[i]) for k in range(K)])
            (indices,) = np.where(distances == distances.min())
            assignments.append(np.random.choice(indices))

        return assignments, centers
