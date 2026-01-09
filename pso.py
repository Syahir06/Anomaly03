import numpy as np

def fitness(price):
    demand = max(0, 500 - 20 * price)
    return -(price * demand)

class PSO:
    def __init__(self, n_particles, iterations):
        self.n_particles = n_particles
        self.iterations = iterations
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5

        self.positions = np.random.uniform(5, 25, n_particles)
        self.velocities = np.random.uniform(-1, 1, n_particles)

        self.pbest_positions = self.positions.copy()
        self.pbest_scores = [fitness(p) for p in self.positions]

        self.gbest_position = self.pbest_positions[np.argmin(self.pbest_scores)]
        self.gbest_score = min(self.pbest_scores)

    def optimize(self):
        history = []

        for _ in range(self.iterations):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(), np.random.rand()

                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                    + self.c2 * r2 * (self.gbest_position - self.positions[i])
                )

                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], 5, 25)

                score = fitness(self.positions[i])

                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i]

            self.gbest_position = self.pbest_positions[np.argmin(self.pbest_scores)]
            self.gbest_score = min(self.pbest_scores)
            history.append(-self.gbest_score)

        return self.gbest_position, -self.gbest_score, history
