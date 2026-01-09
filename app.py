import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Data from GitHub
DATA_URL = "https://raw.githubusercontent.com/Syahir06/Anomaly03/refs/heads/main/cinema_ticket_pricing_clean.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    return df

# 2. Define the PSO Algorithm
class TicketPSO:
    def __init__(self, n_particles, iterations, price_range, demand_model):
        self.n_particles = n_particles
        self.iterations = iterations
        self.min_p, self.max_p = price_range
        self.demand_model = demand_model
        
        # Initialize particles
        self.X = np.random.uniform(self.min_p, self.max_p, n_particles)
        self.V = np.zeros(n_particles)
        self.p_best = self.X.copy()
        self.p_best_val = np.array([self.objective(x) for x in self.X])
        self.g_best = self.X[np.argmax(self.p_best_val)]
        self.g_best_val = np.max(self.p_best_val)
        self.history = []

    def objective(self, price):
        # Revenue = Price * Predicted Demand
        predicted_demand = self.demand_model(price)
        return price * predicted_demand

    def optimize(self, w=0.5, c1=1.5, c2=1.5):
        for _ in range(self.iterations):
            r1, r2 = np.random.rand(2)
            # Update Velocity
            self.V = (w * self.V + 
                      c1 * r1 * (self.p_best - self.X) + 
                      c2 * r2 * (self.g_best - self.X))
            # Update Position
            self.X = np.clip(self.X + self.V, self.min_p, self.max_p)
            
            # Update Bests
            for i in range(self.n_particles):
                fitness = self.objective(self.X[i])
                if fitness > self.p_best_val[i]:
                    self.p_best_val[i] = fitness
                    self.p_best[i] = self.X[i]
            
            if np.max(self.p_best_val) > self.g_best_val:
                self.g_best_val = np.max(self.p_best_val)
                self.g_best = self.p_best[np.argmax(self.p_best_val)]
            
            self.history.append(self.g_best_val)
        return self.g_best, self.g_best_val, self.history

# 3. Streamlit Interface
def main():
    st.title("🎬 Cinema Ticket Price Optimizer (PSO)")
    df = load_data()
    
    st.sidebar.header("PSO Hyperparameters")
    swarm_size = st.sidebar.slider("Swarm Size", 10, 100, 30)
    iters = st.sidebar.slider("Iterations", 10, 200, 50)
    w = st.sidebar.slider("Inertia (w)", 0.1, 1.0, 0.5)
    
    # Simple Demand Model (Linear Regression of Price vs Quantity from your CSV)
    # Note: In a real project, use sklearn to fit this model properly.
    m, b = -2.5, 150 # Placeholder coefficients: Demand = m*Price + b
    demand_fn = lambda p: max(0, m * p + b)

    if st.button("Run Optimization"):
        pso = TicketPSO(swarm_size, iters, (5, 50), demand_fn)
        best_price, best_rev, history = pso.optimize(w=w)
        
        col1, col2 = st.columns(2)
        col1.metric("Optimal Price", f"${best_price:.2f}")
        col2.metric("Max Revenue", f"${best_rev:.2f}")
        
        # Performance Analysis Plot
        fig, ax = plt.subplots()
        ax.plot(history, color='orange', linewidth=2)
        ax.set_title("PSO Convergence Curve")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Revenue")
        st.pyplot(fig)

        st.write("### Data Preview")
        st.dataframe(df.head())

if __name__ == "__main__":
    main()
