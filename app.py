import streamlit as st
import matplotlib.pyplot as plt
from pso import PSO

st.set_page_config(page_title="PSO Ticket Pricing", layout="centered")

st.title("🎟 Cinema Ticket Pricing Optimization (PSO)")

particles = st.slider("Number of Particles", 10, 100, 30)
iterations = st.slider("Iterations", 50, 200, 100)

pso = PSO(particles, iterations)
best_price, best_revenue, history = pso.optimize()

st.subheader("📊 Optimization Results")
st.write(f"**Optimal Ticket Price:** RM {best_price:.2f}")
st.write(f"**Maximum Revenue:** RM {best_revenue:.2f}")

st.subheader("📈 Convergence Curve")
plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("Revenue")
st.pyplot(plt)
