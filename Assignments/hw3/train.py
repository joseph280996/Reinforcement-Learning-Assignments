import gymnasium as gym
import numpy as np
from typing import List
from collections import defaultdict

class NStepSarsa:
    def __init__(
        self,
        env,
        method: str,
        alpha: float = 0.1,
        gamma: float = 0.9,
        n_steps: int = 5,
        epsilon: float = 0.1,
        q_init_low: float = 0,
        q_init_high: float = 1,
    ):
        """
        Initialize the Off-policy n-step SARSA algorithm.
        
        Args:
            env: The environment to train the agent in
            method: The policy update method to use ('sophisticated' = (7.13) + (7.2), 'simplified' = (7.9) + (7.1))
            alpha: Learning rate
            gamma: Discount factor
            n_steps: Number of steps to look ahead
            epsilon: Exploration rate for ε-greedy behavior policy
            q_init_low: Lower bound of Q table initialization distribution
            q_init_high: Upper bound of Q table initialization distribution
        """
        self.alpha = alpha
        self.gamma = gamma
        self.n = n_steps
        self.epsilon = epsilon
        self.q_init_low = q_init_low
        self.q_init_high = q_init_high
        self.method = method

        
        # Create environment (Discrete version)
        self.env = env
        
        # Initialize V and Q tables
        self.Q = defaultdict(self.initialize_q_values)  # State-action values
        
        # Get action space size
        self.n_actions = self.env.action_space.n

    def initialize_q_values(self):
        """
        Initialize Q tables with arbitrary numbers (uniform distribution from 0 to 1 by default)
        """
        return np.random.uniform(self.q_init_low, self.q_init_high, self.env.action_space.n)
 
    def get_target_policy_prob(self, state: int, action: int) -> float:
        """
        Get probability of action under target policy (greedy)
        π(a|s) = 1 if a = argmax Q(s,a'), 0 otherwise
        """
        best_action = self.get_greedy_action(state)
        return 1.0 if action == best_action else 0.0

    def get_behavior_policy_prob(self, state: int, action: int) -> float:
        """
        Get probability of action under behavior policy (ε-greedy)
        b(a|s) = ε/|A| + (1-ε) if a = argmax Q(s,a'), ε/|A| otherwise
        """
        best_action = self.get_greedy_action(state)
        if action == best_action:
            return self.epsilon / self.n_actions + (1 - self.epsilon)
        else:
            return self.epsilon / self.n_actions
    
    def calculate_importance_ratio(self, t: int, h: int) -> float:
        """
        Calculate importance sampling ratio ρ_{t:h-1} using formula (7.10)
        ρ_{t:h-1} = ∏_{i=t}^{min(h,T-1)} π(A_i|S_i) / b(A_i|S_i)
        """
        rho = 1.0
        for i in range(t, min(h, self.T-1) + 1):
            S_i = self.S[i]
            A_i = self.A[i]
            target_prob = self.get_target_policy_prob(S_i, A_i)
            behavior_prob = self.get_behavior_policy_prob(S_i, A_i)
            rho *= target_prob / behavior_prob
        return rho
    
    def calculate_truncated_return(self, t: int, h: int) -> float:
        """
        Calculate the truncated return G_{t:h} using formula (7.13)
        G_{t:h} = ρ_t(R_{t+1} + γG_{t+1:h}) + (1-ρ_t)V_{h-1}(S_t)
        """
        if t == h:
            return self.Q[tuple(self.S[t])][self.A[t]]
        
        if t == self.T:
            return 0
        
        # Get state
        S_t = self.S[t]
        A_t = self.A[t]
        
        # Calculate importance sampling ratio from t to h
        rho = self.calculate_importance_ratio(t, h)
        
        # Get immediate reward
        R_tp1 = self.R[t + 1]
        
        # Recursive calculation of G_{t+1:h}
        G_tp1_h = self.calculate_truncated_return(t + 1, h)
        
        # Get Q_{h-1}(S_t)
        Q_hm1_St = self.Q[tuple(S_t)][A_t]
        
        return rho * (R_tp1 + self.gamma * G_tp1_h) + (1 - rho) * Q_hm1_St
    
    def get_greedy_action(self, state: int) -> int:
        """Get the greedy action for a given state"""
        q_values = self.Q[tuple(state)]
        return np.argmax(q_values)
    
    def select_action(self, state: int) -> int:
        """
        Select action using ε-greedy behavior policy
        With probability ε, select random action
        With probability 1-ε, select greedy action
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_greedy_action(state)
    
    
    def sophisticated_update(self, tau: int) -> None:
        """
        Policy update follows (7.13) and (7.2)
        """
        # Calculate truncated return
        h = tau + self.n
        G = self.calculate_truncated_return(tau, h)
        
        # Update Q-value using (7.2)
        tau_action = self.A[tau]
        self.Q[tuple(self.S[tau])][tau_action] += (
            self.alpha * (G - self.Q[tuple(self.S[tau])][tau_action])
        )
    
    def calculate_n_step_return(self, t: int, n: int) -> float:
        """
        Calculate n-step return using formula (7.1) adapted for Q-values
        G_{t:t+n} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n Q(S_{t+n}, A_{t+n})
        """
        G = 0.0
        end = min(t + n, self.T)
        
        # Sum up rewards
        for i in range(t, end):
            G += self.gamma ** (i - t) * self.R[i + 1]
        
        # Add bootstrap value if not at terminal state
        if t + n < self.T:
            state = self.S[t + n]
            action = self.A[t + n]
            G += self.gamma ** n * self.Q[tuple(state)][action]
            
        return G

    def simplified_update(self, tau: int) -> None:
        """
        Policy update follows (7.9) and (7.1)
        """
        # Calculate n-step return using 7.1
        G = self.calculate_n_step_return(tau, self.n)
        
        # Calculate importance sampling ratio
        rho = self.calculate_importance_ratio(tau, self.n)
        
        # Update Q-value using 7.9 adapted for SARSA
        tau_state = tuple(self.S[tau])
        tau_action = self.A[tau]
        self.Q[tau_state][tau_action] += self.alpha * rho * (G - self.Q[tau_state][tau_action])

    def update(self, tau):
        if self.method == 'sophisticated':
            self.sophisticated_update(tau)
        elif self.method == 'simplified':
            self.simplified_update(tau)
        else:
            raise Exception("Unsupported operation")
    
    def run_episode(self) -> List[float]:
        """Run a complete episode following the algorithm pseudocode"""
        # Initialize state, action, and storage
        S_0, _ = self.env.reset()
        A_0 = self.select_action(S_0)
        
        self.S = [S_0]  # S₀
        self.A = [A_0]  # A₀
        self.R = [0]  # R₀ (dummy value)
        
        self.T = float('inf')
        t = 0
        
        episode_rewards = []
        
        # Loop until τ = T - 1
        while True:

            # Check if we need to take an action
            if t < self.T:
                print("Chosen action:", self.A[t])
                # Take action At
                next_state, reward, terminated, truncated, _ = self.env.step(self.A[t])
                episode_rewards.append(reward)
                print("Current reward:", reward)
                
                # Store next reward and state
                self.R.append(reward)
                self.S.append(next_state)
                
                if terminated or truncated:
                    self.T = t + 1
                else:
                    # Select and store next action using ε-greedy
                    A_tp1 = self.select_action(next_state)
                    print("Chosen next action:", A_tp1)
                    self.A.append(A_tp1)
            
            # τ is the time whose estimate is being updated
            tau = t - self.n + 1
            
            # Perform update if we have enough steps
            if tau >= 0:
                self.update(tau)
            
            # Increment time step
            t += 1
            
            # Termination condition: τ = T - 1
            if tau == self.T - 1:
                break

        
        return episode_rewards

