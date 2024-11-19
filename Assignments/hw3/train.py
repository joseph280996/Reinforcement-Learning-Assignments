import gymnasium as gym
import numpy as np
from typing import List

class CarRacingNStepSarsa:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        n_steps: int = 5,
        epsilon: float = 0.1
    ):
        """
        Initialize the Off-policy n-step SARSA algorithm for Car Racing.
        
        Args:
            alpha: Learning rate
            gamma: Discount factor
            n_steps: Number of steps to look ahead
            epsilon: Exploration rate for ε-greedy behavior policy
        """
        self.alpha = alpha
        self.gamma = gamma
        self.n = n_steps
        self.epsilon = epsilon
        
        # Create environment (Discrete version)
        self.env = gym.make('CarRacing-v2', continuous=False, render_mode="rgb_array")
        
        # Initialize V and Q tables
        self.V = {}  # State values
        self.Q = {}  # State-action values
        
        # Get action space size
        self.n_actions = self.env.action_space.n
    
    def get_value(self, state: int) -> float:
        """Get value of a state, initializing if necessary"""
        if state not in self.V:
            self.V[state] = 0.0
        return self.V[state]
    
    def get_q_value(self, state: int, action: int) -> float:
        """Get Q-value for a state-action pair, initializing if necessary"""
        if (state, action) not in self.Q:
            self.Q[(state, action)] = 0.0
        return self.Q[(state, action)]

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
        ρ_{t:h-1} = ∏_{k=t}^{min(h-1,T-1)} π(A_k|S_k) / b(A_k|S_k)
        """
        rho = 1.0
        for k in range(t, min(h-1, self.T-1) + 1):
            state = self.states[k]
            action = self.actions[k]
            target_prob = self.get_target_policy_prob(state, action)
            behavior_prob = self.get_behavior_policy_prob(state, action)
            if behavior_prob == 0:  # Avoid division by zero
                return 0.0
            rho *= target_prob / behavior_prob
        return rho
    
    def calculate_truncated_return(self, t: int, h: int) -> float:
        """
        Calculate the truncated return G_{t:h} using formula (7.13)
        G_{t:h} = ρ_t(R_{t+1} + γG_{t+1:h}) + (1-ρ_t)V_{h-1}(S_t)
        """
        if t >= h:
            return 0.0
        
        # Get state
        state = self.states[t]
        
        # Calculate importance sampling ratio from t to h-1
        rho = self.calculate_importance_ratio(t, h)
        
        # Get immediate reward
        R_tp1 = self.rewards[t + 1]
        
        # Recursive calculation of G_{t+1:h}
        G_tp1_h = self.calculate_truncated_return(t + 1, h)
        
        # Get V_{h-1}(S_t)
        V_hm1_St = self.get_value(state)
        
        return rho * (R_tp1 + self.gamma * G_tp1_h) + (1 - rho) * V_hm1_St
    
    def get_greedy_action(self, state: int) -> int:
        """Get the greedy action for a given state"""
        q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
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
    
    def update_value_function(self, state: int, G: float) -> None:
        """
        Update the value function using formula (7.2)
        V_{t+n}(S_t) = V_{t+n-1}(S_t) + α[G_{t:t+n} - V_{t+n-1}(S_t)]
        """
        self.V[state] += self.alpha * (G - self.get_value(state))
    
    def run_episode(self) -> List[float]:
        """Run a complete episode following the algorithm pseudocode"""
        # Initialize state, action, and storage
        state, _ = self.env.reset()
        action = self.select_action(state)
        
        self.states = [state]  # S₀
        self.actions = [action]  # A₀
        self.rewards = [0]  # R₀ (dummy value)
        
        self.T = float('inf')
        t = 0
        
        episode_rewards = []
        
        # Loop until τ = T - 1
        while True:
            # Check if we need to take an action
            if t < self.T:
                # Take action At
                next_state, reward, terminated, truncated, _ = self.env.step(self.actions[t])
                episode_rewards.append(reward)
                
                # Store next reward and state
                self.rewards.append(reward)
                self.states.append(next_state)
                
                if terminated or truncated:
                    self.T = t + 1
                else:
                    # Select and store next action using ε-greedy
                    next_action = self.select_action(next_state)
                    self.actions.append(next_action)
            
            # τ is the time whose estimate is being updated
            tau = t - self.n + 1
            
            # Perform update if we have enough steps
            if tau >= 0:
                # Calculate truncated return
                h = min(self.T, tau + self.n)
                G = self.calculate_truncated_return(tau, h)
                
                # Update value function using formula (7.2)
                self.update_value_function(self.states[tau], G)
                
                # Update Q-value
                tau_action = self.actions[tau]
                self.Q[(self.states[tau], tau_action)] += (
                    self.alpha * (G - self.get_q_value(self.states[tau], tau_action))
                )
            
            # Increment time step
            t += 1
            
            # Termination condition: τ = T - 1
            if tau == self.T - 1:
                break
        
        return episode_rewards

