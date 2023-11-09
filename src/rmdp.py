from typing import List, Set, Tuple
import numpy as np
import itertools
import gurobipy as gp
from gurobipy import GRB
import scipy
import sklearn.linear_model

class MDP(object):
    """MDP class for use in the following methods which solve the MDP"""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        num_features: int,
        P,
        phi,
        p_0,
        gamma,
        reward=None,
    ):
        # super(MDP, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.states = np.arange(self.num_states)
        self.actions = np.arange(self.num_actions)
        self.num_features = num_features
        self.gamma = gamma  # discount factor
        # transition probability P \in S x S x A
        self.P = P
        # initial state distribution p_0 \in S
        self.p_0 = p_0
        # reward(S x A) vector itrating by state first
        self.reward = reward
        self.reward_matrix = np.reshape(
            self.reward, (self.num_states, self.num_actions), order="F"
        )
        # features matrix phi \in (SxA)xK where K is the number of features
        self.phi = phi
        self.phi_matrix = phi.reshape(self.num_states, self.num_actions, self.num_features, order='F')

        # Stacked (I - gamma P_a)
        self.IGammaPAStacked = self.construct_design_matrix()
        # QRM matrix
        self.QRM = np.zeros((self.num_states, self.num_actions))
        # c for QRM_step
        self.c = 1
        # occupancy frequency of an expert's policy u[S x A]
        (u_E, opt_return) = self.solve_putterman_dual_LP_for_Opt_policy()
        self.u_E = u_E  # occupancy frequency of the expert's policy
        self.opt_policy = self.occupancy_freq_to_policy(u_E)
        self.opt_return = opt_return  # optimal return of the expert's policy
        (u_rand, rand_return) = self.generate_random_policy_return()
        self.random_return = rand_return
        self.u_rand = u_rand
        # feature expectation of an expert's policy mu_[K]
        self.mu_E = None
        self.weights = np.zeros(num_features)

        self.current_state = 0

        # self.worst_return = self.solve_worst()[1]

    # Methods to make this class work like open-ai gym 
    def reset(self) -> int:
        """
        Reset the MDP to an initial state using p_0
        Returns:
            int: The initial state
        """
        self.current_state = np.random.choice(self.states, p=self.p_0)
        return self.current_state

    def step(self, action) -> Tuple[int, float]:
        """
        Take a step in the MDP
        Args:
            action: The action to take
        Returns:
            int: The next state s'
            float: The reward r(s,a)
        """
        self.current_state = self.next_state(self.current_state, action)
        return self.current_state, self.reward_matrix[self.current_state, action]
    
    def argmax_next_state(self, state: int, action: int) -> int:
        """Given state and action pair, return the most likely next state based on the MDP dynamics"""
        return int(np.argmax(self.P[state, :, action]))

    def next_state(self, state: int, action: int) -> int:
        """Given state and action pair, return the next state based on the MDP dynamics"""
        return np.random.choice(self.states, p=self.P[state, :, action])

    def occupancy_freq_to_policy(self, u) -> np.ndarray:
        """Converts u which is an occupancy frequency matrix of size S x A to a policy of size S x A"""
        S = self.num_states
        A = self.num_actions
        policy = np.zeros((S, A))
        sum_u_s = np.sum(u, axis=1)
        for s in range(S):
            policy[s, :] = u[s, :] / max(sum_u_s[s], 0.0000001)
        return policy

    def generate_samples_from_policy(
        self, num_samples, policy
    ) -> List[Tuple[int, int]]:
        """Generate samples from the given policy
        policy = policy should be an SxA matrix where each row sums to 1
        """
        D = []  # Dataset of (s, a) pairs
        cur_state = np.random.choice(self.states, p=self.p_0)
        for _ in range(num_samples):
            action = np.random.choice(self.actions, p=policy[cur_state, :])
            D.append((cur_state, action))
            cur_state = self.next_state(cur_state, action)
        return D

    def generate_random_policy_return(self) -> Tuple[np.ndarray, float]:
        """Generate the return of a uniformly random policy where pi(a|s) = 1/|A|"""
        # Calculate P_pi for randomized pi
        P_pi = np.sum(self.P, axis=2) / self.num_actions
        r_pi = np.sum(self.reward_matrix, axis=1) / self.num_actions
        d_pi = np.linalg.inv(np.eye(self.num_states) - self.gamma * P_pi.T) @ self.p_0
        u_rand = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                u_rand[s, a] = d_pi[s] * 1/self.num_actions
        return u_rand, d_pi @ r_pi

    def generate_samples_from_occ_freq(
        self, num_samples, occupancy_freq
    ) -> List[Tuple[int, int]]:
        """
        Generate samples from the given occupancy frequency
        occupancy_freq = matrix of size S x A
        num_samples = number of samples to collect
        """
        return self.generate_samples_from_policy(
            num_samples, self.occupancy_freq_to_policy(occupancy_freq)
        )

    def generate_expert_demonstrations(self, num_samples) -> List[Tuple[int, int]]:
        """A wrapper around generate_samples which calls it with the optimal occupancy_freq calculated by
        the dual putterman solution"""
        return self.generate_samples_from_occ_freq(num_samples, self.u_E)

    def generate_samples_from_action_policy(self, horizon, action_policy, behavior_policy) -> List[Tuple[int, int]]:
        """Generate samples from the given action policy, where transition dynamics are governed by the behavior policy"""
        D: List[Tuple[int, int]] = []
        cur_state = np.random.choice(self.states, p=self.p_0)
        for _ in range(horizon):
            D.append((cur_state, np.random.choice(self.actions, p=action_policy[cur_state, :])))
            cur_state = self.next_state(cur_state, np.random.choice(self.actions, p=behavior_policy[cur_state, :]))
        return D

    def generate_off_policy_demonstrations(self, episodes, horizon, action_occ_freq, behavior_occ_freq) -> List[List[Tuple[int, int]]]:
        behavior_policy = self.occupancy_freq_to_policy(behavior_occ_freq)
        action_policy = self.occupancy_freq_to_policy(action_occ_freq)
        D: List[List[Tuple[int, int]]] = []
        for _ in range(episodes):
            D.append(self.generate_samples_from_action_policy(horizon, action_policy, behavior_policy))
        return D

    def generate_all_expert_demonstrations(self) -> List[Tuple[int, int]]:
        """Returns a list of all s,a pairs that the expert follows"""
        policy = self.occupancy_freq_to_policy(self.u_E)
        D = []
        for s in self.states:
            a = np.random.choice(self.actions, p=policy[s, :])
            D.append((s, a))
        return D

    class SampleCollector:
        """This class is soley for the pickling required by the multiprocessing pool
        it remembers the occ_freq for the following call to generate_samples"""

        def __init__(self, mdp, policy, horizon):
            self.policy = policy
            self.mdp = mdp
            self.horizon = horizon

        def __call__(self, _) -> List[Tuple[int, int]]:
            return self.mdp.generate_samples_from_policy(
                self.horizon, policy=self.policy
            )

    def generate_demonstrations_from_occ_freq(
        self, occ_freq, episodes=1, horizon=10, num_samples=None
    ) -> List[List[Tuple[int, int]]]:
        """Generate demonstrations from an occ freq
        Args:
            occ_freq: The occupancy frequency to use
            episodes: The number of episodes to generate
            horizon: The horizon of each episode
            num_samples: The number of samples to generate. If None, then episodes and horizon are used
        Returns:
            A list of episodes, where each episode is a list of (s,a) pairs
        """
        if num_samples:
            return [self.generate_samples_from_occ_freq(num_samples, occ_freq)]
        D: List[List[Tuple[int, int]]] = []
        # This code is left here just in case I want to test parallelization again
        # gen_samples_closure = self.SampleCollector(
        #     self, self.occupancy_freq_to_policy(occ_freq), horizon
        # )
        # with Pool(1) as pool:
        #     D = list(pool.map(gen_samples_closure, range(episodes)))
        policy = self.occupancy_freq_to_policy(occ_freq)
        for _ in range(0, episodes):
            D.append(self.generate_samples_from_policy(horizon, policy))
        return D

    def construct_design_matrix(self) -> np.ndarray:
        """
        Construct the design matrix consisting of (I - gamma P_a) stacked on top of eachother
        Returns an (SA X S) matrix
        """
        arrays = []
        I = np.eye(self.num_states)
        for action in self.actions:
            arrays.append((I - self.gamma * self.P[:,:,action]))
        return np.vstack(arrays)

    def solve_putterman_dual_LP_for_Opt_policy(self) -> Tuple[np.ndarray, float]:
        """This method solves the problem of Bellman Flow Constraint. This
        problem is sometimes called the dual problem of min p0^T v, which finds
        the optimal value function.

        Returns:
            ndarray: The optimal policy
            float: optimal return
        """
        method = "Dual_LP"
        a = self.num_actions
        s = self.num_states
        p_0 = self.p_0
        gamma = self.gamma
        r = self.reward
        W = self.IGammaPAStacked

        # Model
        model = gp.Model(method)
        model.Params.OutputFlag = 0
        # Variables
        u = model.addMVar(shape=(s * a), lb=0.0)
        # Constraints
        model.addMConstr(W.T, u, "==", p_0)
        # setting the objective
        model.setObjective(r @ u, GRB.MAXIMIZE)
        # Solve
        model.optimize()
        if model.Status != GRB.Status.OPTIMAL:
            raise ValueError("DUAL LP DID NOT FIND OPTIMAL SOLUTION")
        dual_return = model.objVal

        u_flat = u.X  # Gurobi way of getting the value of a model variable
        # Check to make sure u is an occupancy frequency
        # assert np.sum(u_flat) - 1 / (1 - gamma) < 10**-2
        # Return the occ_freq and the opt return
        return u_flat.reshape((s, a), order="F"), dual_return

    def QRM_step(self, beta: float, D: List[Tuple]):
        """Performs a single step of the QRM algorithm
        Args:
            D: A list of (s,a) pairs
        Returns:
            A tuple of the new occupancy frequency and the new return
        """
        learning_rate = 1
        for t, (s, a) in enumerate(D):
            r = self.reward_matrix[s, a]
            learning_rate = self.c / (self.c + t)
            QRM_beta_gamma = 8655309 # what is this?
            self.QRM[s,a] = (1-learning_rate) * self.QRM[s,a] + learning_rate * (np.exp(-beta * r) * QRM_beta_gamma)


    def observed(self, state, D: Set[Tuple]) -> Tuple[bool, int]:
        for s, a in D:
            if state == s:
                return (True, a)
        return (False, -1)

    def construct_constraint_vector(self, D: Set[Tuple]) -> np.ndarray:
        """Constructs the constraint vector for u in Upsilon
        Returns a vector of length num_states * num_actions"""
        c = np.zeros((self.num_states, self.num_actions))
        for state in self.states:
            (observed_state, observed_action) = self.observed(state, D)
            for action in self.actions:
                # In order to be in Upsilon, you must observe the state with that action
                # Consistent with the expert
                if observed_state and action != observed_action:
                    # a constraint of 1 means that you shouldnt choose that (s,a) pair
                    c[state, action] = 1
        return c.reshape((self.num_states * self.num_actions), order="F")

    def occ_freq_from_P_pi(self, P_pi: np.ndarray, pi) -> np.ndarray:
        """Compute the matrix U
        given a matrix P_pi of size SxS
        and a function pi that maps states to a simplex over actions
        returns a matrix of size SxA
        """
        u = np.zeros((self.num_states, self.num_actions))
        dtheta = np.linalg.inv(np.eye(self.num_states) - self.gamma * P_pi.T) @ self.p_0
        for s in self.states:
            u[s,:] = pi(s) * dtheta[s]
        return u

    def P_pi(self, pi) -> np.ndarray:
        """Compute the matrix P_pi
        given a function pi that maps states to a simplex over actions
        returns a matrix of size SxS"""
        Ptheta = np.zeros((self.num_states, self.num_states))
        for s in self.states:
            for s_prime in self.states:
                Ptheta[s,s_prime] = pi(s) @ self.P[s,s_prime,:]
        return Ptheta

    def solve_worst(self) -> Tuple[np.ndarray, float]:
        """Solve the worst-case bellman flow problem
        returns a tuple of the worst-case occupancy frequency and the 
        corresponding worst-case return"""
        method = "Worst"
        a = self.num_actions
        s = self.num_states
        p_0 = self.p_0
        r = self.reward
        W = self.IGammaPAStacked

        # Model
        model = gp.Model(method)
        model.Params.OutputFlag = 0
        # Variables
        u = model.addMVar(shape=(s * a), lb=0.0)
        # Constraints
        model.addMConstr(W.T, u, "==", p_0)
        # setting the objective
        model.setObjective(r @ u, GRB.MINIMIZE)
        # Solve
        model.optimize()
        if model.Status != GRB.Status.OPTIMAL:
            raise ValueError("MIN DUAL LP DID NOT FIND OPTIMAL SOLUTION")
        dual_return = model.objVal

        u_flat = u.X  # Gurobi way of getting the value of a model variable
        # Check to make sure u is an occupancy frequency
        # assert np.sum(u_flat) - 1 / (1 - gamma) < 10**-2
        # Return the occ_freq and the opt return
        return u_flat.reshape((s, a), order="F"), dual_return
