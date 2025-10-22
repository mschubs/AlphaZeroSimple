import torch
import math
import numpy as np


def add_dirichlet_noise(action_probs, valid_moves, alpha=0.3, epsilon=0.25):
    """
    Add Dirichlet noise to action probabilities for exploration during training.
    
    Args:
        action_probs: Neural network's action probabilities
        valid_moves: Binary array indicating valid moves
        alpha: Dirichlet concentration parameter (0.3 for Connect4, 0.03 for Go)
        epsilon: Mixing weight (0.25 is standard)
    
    Returns:
        Noisy action probabilities: (1-ε)*p + ε*noise
    """
    valid_actions = [i for i, valid in enumerate(valid_moves) if valid]
    
    if len(valid_actions) == 0:
        return action_probs
    
    # Generate Dirichlet noise only for valid actions
    noise = np.zeros_like(action_probs)
    dirichlet_noise = np.random.dirichlet([alpha] * len(valid_actions))
    
    # Assign noise to valid actions
    for i, action in enumerate(valid_actions):
        noise[action] = dirichlet_noise[i]
    
    # Mix original probabilities with noise
    noisy_probs = (1 - epsilon) * action_probs + epsilon * noise
    
    # Ensure probabilities are normalized and valid
    noisy_probs = noisy_probs * valid_moves  # Zero out invalid moves
    if np.sum(noisy_probs) > 0:
        noisy_probs /= np.sum(noisy_probs)
    else:
        # Fallback to uniform over valid moves
        noisy_probs = np.array(valid_moves, dtype=float)
        noisy_probs /= np.sum(noisy_probs)
    
    return noisy_probs


def ucb_score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score


class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, state, to_play, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        self.to_play = to_play
        self.state = state
        for a, prob in enumerate(action_probs):
            if prob != 0:
                self.children[a] = Node(prior=prob, to_play=self.to_play * -1)

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())


class MCTS:

    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args

    def run(self, model, state, to_play, add_noise=False):
        """
        Run MCTS simulations and return the root node.
        
        Args:
            model: Neural network model
            state: Current game state  
            to_play: Player to move (+1 or -1)
            add_noise: Whether to add Dirichlet noise for exploration (training only)
        """
        root = Node(0, to_play)

        # EXPAND root
        action_probs, value = model.predict(state)
        valid_moves = self.game.get_valid_moves(state)
        action_probs = action_probs * valid_moves  # mask invalid moves
        action_probs /= np.sum(action_probs)
        
        # Add Dirichlet noise for exploration during training
        if add_noise:
            alpha = self.args.get('dirichlet_alpha', 0.3)  # 0.3 for Connect4
            epsilon = self.args.get('dirichlet_epsilon', 0.25)  # Standard value
            action_probs = add_dirichlet_noise(action_probs, valid_moves, alpha, epsilon)
        
        root.expand(state, to_play, action_probs)

        for _ in range(self.args['num_simulations']):
            node = root
            search_path = [node]

            # SELECT
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            state = parent.state
            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            next_state, _ = self.game.get_next_state(state, player=1, action=action)
            # Get the board from the perspective of the other player
            next_state = self.game.get_canonical_board(next_state, player=-1)

            # The value of the new state from the perspective of the other player
            value = self.game.get_reward_for_player(next_state, player=1)
            if value is None:
                # If the game has not ended:
                # EXPAND
                action_probs, value = model.predict(next_state)
                valid_moves = self.game.get_valid_moves(next_state)
                action_probs = action_probs * valid_moves  # mask invalid moves
                action_probs /= np.sum(action_probs)
                node.expand(next_state, parent.to_play * -1, action_probs)

            self.backpropagate(search_path, value, parent.to_play * -1)

        return root

    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
