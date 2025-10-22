import os
import numpy as np
from random import shuffle
import random

import torch
import torch.optim as optim

from monte_carlo_tree_search import MCTS
from symmetry_utils import augment_training_data

class Trainer:

    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.device = model.device  # Get device from model
        self.mcts = MCTS(self.game, self.model, self.args)

    def get_random_move(self, board):
        """Get a random valid move"""
        valid_moves = self.game.get_valid_moves(board)
        valid_actions = [i for i, valid in enumerate(valid_moves) if valid]
        return random.choice(valid_actions) if valid_actions else None

    def get_weak_ai_move(self, board, player):
        """Get a move from a weak AI (fewer MCTS simulations)"""
        args = {'num_simulations': 10}  # Much fewer simulations
        mcts = MCTS(self.game, self.model, args)
        canonical_board = self.game.get_canonical_board(board, player)
        root = mcts.run(self.model, canonical_board, to_play=1)
        return root.select_action(temperature=0.5)  # Higher temperature for more randomness

    def get_variable_strength_move(self, board, player, strength=0.5):
        """
        Get a move with variable strength (generalizable across games)
        strength: 0.0 = random, 1.0 = full strength
        """
        valid_moves = self.game.get_valid_moves(board)
        valid_actions = [i for i, valid in enumerate(valid_moves) if valid]
        
        if not valid_actions:
            return None
        
        # Pure random play
        if strength == 0.0:
            return random.choice(valid_actions)
        
        # Pure optimal play
        if strength == 1.0:
            return self.get_weak_ai_move(board, player)
        
        # Mixed play: blend between random and AI
        if random.random() < strength:
            # Play like AI but with limited simulations based on strength
            sim_count = int(self.args['num_simulations'] * strength)
            sim_count = max(5, min(sim_count, self.args['num_simulations']))
            
            args = {'num_simulations': sim_count}
            mcts = MCTS(self.game, self.model, args)
            canonical_board = self.game.get_canonical_board(board, player)
            root = mcts.run(self.model, canonical_board, to_play=1)
            
            # Use higher temperature for weaker play
            temperature = 2.0 - strength  # strength 0.5 -> temp 1.5, strength 0.8 -> temp 1.2
            return root.select_action(temperature=temperature)
        else:
            # Play randomly
            return random.choice(valid_actions)

    def exceute_episode(self, iteration=1, opponent_strength=1.0):
        """
        Execute an episode with variable opponent strength (generalizable)
        opponent_strength: 1.0 = self-play, 0.0 = random, 0.1-0.9 = mixed strength
        """
        train_examples = []
        current_player = 1
        state = self.game.get_init_board()

        while True:
            canonical_board = self.game.get_canonical_board(state, current_player)

            # Decide who plays this turn
            if current_player == 1:
                # AI (player 1) always plays optimally and generates training data
                self.mcts = MCTS(self.game, self.model, self.args)
                root = self.mcts.run(self.model, canonical_board, to_play=1)

                action_probs = [0 for _ in range(self.game.get_action_size())]
                for k, v in root.children.items():
                    action_probs[k] = v.visit_count

                action_probs = action_probs / np.sum(action_probs)
                train_examples.append((canonical_board, current_player, action_probs))

                # Use temperature > 0 for exploration during training
                temperature = 1.0 if iteration <= self.args['numIters'] * 0.7 else 0.1
                action = root.select_action(temperature=temperature)
            else:
                # Opponent (player -1) plays with variable strength
                if opponent_strength == 1.0 or isinstance(self.game, Connect4Game):
                    # Full self-play: opponent also plays optimally
                    self.mcts = MCTS(self.game, self.model, self.args)
                    root = self.mcts.run(self.model, canonical_board, to_play=1)
                    
                    # Also generate training data for opponent in self-play
                    action_probs = [0 for _ in range(self.game.get_action_size())]
                    for k, v in root.children.items():
                        action_probs[k] = v.visit_count
                    action_probs = action_probs / np.sum(action_probs)
                    train_examples.append((canonical_board, current_player, action_probs))
                    
                    temperature = 1.0 if iteration <= self.args['numIters'] * 0.7 else 0.1
                    action = root.select_action(temperature=temperature)
                else:
                    # Variable strength opponent
                    action = self.get_variable_strength_move(state, current_player, opponent_strength)
                
                if action is None:  # No valid moves
                    break

            state, current_player = self.game.get_next_state(state, current_player, action)
            reward = self.game.get_reward_for_player(state, current_player)

            if reward is not None:
                ret = []
                for hist_state, hist_current_player, hist_action_probs in train_examples:
                    # [Board, currentPlayer, actionProbabilities, Reward]
                    ret.append((hist_state, hist_action_probs, reward * ((-1) ** (hist_current_player != current_player))))

                return ret

    def learn(self):
        for i in range(1, self.args['numIters'] + 1):

            print("{}/{}".format(i, self.args['numIters']))

            train_examples = []

            # Curriculum learning: mix of opponent strengths
            for eps in range(self.args['numEps']):
                # Dynamic curriculum: start with more weak opponents, gradually increase self-play
                curriculum_progress = i / self.args['numIters']  # 0.0 to 1.0
                
                rand_val = random.random()
                
                # Early training: more variety (lots of weak opponents)
                # Late training: more self-play for refinement
                if curriculum_progress < 0.3:
                    # Early: 30% self-play, 70% varied strength
                    if rand_val < 0.3:
                        opponent_strength = 1.0  # Self-play
                    else:
                        opponent_strength = random.uniform(0.0, 0.8)  # Weak to medium
                elif curriculum_progress < 0.7:
                    # Mid: 50% self-play, 50% varied strength
                    if rand_val < 0.5:
                        opponent_strength = 1.0  # Self-play
                    else:
                        opponent_strength = random.uniform(0.2, 0.9)  # Medium strength
                else:
                    # Late: 70% self-play, 30% strong opponents
                    if rand_val < 0.7:
                        opponent_strength = 1.0  # Self-play
                    else:
                        opponent_strength = random.uniform(0.6, 1.0)  # Strong opponents
                
                iteration_train_examples = self.exceute_episode(iteration=i, opponent_strength=opponent_strength)
                if iteration_train_examples:  # Only add if episode completed successfully
                    train_examples.extend(iteration_train_examples)

            # Augment with symmetries
            train_examples = augment_training_data(train_examples, self.game)
            shuffle(train_examples)
            self.train(train_examples)
            filename = self.args['checkpoint_path']
            self.save_checkpoint(folder=".", filename=filename)

    def train(self, examples):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)  # Reduced from 5e-4
        pi_losses = []
        v_losses = []

        for epoch in range(self.args['epochs']):
            self.model.train()

            batch_idx = 0

            while batch_idx < int(len(examples) / self.args['batch_size']):
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                # boards = boards.contiguous().cuda()
                # target_pis = target_pis.contiguous().cuda()
                # target_vs = target_vs.contiguous().cuda()

                boards = boards.contiguous().to(self.device)
                target_pis = target_pis.contiguous().to(self.device)
                target_vs = target_vs.contiguous().to(self.device)

                # compute output
                out_pi, out_v = self.model(boards)
                
                # Check for NaN in model outputs
                if torch.isnan(out_pi).any() or torch.isnan(out_v).any():
                    print("WARNING: Model output contains NaN!")
                    print(f"out_pi has NaN: {torch.isnan(out_pi).any()}")
                    print(f"out_v has NaN: {torch.isnan(out_v).any()}")
                    print(f"Input boards: {boards}")
                    continue  # Skip this batch
                
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                
                # Check for NaN in losses
                if torch.isnan(total_loss):
                    print("WARNING: Loss is NaN, skipping batch")
                    continue

                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()

                batch_idx += 1

            print()
            print("Policy Loss", np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))
            print("Examples:")
            print(out_pi[0].detach())
            print(target_pis[0])

    def loss_pi(self, targets, outputs):
        # Add small epsilon to prevent log(0) = -inf
        eps = 1e-8
        outputs = torch.clamp(outputs, eps, 1.0)
        loss = -(targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets, outputs):
        loss = torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
        return loss

    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save({
            'state_dict': self.model.state_dict(),
        }, filepath)
