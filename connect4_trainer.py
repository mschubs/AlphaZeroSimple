import os
import numpy as np
from random import shuffle
import random
import time

import torch
import torch.optim as optim

from monte_carlo_tree_search import MCTS
from symmetry_utils import augment_training_data

from connect4_model import Connect4Model

class Connect4Trainer:

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
        episode_start = time.time()
        train_examples = []
        current_player = 1
        state = self.game.get_init_board()
        move_count = 0

        while True:
            move_start = time.time()
            canonical_board = self.game.get_canonical_board(state, current_player)

            # Run MCTS to get move probabilities (with noise for exploration)
            self.mcts = MCTS(self.game, self.model, self.args)
            root = self.mcts.run(self.model, canonical_board, to_play=1, add_noise=True)

            action_probs = [0 for _ in range(self.game.get_action_size())]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count

            if sum(action_probs) == 0:
                break  # No valid moves

            action_probs = action_probs / np.sum(action_probs)
            train_examples.append((canonical_board, current_player, action_probs))

            # Use temperature > 0 for exploration during training
            temperature = 1.0 if iteration <= self.args['numIters'] * 0.7 else 0.1
            action = root.select_action(temperature=temperature)
        
            if action is None:  # No valid moves
                break

            state, current_player = self.game.get_next_state(state, current_player, action)
            reward = self.game.get_reward_for_player(state, current_player)
            
            move_time = time.time() - move_start
            move_count += 1

            if reward is not None:
                # Game ended - assign rewards to all moves
                ret = []
                for hist_state, hist_current_player, hist_action_probs in train_examples:
                    # [Board, currentPlayer, actionProbabilities, Reward]
                    ret.append((hist_state, hist_action_probs, reward * ((-1) ** (hist_current_player != current_player))))

                episode_time = time.time() - episode_start
                winner = "Player 1" if reward == 1 else "Player -1" if reward == -1 else "Draw"
                # Only print detailed timing for first few episodes to avoid spam
                if move_count <= 5:
                    print(f"     Self-play game: {move_count} moves in {episode_time:.1f}s ({winner})")
                
                return ret

    def learn(self):
        model_version = 0
        for i in range(1, self.args['numIters'] + 1):
            iteration_start = time.time()  # Time each iteration
            
            print("{}/{}".format(i, self.args['numIters']))

            train_examples = []

            # Generate training examples from self-play episodes
            episodes_start = time.time()
            for eps in range(self.args['numEps']):
                episode_start = time.time()
                
                iteration_train_examples = self.exceute_episode(iteration=i, opponent_strength=1)
                if iteration_train_examples:  # Only add if episode completed successfully
                    train_examples.extend(iteration_train_examples)
                
                episode_time = time.time() - episode_start
                print(f"   Episode {eps+1}/{self.args['numEps']}: {episode_time:.1f}s ({len(iteration_train_examples) if iteration_train_examples else 0} examples)")

            episodes_time = time.time() - episodes_start
            print(f"📚 Generated {len(train_examples)} training examples in {episodes_time:.1f}s")

            # Augment with symmetries and train
            augment_start = time.time()
            train_examples = augment_training_data(train_examples, self.game)
            shuffle(train_examples)
            augment_time = time.time() - augment_start
            print(f"🔄 Augmented to {len(train_examples)} examples in {augment_time:.1f}s")
            
            self.train(train_examples)

            # fade candidate evaluation for time purposes
            self.save_checkpoint(folder=".", filename=self.args['checkpoint_path'])

            # # Evaluate candidate vs latest model
            # eval_start = time.time()
            # if self.candidate_outperforms_latest(model_version):
            #     self.save_checkpoint(folder=".", filename=self.args['checkpoint_path'])
            #     model_version += 1
            #     print(f"✅ Model v{model_version} promoted!")
            # else:
            #     checkpoint = torch.load(self.args['checkpoint_path'], map_location=self.device)
            #     self.model.load_state_dict(checkpoint['state_dict'])
            #     print(f"❌ Model rejected, reverted to v{model_version}")
            
            # eval_time = time.time() - eval_start
            # iteration_time = time.time() - iteration_start
            # print(f"⏱️  Iteration {i}: {iteration_time:.1f}s total (eval: {eval_time:.1f}s)\n")

    def train(self, examples):
        training_start = time.time()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)  # Reduced from 5e-4
        pi_losses = []
        v_losses = []
        
        num_batches = int(len(examples) / self.args['batch_size'])
        print(f"🎯 Training on {len(examples)} examples for {self.args['epochs']} epochs ({num_batches} batches each)")

        for epoch in range(self.args['epochs']):
            epoch_start = time.time()
            self.model.train()
            
            epoch_pi_losses = []
            epoch_v_losses = []
            batch_idx = 0

            while batch_idx < num_batches:
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                boards = boards.contiguous().to(self.device)
                target_pis = target_pis.contiguous().to(self.device)
                target_vs = target_vs.contiguous().to(self.device)

                # Compute output and losses
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
                epoch_pi_losses.append(float(l_pi))
                epoch_v_losses.append(float(l_v))

                # Gradient update
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_idx += 1

            epoch_time = time.time() - epoch_start
            avg_pi_loss = np.mean(epoch_pi_losses) if epoch_pi_losses else 0
            avg_v_loss = np.mean(epoch_v_losses) if epoch_v_losses else 0
            print(f"   Epoch {epoch+1}/{self.args['epochs']}: {epoch_time:.1f}s | Policy: {avg_pi_loss:.4f} | Value: {avg_v_loss:.4f}")

        training_time = time.time() - training_start
        final_pi_loss = np.mean(pi_losses) if pi_losses else 0
        final_v_loss = np.mean(v_losses) if v_losses else 0
        print(f"🧠 Training complete: {training_time:.1f}s | Final Policy: {final_pi_loss:.4f} | Final Value: {final_v_loss:.4f}")

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

    def candidate_outperforms_latest(self, model_version):
        """Check if candidate model outperforms latest checkpoint"""
        eval_start = time.time()

        if not os.path.exists(self.args['checkpoint_path']):
            print(f"🆕 No checkpoint exists - candidate becomes model v{model_version + 1}")
            return True

        # Load previous best model
        load_start = time.time()
        latest_checkpoint = torch.load(self.args['checkpoint_path'], map_location=self.device)
        latest_model = Connect4Model(self.game.get_board_size(), self.game.get_action_size(), self.device)
        latest_model.load_state_dict(latest_checkpoint['state_dict'])
        load_time = time.time() - load_start

        # Set models to evaluation mode
        self.model.eval()
        latest_model.eval()

        # Run evaluation games
        num_evaluations = self.args.get('num_evaluations', 20)
        print(f"🏆 Evaluating candidate vs model v{model_version} ({num_evaluations} games)")
        
        games_start = time.time()
        num_candidate_wins = 0
        for i in range(num_evaluations):
            game_start = time.time()
            result = self.candidate_wins(latest_model, self.model)
            num_candidate_wins += result
            game_time = time.time() - game_start
            
            # Progress update every 5 games
            if (i + 1) % 5 == 0 or i == 0:
                win_rate = num_candidate_wins / (i + 1)
                print(f"   Game {i+1}/{num_evaluations}: {game_time:.1f}s | Win rate: {win_rate:.2f}")

        games_time = time.time() - games_start
        
        # Restore training mode
        self.model.train()
        latest_model.train()

        # Calculate results
        win_rate = num_candidate_wins / num_evaluations
        threshold = 0.55
        eval_time = time.time() - eval_start
        
        print(f"📊 Evaluation complete: {eval_time:.1f}s (load: {load_time:.1f}s, games: {games_time:.1f}s)")
        print(f"   Result: {num_candidate_wins}/{num_evaluations} wins ({win_rate:.3f}) vs threshold {threshold}")

        return win_rate > threshold


    def candidate_wins(self, latest_model, candidate_model):
        """Play single evaluation game between candidate and latest model"""
        
        # Initialize game state
        state = self.game.get_init_board()
        current_player = 1
        
        # Randomly assign which model plays as which player (fairness)
        candidate_player = 1 if random.random() < 0.5 else -1
        model1 = candidate_model if candidate_player == 1 else latest_model
        model2 = latest_model if candidate_player == 1 else candidate_model
        
        move_count = 0
        max_moves = 42  # Prevent infinite games
        
        while move_count < max_moves:
            move_start = time.time()
            
            # Get move from current player's model
            if current_player == 1:
                state, current_player = self.get_move(model1, state, current_player)
            else:
                state, current_player = self.get_move(model2, state, current_player)
            
            move_time = time.time() - move_start
            move_count += 1
            
            # Check for game end
            reward = self.game.get_reward_for_player(state, current_player)
            if reward is not None:
                winner = "candidate" if reward == candidate_player else "latest"
                return reward == candidate_player
            
            # Check for draw
            if not self.game.has_legal_moves(state):
                return False  # Draw counts as loss for candidate
        
        # Max moves reached - treat as draw
        return False

    def get_move(self, model, state, current_player):
        """Get AI move using MCTS"""
        mcts_start = time.time()
        
        # Get canonical board and run MCTS (no noise during evaluation)
        canonical_board = self.game.get_canonical_board(state, current_player)
        mcts = MCTS(self.game, model, self.args)
        root = mcts.run(model, canonical_board, to_play=1, add_noise=False)
        
        # Select best move using low temperature for strong play
        move = root.select_action(temperature=0.1)
        
        if move is not None:
            state, current_player = self.game.get_next_state(state, current_player, move)
        
        mcts_time = time.time() - mcts_start
        
        return state, current_player
