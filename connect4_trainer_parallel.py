import os
import numpy as np
from random import shuffle
import random
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import torch.optim as optim
from functools import partial

from monte_carlo_tree_search import MCTS
from symmetry_utils import augment_training_data
from connect4_model import Connect4Model
from connect4_game import Connect4Game


def run_episode_worker(args_tuple):
    """Worker function to run a single episode in parallel"""
    iteration, opponent_strength, game_class, model_state_dict, mcts_args, device_str = args_tuple
    
    # Create game and model in worker process
    game = game_class()
    device = torch.device(device_str)
    model = Connect4Model(game.get_board_size(), game.get_action_size(), device)
    model.load_state_dict(model_state_dict)
    model.eval()  # Set to evaluation mode for inference
    
    # Run the episode
    episode_start = time.time()
    train_examples = []
    current_player = 1
    state = game.get_init_board()
    move_count = 0

    while True:
        canonical_board = game.get_canonical_board(state, current_player)
        
        # Create MCTS instance (with noise for exploration during training)
        mcts = MCTS(game, model, mcts_args)
        root = mcts.run(model, canonical_board, to_play=1, add_noise=True)

        action_probs = [0 for _ in range(game.get_action_size())]
        for k, v in root.children.items():
            action_probs[k] = v.visit_count

        if sum(action_probs) == 0:
            break

        action_probs = action_probs / np.sum(action_probs)
        train_examples.append((canonical_board, current_player, action_probs))

        # Use temperature for exploration
        temperature = 1.0 if iteration <= mcts_args.get('numIters', 50) * 0.7 else 0.1
        action = root.select_action(temperature=temperature)
        
        if action is None:
            break

        state, current_player = game.get_next_state(state, current_player, action)
        reward = game.get_reward_for_player(state, current_player)
        move_count += 1

        if reward is not None:
            # Game ended - assign rewards
            ret = []
            for hist_state, hist_current_player, hist_action_probs in train_examples:
                ret.append((hist_state, hist_action_probs, reward * ((-1) ** (hist_current_player != current_player))))

            episode_time = time.time() - episode_start
            winner = "Player 1" if reward == 1 else "Player -1" if reward == -1 else "Draw"
            
            return {
                'examples': ret,
                'episode_time': episode_time,
                'move_count': move_count,
                'winner': winner,
                'process_id': os.getpid()
            }
    
    return {'examples': [], 'episode_time': time.time() - episode_start, 'move_count': 0, 'winner': 'Incomplete'}


class Connect4TrainerParallel:
    """Parallel version of Connect4Trainer using multiprocessing"""
    
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.device = model.device
        self.mcts = MCTS(self.game, self.model, self.args)
        
        # Parallel processing settings
        self.num_workers = args.get('num_workers', mp.cpu_count())
        print(f"🔄 Using {self.num_workers} parallel workers")

    def learn(self):
        model_version = 0
        for i in range(1, self.args['numIters'] + 1):
            iteration_start = time.time()
            
            print(f"{i}/{self.args['numIters']}")

            # Generate training examples in parallel
            train_examples = self.generate_parallel_episodes(i)
            
            if not train_examples:
                print("⚠️ No training examples generated, skipping iteration")
                continue

            # Augment and train
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
            # print(f"⏱️ Iteration {i}: {iteration_time:.1f}s total (eval: {eval_time:.1f}s)\n")

    def generate_parallel_episodes(self, iteration):
        """Generate training examples using parallel episodes"""
        episodes_start = time.time()
        
        # Prepare arguments for worker processes
        model_state_dict = self.model.state_dict()
        device_str = str(self.device)
        
        # Create argument tuples for each episode
        worker_args = [
            (iteration, 1.0, Connect4Game, model_state_dict, self.args, device_str)
            for _ in range(self.args['numEps'])
        ]
        
        train_examples = []
        completed_episodes = 0
        total_episode_time = 0
        
        print(f"🚀 Starting {self.args['numEps']} parallel episodes...")
        
        # Run episodes in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_episode = {
                executor.submit(run_episode_worker, args): i 
                for i, args in enumerate(worker_args)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_episode):
                episode_num = future_to_episode[future]
                
                try:
                    result = future.result()
                    
                    if result['examples']:
                        train_examples.extend(result['examples'])
                        completed_episodes += 1
                        total_episode_time += result['episode_time']
                        
                        # Progress update
                        if completed_episodes % max(1, self.args['numEps'] // 10) == 0:
                            avg_time = total_episode_time / completed_episodes
                            print(f"   📊 {completed_episodes}/{self.args['numEps']} episodes complete (avg: {avg_time:.1f}s)")
                
                except Exception as e:
                    print(f"❌ Episode {episode_num} failed: {e}")
        
        episodes_time = time.time() - episodes_start
        avg_episode_time = total_episode_time / max(1, completed_episodes)
        speedup = total_episode_time / episodes_time if episodes_time > 0 else 1
        
        print(f"📚 Generated {len(train_examples)} examples from {completed_episodes} episodes")
        print(f"⚡ Parallel speedup: {speedup:.1f}x (total: {episodes_time:.1f}s vs sequential: {total_episode_time:.1f}s)")
        
        return train_examples

    def train(self, examples):
        """Same training logic as original"""
        training_start = time.time()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
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

                # Forward pass
                out_pi, out_v = self.model(boards)
                
                # Skip NaN batches
                if torch.isnan(out_pi).any() or torch.isnan(out_v).any():
                    print("WARNING: Model output contains NaN, skipping batch")
                    continue
                
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                
                if torch.isnan(total_loss):
                    print("WARNING: Loss is NaN, skipping batch")
                    continue

                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))
                epoch_pi_losses.append(float(l_pi))
                epoch_v_losses.append(float(l_v))

                # Backward pass
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

    def candidate_outperforms_latest(self, model_version):
        """Same evaluation logic as original"""
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
            
            if (i + 1) % 5 == 0 or i == 0:
                win_rate = num_candidate_wins / (i + 1)
                print(f"   Game {i+1}/{num_evaluations}: {game_time:.1f}s | Win rate: {win_rate:.2f}")

        games_time = time.time() - games_start
        
        # Restore training mode
        self.model.train()
        latest_model.train()

        win_rate = num_candidate_wins / num_evaluations
        threshold = 0.55
        eval_time = time.time() - eval_start
        
        print(f"📊 Evaluation complete: {eval_time:.1f}s (load: {load_time:.1f}s, games: {games_time:.1f}s)")
        print(f"   Result: {num_candidate_wins}/{num_evaluations} wins ({win_rate:.3f}) vs threshold {threshold}")

        return win_rate > threshold

    def candidate_wins(self, latest_model, candidate_model):
        """Same game logic as original"""
        state = self.game.get_init_board()
        current_player = 1
        
        candidate_player = 1 if random.random() < 0.5 else -1
        model1 = candidate_model if candidate_player == 1 else latest_model
        model2 = latest_model if candidate_player == 1 else candidate_model
        
        move_count = 0
        max_moves = 42
        
        while move_count < max_moves:
            if current_player == 1:
                state, current_player = self.get_move(model1, state, current_player)
            else:
                state, current_player = self.get_move(model2, state, current_player)
            
            move_count += 1
            
            reward = self.game.get_reward_for_player(state, current_player)
            if reward is not None:
                return reward == candidate_player
            
            if not self.game.has_legal_moves(state):
                return False
        
        return False

    def get_move(self, model, state, current_player):
        """Same move logic as original"""
        canonical_board = self.game.get_canonical_board(state, current_player)
        mcts = MCTS(self.game, model, self.args)
        root = mcts.run(model, canonical_board, to_play=1, add_noise=False)
        
        move = root.select_action(temperature=0.1)
        
        if move is not None:
            state, current_player = self.game.get_next_state(state, current_player, move)
        
        return state, current_player

    def loss_pi(self, targets, outputs):
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