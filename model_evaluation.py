#!/usr/bin/env python3

import torch
import numpy as np
import random
import os
from datetime import datetime
import json
from connect4_game import Connect4Game
from connect4_model import Connect4Model
from monte_carlo_tree_search import MCTS

class ModelEvaluator:
    """
    AlphaZero-style model evaluation system for Connect 4
    
    After each training iteration:
    1. Candidate network plays against current best network
    2. If candidate wins >55% of games, it becomes the new best
    3. Otherwise, candidate is discarded and old network remains
    """
    
    def __init__(self, game, args, evaluation_games=400, win_threshold=0.55):
        self.game = game
        self.args = args
        self.evaluation_games = evaluation_games
        self.win_threshold = win_threshold
        self.device = torch.device('cpu')
        
        # File paths
        self.current_best_path = 'latest.pth'  # Current champion
        self.candidate_path = 'candidate.pth'  # Newly trained model
        self.evaluation_log = 'evaluation_history.json'
        
        # Load or initialize evaluation history
        self.history = self.load_evaluation_history()
        
        print(f"🏆 MODEL EVALUATOR INITIALIZED")
        print(f"   Evaluation games: {self.evaluation_games}")
        print(f"   Win threshold: {self.win_threshold:.1%}")
        print(f"   Champion model: {self.current_best_path}")
        print(f"   Candidate model: {self.candidate_path}")
    
    def load_evaluation_history(self):
        """Load previous evaluation results"""
        if os.path.exists(self.evaluation_log):
            try:
                with open(self.evaluation_log, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_evaluation_history(self):
        """Save evaluation results to file"""
        with open(self.evaluation_log, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_model(self, path):
        """Load a model from checkpoint"""
        model = Connect4Model(self.game.get_board_size(), self.game.get_action_size(), self.device)
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            return model
        else:
            print(f"⚠️  Model not found: {path}")
            return None
    
    def play_game(self, model1, model2, model1_starts=True):
        """
        Play a single game between two models
        
        Returns:
        - 1 if model1 wins
        - -1 if model2 wins  
        - 0 if draw
        """
        state = self.game.get_init_board()
        current_player = 1
        
        # Determine which model plays which side
        if model1_starts:
            player1_model = model1  # model1 plays as player 1
            player2_model = model2  # model2 plays as player -1
        else:
            player1_model = model2  # model2 plays as player 1
            player2_model = model1  # model1 plays as player -1
        
        move_count = 0
        max_moves = 42  # Maximum possible moves in Connect 4
        
        while move_count < max_moves:
            # Select the appropriate model
            if current_player == 1:
                active_model = player1_model
            else:
                active_model = player2_model
            
            # Get move from MCTS
            mcts = MCTS(self.game, active_model, self.args)
            canonical_board = self.game.get_canonical_board(state, current_player)
            root = mcts.run(active_model, canonical_board, to_play=1)
            
            # Use low temperature for strong play
            action = root.select_action(temperature=0.1)
            
            # Make the move
            state, current_player = self.game.get_next_state(state, current_player, action)
            move_count += 1
            
            # Check for game end
            reward = self.game.get_reward_for_player(state, 1)
            if reward is not None:
                # Game ended, determine winner from model1's perspective
                if model1_starts:
                    # model1 was player 1
                    return reward
                else:
                    # model1 was player -1
                    return -reward
        
        # Game ended in draw
        return 0
    
    def evaluate_candidate(self, iteration=None):
        """
        Evaluate candidate model against current best
        
        Returns:
        - True if candidate should be promoted
        - False if candidate should be discarded
        """
        print(f"\n🏟️  MODEL EVALUATION - ITERATION {iteration if iteration else '??'}")
        print("=" * 60)
        
        # Load models
        candidate_model = self.load_model(self.candidate_path)
        if candidate_model is None:
            print("❌ Cannot evaluate: no candidate model found")
            return False
        
        # Load current champion (or use candidate as initial champion)
        if os.path.exists(self.current_best_path):
            best_model = self.load_model(self.current_best_path)
            print(f"✅ Loaded current champion model: {self.current_best_path}")
        else:
            print(f"🆕 No current champion - candidate becomes first champion model")
            self.promote_candidate()
            return True
        
        print(f"🥊 Starting head-to-head evaluation...")
        print(f"   Candidate vs Current Champion")
        print(f"   {self.evaluation_games} games, need >{self.win_threshold:.1%} win rate")
        
        # Track results
        candidate_wins = 0
        best_wins = 0
        draws = 0
        
        # Play evaluation games
        for game_num in range(self.evaluation_games):
            # Alternate who starts (fairness)
            candidate_starts = (game_num % 2 == 0)
            
            result = self.play_game(candidate_model, best_model, candidate_starts)
            
            if result == 1:
                candidate_wins += 1
            elif result == -1:
                best_wins += 1
            else:
                draws += 1
            
            # Progress reporting
            if (game_num + 1) % 50 == 0 or game_num == 0:
                current_win_rate = candidate_wins / (game_num + 1)
                print(f"   Game {game_num + 1:3d}: Candidate {candidate_wins:3d}-{best_wins:3d}-{draws:2d} Champion "
                      f"(win rate: {current_win_rate:.3f})")
        
        # Calculate final results
        total_decisive_games = candidate_wins + best_wins
        if total_decisive_games > 0:
            win_rate = candidate_wins / total_decisive_games
        else:
            win_rate = 0.5  # All draws
        
        win_rate_including_draws = candidate_wins / self.evaluation_games
        
        # Log results
        evaluation_result = {
            'timestamp': datetime.now().isoformat(),
            'iteration': iteration,
            'candidate_wins': candidate_wins,
            'best_wins': best_wins,
            'draws': draws,
            'total_games': self.evaluation_games,
            'win_rate_decisive': win_rate,
            'win_rate_total': win_rate_including_draws,
            'threshold': self.win_threshold,
            'promoted': win_rate > self.win_threshold
        }
        
        self.history.append(evaluation_result)
        self.save_evaluation_history()
        
        # Print detailed results
        print(f"\n📊 EVALUATION RESULTS:")
        print(f"   Candidate wins: {candidate_wins}")
        print(f"   Current champion wins: {best_wins}")
        print(f"   Draws: {draws}")
        print(f"   Win rate (decisive games): {win_rate:.3f}")
        print(f"   Win rate (all games): {win_rate_including_draws:.3f}")
        print(f"   Threshold: {self.win_threshold:.3f}")
        
        # Decision
        if win_rate > self.win_threshold:
            print(f"🎉 PROMOTION: Candidate wins {win_rate:.1%} > {self.win_threshold:.1%}")
            print(f"   Candidate promoted to new champion!")
            self.promote_candidate()
            return True
        else:
            print(f"❌ REJECTION: Candidate wins {win_rate:.1%} ≤ {self.win_threshold:.1%}")
            print(f"   Candidate discarded, champion model retained.")
            self.discard_candidate()
            return False
    
    def promote_candidate(self):
        """Promote candidate model to current champion"""
        if os.path.exists(self.candidate_path):
            # Copy candidate to champion
            checkpoint = torch.load(self.candidate_path, map_location=self.device)
            torch.save(checkpoint, self.current_best_path)
            print(f"💾 Candidate promoted: {self.candidate_path} → {self.current_best_path}")
        else:
            print(f"⚠️  Cannot promote: candidate not found")
    
    def discard_candidate(self):
        """Discard candidate model and prepare for next training iteration"""
        if os.path.exists(self.candidate_path):
            os.remove(self.candidate_path)
            print(f"🗑️  Candidate discarded: {self.candidate_path}")
        
        # Note: The trainer should load from latest.pth for the next iteration
        print(f"📋 Next training will continue from champion: {self.current_best_path}")
    
    def prepare_for_training(self):
        """Prepare the training environment by ensuring latest.pth exists for training"""
        if not os.path.exists(self.current_best_path):
            print(f"⚠️  No champion model found. Training will start from scratch.")
            return False
        else:
            print(f"✅ Training will continue from champion: {self.current_best_path}")
            return True
    
    def get_current_best_model(self):
        """Get the current champion model for gameplay"""
        if os.path.exists(self.current_best_path):
            return self.load_model(self.current_best_path)
        elif os.path.exists(self.candidate_path):
            print(f"⚠️  No champion model, using candidate")
            return self.load_model(self.candidate_path)
        else:
            print(f"❌ No models available")
            return None
    
    def print_evaluation_summary(self):
        """Print summary of all evaluations"""
        if not self.history:
            print("📊 No evaluation history available")
            return
        
        print(f"\n📈 EVALUATION HISTORY SUMMARY")
        print("=" * 60)
        
        total_evaluations = len(self.history)
        promotions = sum(1 for result in self.history if result['promoted'])
        rejection_rate = (total_evaluations - promotions) / total_evaluations if total_evaluations > 0 else 0
        
        print(f"Total evaluations: {total_evaluations}")
        print(f"Promotions: {promotions}")
        print(f"Rejections: {total_evaluations - promotions}")
        print(f"Rejection rate: {rejection_rate:.1%}")
        
        print(f"\nRecent evaluations:")
        for result in self.history[-5:]:  # Last 5
            status = "✅ PROMOTED" if result['promoted'] else "❌ REJECTED"
            iteration = result.get('iteration', '??')
            win_rate = result['win_rate_decisive']
            print(f"  Iter {iteration:2s}: {win_rate:.3f} win rate - {status}")

def integrate_with_trainer():
    """Show how to integrate evaluation with existing trainer"""
    
    integration_code = '''
# Add this to your trainer.py or connect4_main.py

from model_evaluation import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(game, args, evaluation_games=400, win_threshold=0.55)

# In your training loop, after each iteration:
for i in range(1, args['numIters'] + 1):
    # Load from latest.pth if it exists (champion model)
    if i > 1 and os.path.exists('latest.pth'):
        trainer.load_checkpoint('latest.pth')
    
    # ... existing training code ...
    
    # Save candidate model
    trainer.save_checkpoint(folder=".", filename='candidate.pth')
    
    # Evaluate candidate vs current champion
    if i % 5 == 0:  # Evaluate every 5 iterations (adjust as needed)
        print(f"\\n🏆 EVALUATING ITERATION {i}")
        promoted = evaluator.evaluate_candidate(iteration=i)
        
        if promoted:
            print(f"🎉 New champion model at iteration {i}!")
            # candidate.pth is now copied to latest.pth
        else:
            print(f"❌ Candidate rejected, continuing with champion")
            # candidate.pth is discarded, will train from latest.pth next
    
    # Optional: Print evaluation history
    if i % 20 == 0:
        evaluator.print_evaluation_summary()
'''
    
    print("🔧 TRAINER INTEGRATION")
    print("=" * 60)
    print(integration_code)

def main():
    """Main evaluation demonstration"""
    print("🏆 ALPHAZERO-STYLE MODEL EVALUATION FOR CONNECT 4")
    print("=" * 60)
    
    # Setup
    device = torch.device('cpu')
    game = Connect4Game()
    
    # MCTS args for evaluation (use fewer simulations for speed)
    args = {
        'num_simulations': 200,  # Reduced for faster evaluation
        'batch_size': 64,
        'numIters': 50,
        'numEps': 25,
        'epochs': 4,
        'checkpoint_path': 'candidate.pth'
    }
    
    # Create evaluator
    evaluator = ModelEvaluator(game, args, evaluation_games=100, win_threshold=0.55)  # Reduced games for demo
    
    # Show current status
    print(f"\n📋 CURRENT STATUS:")
    print(f"   Champion model exists: {os.path.exists(evaluator.current_best_path)}")
    print(f"   Candidate exists: {os.path.exists(evaluator.candidate_path)}")
    
    # If both models exist, run evaluation
    if os.path.exists(evaluator.candidate_path):
        print(f"\n🚀 Running evaluation...")
        promoted = evaluator.evaluate_candidate(iteration="demo")
        
        if promoted:
            print(f"🎉 Candidate was promoted to champion!")
        else:
            print(f"❌ Candidate rejected, champion retained")
    else:
        print(f"\n⚠️  No candidate model found for evaluation")
        print(f"   Train a model first with: python connect4_main_with_evaluation.py")
    
    # Show evaluation history
    evaluator.print_evaluation_summary()
    
    # Show integration instructions
    integrate_with_trainer()

if __name__ == "__main__":
    main() 