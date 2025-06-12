"""
Mining Process Simulation

This module provides an interactive simulation of the cryptocurrency mining process,
demonstrating how miners compete to find valid blocks and how difficulty adjusts.
"""

import time
import random
import hashlib
import threading
import json
from typing import Dict, List, Any, Tuple, Optional, Callable
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

class MiningSimulation:
    """Interactive mining process simulation."""
    
    def __init__(self, 
                 difficulty: int = 4,
                 miners: int = 5,
                 target_block_time: int = 600,  # 10 minutes
                 simulation_speed: float = 100.0,  # Speed multiplier
                 max_blocks: int = 100):
        """
        Initialize mining simulation.
        
        Args:
            difficulty: Initial mining difficulty (leading zeros)
            miners: Number of miners in the network
            target_block_time: Target time between blocks in seconds
            simulation_speed: Simulation speed multiplier
            max_blocks: Maximum blocks to mine
        """
        self.difficulty = difficulty
        self.num_miners = miners
        self.target_block_time = target_block_time
        self.simulation_speed = simulation_speed
        self.max_blocks = max_blocks
        
        # Simulation state
        self.blocks = []
        self.miners = []
        self.running = False
        self.start_time = 0
        self.current_time = 0
        self.block_times = []
        self.hashrates = {}
        self.total_hashes = 0
        
        # For visualization
        self.block_history = []
        self.difficulty_history = []
        self.hashrate_history = []
        self.reward_history = {}
        
        # Initialize miners
        self._init_miners()
    
    def _init_miners(self) -> None:
        """Initialize miners with different hashrates."""
        self.miners = []
        total_hashrate = 0
        
        # Create miners with different hashrates following a power law distribution
        for i in range(self.num_miners):
            # Power law distribution: few miners have high hashrate
            hashrate = 10 ** (random.uniform(1, 3))  # 10-1000 H/s
            total_hashrate += hashrate
            
            self.miners.append({
                "id": f"miner_{i+1}",
                "hashrate": hashrate,
                "blocks_mined": 0,
                "rewards": 0.0,
                "active": True
            })
            
            self.hashrates[f"miner_{i+1}"] = hashrate
            self.reward_history[f"miner_{i+1}"] = []
        
        # Normalize hashrates to make simulation faster
        for miner in self.miners:
            miner["hashrate"] = miner["hashrate"] / total_hashrate * 1000 * self.simulation_speed
    
    def get_target(self) -> str:
        """Get target hash based on current difficulty."""
        return '0' * self.difficulty + 'f' * (64 - self.difficulty)
    
    def adjust_difficulty(self) -> None:
        """Adjust mining difficulty based on recent block times."""
        if len(self.block_times) < 3:
            return
            
        # Calculate average time between recent blocks
        recent_times = self.block_times[-3:]
        time_diffs = [t2 - t1 for t1, t2 in zip(recent_times[:-1], recent_times[1:])]
        avg_time = sum(time_diffs) / len(time_diffs)
        
        # Adjust difficulty based on ratio to target time
        ratio = avg_time / self.target_block_time
        
        if ratio < 0.5:
            # Blocks coming too fast, increase difficulty
            self.difficulty += 1
        elif ratio > 2.0:
            # Blocks coming too slow, decrease difficulty
            self.difficulty = max(1, self.difficulty - 1)
        
        # Record for visualization
        self.difficulty_history.append({
            "time": self.current_time,
            "difficulty": self.difficulty,
            "avg_block_time": avg_time
        })
    
    def mine_block(self, miner_id: str, previous_hash: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to mine a block.
        
        Args:
            miner_id: ID of the miner
            previous_hash: Hash of the previous block
            
        Returns:
            Block data if successful, None otherwise
        """
        # Find miner by ID
        miner = next((m for m in self.miners if m["id"] == miner_id), None)
        if not miner or not miner["active"]:
            return None
        
        # Get target
        target = self.get_target()
        
        # Simulate mining process
        nonce = random.randint(0, 1000000)
        block_data = f"{previous_hash}{miner_id}{nonce}"
        block_hash = hashlib.sha256(block_data.encode()).hexdigest()
        
        # Simulate hashrate by probability
        hashrate = miner["hashrate"]
        success_probability = hashrate / (16 ** self.difficulty)
        
        # Update total hashes
        self.total_hashes += hashrate
        
        # Check if block found (probabilistic)
        if random.random() < success_probability:
            # Block found!
            block = {
                "index": len(self.blocks),
                "timestamp": self.current_time,
                "miner": miner_id,
                "previous_hash": previous_hash,
                "hash": block_hash,
                "nonce": nonce,
                "difficulty": self.difficulty
            }
            
            # Update miner stats
            miner["blocks_mined"] += 1
            reward = 6.25  # BTC reward
            miner["rewards"] += reward
            
            # Record for visualization
            self.reward_history[miner_id].append({
                "time": self.current_time,
                "reward": reward,
                "total": miner["rewards"]
            })
            
            return block
        
        return None
    
    def run_simulation_step(self) -> Optional[Dict[str, Any]]:
        """
        Run one step of the simulation.
        
        Returns:
            New block if mined, None otherwise
        """
        if not self.blocks:
            # Create genesis block
            genesis = {
                "index": 0,
                "timestamp": self.current_time,
                "miner": "genesis",
                "previous_hash": "0" * 64,
                "hash": hashlib.sha256(b"genesis").hexdigest(),
                "nonce": 0,
                "difficulty": self.difficulty
            }
            self.blocks.append(genesis)
            self.block_times.append(self.current_time)
            self.block_history.append(genesis)
            return genesis
        
        # Get latest block
        latest_block = self.blocks[-1]
        
        # Each miner tries to mine a block
        for miner in self.miners:
            if not miner["active"]:
                continue
                
            block = self.mine_block(miner["id"], latest_block["hash"])
            if block:
                self.blocks.append(block)
                self.block_times.append(self.current_time)
                self.block_history.append(block)
                
                # Adjust difficulty periodically
                if len(self.blocks) % 10 == 0:
                    self.adjust_difficulty()
                
                return block
        
        return None
    
    def run_simulation(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        """
        Run the full simulation.
        
        Args:
            callback: Function to call after each block is mined
        """
        self.running = True
        self.start_time = time.time()
        self.current_time = 0
        
        while self.running and len(self.blocks) < self.max_blocks:
            # Run simulation step
            block = self.run_simulation_step()
            
            # Update simulation time
            self.current_time += 1
            
            # Record hashrate history
            if self.current_time % 10 == 0:
                self.hashrate_history.append({
                    "time": self.current_time,
                    "total_hashrate": self.total_hashes / 10,
                    "miner_hashrates": {m["id"]: m["hashrate"] for m in self.miners}
                })
                self.total_hashes = 0
            
            # Call callback if block was mined
            if block and callback:
                callback(block)
            
            # Small delay to not hog CPU
            time.sleep(0.01)
    
    def stop_simulation(self) -> None:
        """Stop the simulation."""
        self.running = False
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """
        Get simulation statistics.
        
        Returns:
            Dictionary of simulation statistics
        """
        if not self.blocks:
            return {}
            
        # Calculate statistics
        total_time = self.current_time
        blocks_mined = len(self.blocks) - 1  # Exclude genesis
        avg_block_time = total_time / blocks_mined if blocks_mined > 0 else 0
        
        # Calculate miner statistics
        miner_stats = []
        for miner in self.miners:
            miner_stats.append({
                "id": miner["id"],
                "hashrate": miner["hashrate"] / self.simulation_speed,
                "blocks_mined": miner["blocks_mined"],
                "rewards": miner["rewards"],
                "share": miner["blocks_mined"] / blocks_mined if blocks_mined > 0 else 0
            })
        
        return {
            "total_time": total_time,
            "blocks_mined": blocks_mined,
            "avg_block_time": avg_block_time,
            "current_difficulty": self.difficulty,
            "miners": miner_stats
        }
    
    def plot_block_times(self) -> plt.Figure:
        """
        Plot block times over the simulation.
        
        Returns:
            Matplotlib figure
        """
        if len(self.block_times) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough blocks mined", ha='center', va='center')
            return fig
            
        # Calculate time differences
        times = self.block_times
        time_diffs = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(time_diffs) + 1), time_diffs, marker='o')
        ax.axhline(y=self.target_block_time, color='r', linestyle='--', label=f'Target ({self.target_block_time}s)')
        
        ax.set_xlabel('Block Number')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Block Mining Times')
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def plot_difficulty(self) -> plt.Figure:
        """
        Plot difficulty changes over the simulation.
        
        Returns:
            Matplotlib figure
        """
        if not self.difficulty_history:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No difficulty adjustments", ha='center', va='center')
            return fig
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times = [d["time"] for d in self.difficulty_history]
        difficulties = [d["difficulty"] for d in self.difficulty_history]
        
        ax.plot(times, difficulties, marker='o')
        
        ax.set_xlabel('Simulation Time')
        ax.set_ylabel('Difficulty')
        ax.set_title('Mining Difficulty Adjustments')
        ax.grid(True)
        
        return fig
    
    def plot_hashrates(self) -> plt.Figure:
        """
        Plot hashrates over the simulation.
        
        Returns:
            Matplotlib figure
        """
        if not self.hashrate_history:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No hashrate data", ha='center', va='center')
            return fig
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times = [h["time"] for h in self.hashrate_history]
        total_hashrates = [h["total_hashrate"] / self.simulation_speed for h in self.hashrate_history]
        
        ax.plot(times, total_hashrates, label='Total Network', linewidth=2)
        
        # Plot individual miner hashrates
        for miner in self.miners:
            miner_id = miner["id"]
            miner_hashrates = [h["miner_hashrates"].get(miner_id, 0) / self.simulation_speed 
                              for h in self.hashrate_history]
            ax.plot(times, miner_hashrates, label=miner_id, alpha=0.7)
        
        ax.set_xlabel('Simulation Time')
        ax.set_ylabel('Hashrate (H/s)')
        ax.set_title('Mining Hashrates')
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def plot_rewards(self) -> plt.Figure:
        """
        Plot miner rewards over the simulation.
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for miner_id, rewards in self.reward_history.items():
            if not rewards:
                continue
                
            times = [r["time"] for r in rewards]
            cumulative = [r["total"] for r in rewards]
            
            ax.plot(times, cumulative, marker='o', label=miner_id)
        
        ax.set_xlabel('Simulation Time')
        ax.set_ylabel('Cumulative Rewards')
        ax.set_title('Miner Rewards')
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def export_simulation_data(self) -> str:
        """
        Export simulation data as JSON.
        
        Returns:
            JSON string of simulation data
        """
        data = {
            "blocks": self.block_history,
            "difficulty": self.difficulty_history,
            "hashrates": self.hashrate_history,
            "rewards": self.reward_history,
            "stats": self.get_simulation_stats()
        }
        
        return json.dumps(data, indent=2)

# Example usage
if __name__ == "__main__":
    # Create simulation
    simulation = MiningSimulation(
        difficulty=3,
        miners=5,
        target_block_time=60,  # 1 minute for faster simulation
        simulation_speed=1000.0,
        max_blocks=50
    )
    
    # Define callback
    def block_callback(block):
        print(f"Block #{block['index']} mined by {block['miner']} with difficulty {block['difficulty']}")
    
    # Run simulation
    print("Starting mining simulation...")
    simulation.run_simulation(block_callback)
    
    # Print statistics
    stats = simulation.get_simulation_stats()
    print("\nSimulation Statistics:")
    print(f"Total time: {stats['total_time']:.2f} seconds")
    print(f"Blocks mined: {stats['blocks_mined']}")
    print(f"Average block time: {stats['avg_block_time']:.2f} seconds")
    print(f"Final difficulty: {stats['current_difficulty']}")
    
    print("\nMiner Statistics:")
    for miner in stats['miners']:
        print(f"{miner['id']}: {miner['blocks_mined']} blocks, "
              f"{miner['rewards']:.2f} rewards, "
              f"{miner['share']*100:.1f}% share")
    
    # Generate plots
    simulation.plot_block_times().savefig("block_times.png")
    simulation.plot_difficulty().savefig("difficulty.png")
    simulation.plot_hashrates().savefig("hashrates.png")
    simulation.plot_rewards().savefig("rewards.png")
    
    # Export data
    with open("simulation_data.json", "w") as f:
        f.write(simulation.export_simulation_data())