import json
import itertools
from typing import List, Tuple, Dict, Any
import os
from pathlib import Path


class EnvironmentVariationGenerator:
    """Generate all variations of an environment with different key and goal positions"""
    
    def __init__(self, json_file_path: str):
        """Initialize with a base environment JSON file"""
        with open(json_file_path, 'r') as f:
            self.base_env = json.load(f)
        
        # Extract grid dimensions from training command or default
        self.grid_height, self.grid_width = self._extract_grid_dimensions()
        
    def _extract_grid_dimensions(self) -> Tuple[int, int]:
        """Extract grid dimensions from training command or use defaults"""
        if 'training_command' in self.base_env:
            cmd = self.base_env['training_command']
            height = None
            width = None
            
            # Parse grid_height and grid_width from command
            parts = cmd.split()
            for i, part in enumerate(parts):
                if part == '--grid_height' and i + 1 < len(parts):
                    height = int(parts[i + 1])
                elif part == '--grid_width' and i + 1 < len(parts):
                    width = int(parts[i + 1])
            
            print(f"Height: {height}, Width: {width}")
            if height and width:
                return height, width
        
        # Fallback: infer from max coordinates + 1
        all_positions = []
        for key in ['agent_pos', 'box_pos', 'trap_pos', 'goal_pos', 'wall_pos', 'helper_pos', 'key_pos']:
            if key in self.base_env:
                all_positions.extend(self.base_env[key])
        
        if all_positions:
            max_row = max(pos[0] for pos in all_positions) + 1
            max_col = max(pos[1] for pos in all_positions) + 1
            return max_row, max_col
        
        # Default fallback
        return 7, 7
    
    def get_occupied_positions(self) -> List[Tuple[int, int]]:
        """Get all positions occupied by entities (excluding key and goal)"""
        occupied = []
        
        # Add all entity positions except key and goal
        for key in ['agent_pos', 'box_pos', 'trap_pos', 'wall_pos', 'helper_pos']:
            if key in self.base_env:
                occupied.extend([tuple(pos) for pos in self.base_env[key]])
        
        return list(set(occupied))  # Remove duplicates
    
    def get_free_positions(self) -> List[Tuple[int, int]]:
        """Get all unoccupied positions in the grid (including current key/goal positions)"""
        occupied = set(self.get_occupied_positions())
        
        # Generate all possible positions
        all_positions = [(r, c) for r in range(self.grid_height) 
                        for c in range(self.grid_width)]
        
        # Return positions not occupied by fixed entities
        free_positions = [pos for pos in all_positions if pos not in occupied]
        return free_positions
    
    def generate_all_variations(self) -> List[Dict[str, Any]]:
        """Generate all possible variations with different key and goal positions"""
        free_positions = self.get_free_positions()
        
        print(f"Grid size: {self.grid_height}x{self.grid_width} = {self.grid_height * self.grid_width} total positions")
        print(f"Occupied positions: {len(self.get_occupied_positions())}")
        print(f"Free positions: {len(free_positions)}")
        print(f"Total variations (permutations of 2): {len(free_positions) * (len(free_positions) - 1)}")
        
        variations = []
        
        # Generate all permutations of 2 positions from free positions (key, goal)
        for key_pos, goal_pos in itertools.permutations(free_positions, 2):
            # Create a copy of the base environment
            variation = self.base_env.copy()
            
            # Update key and goal positions
            variation['key_pos'] = [list(key_pos)]
            variation['goal_pos'] = [list(goal_pos)]
            
            # Remove metadata fields for cleaner training files
            for meta_key in ['training_command', 'wandbids', 'status']:
                variation.pop(meta_key, None)
            
            variations.append(variation)
        
        return variations
    
    def save_variations(self, output_dir: str = "environment_variations", 
                       base_filename: str = None) -> List[str]:
        """Save all variations to separate JSON files"""
        variations = self.generate_all_variations()
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        if base_filename is None:
            base_filename = "env_variation"
        
        saved_files = []
        
        for i, variation in enumerate(variations):
            filename = f"{base_filename}_{i:04d}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(variation, f, indent=2)
            
            saved_files.append(filepath)
        
        print(f"Saved {len(variations)} variations to {output_dir}/")
        return saved_files
    
    def create_training_batch_file(self, output_dir: str = "environment_variations",
                                  batch_filename: str = "all_variations.json") -> str:
        """Create a single JSON file containing all variations for batch training"""
        variations = self.generate_all_variations()
        
        Path(output_dir).mkdir(exist_ok=True)
        batch_filepath = os.path.join(output_dir, batch_filename)
        
        with open(batch_filepath, 'w') as f:
            json.dump(variations, f, indent=2)
        
        print(f"Saved batch file with {len(variations)} variations to {batch_filepath}")
        return batch_filepath


def main():
    """Example usage"""
    # Generate variations for the embodied_moving_boxes_1.json
    generator = EnvironmentVariationGenerator("embodied_moving_boxes_1.json")
    
    print("=== Environment Analysis ===")
    print(f"Grid dimensions: {generator.grid_height}x{generator.grid_width}")
    print(f"Occupied positions: {generator.get_occupied_positions()}")
    print(f"Free positions: {generator.get_free_positions()}")
    
    print("\n=== Generating Variations ===")
    # Save individual files
    saved_files = generator.save_variations(
        output_dir="embodied_moving_boxes_variations",
        base_filename="moving_boxes_var"
    )
    
    # Create batch file
    batch_file = generator.create_training_batch_file(
        output_dir="embodied_moving_boxes_variations",
        batch_filename="all_moving_boxes_variations.json"
    )
    
    print(f"\nFirst few saved files: {saved_files[:5]}")
    print(f"Batch file: {batch_file}")


if __name__ == "__main__":
    main()
