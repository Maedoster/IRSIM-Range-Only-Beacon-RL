import os
import glob
import shutil
import yaml
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "IRSimDataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "IRSimDataset_Sorted")


# ==========================================
# GEOMETRY HELPERS
# ==========================================
def shoelace_area(vertices):
    """Calculates the area of a polygon given its vertices [x, y] using the Shoelace formula."""
    if len(vertices) < 3:
        return 0.0
    
    x = np.array([v[0] for v in vertices])
    y = np.array([v[1] for v in vertices])
    
    # Shoelace formula implementation
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def calculate_yaml_difficulty(yaml_path):
    """
    Parses an IRSim YAML file, calculates total obstacle area, 
    and normalizes it by the world area to get an obstacle density ratio.
    """
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        # 1. Calculate Total World Area
        world_data = data.get('world', {})
        width = world_data.get('width', 0.0)
        height = world_data.get('height', 0.0)
        world_area = width * height
        
        if world_area <= 0:
            return 0.0 # Guard against malformed world bounds
            
        total_obstacle_area = 0.0
        
        if 'obstacle' not in data or not data['obstacle']:
            return 0.0 
            
        # 2. Calculate sum of all obstacle areas
        for obs in data['obstacle']:
            if 'shape' not in obs:
                continue
                
            shape_data = obs['shape']
            shape_name = shape_data.get('name')
            
            if shape_name == 'rectangle':
                total_obstacle_area += shape_data.get('length', 0.0) * shape_data.get('width', 0.0)
            elif shape_name == 'circle':
                total_obstacle_area += np.pi * (shape_data.get('radius', 0.0) ** 2)
            elif shape_name == 'polygon':
                vertices = shape_data.get('vertices', [])
                if vertices:
                    total_obstacle_area += shoelace_area(vertices)
                    
        # 3. Return the density ratio (0.0 means empty, closer to 1.0 means fully blocked)
        density_ratio = total_obstacle_area / world_area
        return density_ratio
        
    except Exception as e:
        print(f"Error parsing {yaml_path}: {e}")
        return None

# ==========================================
# MAIN SORTING & COPYING LOGIC
# ==========================================
def main():
    search_pattern = os.path.join(INPUT_DIR, "*.yaml")
    yaml_files = glob.glob(search_pattern)
    
    if not yaml_files:
        print(f"No .yaml files found in {INPUT_DIR}!")
        return

    print(f"Found {len(yaml_files)} YAML maps. Calculating mathematical difficulty...")
    map_scores = []
    
    # 1. Score every map
    for i, yaml_path in enumerate(yaml_files):
        obstacle_area = calculate_yaml_difficulty(yaml_path)
        
        if obstacle_area is not None:
            map_scores.append({
                "path": yaml_path, 
                "filename": os.path.basename(yaml_path),
                "area": obstacle_area
            })
            
        if (i + 1) % 5000 == 0:
            print(f"Analyzed {i + 1}/{len(yaml_files)}...")

    # 2. Sort by total obstacle area (Easy -> Hard)
    map_scores.sort(key=lambda x: x["area"])

    # 3. Divide into 3 equal layers
    total_valid = len(map_scores)
    third = total_valid // 3

    tiers = {
        "Easy": map_scores[:third],
        "Medium": map_scores[third:2*third],
        "Hard": map_scores[2*third:]
    }

    # 4. Create directories and copy files
    print("\nStarting file copying process...")
    
    for tier_name, maps in tiers.items():
        tier_dir = os.path.join(OUTPUT_DIR, tier_name)
        os.makedirs(tier_dir, exist_ok=True)
        
        print(f"Copying {len(maps)} files to {tier_dir}...")
        
        for m in maps:
            src_yaml = m["path"]
            dst_yaml = os.path.join(tier_dir, m["filename"])
            shutil.copy2(src_yaml, dst_yaml)

    print("\n" + "="*50)
    print("SORTING AND COPYING COMPLETE!")
    print(f"Easy folder   : {len(tiers['Easy'])} maps (Max Obstacle Area: {tiers['Easy'][-1]['area']:.2f} m²)")
    print(f"Medium folder : {len(tiers['Medium'])} maps (Max Obstacle Area: {tiers['Medium'][-1]['area']:.2f} m²)")
    print(f"Hard folder   : {len(tiers['Hard'])} maps (Max Obstacle Area: {tiers['Hard'][-1]['area']:.2f} m²)")
    print("="*50)

if __name__ == "__main__":
    main()