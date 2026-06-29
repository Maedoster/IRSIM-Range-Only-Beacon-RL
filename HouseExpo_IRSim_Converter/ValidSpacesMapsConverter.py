import os
import json
import numpy as np
import cv2

# Constants, similar to those used in the world.yaml
WALL_THICKNESS = 0.13
MARGIN = 1.0
RESOLUTION = 0.05  # 5cm per pixel
ROBOT_RADIUS = 0.2 + 0.15 #Size of robot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "OriginalDataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "OccupancyGrids")

def load_layout(path):
    with open(path, "r") as f:
        return json.load(f)

def compute_bbox(verts):
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    return min(xs), min(ys), max(xs), max(ys)

def shift_verts(verts, dx, dy):
    return [[x + dx, y + dy] for x, y in verts]

def generate_grid(layout, filename):
    polygons = layout.get("polygons", [layout["verts"]])

    #Original bounding box
    all_verts = [v for poly in polygons for v in poly]
    min_x, min_y, max_x, max_y = compute_bbox(all_verts)

    width_m = (max_x - min_x) + 2 * MARGIN
    height_m = (max_y - min_y) + 2 * MARGIN

    grid_w = int(width_m / RESOLUTION)
    grid_h = int(height_m / RESOLUTION)

    # Everything is initialized as black 
    grid = np.ones((grid_h, grid_w), dtype=np.uint8)

    def to_pixels(poly):
        shifted = shift_verts(poly, -min_x + MARGIN, -min_y + MARGIN)
        return (np.array(shifted) / RESOLUTION).astype(np.int32)

    # Xor and parity rules for filling
    mask = np.zeros_like(grid, dtype=np.uint8)
    for poly in polygons:
        px = to_pixels(poly)
        temp = np.zeros_like(grid, dtype=np.uint8)
        cv2.fillPoly(temp, [px], 1)
        mask = cv2.bitwise_xor(mask, temp)

    binary = (mask == 1).astype(np.uint8)

    wall_thickness_px = max(1, int(WALL_THICKNESS / RESOLUTION))
    for poly in polygons:
        px = to_pixels(poly)
        # Draw the walls as 0 (obstacles) so they physically restrict the flood mask
        cv2.polylines(binary, [px], isClosed=True, color=0, thickness=wall_thickness_px)

    flood = binary.copy()
    h, w = flood.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Riempiamo dall'angolo (0,0) che è sicuramente fuori grazie al MARGIN
    cv2.floodFill(flood, flood_mask, (0, 0), 2)
    
    reachable = (flood == 1).astype(np.uint8)

    robot_radius_px = int(ROBOT_RADIUS / RESOLUTION)
    
    if robot_radius_px > 0:
        kernel_size = robot_radius_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        #Invert reachable space to get OBSTACLES (walls + outside = 1)
        obstacles = (reachable == 0).astype(np.uint8)
        
        # DILATE obstacles. 
        # This forces thin walls to expand. If a door is narrower than the robot, 
        # the expanded walls will overlap and completely seal the door.
        inflated_obstacles = cv2.dilate(obstacles, kernel)
        
        # Get the valid Configuration Space (navigable areas = 1)
        c_space = (inflated_obstacles == 0).astype(np.uint8)
        
        # Find the largest isolated main room in this safe C-Space
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(c_space, connectivity=4)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            largest_c_space = (labels == largest_label).astype(np.uint8)
            
            # Restore the main room's original physical size by dilating it back outwards
            restored_reachable = cv2.dilate(largest_c_space, kernel)
            
            # Mask against the original reachable area to ensure we don't bleed into actual walls
            reachable = cv2.bitwise_and(restored_reachable, reachable)
            

    grid = np.ones_like(binary, dtype=np.uint8)
    grid[reachable == 1] = 0

    # Draw walls again as 1 (Obstacles) to ensure outer borders are crisp
    for poly in polygons:
        px = to_pixels(poly)
        cv2.polylines(grid, [px], isClosed=True, color=1, thickness=wall_thickness_px)

    # Saving and metadata
    base_name = filename.replace(".json", "")
    np.save(os.path.join(OUTPUT_DIR, f"{base_name}.npy"), grid)
    
    img = (1 - grid) * 255
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}.png"), img)
    
    meta = {
        "resolution": RESOLUTION,
        "min_x": 0, 
        "min_y": 0,
        "margin": 0, 
        "grid_shape": grid.shape
    }
    with open(os.path.join(OUTPUT_DIR, f"{base_name}_meta.json"), "w") as f:
        json.dump(meta, f)
    

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

    for file in files:
        try:
            layout = load_layout(os.path.join(INPUT_DIR, file))
            generate_grid(layout, file)
            print(f"Successo: {file} convertito in Griglia Occupazione")
        except Exception as e:
            print(f"Errore su {file}: {e}")

if __name__ == "__main__":
    main()