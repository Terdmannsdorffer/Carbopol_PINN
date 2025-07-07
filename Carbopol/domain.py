# domain_regional.py - Enhanced domain with regional point tracking
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# PIV-COMPATIBLE dimensions from experimental data
L_up = 0.097      # Upper horizontal length
L_down = 0.174    # Total horizontal length
H_left = 0.119    # Left vertical height
H_right = 0.019   # Right vertical height

# PIV coordinate offset
PIV_X_OFFSET = 0.002745
PIV_Y_OFFSET = -0.121043

def inside_L(x, y):
    main_rect = (0.0 <= x <= L_down) and (0.0 <= y <= H_left)
    # Cambia el rango de y: ahora recorta desde H_right hasta H_left (la esquina superior)
    corner_rect = (L_up <= x <= L_down) and (H_right <= y <= H_left)
    return main_rect and not corner_rect


def get_domain_info(device=None):
    """
    Devuelve toda la información necesaria para visualizar y enmascarar el dominio L-pipe:
      - rangos x/y
      - función inside_L_torch (para máscara en visualización)
      - regiones PIV completas
    """
    # Rango seguro para la malla
    x_min = 0.0
    x_max = L_down
    y_min = 0.0
    y_max = H_left

    # --- Función compatible con Torch para la máscara de dominio ---
    def inside_L_torch(points):
        x = points[:, 0]
        y = points[:, 1]
        main_rect = (x >= 0) & (x <= L_down) & (y >= 0) & (y <= H_left)
        # Cambia el rango de y aquí también
        corner_rect = (x >= L_up) & (x <= L_down) & (y >= H_right) & (y <= H_left)
        return main_rect & ~corner_rect


    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'inside_L_torch': inside_L_torch,
        'regions': get_piv_regions()
    }


def get_region_for_point(x, y):
    """Determine which region a point belongs to"""
    regions = get_piv_regions()
    
    for region_name, region_info in regions.items():
        x_min, x_max = region_info['x_range']
        y_min, y_max = region_info['y_range']
        
        if x_min <= x <= x_max and y_min <= y <= y_max and inside_L(x, y):
            return region_name
    
    # If not in any specific region, classify as general domain
    if inside_L(x, y):
        return 'general'
    return None

def get_piv_regions():
    """Define regiones PIV con límites detallados y convención y positiva."""
    L_up = 0.097      # Upper horizontal length
    L_down = 0.174    # Total horizontal length
    H_left = 0.119    # Left vertical height
    H_right = 0.019   # Right vertical height

def get_piv_regions():
    regions = {
        'inlet': {
            'x_range': [0.01990, 0.07289],
            'y_range': [0.09566, 0.11685],  # flipped
            'u_mean': -4.5112e-05,
            'u_std': 1.6e-04,
            'v_mean': -2.2786e-03,
            'v_std': 7.3e-04,
            'mag_mean': 2.2824e-03,
            'mag_std': 2.8215e-04,
            'stagnation_fraction': 0.0,
            'weight': 2.0,
            'description': 'inlet'
        },
        'middle': {
            'x_range': [0.02090, 0.07421],
            'y_range': [0.00823, 0.09599],  # flipped
            'u_mean': 6.8280e-04,
            'u_std': 8.2e-04,
            'v_mean': -1.8082e-03,
            'v_std': 8.8e-04,
            'mag_mean': 2.0331e-03,
            'mag_std': 8.2295e-04,
            'stagnation_fraction': 2.8986e-02,
            'weight': 2.5,
            'description': 'middle'
        },
        'corner': {
            'x_range': [0.090, 0.105],     # Justo en la esquina
            'y_range': [0.005, 0.025],     # Desde adentro del outlet hasta un poco arriba del borde superior del outlet horizontal
            'u_mean': 4.2945e-03,
            'u_std': 3.0e-03,
            'v_mean': -2.8314e-03,
            'v_std': 1.4e-03,
            'mag_mean': 5.6540e-03,
            'mag_std': 3.0406e-03,
            'stagnation_fraction': 0.0,
            'weight': 6.0,
            'description': 'corner'
        },



        'outlet': {
            'x_range': [0.097, 0.174],      # Covers full horizontal leg
            'y_range': [0.000, 0.030],      # Goes from bottom up to y=0.03
            'u_mean': 9.1711e-03,
            'u_std': 3.2720e-03,
            'v_mean': -2.0441e-04,
            'v_std': 9.8e-04,
            'mag_mean': 9.2357e-03,
            'mag_std': 3.2720e-03,
            'stagnation_fraction': 0.0,
            'weight': 7.0,
            'description': 'outlet'
        },
        'right_wall': {
            'x_range': [0.07355, 0.09309],
            'y_range': [0.03837, 0.11718],  # flipped
            'u_mean': 1.7259e-04,
            'u_std': 6.2e-04,
            'v_mean': -1.7618e-03,
            'v_std': 7.7e-04,
            'mag_mean': 1.7748e-03,
            'mag_std': 6.2003e-04,
            'stagnation_fraction': 0.0,
            'weight': 1.5,
            'description': 'right_wall'
        },
        'left_wall': {
            'x_range': [0.00235, 0.01990],
            'y_range': [0.0, 0.11652],  # flipped
            'u_mean': 1.9377e-05,
            'u_std': 2.5e-04,
            'v_mean': -1.8709e-04,
            'v_std': 2.1e-04,
            'mag_mean': 2.3271e-04,
            'mag_std': 2.5319e-04,
            'stagnation_fraction': 4.5342e-01,
            'weight': 3.0,
            'description': 'left_wall'
        }
    }
    return regions




def export_summary_dataframe(metrics):
    import pandas as pd
    regional_summary = []
    for region, vals in metrics['regional_accuracy'].items():
        regional_summary.append({
            'Region': region,
            'Magnitude_Error_%': vals['mag_error'] * 100,
            'Direction_Accuracy': vals['direction_accuracy'],
            'Achieved_Mean': vals['achieved_mean'],
            'Target_Mean': vals['target_mean'],
            'Stagnation_Fraction': vals['stagnation_fraction']
        })
    metrics['summary'] = pd.DataFrame(regional_summary)
    return metrics
def generate_domain_points_regional(n_total=3000, adaptive_sampling=True):
    """Generate domain points with regional tracking and adaptive sampling"""
    print(f"\nGenerating {n_total} domain points with regional tracking...")
    
    regions = get_piv_regions()
    points_by_region = {name: [] for name in regions.keys()}
    points_by_region['general'] = []
    
    if adaptive_sampling:
        # Allocate points based on region complexity and stagnation
        total_complexity = 0
        region_scores = {}
        
        for name, info in regions.items():
            # Score based on: variability, stagnation, and importance
            variability = (info['u_std'] + info['v_std']) / (info['mag_mean'] + 1e-6)
            stagnation = info['stagnation_fraction']
            importance = info['weight']
            
            score = importance * (1 + variability + 2 * stagnation)
            region_scores[name] = score
            total_complexity += score
        
        # Allocate points proportionally
        points_allocated = {}
        remaining = n_total
        
        for name, score in region_scores.items():
            n_region = int(n_total * score / total_complexity)
            points_allocated[name] = n_region
            remaining -= n_region
        
        # Add remaining points to most complex region
        most_complex = max(region_scores.keys(), key=lambda k: region_scores[k])
        points_allocated[most_complex] += remaining
        
        print("\nAdaptive point allocation:")
        for name, n_pts in points_allocated.items():
            print(f"  {name}: {n_pts} points ({n_pts/n_total*100:.1f}%)")
        
        # Generate points for each region
        for region_name, n_points in points_allocated.items():
            region_info = regions[region_name]
            x_min, x_max = region_info['x_range']
            y_min, y_max = region_info['y_range']
            
            generated = 0
            attempts = 0
            max_attempts = n_points * 10
            
            while generated < n_points and attempts < max_attempts:
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                
                if inside_L(x, y):
                    points_by_region[region_name].append([x, y])
                    generated += 1
                
                attempts += 1
            
            if generated < n_points:
                print(f"  Warning: Only generated {generated}/{n_points} points for {region_name}")
    
    else:
        # Uniform sampling
        generated = 0
        while generated < n_total:
            x = np.random.uniform(0, L_down)
            y = np.random.uniform(0, H_left)
            
            if inside_L(x, y):
                region = get_region_for_point(x, y)
                if region:
                    points_by_region[region].append([x, y])
                    generated += 1
    
    # Convert to numpy arrays and create combined array
    all_points = []
    region_indices = {}
    current_idx = 0
    
    for region_name, points in points_by_region.items():
        if len(points) > 0:
            points_array = np.array(points)
            n_pts = len(points_array)
            
            region_indices[region_name] = (current_idx, current_idx + n_pts)
            all_points.extend(points)
            current_idx += n_pts
            
            print(f"  {region_name}: {n_pts} points generated")
    
    all_points = np.array(all_points)
    
    # Print statistics
    print(f"\nTotal points generated: {len(all_points)}")
    print("\nRegional statistics:")
    for region_name, (start_idx, end_idx) in region_indices.items():
        n_pts = end_idx - start_idx
        if region_name != 'general':
            info = regions.get(region_name, {})
            print(f"  {region_name}: {n_pts} points, stagnation={info.get('stagnation_fraction', 0):.1%}")
    
    return all_points, points_by_region, region_indices

def generate_boundary_points_regional(n_boundary_per_segment=50):
    """Generate boundary points with regional classification"""
    print(f"\nGenerating boundary points with regional tracking...")
    
    # Define wall segments
    wall_segments = [
        [(0, 0), (0, H_left)],                    # Left wall
        [(0, H_left), (L_up, H_left)],           # Top wall (inlet)
        [(L_up, H_left), (L_up, H_right)],       # Right upper wall
        [(L_up, H_right), (L_down, H_right)],    # Step wall
        [(L_down, H_right), (L_down, 0)],        # Right wall (outlet)
        [(L_down, 0), (0, 0)]                    # Bottom wall
    ]
    
    boundary_points_by_type = {
        'inlet': [],
        'outlet': [],
        'wall': [],
        'wall_normals': []
    }
    
    for i, ((x1, y1), (x2, y2)) in enumerate(wall_segments):
        # Higher resolution for critical segments
        if i == 1:  # Inlet
            n_points = int(n_boundary_per_segment * 2.0)
        elif i == 3:  # Outlet
            n_points = int(n_boundary_per_segment * 2.0)
        elif i == 2:  # Corner
            n_points = int(n_boundary_per_segment * 1.5)
        else:
            n_points = n_boundary_per_segment
        
        t_vals = np.linspace(0, 1, n_points)
        
        for t in t_vals:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # Calculate inward normal
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                normal = (-dy/length, dx/length)
            else:
                normal = (0, 0)
            
            if i == 1:  # Top wall (inlet)
                boundary_points_by_type['inlet'].append([x, y])
            elif i == 4:  # Right wall (outlet)
                boundary_points_by_type['outlet'].append([x, y])
            else:  # Other walls
                boundary_points_by_type['wall'].append([x, y])
                boundary_points_by_type['wall_normals'].append(normal)
    
    # Convert to arrays
    for key in boundary_points_by_type:
        if key != 'wall_normals' and len(boundary_points_by_type[key]) > 0:
            boundary_points_by_type[key] = np.array(boundary_points_by_type[key])
        elif key == 'wall_normals':
            boundary_points_by_type['wall_normals'] = np.array(boundary_points_by_type['wall_normals'])
    
    print(f"Generated boundary points:")
    print(f"  Inlet points: {len(boundary_points_by_type['inlet'])}")
    print(f"  Outlet points: {len(boundary_points_by_type['outlet'])}")
    print(f"  Wall points: {len(boundary_points_by_type['wall'])}")
    boundary_points_by_type['wall_segments'] = wall_segments
    return boundary_points_by_type

def plot_domain_with_regional_points(points_by_region, save_path=None):
    """Visualize domain with color-coded regional points"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left plot: Domain structure with regions
    regions = get_piv_regions()
    
    # Draw main rectangle
    main_rect = patches.Rectangle((0, 0), L_down, H_left, 
                                 linewidth=2, edgecolor='black', 
                                 facecolor='lightgray', alpha=0.3)
    ax1.add_patch(main_rect)
    
    # Draw corner cutout
    corner_rect = patches.Rectangle((L_up, H_right), L_down-L_up, H_left-H_right,
                                   linewidth=2, edgecolor='black',
                                   facecolor='white', alpha=1.0)
    ax1.add_patch(corner_rect)
    
    # Draw regions with transparency
    colors = plt.cm.tab10(np.linspace(0, 1, len(regions)))
    
    for (name, region), color in zip(regions.items(), colors):
        x_min, x_max = region['x_range']
        y_min, y_max = region['y_range']
        
        rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                               linewidth=2, edgecolor=color,
                               facecolor=color, alpha=0.3,
                               label=f"{name} (stag={region['stagnation_fraction']:.1%})")
        ax1.add_patch(rect)
    
    ax1.set_xlim(-0.01, L_down+0.01)
    ax1.set_ylim(-0.01, H_left+0.01)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Domain Regions with Stagnation Info', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Right plot: Point distribution
    for (name, points), color in zip(points_by_region.items(), colors):
        if len(points) > 0 and name != 'general':
            points_array = np.array(points)
            ax2.scatter(points_array[:, 0], points_array[:, 1], 
                       c=[color], s=20, alpha=0.6, label=f"{name} ({len(points)} pts)")
    
    # Draw domain outline
    domain_x = [0, 0, L_up, L_up, L_down, L_down, 0]
    domain_y = [0, H_left, H_left, H_right, H_right, 0, 0]
    ax2.plot(domain_x, domain_y, 'k-', linewidth=2)
    
    ax2.set_xlim(-0.01, L_down+0.01)
    ax2.set_ylim(-0.01, H_left+0.01)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title('Adaptive Point Distribution by Region', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Regional domain plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

def get_regional_physics_parameters():
    """Get region-specific physics parameters for Carbopol"""
    return {
        'global': {
            'rho': 0.101972,  # kg/m³
            'g_x': 0.0,
            'g_y': -9.81,
            'tau_y': 30.0,    # Pa
            'k': 2.8,
            'n': 0.65
        },
        'inlet': {
            'expected_re': 0.001,  # Very low Reynolds
            'yield_threshold': 0.9,  # 90% might be unyielded
        },
        'middle': {
            'expected_re': 0.001,
            'yield_threshold': 0.85,  # High stagnation
        },
        'corner': {
            'expected_re': 0.01,   # Higher Re in corner
            'yield_threshold': 0.5,  # More yielded
        },
        'outlet': {
            'expected_re': 0.01,
            'yield_threshold': 0.6,
        },
        'left_wall': {
            'expected_re': 0.0001,  # Nearly stagnant
            'yield_threshold': 0.95,  # Mostly unyielded
        },
        'right_wall': {
            'expected_re': 0.002,
            'yield_threshold': 0.8,
        }
    }

# Backward compatibility exports
generate_domain_points = generate_domain_points_regional

def generate_boundary_points(n_boundary_per_segment=50):
    """Wrapper for backward compatibility"""
    boundary_data = generate_boundary_points_regional(n_boundary_per_segment)
    
    # Return in old format: (wall_points, wall_normals, inlet_points, outlet_points, wall_segments)
    return (
        boundary_data['wall'],
        boundary_data['wall_normals'],
        boundary_data['inlet'],
        boundary_data['outlet'],
        []  # wall_segments (empty for compatibility)
    )

if __name__ == "__main__":
    # Test regional generation
    print("Testing regional domain generation...")
    
    all_points, points_by_region, region_indices = generate_domain_points_regional(n_total=3000)
    boundary_points = generate_boundary_points_regional()
    
    # Create visualization
    plot_domain_with_regional_points(points_by_region, "domain_regional_distribution.png")
    
    # Print physics parameters
    physics = get_regional_physics_parameters()
    print("\nRegional physics parameters:")
    for region in ['inlet', 'middle', 'corner', 'outlet']:
        print(f"  {region}: Re={physics[region]['expected_re']}, yield_threshold={physics[region]['yield_threshold']}")


