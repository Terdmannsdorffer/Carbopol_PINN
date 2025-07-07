# training.py - ENHANCED VERSION with Magnitude Calibration
import torch
import numpy as np
import matplotlib.pyplot as plt
from domain import generate_domain_points_regional, get_piv_regions, inside_L
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_training_progress(model, epoch, device):
    """Create debug plots during training to monitor flow development"""
    model.eval()
    
    # Create grid for visualization
    x = np.linspace(0, 0.174, 50)
    y = np.linspace(0, 0.119, 40)
    X, Y = np.meshgrid(x, y)
    
    points = []
    for i in range(len(y)):
        for j in range(len(x)):
            if inside_L(X[i,j], Y[i,j]):
                points.append([X[i,j], Y[i,j]])
    
    points = np.array(points)
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        pred = model(points_tensor)
        u = pred[:, 0].cpu().numpy()
        v = pred[:, 1].cpu().numpy()
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot u-component
    scatter1 = ax1.scatter(points[:, 0], points[:, 1], c=u, cmap='RdBu', s=20, vmin=-0.01, vmax=0.01)
    ax1.set_title(f'U-velocity (Epoch {epoch})')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1)
    
    # Plot v-component  
    scatter2 = ax2.scatter(points[:, 0], points[:, 1], c=v, cmap='RdBu', s=20, vmin=-0.01, vmax=0.01)
    ax2.set_title(f'V-velocity (Epoch {epoch})')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2)
    
    # Plot velocity vectors
    skip = 5
    ax3.quiver(points[::skip, 0], points[::skip, 1], 
               u[::skip], v[::skip], 
               np.sqrt(u[::skip]**2 + v[::skip]**2),
               cmap='viridis', scale=0.05)
    ax3.set_title(f'Velocity Vectors (Epoch {epoch})')
    ax3.set_aspect('equal')
    
    # Add domain outline
    for ax in [ax1, ax2, ax3]:
        domain_x = [0, 0, 0.097, 0.097, 0.174, 0.174, 0]
        domain_y = [0, 0.119, 0.119, 0.019, 0.019, 0, 0]
        ax.plot(domain_x, domain_y, 'k-', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(f'plots/debug_epoch_{epoch:04d}.png', dpi=150)
    plt.close()
    
    # Print statistics
    print(f"\n[Epoch {epoch}] Flow Statistics:")
    print(f"  U range: [{u.min():.6f}, {u.max():.6f}] m/s")
    print(f"  V range: [{v.min():.6f}, {v.max():.6f}] m/s")
    print(f"  Mean |V|: {np.sqrt(u**2 + v**2).mean():.6f} m/s")
    
    model.train()

def compute_no_slip_wall_loss(model, wall_points, device):
    if len(wall_points) == 0:
        return torch.tensor(0.0, device=device)
    n_wall = min(200, len(wall_points))
    wall_indices = torch.randperm(len(wall_points))[:n_wall]
    wall_subset = [wall_points[i] for i in wall_indices]
    wall_tensor = torch.tensor(wall_subset, dtype=torch.float32, device=device)
    wall_pred = model(wall_tensor)
    u_wall = wall_pred[:, 0]
    v_wall = wall_pred[:, 1]
    return torch.mean(u_wall**2) + torch.mean(v_wall**2)


# Replace your compute_power_constraint function with this:

def compute_power_constraint(model, all_points, device):
    """Reasonable power constraint based on actual PIV velocities"""
    domain_tensor = torch.tensor(all_points, dtype=torch.float32, device=device, requires_grad=True)
    
    output = model(domain_tensor)
    u, v = output[:, 0], output[:, 1]
    
    velocity_magnitude = torch.sqrt(u**2 + v**2)
    
    # Target based on PIV average (not ultra-low)
    target_velocity = 0.003  # 3 mm/s (more reasonable than 1 mm/s)
    
    # Allow velocities up to PIV maximum
    max_allowed = 0.01  # 10 mm/s (PIV shows up to 9 mm/s)
    
    # Penalize mean being too far from target
    mean_velocity = torch.mean(velocity_magnitude)
    mean_loss = (mean_velocity - target_velocity)**2
    
    # Only penalize velocities above maximum
    max_penalty = torch.mean(torch.relu(velocity_magnitude - max_allowed)**2)
    
    # Don't penalize low velocities - we want variation!
    power_loss = mean_loss + 10.0 * max_penalty
    
    return power_loss




def train_enhanced_staged(model, optimizer, all_points, inlet_points, outlet_points, wall_points, wall_normals,
                          region_points, region_targets, piv_reference_data):
    """
    Enhanced 4-stage training with magnitude calibration and mass conservation
    
    Key improvements:
    1. Learnable velocity scaling parameters
    2. Mass conservation enforcement
    3. PIV magnitude calibration stage
    4. Better boundary conditions
    """
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    
    loss_history = []
    physics_loss_history = []
    bc_loss_history = []
    magnitude_loss_history = []
    
    # Physical parameters
    rho = 0.998 
    g_x, g_y = 0.0, -9.81
    L_up, L_down = 0.097, 0.174
    H_left, H_right = 0.119, 0.019
    
    # PIV reference data handling
    if piv_reference_data:
        piv_u_mean = piv_reference_data.get('u_mean', 0.0)
        piv_v_mean = piv_reference_data.get('v_mean', -0.005)
        piv_v_mean = -abs(piv_v_mean)
        piv_mag_mean = piv_reference_data.get('mag_mean', 0.005)
        piv_mag_std = piv_reference_data.get('mag_std', 0.002)
        target_mass_flow = piv_reference_data.get('mass_flow_rate', 0.001)
        print(f"Using PIV reference: mag={piv_mag_mean:.6f}±{piv_mag_std:.6f} m/s")
    else:
        piv_u_mean, piv_v_mean = 0.0, -0.005
        piv_mag_mean, piv_mag_std = 0.005, 0.002
        target_mass_flow = 0.001
        print("Using default reference values")
    print(f"\n🔍 DEBUG PIV Reference:")
    print(f"  piv_v_mean = {piv_v_mean:.6f} (should be negative for downward flow)")
    print(f"  piv_u_mean = {piv_u_mean:.6f}")    
    # Carbopol rheology
    tau_y, k, n = 35.55, 2.32, 0.74

    def compute_velocity_scale_calibration(model, all_points, piv_reference_data, device):
        """FIXED: Conservative velocity scale calibration"""
        with torch.no_grad():
            n_sample = min(500, len(all_points))
            indices = torch.randperm(len(all_points))[:n_sample]
            sample_points = [all_points[i] for i in indices]
            sample_tensor = torch.tensor(sample_points, dtype=torch.float32, device=device)
            
            pred = model(sample_tensor)
            u_pred = pred[:, 0].cpu().numpy()
            v_pred = pred[:, 1].cpu().numpy()
            
            # Use RMS instead of mean for better scaling
            current_u_rms = np.sqrt(np.mean(u_pred**2))
            current_v_rms = np.sqrt(np.mean(v_pred**2))
            
            # Target values from PIV
            target_u_rms = 0.004  # Based on PIV data showing u up to 0.009
            target_v_rms = 0.003  # Based on PIV data showing v up to -0.006
            
            # Calculate required scaling adjustments
            u_scale_factor = target_u_rms / (current_u_rms + 1e-8)
            v_scale_factor = target_v_rms / (current_v_rms + 1e-8)
            
            # CRITICAL: Apply CONSERVATIVE scaling - don't change too much at once!
            u_scale_factor = np.clip(u_scale_factor, 0.5, 2.0)  # Max 2x change
            v_scale_factor = np.clip(v_scale_factor, 0.5, 2.0)  # Max 2x change
            
            # Update model scales CONSERVATIVELY
            current_u_scale = model.u_scale.item()
            current_v_scale = model.v_scale.item()
            
            # Only update if the change is significant
            if abs(u_scale_factor - 1.0) > 0.1:
                model.u_scale.data = torch.tensor(current_u_scale * u_scale_factor)
            if abs(v_scale_factor - 1.0) > 0.1:
                model.v_scale.data = torch.tensor(current_v_scale * v_scale_factor)
            
            print(f"Velocity scale calibration:")
            print(f"  u_scale adjusted by {u_scale_factor:.2f}x -> {model.u_scale.item():.6f}")
            print(f"  v_scale adjusted by {v_scale_factor:.2f}x -> {model.v_scale.item():.6f}")

    def compute_flow_guidance_loss(model, points_by_region, device):
        """Guide flow to follow L-shaped pattern"""
        guidance_loss = torch.tensor(0.0, device=device)
        
        # Upper region (before corner): flow should be mainly downward
        if 'inlet' in points_by_region and len(points_by_region['inlet']) > 0:
            upper_points = torch.tensor(points_by_region['inlet'], dtype=torch.float32, device=device)
            upper_pred = model(upper_points)
            u_upper, v_upper = upper_pred[:, 0], upper_pred[:, 1]
            
            # Penalize horizontal flow in upper region
            guidance_loss += 10.0 * torch.mean(u_upper**2)
            # Ensure downward flow
            guidance_loss += 10.0 * torch.mean(torch.relu(v_upper)**2)
        
        # Middle region: transition zone
        if 'middle' in points_by_region and len(points_by_region['middle']) > 0:
            middle_points = torch.tensor(points_by_region['middle'], dtype=torch.float32, device=device)
            middle_pred = model(middle_points)
            u_middle, v_middle = middle_pred[:, 0], middle_pred[:, 1]
            
            # Allow both components but encourage transition
            # Points closer to bottom should have more horizontal flow
            y_coords = middle_points[:, 1]
            y_normalized = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min() + 1e-6)
            
            # Weight: bottom points (low y) should be horizontal, top points vertical
            horizontal_weight = 1.0 - y_normalized
            vertical_weight = y_normalized
            
            # Encourage appropriate flow direction based on position
            guidance_loss += 5.0 * torch.mean(horizontal_weight * v_middle**2)
            guidance_loss += 5.0 * torch.mean(vertical_weight * u_middle**2)
        
        # Corner region: strong flow redirection
        if 'corner' in points_by_region and len(points_by_region['corner']) > 0:
            corner_points = torch.tensor(points_by_region['corner'], dtype=torch.float32, device=device)
            corner_pred = model(corner_points)
            u_corner, v_corner = corner_pred[:, 0], corner_pred[:, 1]
            
            # Corner should have significant flow in both directions
            # Penalize if either component is too small
            min_corner_speed = 0.003
            guidance_loss += 10.0 * torch.mean(torch.relu(min_corner_speed - torch.abs(u_corner)))
            guidance_loss += 10.0 * torch.mean(torch.relu(min_corner_speed - torch.abs(v_corner)))
        
        # Outlet region: flow should be mainly horizontal
        if 'outlet' in points_by_region and len(points_by_region['outlet']) > 0:
            outlet_points = torch.tensor(points_by_region['outlet'], dtype=torch.float32, device=device)
            outlet_pred = model(outlet_points)
            u_outlet, v_outlet = outlet_pred[:, 0], outlet_pred[:, 1]
            
            # Penalize vertical flow in outlet region
            guidance_loss += 10.0 * torch.mean(v_outlet**2)
            # Ensure rightward flow
            guidance_loss += 10.0 * torch.mean(torch.relu(-u_outlet)**2)
        
        return guidance_loss

    def compute_boundary_loss_enhanced():
        """FIXED boundary conditions - properly enforce L-shaped flow"""
        bc_loss = torch.tensor(0.0, device=device)
        
        # Inlet: STRONG vertical downward flow
        if len(inlet_points) > 0:
            inlet_tensor = torch.tensor(inlet_points, dtype=torch.float32, device=device)
            inlet_pred = model(inlet_tensor)
            u_inlet, v_inlet = inlet_pred[:, 0], inlet_pred[:, 1]
            
            # u should be exactly 0, v should be negative (downward)
            bc_loss += 100.0 * torch.mean(u_inlet**2)  # Increased weight
            
            # Force v to match PIV inlet velocity
            target_v_inlet = -0.002  # Negative for downward
            bc_loss += 100.0 * torch.mean((v_inlet - target_v_inlet)**2)
            
            # Penalize any upward flow
            bc_loss += 50.0 * torch.mean(torch.relu(v_inlet)**2)
        
        # Outlet: STRONG horizontal rightward flow  
        if len(outlet_points) > 0:
            outlet_tensor = torch.tensor(outlet_points, dtype=torch.float32, device=device)
            outlet_pred = model(outlet_tensor)
            u_outlet, v_outlet = outlet_pred[:, 0], outlet_pred[:, 1]
            
            # v should be exactly 0, u should be positive (rightward)
            bc_loss += 100.0 * torch.mean(v_outlet**2)  # Increased weight
            
            # Force u to match expected outlet velocity
            target_u_outlet = 0.009  # Based on PIV data
            bc_loss += 100.0 * torch.mean((u_outlet - target_u_outlet)**2)
            
            # Penalize any leftward flow
            bc_loss += 50.0 * torch.mean(torch.relu(-u_outlet)**2)
        
        # Walls: strict no-slip
        if len(wall_points) > 0:
            n_wall = min(200, len(wall_points))
            wall_indices = torch.randperm(len(wall_points))[:n_wall]
            wall_subset = [wall_points[i] for i in wall_indices]
            wall_tensor = torch.tensor(wall_subset, dtype=torch.float32, device=device)
            wall_pred = model(wall_tensor)
            u_wall, v_wall = wall_pred[:, 0], wall_pred[:, 1]
            
            # Much stronger penalty for wall velocities
            bc_loss += 100.0 * torch.mean(u_wall**2 + v_wall**2)
        
        return bc_loss
    
    def compute_mass_conservation_loss():
        """Enhanced mass conservation with expected velocity scales"""
        mass_loss = torch.tensor(0.0, device=device)
        
        # Velocidades esperadas basadas en el rango PIV
        expected_inlet_velocity = -0.002   # m/s (negativo = hacia abajo)
        expected_outlet_velocity = 0.002   # m/s (positivo = hacia la derecha)
        
        # 1. Conservación de masa en inlet/outlet
        if len(inlet_points) > 0 and len(outlet_points) > 0:
            inlet_tensor = torch.tensor(inlet_points, dtype=torch.float32, device=device)
            outlet_tensor = torch.tensor(outlet_points, dtype=torch.float32, device=device)
            
            inlet_pred = model(inlet_tensor)
            outlet_pred = model(outlet_tensor)
            
            v_inlet = inlet_pred[:, 1]  # componente vertical
            u_outlet = outlet_pred[:, 0]  # componente horizontal
            
            # Calcular flujos promedio
            avg_inlet_v = torch.mean(v_inlet)
            avg_outlet_u = torch.mean(u_outlet)
            
            # a) Conservación de masa tradicional
            inlet_flow = avg_inlet_v * L_up
            outlet_flow = avg_outlet_u * H_left
            mass_conservation_error = (inlet_flow + outlet_flow)**2
            
            # b) Penalizar desviaciones de velocidades esperadas
            inlet_velocity_loss = (avg_inlet_v - expected_inlet_velocity)**2
            outlet_velocity_loss = (avg_outlet_u - expected_outlet_velocity)**2
            
            # c) Penalizar variabilidad excesiva
            inlet_std = torch.std(v_inlet)
            outlet_std = torch.std(u_outlet)
            uniformity_loss = inlet_std**2 + outlet_std**2
            
            # Combinar todas las pérdidas
            mass_loss = (10.0 * mass_conservation_error + 
                        5.0 * inlet_velocity_loss + 
                        5.0 * outlet_velocity_loss + 
                        0.1 * uniformity_loss)
        
        # 2. Continuidad en el dominio (opcional pero útil)
        n_continuity_points = min(200, len(all_points))
        if n_continuity_points > 0:
            indices = np.random.choice(len(all_points), n_continuity_points, replace=False)
            continuity_points = all_points[indices]
            
            cont_tensor = torch.tensor(continuity_points, dtype=torch.float32, device=device, requires_grad=True)
            cont_pred = model(cont_tensor)
            u_cont, v_cont = cont_pred[:, 0], cont_pred[:, 1]
            
            # Calcular divergencia
            u_x = torch.autograd.grad(u_cont.sum(), cont_tensor, create_graph=True)[0][:, 0]
            v_y = torch.autograd.grad(v_cont.sum(), cont_tensor, create_graph=True)[0][:, 1]
            
            divergence = u_x + v_y
            continuity_loss = torch.mean(divergence**2)
            
            mass_loss += 0.1 * continuity_loss
        
        return mass_loss
    
    def compute_magnitude_matching_loss():
        """PIV magnitude matching - the key innovation for scaling"""
        if len(all_points) == 0:
            return torch.tensor(0.0, device=device)
        
        # Sample domain points
        n_sample = min(400, len(all_points))
        indices = torch.randperm(len(all_points))[:n_sample]
        sample_points = [all_points[i] for i in indices]
        
        sample_tensor = torch.tensor(sample_points, dtype=torch.float32, device=device)
        pred = model(sample_tensor)
        u, v = pred[:, 0], pred[:, 1]
        magnitude = torch.sqrt(u**2 + v**2)
        
        # Match PIV magnitude statistics
        mag_mean_loss = (torch.mean(magnitude) - piv_mag_mean)**2
        mag_std_loss = (torch.std(magnitude) - piv_mag_std)**2
        
        # Match component means
        u_component_loss = (torch.mean(u) - piv_u_mean)**2
        v_component_loss = (torch.mean(v) - piv_v_mean)**2
        
        return mag_mean_loss + 0.5 * mag_std_loss + u_component_loss + v_component_loss
    



    def compute_regional_component_loss(model, points_by_region, device):
        """Enforce expected velocity components in each region based on PIV"""
        component_loss = torch.tensor(0.0, device=device)
        
        # Expected values from PIV analysis
        regional_targets = {
            'inlet': {'u': 0.0, 'v': -0.002, 'u_tol': 0.0005, 'v_tol': 0.001},
            'middle': {'u': 0.0007, 'v': -0.0018, 'u_tol': 0.001, 'v_tol': 0.001},
            'corner': {'u': 0.004, 'v': -0.003, 'u_tol': 0.002, 'v_tol': 0.002},
            'outlet': {'u': 0.009, 'v': 0.0, 'u_tol': 0.003, 'v_tol': 0.0005},
            'left_wall': {'u': 0.0, 'v': -0.0002, 'u_tol': 0.0002, 'v_tol': 0.0002},
            'right_wall': {'u': 0.0002, 'v': -0.0018, 'u_tol': 0.0005, 'v_tol': 0.001}
        }
        
        for region_name, targets in regional_targets.items():
            if region_name not in points_by_region or len(points_by_region[region_name]) == 0:
                continue
                
            points = points_by_region[region_name]
            n_points = min(100, len(points))
            indices = np.random.choice(len(points), n_points, replace=False)
            sample_points = [points[i] for i in indices]
            
            points_tensor = torch.tensor(sample_points, dtype=torch.float32, device=device)
            pred = model(points_tensor)
            u_pred, v_pred = pred[:, 0], pred[:, 1]
            
            # Component-wise loss with tolerance
            u_error = torch.abs(u_pred - targets['u']) - targets['u_tol']
            v_error = torch.abs(v_pred - targets['v']) - targets['v_tol']
            
            u_loss = torch.mean(torch.relu(u_error)**2)
            v_loss = torch.mean(torch.relu(v_error)**2)
            
            # Weight by region importance
            region_weight = {
                'inlet': 5.0,
                'outlet': 5.0,
                'corner': 3.0,
                'middle': 2.0,
                'left_wall': 1.0,
                'right_wall': 1.0
            }.get(region_name, 1.0)
            
            component_loss += region_weight * (u_loss + v_loss)
        
        return component_loss

    def compute_regional_magnitude_loss(model, region_points, region_targets, device):
        region_weights = {
            'inlet': 1.0,
            'middle': 1.0,
            'corner': 3.0,
            'outlet': 5.0,
            'right_wall': 1.0,
            'left_wall': 1.0
        }
        regional_loss = torch.tensor(0.0, device=device)
        for region in ['left_wall', 'outlet']:
            points = region_points.get(region, None)
            if points is None or len(points) == 0:
                continue
            target_mag = region_targets[region]
            tensor = torch.tensor(points, dtype=torch.float32, device=device)
            pred = model(tensor)
            u, v = pred[:, 0], pred[:, 1]
            mag = torch.sqrt(u ** 2 + v ** 2)
            weight = region_weights.get(region, 1.0)
            regional_loss = regional_loss + weight * (torch.mean(mag) - target_mag) ** 2
        return regional_loss


    def compute_reynolds_constraint(model, all_points, device):
        """Asegurar Re << 1 en todo el dominio"""
        n_points = min(500, len(all_points))
        indices = np.random.choice(len(all_points), n_points, replace=False)
        sample_points = all_points[indices]
        
        sample_tensor = torch.tensor(sample_points, dtype=torch.float32, device=device, requires_grad=True)
        output = model(sample_tensor)
        u, v = output[:, 0], output[:, 1]
        
        # Calcular shear rate para viscosidad aparente
        u_x = torch.autograd.grad(u.sum(), sample_tensor, create_graph=True)[0][:, 0]
        u_y = torch.autograd.grad(u.sum(), sample_tensor, create_graph=True)[0][:, 1]
        v_x = torch.autograd.grad(v.sum(), sample_tensor, create_graph=True)[0][:, 0]
        v_y = torch.autograd.grad(v.sum(), sample_tensor, create_graph=True)[0][:, 1]
        
        shear_rate = torch.sqrt(2.0 * (u_x**2 + v_y**2 + 0.5*(u_y + v_x)**2))
        
        # Viscosidad aparente Herschel-Bulkley
        eta_app = torch.where(shear_rate > 1e-6,
                            tau_y / shear_rate + k * shear_rate**(n-1),
                            1000.0)  # Alta viscosidad cuando shear_rate → 0
        
        # Reynolds local
        velocity_mag = torch.sqrt(u**2 + v**2)
        Re_local = rho * velocity_mag * 0.02 / eta_app  # 0.02m es escala característica
        
        # Para Carbopol, Re debe ser << 1
        re_loss = torch.mean(torch.relu(Re_local - 0.01)**2)
        
        return re_loss   
    def compute_physics_loss_stable(all_points):
        """Stable physics loss computation"""
        if len(all_points) == 0:
            return torch.tensor(0.0, device=device)
        
        n_points = min(1000, len(all_points))
        indices = torch.randperm(len(all_points))[:n_points]
        selected_points = [all_points[i] for i in indices]
        
        xy = torch.tensor(selected_points, dtype=torch.float32, device=device, requires_grad=True)
        output = model(xy)
        u, v, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]

        u_x, u_y = torch.autograd.grad(u.sum(), xy, create_graph=True)[0].split(1, dim=1)
        v_x, v_y = torch.autograd.grad(v.sum(), xy, create_graph=True)[0].split(1, dim=1)
        p_x, p_y = torch.autograd.grad(p.sum(), xy, create_graph=True)[0].split(1, dim=1)

        shear_rate = torch.sqrt(2*((u_x)**2 + (v_y)**2) + (u_y + v_x)**2 + 1e-8)
        eta_eff = (tau_y / (shear_rate + 1e-8)) + k * torch.pow(shear_rate + 1e-8, n - 1)
        eta_eff = torch.clamp(eta_eff, min=0.01, max=500.0)

        continuity = u_x + v_y
        f_x = p_x - eta_eff * (u_x + u_y) + rho * g_x
        f_y = p_y - eta_eff * (v_x + v_y) + rho * g_y
        
        # AGREGAR: Penalización por velocidades altas
        velocity_magnitude = torch.sqrt(u**2 + v**2)
        high_velocity_mask = velocity_magnitude > 0.002
        velocity_penalty = torch.mean(torch.where(
            high_velocity_mask,
            1000.0 * (velocity_magnitude - 0.002)**2,
            torch.zeros_like(velocity_magnitude)
        ))
        
        return torch.mean(continuity**2) + torch.mean(f_x**2 + f_y**2) + velocity_penalty
    
###datso para entrenamiento regional###
    all_points, points_by_region, region_indices = generate_domain_points_regional(n_total=3000)

    if 'outlet' not in points_by_region or len(points_by_region['outlet']) == 0:
        print("Manually adding outlet points...")
        outlet_points = []
        for i in range(100):
            x = 0.173  # Near right edge
            y = 0.02 + (0.11 - 0.02) * np.random.rand()  # Random y in outlet range
            if inside_L(x, y):
                outlet_points.append([x, y])
        points_by_region['outlet'] = outlet_points
        print(f"Added {len(outlet_points)} outlet points")


    piv_regions = get_piv_regions()
    region_targets = {region: info['mag_mean'] for region, info in piv_regions.items()}

    middle_points = np.array(points_by_region['middle'])
    zero_middle_mask = middle_points[:, 1] < 0.04

    # Training stages
    print(f"\nEnhanced 4-Stage Training")
    print(f"Target magnitude: {piv_mag_mean:.6f} ± {piv_mag_std:.6f} m/s")
    
    # Use your existing training loop structure but with the fixed loss functions
    # Stage 1: Pattern Learning
    print(f"\nStage 1: Pattern Learning (1000 epochs)...")
    for epoch in range(1000):
        model.train()
        
        bc_loss = compute_boundary_loss_enhanced()
        total_loss = bc_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        if epoch % 250 == 0:
            print(f"[Stage 1: {epoch:04d}] BC Loss = {bc_loss.item():.6f}")
    
    # Stage 2: Add physics
    print(f"\nStage 2: Physics integration (500 epochs)...")
    optimizer.param_groups[0]['lr'] = 1e-4
    
    for epoch in range(500):
        bc_loss = compute_boundary_loss_enhanced()
        mass_loss = compute_mass_conservation_loss()
        physics_loss = compute_physics_loss_stable(all_points)
        magnitude_loss = compute_magnitude_matching_loss()
        
        total_loss = 5.0 * bc_loss + 10.0 * mass_loss + 0.01 * physics_loss + 10.0 * magnitude_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"[Stage 2: {epoch:03d}] BC={bc_loss.item():.6f}, Mass={mass_loss.item():.6f}, "
                  f"Physics={physics_loss.item():.6f}, Mag={magnitude_loss.item():.6f}")
    
    # STAGE 3: Strong regional enforcement (1000 epochs)
    print("\nStage 3: Strong regional magnitude enforcement (1000 epochs)...")
    optimizer.param_groups[0]['lr'] = 5e-5
    
    for epoch in range(1000):
        bc_loss = compute_boundary_loss_enhanced()
        regional_loss = compute_regional_magnitude_loss(model, points_by_region, region_targets, device)
        component_loss = compute_regional_component_loss(model, points_by_region, device)
        magnitude_loss = compute_magnitude_matching_loss()
        
        # Very strong regional enforcement
        total_loss = (
            1.0 * bc_loss +
            50.0 * regional_loss +  # Very high weight!
            20.0 * component_loss +
            10.0 * magnitude_loss
        )
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"[Stage 3: {epoch:04d}] Regional={regional_loss.item():.6f}, Component={component_loss.item():.6f}")
            
            # Check actual vs target
            with torch.no_grad():
                for region in ['inlet', 'outlet', 'corner']:
                    if region not in region_points:
                        continue
                    points = region_points[region]
                    if len(points) == 0:
                        continue
                    tensor = torch.tensor(points[:50], dtype=torch.float32, device=device)
                    pred = model(tensor)
                    u, v = pred[:, 0].cpu().numpy(), pred[:, 1].cpu().numpy()
                    mag = np.sqrt(u**2 + v**2)
                    target = region_targets[region]
                    print(f"   {region}: |V|={mag.mean():.4e} (target: {target:.4e}, ratio: {mag.mean()/target:.2f})")
    
    return model
def load_piv_reference_data(piv_filepath):
    """Load PIV data for reference statistics"""
    try:
        import pandas as pd
        
        with open(piv_filepath, 'r') as f:
            lines = f.readlines()
        
        header_idx = None
        for i, line in enumerate(lines):
            if 'x [m]' in line and 'y [m]' in line:
                header_idx = i
                break
        
        if header_idx is None:
            return None
        
        piv_df = pd.read_csv(piv_filepath, skiprows=header_idx)
        
        required_columns = ['x [m]', 'y [m]', 'u [m/s]', 'v [m/s]']
        if not all(col in piv_df.columns for col in required_columns):
            return None
        
        valid_mask = (
            np.isfinite(piv_df['x [m]']) & 
            np.isfinite(piv_df['y [m]']) & 
            np.isfinite(piv_df['u [m/s]']) & 
            np.isfinite(piv_df['v [m/s]'])
        )
        
        piv_clean = piv_df[valid_mask].copy()
        
        if len(piv_clean) == 0:
            return None
        
        # Handle coordinate flip if needed
        max_y = piv_clean['y [m]'].max()
        if max_y > 0.2:
            piv_clean['y [m]'] = max_y - piv_clean['y [m]']
            piv_clean['v [m/s]'] = -piv_clean['v [m/s]']
        
        u_data = piv_clean['u [m/s]'].values
        v_data = piv_clean['v [m/s]'].values
        magnitude = np.sqrt(u_data**2 + v_data**2)
        
        # Estimate mass flow
        avg_inlet_v = np.mean(v_data[v_data < 0])
        estimated_mass_flow = abs(avg_inlet_v) * 0.097 * 0.101972
        
        return {
            'u_mean': np.mean(u_data),
            'v_mean': np.mean(v_data),
            'mag_mean': np.mean(magnitude),
            'mag_std': np.std(magnitude),
            'mass_flow_rate': estimated_mass_flow,
            'n_points': len(piv_clean)
        }
        
    except Exception as e:
        print(f"Error loading PIV data: {e}")
        return None


# Compatibility alias
train_staged = train_enhanced_staged