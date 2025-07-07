import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from domain import inside_L, get_piv_regions, L_down, H_left

# Grid for mask visualization
x = np.linspace(0, L_down, 300)
y = np.linspace(0, H_left, 300)
X, Y = np.meshgrid(x, y)
mask = np.vectorize(inside_L)(X, Y)

plt.figure(figsize=(9, 7))
plt.pcolormesh(X, Y, mask, cmap='Greys', alpha=0.3, shading='auto')

# Draw L-outline for reference
domain_x = [0, 0, 0.097, 0.097, 0.174, 0.174, 0]
domain_y = [0, 0.119, 0.119, 0.019, 0.019, 0, 0]
plt.plot(domain_x, domain_y, 'k-', lw=2, label='L-domain outline')

# Overlay each region as a rectangle
regions = get_piv_regions()
colors = ['red', 'orange', 'purple', 'blue', 'green', 'brown']
for idx, (name, info) in enumerate(regions.items()):
    x0, x1 = info['x_range']
    y0, y1 = info['y_range']
    rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, 
                             linewidth=2, edgecolor=colors[idx % len(colors)],
                             facecolor='none', label=name, linestyle='--')
    plt.gca().add_patch(rect)

    # Annotate region name
    plt.text((x0 + x1)/2, (y0 + y1)/2, name, color=colors[idx % len(colors)], 
             fontsize=11, ha='center', va='center')

plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Domain mask (gray) with PIV Regions overlay')
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()
