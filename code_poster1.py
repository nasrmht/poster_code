import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import ListedColormap

# Configuration des plots
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'text.usetex': False,  # Mettre True si vous avez LaTeX installé
})

# Créer une figure principale avec une grille pour plusieurs subplots
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.5], height_ratios=[1, 1])

# Style commun
background_color = '#f8f9fa'
arrow_props = dict(arrowstyle='-|>', lw=2, mutation_scale=20, color='r')
text_props = dict(ha='center', va='center', fontsize=14)

# 1. Visualisation de l'espace des paramètres d'entrée (espace de faible dimension)
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax1.set_title('Espace des paramètres de simulation', fontweight='bold')
ax1.set_xlabel(r'$\theta_1$')
ax1.set_ylabel(r'$\theta_2$')
ax1.set_zlabel(r'$\theta_3$')

# Générer quelques points pour représenter des paramètres d'entrée
np.random.seed(42)
n_samples = 10
theta = np.random.uniform(-1, 1, size=(n_samples, 3))
ax1.scatter(theta[:, 0], theta[:, 1], theta[:, 2], c='blue', marker='o', s=80, label='Points d\'échantillonnage')

# Ajouter une sphère semi-transparente pour délimiter l'espace des paramètres
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
x = 1.1 * np.outer(np.cos(u), np.sin(v))
y = 1.1 * np.outer(np.sin(u), np.sin(v))
z = 1.1 * np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_surface(x, y, z, color='gray', alpha=0.1)

# Mettre en évidence un point spécifique comme exemple
highlighted_point = theta[0]
ax1.scatter([highlighted_point[0]], [highlighted_point[1]], [highlighted_point[2]], 
           color='red', s=200, edgecolor='k', marker='o', label='Point sélectionné')

ax1.view_init(elev=30, azim=45)
ax1.grid(False)#True, alpha=0.00001)

# 2. Visualisation de l'espace de sortie à haute dimension (champs physiques)
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_title('Espace de sortie (champs physiques)', fontweight='bold')

# Simuler un champ 2D comme exemple
x_grid = np.linspace(0, 1, 20)
y_grid = np.linspace(0, 1, 20)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# Créer quelques champs différents pour illustrer la haute dimensionnalité
fields = []
for i in range(6):
    # Générer des champs différents selon les paramètres
    sigma = 0.1 + 0.05 * i
    center_x = 0.3 + 0.1 * (i % 3)
    center_y = 0.3 + 0.1 * (i // 3)
    
    field = np.exp(-((X_grid - center_x)**2 + (Y_grid - center_y)**2) / (2*sigma**2))
    fields.append(field)

# Afficher un arrangement de plusieurs champs pour montrer la haute dimensionnalité
grid_size = 2
field_size = 2.5
spacing = 0.2

for i in range(min(6, len(fields))):
    row, col = i // 3, i % 3
    pos_x = col * (field_size + spacing)
    pos_y = row * (field_size + spacing)
    
    # Ajouter un cadre pour chaque champ
    rect = Rectangle((pos_x, pos_y), field_size, field_size, 
                     edgecolor='gray', facecolor='none', linewidth=1, alpha=0.7)
    ax2.add_patch(rect)
    
    # Afficher le champ
    ax2.imshow(fields[i], extent=[pos_x, pos_x+field_size, pos_y, pos_y+field_size], 
               cmap='viridis', interpolation='nearest', origin='lower')
    
    # Ajouter un label
    ax2.text(pos_x + field_size/2, pos_y + field_size/2, f'Champ {i+1}', color='white', **text_props)

# Ajouter un point de référence et une flèche pour le champ sélectionné
field_idx = 0
selected_field_pos = (field_idx % 3 * (field_size + spacing) + field_size/2, 
                     field_idx // 3 * (field_size + spacing) + field_size/2)
ax2.scatter([selected_field_pos[0]], [selected_field_pos[1]], s=200, facecolor='none', 
           edgecolor='red', linewidth=2, zorder=10)

ax2.set_xlim(-0.5, 3 * (field_size + spacing) - spacing + 0.5)
ax2.set_ylim(-0.5, 2 * (field_size + spacing) - spacing + 0.5)
ax2.axis('off')

# 3. Flèche de métamodèle direct (entrée -> sortie)
ax_arrow1 = fig.add_subplot(gs[0, 1])
ax_arrow1.set_title('Métamodèle direct', fontweight='bold')
ax_arrow1.annotate('', xy=(1, 0.5), xytext=(0, 0.5), 
                  arrowprops=dict(arrowstyle='-|>', lw=3, color='red', shrinkA=0, shrinkB=0))
ax_arrow1.annotate('$\mathcal{M}(\\theta)$', xy=(0.5, 0.6), color='red', fontsize=16)
ax_arrow1.axis('off')

# 4. Visualisation de l'espace latent (après ACP)
ax3 = fig.add_subplot(gs[1, 0], projection='3d')
ax3.set_title('Espace latent (après ACP)', fontweight='bold')
ax3.set_xlabel('Composante 1')
ax3.set_ylabel('Composante 2')
ax3.set_zlabel('Composante 3')

# Générer des points dans l'espace latent
latent_points = np.random.multivariate_normal(
    mean=[0, 0, 0], 
    cov=np.diag([0.8, 0.4, 0.2]), 
    size=n_samples
)

# Tracer les points dans l'espace latent
ax3.scatter(latent_points[:, 0], latent_points[:, 1], latent_points[:, 2], c='blue', marker='o', s=80)

# Point latent correspondant au champ sélectionné
highlighted_latent = latent_points[0]
ax3.scatter([highlighted_latent[0]], [highlighted_latent[1]], [highlighted_latent[2]], 
           color='red', s=200, edgecolor='k', marker='o')

# Ajouter les axes principaux pour illustrer l'ACP
for i, (color, length) in enumerate(zip(['r', 'g', 'b'], [1.0, 0.7, 0.5])):
    vector = np.zeros(3)
    vector[i] = length
    ax3.quiver(0, 0, 0, *vector, color=color, arrow_length_ratio=0.1, lw=2)

ax3.view_init(elev=30, azim=45)
ax3.grid(False) #, alpha=0.0003)

# 5. Représentation du processus complet avec flèches
ax_workflow = fig.add_subplot(gs[1, 1:])
ax_workflow.set_title('Workflow de métamodélisation avec ACP', fontweight='bold')

# Éléments du workflow
elements = {
    'input': {'pos': (0.1, 0.5), 'label': 'Espace des paramètres\n$\\theta \\in \\mathbb{R}^d$', 'radius': 0.1},
    'latent': {'pos': (0.5, 0.8), 'label': 'Espace latent\n(composantes ACP)', 'radius': 0.1},
    'output': {'pos': (0.9, 0.5), 'label': 'Espace des champs\n$\\mathcal{Y} \\in \\mathbb{R}^D$ (D>>d)', 'radius': 0.1},
}

# Dessiner les ellipses pour chaque espace
for key, element in elements.items():
    if key == 'latent':
        ellipse = Ellipse(element['pos'], width=0.2, height=0.15, 
                        edgecolor='blue', facecolor='lightgrey', alpha=0.13)
    elif key == 'input':
        ellipse = Ellipse(element['pos'], width=0.15, height=0.25, 
                        edgecolor='green', facecolor='lightgrey', alpha=0.13)
    else:  # output
        ellipse = Ellipse(element['pos'], width=0.2, height=0.25, 
                        edgecolor='purple', facecolor='lavender', alpha=0.13)
    
    ax_workflow.add_patch(ellipse)
    ax_workflow.text(element['pos'][0], element['pos'][1], element['label'], 
                    ha='center', va='center', fontsize=12, fontweight='bold')

# Flèches du workflow
# Chemin direct
direct_arrow = FancyArrowPatch(elements['input']['pos'], elements['output']['pos'], 
                              connectionstyle="arc3,rad=0.0", **arrow_props)
ax_workflow.add_patch(direct_arrow)
ax_workflow.text(0.5, 0.4, '$\\mathcal{M}$ direct (coûteux)', fontsize=12, color='r')

# Chemin avec ACP
to_latent_arrow = FancyArrowPatch(elements['input']['pos'], elements['latent']['pos'], 
                                 connectionstyle="arc3,rad=0.3", **arrow_props)
from_latent_arrow = FancyArrowPatch(elements['latent']['pos'], elements['output']['pos'], 
                                   connectionstyle="arc3,rad=0.3", **arrow_props)
ax_workflow.add_patch(to_latent_arrow)
ax_workflow.add_patch(from_latent_arrow)

# Ajouter les labels des chemins
ax_workflow.text(0.25, 0.75, 'GP + ACP', fontsize=12, color='r', rotation=45)
ax_workflow.text(0.75, 0.75, 'Reconstruction', fontsize=12, color='r', rotation=-45)

# Ajouter des équations
ax_workflow.text(0.5, 0.2, r'$\mathcal{M}({\theta}) \approx \mathcal{R}(\mathcal{GP}(\mathcal{P}({\theta})))$', 
                fontsize=16, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

# Légende des équations
ax_workflow.text(0.3, 0.1, r'$\mathcal{P}$: Projection ACP', fontsize=12)
ax_workflow.text(0.5, 0.05, r'$\mathcal{GP}$: Processus Gaussien', fontsize=12)
ax_workflow.text(0.7, 0.1, r'$\mathcal{R}$: Reconstruction', fontsize=12)

ax_workflow.set_xlim(0, 1)
ax_workflow.set_ylim(0, 1)
ax_workflow.axis('off')

plt.tight_layout()
plt.savefig('mogp_pca_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Deuxième figure pour illustrer la métamodélisation des composantes principales
fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle('Métamodélisation par GP des composantes principales', fontweight='bold', fontsize=16)

# Définir des données d'exemple pour les GP
X_train = np.sort(np.random.uniform(0, 1, 10)).reshape(-1, 1)
y_train_pc1 = np.sin(X_train * 6) + 0.1 * np.random.randn(10, 1)
y_train_pc2 = np.cos(X_train * 6) + 0.1 * np.random.randn(10, 1)

X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_test_pc1 = np.sin(X_test * 6)
y_test_pc2 = np.cos(X_test * 6)

# Ajouter de l'incertitude pour le GP
sigma1 = 0.2 * np.ones_like(X_test) * (1 + 0.5 * np.sin(X_test * 3 * np.pi))
sigma2 = 0.15 * np.ones_like(X_test) * (1 + 0.5 * np.cos(X_test * 3 * np.pi))

# 1. GP pour la première composante principale
ax = axes[0, 0]
ax.set_title('GP pour Composante Principale 1')
ax.plot(X_test, y_test_pc1, 'b-', lw=1, label='Fonction vraie')
ax.plot(X_test, y_test_pc1, 'r--', lw=2, label='Prédiction GP')
ax.fill_between(X_test.ravel(), 
                (y_test_pc1 - 1.96 * sigma1).ravel(), 
                (y_test_pc1 + 1.96 * sigma1).ravel(), 
                alpha=0.2, color='grey', label='Intervalle de confiance')
ax.scatter(X_train, y_train_pc1, c='k', s=50, label='Points d\'échantillonnage')
ax.set_xlabel('Paramètre $\\theta$')
ax.set_ylabel('Valeur CP1')
ax.grid(False) #, alpha=0.3)
ax.legend()

# 2. GP pour la deuxième composante principale
ax = axes[0, 1]
ax.set_title('GP pour Composante Principale 2')
ax.plot(X_test, y_test_pc2, 'b-', lw=1, label='Fonction vraie')
ax.plot(X_test, y_test_pc2, 'r--', lw=2, label='Prédiction GP')
ax.fill_between(X_test.ravel(), 
                (y_test_pc2 - 1.96 * sigma2).ravel(), 
                (y_test_pc2 + 1.96 * sigma2).ravel(), 
                alpha=0.2, color='grey', label='Intervalle de confiance')
ax.scatter(X_train, y_train_pc2, c='k', s=50, label='Points d\'échantillonnage')
ax.set_xlabel('Paramètre $\\theta$')
ax.set_ylabel('Valeur CP2')
ax.grid(False) #, alpha=0.3)
ax.legend()

# 3. Visualisation 2D des prédictions de GP dans l'espace latent
ax = axes[1, 0]
ax.set_title('Prédictions dans l\'espace latent (CP1 vs CP2)')
ax.scatter(y_train_pc1, y_train_pc2, c='k', s=80, label='Points d\'échantillonnage')

# Tracer quelques prédictions avec leurs incertitudes
for i in range(0, len(X_test), 10):
    ax.plot(y_test_pc1[i], y_test_pc2[i], 'ro', alpha=0.6)
    # Ellipse d'incertitude
    ellipse = Ellipse((y_test_pc1[i][0], y_test_pc2[i][0]), 
                      width=sigma1[i][0]*3.92, height=sigma2[i][0]*3.92,
                      alpha=0.1, color='grey', edgecolor='r')
    ax.add_patch(ellipse)

# Ajouter une trajectoire de prédiction
ax.plot(y_test_pc1, y_test_pc2, 'r--', lw=1, alpha=0.5, label='Trajectoire de prédiction')
ax.set_xlabel('Composante Principale 1')
ax.set_ylabel('Composante Principale 2')
ax.grid(True, alpha=0.3)
ax.legend()

# 4. Visualisation de la reconstruction des champs
ax = axes[1, 1]
ax.set_title('Reconstruction des champs physiques')

# Simuler une grille régulière pour un champ
nx, ny = 20, 20
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# Générer un champ de base
base_field = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.1)

# Reconstruire plusieurs variantes pour illustrer l'effet des composantes principales
fields = []
n_fields = 4
grid_size = 2

for i in range(n_fields):
    # Simuler l'effet de différentes composantes principales
    scale1 = 0.5 + i * 0.25
    scale2 = 0.5 + (n_fields - i) * 0.15
    shift_x = 0.1 * np.sin(i * np.pi / 2)
    shift_y = 0.1 * np.cos(i * np.pi / 2)
    
    field = scale1 * np.exp(-((X - (0.5 + shift_x))**2 + (Y - (0.5 + shift_y))**2) / (0.1 * scale2))
    fields.append(field)

# Afficher les champs reconstruits
for i, field in enumerate(fields):
    row, col = i // 2, i % 2
    pos_x = col * 0.5
    pos_y = row * 0.5
    
    ax.imshow(field, extent=[pos_x, pos_x+0.45, pos_y, pos_y+0.45], 
              cmap='viridis', origin='lower', interpolation='bilinear')
    
    # Ajouter un label
    ax.text(pos_x + 0.45/2, pos_y + 0.45/2, f'Variante {i+1}', color='white', fontsize=10, ha='center')

ax.axis('off')

plt.tight_layout()
plt.savefig('mogp_pca_components.png', dpi=300, bbox_inches='tight')
plt.show()

# Troisième figure : Visualisation 3D du métamodèle avec incertitude
fig3 = plt.figure(figsize=(15, 10))
gs3 = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

# 1. Représentation 3D d'un champ physique
ax_3d = fig3.add_subplot(gs3[0], projection='3d')
ax_3d.set_title('Visualisation 3D d\'un champ physique', fontweight='bold')

# Créer une grille 3D plus fine pour le champ
nx, ny = 30, 30
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# Générer un champ avec une structure plus complexe
Z = 0.7 * np.exp(-((X - 0.3)**2 + (Y - 0.3)**2) / 0.05) + \
    0.5 * np.exp(-((X - 0.7)**2 + (Y - 0.7)**2) / 0.1) + \
    0.2 * np.sin(X * 6 * np.pi) * np.cos(Y * 4 * np.pi)

# Créer la surface 3D
surf = ax_3d.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

# Ajouter un point de référence dans l'espace des paramètres
param_point = np.array([0.5, 0.5])
field_value = Z[15, 15]  # Valeur du champ au point milieu
ax_3d.scatter([param_point[0]], [param_point[1]], [field_value], color='red', s=100, edgecolor='k')

# Ajouter quelques lignes de niveau sur le sol pour mieux visualiser
ax_3d.contour(X, Y, Z, zdir='z', offset=-0.2, cmap='viridis', alpha=0.5)

ax_3d.set_xlabel('Coordonnée X')
ax_3d.set_ylabel('Coordonnée Y')
ax_3d.set_zlabel('Valeur du champ')
ax_3d.set_zlim(-0.2, 1.2)

# Ajouter une barre de couleur
cbar = fig3.colorbar(surf, ax=ax_3d, shrink=0.7, aspect=10)
cbar.set_label('Intensité')

# 2. Visualisation de l'incertitude du métamodèle
ax_uncertainty = fig3.add_subplot(gs3[1])
ax_uncertainty.set_title('Incertitude du métamodèle dans l\'espace physique', fontweight='bold')

# Générer un champ d'incertitude
uncertainty = 0.05 + 0.2 * np.exp(-((X - 0.6)**2 + (Y - 0.4)**2) / 0.15)

# Points d'échantillonnage
n_samples = 12
sample_x = np.random.uniform(0, 1, n_samples)
sample_y = np.random.uniform(0, 1, n_samples)

# Tracer le champ d'incertitude
contour = ax_uncertainty.contourf(X, Y, uncertainty, 20, cmap='YlOrRd')
ax_uncertainty.scatter(sample_x, sample_y, color='black', s=80, marker='o', label='Points d\'échantillonnage')

# Ajouter des annotations pour les zones
ax_uncertainty.annotate('Zone de haute\nincertitude', xy=(0.6, 0.4), xytext=(0.7, 0.5),
                      arrowprops=dict(arrowstyle='->'), fontsize=12)
ax_uncertainty.annotate('Zone de basse\nincertitude', xy=(0.2, 0.7), xytext=(0.1, 0.8),
                      arrowprops=dict(arrowstyle='->'), fontsize=12)

# Ajouter une barre de couleur
cbar = fig3.colorbar(contour, ax=ax_uncertainty)
cbar.set_label('Écart-type de prédiction')

ax_uncertainty.set_xlabel('Coordonnée X')
ax_uncertainty.set_ylabel('Coordonnée Y')
ax_uncertainty.legend()
ax_uncertainty.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mogp_uncertainty.png', dpi=300, bbox_inches='tight')
plt.show()