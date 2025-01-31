import numpy as np

class IntensityJitter:  # Variaciones en la intensidad
    def __init__(self, jitter_range_dust=69, jitter_range_no_dust=408):
        self.jitter_range_dust = jitter_range_dust
        self.jitter_range_no_dust = jitter_range_no_dust
        self.required_feature = 'intensity'

    def __call__(self, points, labels, feature_positions):
        if self.required_feature in feature_positions:
            start, end = feature_positions[self.required_feature]
            # Aplica diferentes valores de jitter según la clase
            dust_mask = labels == 0
            no_dust_mask = labels == 1
            points[dust_mask, start:end] += np.random.uniform(-self.jitter_range_dust, self.jitter_range_dust, size=(np.sum(dust_mask), 1))
            points[no_dust_mask, start:end] += np.random.uniform(-self.jitter_range_no_dust, self.jitter_range_no_dust, size=(np.sum(no_dust_mask), 1))
            points[:, start:end] = np.clip(points[:, start:end], 0, 1)  # Clipa intensidad entre 0 y 1
        return points, labels

class RandomOcclusion:  # Elimina puntos aleatoriamente (solo para polvo)
    def __init__(self, occlusion_prob_dust=0.0, occlusion_prob_no_dust=0.0):
        self.occlusion_prob_dust = occlusion_prob_dust
        self.occlusion_prob_no_dust = occlusion_prob_no_dust

    def __call__(self, points, labels, feature_positions):
        dust_mask = labels == 0
        no_dust_mask = labels == 1
        # Aplica la probabilidad de oclusión en polvo y no polvo
        dust_occlusion = np.random.uniform(0, 1, size=(np.sum(dust_mask),)) > self.occlusion_prob_dust
        no_dust_occlusion = np.random.uniform(0, 1, size=(np.sum(no_dust_mask),)) > self.occlusion_prob_no_dust
        mask = np.concatenate([dust_occlusion, no_dust_occlusion])
        return points[mask], labels[mask]

class DustDispersion:  # Ruido en la posición, solo para polvo
    def __init__(self, dispersion_level=0.231):
        self.dispersion_level = dispersion_level

    def __call__(self, points, labels, feature_positions):
        mask = labels == 0  # Aplica solo a puntos etiquetados como polvo
        points[mask, :3] += np.random.normal(0, self.dispersion_level, size=(np.sum(mask), 3))
        return points, labels

class FlowVariation:  # Variaciones en el scene flow
    def __init__(self, flow_variation_dust=0.005, flow_variation_no_dust=0.004):
        self.flow_variation_dust = flow_variation_dust
        self.flow_variation_no_dust = flow_variation_no_dust
        self.required_feature = 'flow'

    def __call__(self, points, labels, feature_positions):
        if self.required_feature in feature_positions:
            start, end = feature_positions[self.required_feature]
            dust_mask = labels == 0
            no_dust_mask = labels == 1
            # Aplica variaciones de scene flow según la clase
            points[dust_mask, start:end] += np.random.normal(0, self.flow_variation_dust, size=(np.sum(dust_mask), end - start))
            points[no_dust_mask, start:end] += np.random.normal(0, self.flow_variation_no_dust, size=(np.sum(no_dust_mask), end - start))
        return points, labels

class LocalScaling:  # Escala todos los puntos
    def __init__(self, scaling_range=(0.9, 1.1)):
        self.scaling_range = scaling_range

    def __call__(self, points, labels, feature_positions):
        scale_factor = np.random.uniform(*self.scaling_range)
        points[:, :3] *= scale_factor
        return points, labels

class RandomRotation:  # Rotación de los puntos
    def __init__(self, angle_range=(0, np.pi / 4)):
        self.angle_range = angle_range
        self.required_feature = 'flow'

    def __call__(self, points, labels, feature_positions):
        angle = np.random.uniform(*self.angle_range)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        # Aplica rotación a las coordenadas y al scene flow si está presente
        if 'coord' in feature_positions:
            start, end = feature_positions['coord']
            points[:, start:end] = points[:, start:end].dot(rotation_matrix)
        if self.required_feature in feature_positions:
            start, end = feature_positions[self.required_feature]
            points[:, start:end] = points[:, start:end].dot(rotation_matrix)
        return points, labels

class PositionJitter:
    def __init__(self, coord_jitter=0.1, uniform=False):
        self.coord_jitter = coord_jitter
        self.uniform = uniform

    def __call__(self, points, labels, feature_positions):
        # Aplica jitter en las coordenadas
        if self.uniform:
            # Usa el mismo valor de desplazamiento para todos los puntos
            jitter_value = np.random.uniform(-self.coord_jitter, self.coord_jitter)
            points[:, :3] += jitter_value
            print("Uniform jitter for all points:", jitter_value)  # Para verificar el valor aplicado
        else:
            # Aplica jitter independiente en cada punto
            points[:, :3] += np.random.uniform(-self.coord_jitter, self.coord_jitter, size=(points.shape[0], 3))
        
        return points, labels


# Transformación conjunta
class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points, labels, feature_positions):
        for transform in self.transforms:  # Revisa si son compatibles con las características que se estén usando
            if hasattr(transform, 'required_feature'):
                if transform.required_feature in feature_positions:
                    points, labels = transform(points, labels, feature_positions)
            else:
                points, labels = transform(points, labels, feature_positions)
        return points, labels
