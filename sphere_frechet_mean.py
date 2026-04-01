#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
class Manifold:
    def dist(self, x, y):
        raise NotImplementedError("Every manifold must define a distance metric.")

    def exp(self, x, v):
        raise NotImplementedError("Every manifold must define an exponential map.")
    
    def proj(self, x, v):
        raise NotImplementedError("Every manifold must define a projection map.")
    
    def log(self, x, y):
        raise NotImplementedError("Every manifold must define a logarithmic map.")
    
    

class Sphere(Manifold):
    def dist(self, x, y):
        dot_product = np.clip(np.dot(x,y), -1,1)
        return np.arccos(dot_product)
    
    def proj(self, x, v):
        radial_length = np.dot(x, v)
        radial_vector = radial_length * x
        tangent_vector = v - radial_vector
        return tangent_vector
    
    def log(self, x, y):
        distance = self.dist(x, y)
        
        if distance < 1e-15:
            return np.zeros_like(x)
        
        direction = self.proj(x,y)
        magnitude = np.linalg.norm(direction)
        normalised = direction / magnitude
        
        return normalised * distance
    
    def exp(self, x, v):
 
        magnitude = np.linalg.norm(v)
        if magnitude < 1e-15:
            return x
        normalised_v = v / magnitude 
        return (np.cos(magnitude) * x) + (np.sin(magnitude) * normalised_v)
    

    def frechet_mean(self, data_points, max_iter = 100):
        
        mu = data_points[0]
        
        for iterations in range(max_iter):
            tangent_vectors = [self.log(mu, p) for p in data_points]
            averaged_vector = np.mean(tangent_vectors, axis = 0)
            
            if np.linalg.norm(averaged_vector) < 1e-9:
                break
            
            mu = self.exp(mu, averaged_vector)
        
        return mu
        

class Cylinder(Manifold):
    def __init__(self, sphere_dim = 2):
        self.sphere_dim = sphere_dim
        
    def dist(self, x, y):
        
        x_sphere = x[:self.sphere_dim]
        y_sphere = y[:self.sphere_dim]
        
        dot_product = np.clip(np.dot(x_sphere, y_sphere),-1,1)
        
        d_flat = np.linalg.norm(x[self.sphere_dim:] - y[self.sphere_dim:])
        d_sphere = np.arccos(dot_product)
        return  np.sqrt(d_sphere ** 2 + d_flat ** 2)
        
    def proj(self, x, v):
        
        x_sphere = x[:self.sphere_dim]
        v_sphere = v[:self.sphere_dim]
        
        radial_length = np.dot(x_sphere,v_sphere)
        radial_vector = radial_length * x_sphere
        tangent_vector = v_sphere - radial_vector
        
        v_flat = v[self.sphere_dim:]
        return np.concatenate([tangent_vector, v_flat])
        
    def log(self, x, y):
        
        x_sphere = x[:self.sphere_dim]
        y_sphere = y[:self.sphere_dim]
        
        x_flat = x[self.sphere_dim:]
        y_flat = y[self.sphere_dim:]
        
        m1 = Sphere()
        sphere_tangent = m1.proj(x_sphere, y_sphere)
        sphere_magnitude = np.linalg.norm(sphere_tangent)
        if sphere_magnitude < 1e-12:
            scaled_sphere = np.zeros_like(x_sphere)
        else:
            sphere_normalised = sphere_tangent / sphere_magnitude
        
            scaled_sphere = np.arccos(np.clip(
            np.dot(x_sphere, y_sphere), -1,1)) * sphere_normalised
        

            
        flat_tangent = y_flat - x_flat
        
        return np.concatenate([scaled_sphere, flat_tangent])
    
    def exp(self, x, v):
        x_sphere = x[:self.sphere_dim]
        v_sphere = v[:self.sphere_dim]
        
        x_flat = x[self.sphere_dim:]
        v_flat = v[self.sphere_dim:]
        
        distance_sphere = np.linalg.norm(v_sphere)
        
        if distance_sphere < 1e-12:
            new_x_sphere = x_sphere
        else:
            new_x_sphere = (np.cos(distance_sphere) * x_sphere + 
                            (np.sin(distance_sphere) / distance_sphere) 
                            * v_sphere)

        new_x_flat = x_flat + v_flat
        
        return np.concatenate([new_x_sphere, new_x_flat])



