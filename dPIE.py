from functools import partial
import os
import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import glee2


class DPIE_GPU:
    def __init__(self, size_x, size_y, pixelsize, N, xx = None, yy =None,  IfGLad = False):
        self.piemd =  Piemd_GPU(size_x, size_y,pixelsize, N, xx, yy, IfGLad)
        self.size_x = size_x
        self.size_y = size_y
        self._r_soft = 1e-5
        self.pixelsize = pixelsize
        self.N = N
        if xx is not None and yy is not None:
            self.xx, self.yy = xx.flatten(), yy.flatten()
        elif IfGLad:
            assert size_x == size_y, "please let size_x = size_y, when using GlaD, for the reason I'm lazy to implement otherwise:)"
            self.xx, self.yy =  self.grid_glad()
        else:
            print("Error: either gives position or image size for GLaD!")


    @partial(jax.jit, static_argnums=0)
    def grid_glad(self,): # put thr galaxy centre as (0,0), rotate the galaxy such that the major axis along the x axis,
        count = jnp.arange(self.size_x) - ((self.size_x-1)/2)
        xx = (jnp.zeros((self.size_x, self.size_x)) + count)*self.pixelsize
        yy = -jnp.transpose(xx)
        xx = xx.flatten()
        yy = yy.flatten()
        return xx, yy

    @partial(jax.jit, static_argnums=0)
    def rotation(self, xx, yy, phi):
        cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
        xx_rot = cos_phi*xx + sin_phi*yy
        yy_rot = -sin_phi*xx + cos_phi*yy
        return xx_rot, yy_rot

    @partial(jax.jit, static_argnums=0)
    def normalization_glee(self, theta_E, r_core,r_tr):
        w2 = r_core**2
        s2 =r_tr**2
        theta_E2 = theta_E**2
        theta_E_scaled = theta_E2 / ( (jnp.sqrt(w2 + theta_E2) - r_core) - (jnp.sqrt(s2 + theta_E2) -r_tr) )
        return theta_E_scaled

    @partial(jax.jit, static_argnums=0)
    def _check_radii(self, w,s):
         # make sure the core radius parameters do not go below some small value for numerical stability
        w = jnp.where(w < self._r_soft, self._r_soft, w)
        w_ = jnp.where(s < w, s, w)
        s_ = jnp.where(s < w, w, s)
        return w_, s_


    @partial(jax.jit, static_argnums=0)
    def dpie_pro(self, x_centre, y_centre, q, pa, theta_E, r_core,r_tr, flags):
        r_core,r_tr = self._check_radii(r_core,r_tr)
        w2 = r_core**2
        s2 =r_tr**2
        theta_E = jnp.where(flags, self.normalization_glee(theta_E, r_core,r_tr), theta_E * s2 / (s2 - w2))
        flags_piemd = jnp.zeros_like(x_centre, dtype=bool)
        kappa_w = self.piemd.piemd_pro(x_centre, y_centre, q, pa, theta_E, r_core, flags_piemd)
        kappa_s = self.piemd.piemd_pro(x_centre, y_centre, q, pa, theta_E,r_tr, flags_piemd)
        kappa = kappa_w - kappa_s
        return kappa

    @partial(jax.jit, static_argnums=0)
    def kappa(self, *args):
        Intensity = vmap(self.dpie_pro)(*args)
        Intensity = jnp.sum(Intensity, axis = 0)
        return Intensity

    @partial(jax.jit, static_argnums=0)
    def _deflection_angle(self,x_center, y_center, q, pa, theta_E, r_core,r_tr, flags):
        r_core,r_tr = self._check_radii(r_core,r_tr)
        w2 = r_core**2
        s2 =r_tr**2
        theta_E = jnp.where(flags, self.normalization_glee(theta_E, r_core,r_tr), theta_E * s2 / (s2 - w2))
        flags_piemd = jnp.zeros_like(x_center, dtype=bool)
        alpha_w = self.piemd._deflection_angle(x_center, y_center, q, pa, theta_E, r_core, flags_piemd)
        alpha_s = self.piemd._deflection_angle(x_center, y_center, q, pa, theta_E,r_tr, flags_piemd)
        alpha_x = alpha_w[0] - alpha_s[0]
        alpha_y = alpha_w[1] - alpha_s[1]
        return alpha_x, alpha_y

    @partial(jax.jit, static_argnums=0)
    def get_deflection_angle(self, x_center, y_center, q, pa, theta_E, r_core,r_tr, flags):
        deflection_1, deflection_2  = vmap(self._deflection_angle)(x_center, y_center, q, pa, theta_E, r_core,r_tr, flags)
        deflection_1 = jnp.sum(deflection_1, axis = 0).flatten()
        deflection_2 = jnp.sum(deflection_2, axis = 0).flatten()
        return jnp.array([deflection_1, deflection_2])

   
  