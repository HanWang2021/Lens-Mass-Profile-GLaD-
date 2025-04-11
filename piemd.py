from functools import partial
import os
import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import glee2
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Disables JAX preallocation
jax.config.update("jax_enable_x64", True)


class Piemd_GPU: # with core radius, this object includes two types of grid, one for glee, one for glad, kappa, deflection angle, potential, hessen matrix (so gamma also calculated), one function for mge is the big sigma in Anowar (2019) paper
    def __init__(self, size_x, size_y, pixelsize, N, xx = None, yy =None,  IfGLad = False): # depth is the number of the piemd profiles in the config file, calculate them at once, no need for loop
        self.size_x = size_x # dosen't matter for glee
        self.size_y = size_y
        self.pixelsize = pixelsize
        self.N = N
        self.core_limit = 15 #arcsecond
        if xx is not None and yy is not None:
            self.xx, self.yy = xx.flatten(), yy.flatten() # give the xx and yy here to calculate the a1, a2, potential..(xx, yy)
        elif IfGLad:
            assert size_x == size_y, "please let size_x = size_y, when using GlaD, for the reason I'm lazy to implement otherwise:)"
            self.xx, self.yy =  self.grid_glad()
        #else:
            #print("Error: either gives position or image size for GLaD!")



    @partial(jax.jit, static_argnums=0)
    def rotation(self, xx, yy, phi):
        cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
        xx_rot = cos_phi*xx + sin_phi*yy
        yy_rot = -sin_phi*xx + cos_phi*yy
        return xx_rot, yy_rot


    @partial(jax.jit, static_argnums=0)
    def piemd_pro(self, x_centre, y_centre, q, pa, theta_E, w, Flags):
        xx = self.xx - x_centre
        yy = self.yy - y_centre
        xx, yy = self.rotation(xx, yy, pa)
        r2 = xx**2 + yy**2/(q*q)
        theta_E = jnp.where(Flags, theta_E**2/(jnp.sqrt(theta_E*theta_E + w * w) - w), theta_E)
        Intensity = (theta_E/(1+q))*jnp.sqrt((1/(4*w**2/(1+q)**2 + r2)))
        return Intensity


    @partial(jax.jit, static_argnums=0)
    def kappa(self, *args):  ##both glee and glad need to use this function, size_x, size_y shouldn't be used here
        # no sampling, no convolution, returns kappa
        Intensity  = vmap(self.piemd_pro, in_axes=0)(*args)
        Intensity= jnp.sum(Intensity, axis = 0) # already sum everthing up
        #Intensity = Intensity.reshape(self.size_x, self.size_y)
        return Intensity

    @partial(jax.jit, static_argnums=0)
    def _poten_q(self, x_centre, y_centre, e, pa, theta_E, w):
        # potential, given q != 1
        xx = self.xx - x_centre
        yy = self.yy - y_centre
        xx, yy = self.rotation(xx, yy, pa)
        #jax.debug.print("theta_E: {}", theta_E)
        rem = jnp.sqrt(xx*xx/((1+e)*(1+e))+yy*yy/((1-e)*(1-e)))
        rm = jnp.sqrt(xx*xx + yy*yy)
        sang = yy/rm
        cang = xx/rm
        eta = -0.5*jnp.arcsinh((2.*jnp.sqrt(e)/(1.-e)) * sang) + 0.5j * jnp.arcsin((2.*jnp.sqrt(e)/(1.+e)) * cang)
        zeta = 0.5*jnp.log((rem + jnp.sqrt(rem*rem + w*w))/w) + 0.0j
        cosheta = jnp.cosh(eta)
        coshplus = jnp.cosh(eta+zeta)
        coshminus = jnp.cosh(eta-zeta)
        Kstar = jnp.sinh(2.*eta) * jnp.log(cosheta*cosheta/(coshplus*coshminus)) + jnp.sinh(2.*zeta) * jnp.log(coshplus/coshminus)
        phi = (theta_E*w*(1.-e*e)/(2.*rem*jnp.sqrt(e))) * jnp.imag((xx - 1j*yy) * Kstar)
        return phi

    @partial(jax.jit, static_argnums=0)
    def _poten_spherical(self, x_centre, y_centre, _, pa, theta_E, w):
        # potential, given q = 1
        xx = self.xx - x_centre
        yy = self.yy - y_centre
        xx, yy = self.rotation(xx, yy, pa)
        rm = jnp.sqrt((xx)**2 + (yy)**2)
        phi = theta_E *(jnp.sqrt(rm*rm+w*w) - w*jnp.log(w+jnp.sqrt(rm*rm+w*w)) - w + w*jnp.log(2.*w) )
        return phi

    @partial(jax.jit, static_argnums=0)
    def _potential(self, x_centre, y_centre, q, pa ,theta_E, w, flags):  # flags should be also an array
        # return lens potential
        e = (1.-q)/(1.+q)
        theta_E = jnp.where(flags, theta_E**2/(jnp.sqrt(theta_E*theta_E + w * w) - w), theta_E)
        return jax.lax.cond(e!=0, self._poten_q, self._poten_spherical, x_centre, y_centre, e, pa, theta_E, w)

    @partial(jax.jit, static_argnums=0)
    def get_potential(self, x_centre, y_centre, q, pa ,theta_E, w, flags):
        return jnp.sum(vmap(self._potential)(x_centre, y_centre, q, pa ,theta_E, w, flags), axis = 0)

    # jax.lax.complex can take two arrays
    @partial(jax.jit, static_argnums=0)
    def _def_q(self, x_centre, y_centre, e, pa, E0, w):
        # return deflection_angle, given q!=1
        dx =  self.xx - x_centre # here is correct, need to keep the centre position
        dy =  self.yy - y_centre
        dx, dy = self.rotation(dx, dy, pa)
        Istar = ((-0.5j) * (1 - (e * e)) * E0 * jnp.log((((1 - e) * dx) / (1 + e) - (1j * (1 + e) * dy) / (1 - e) + (2j) * jnp.sqrt(e) * jnp.sqrt(w * w + (dx * dx) / ((1 + e) * (1 + e)) + (dy * dy) / ((1 - e) * (1 - e)))) / ((2j) * jnp.sqrt(e) * w + dx - (1j) * dy))) / jnp.sqrt(e)
        alp_x, alp_y = jnp.real(Istar), jnp.imag(Istar)
        alp_x, alp_y = self.rotation(alp_x, alp_y, -pa)
        return alp_x, alp_y

    @partial(jax.jit, static_argnums=0)
    def _def_spherical(self, x_centre, y_centre, _, pa, theta_E, w):
        # return deflection_angle, q = 1, e is not used here, passing it as input is only because of the syntax of jax.lax.cond()
        tol = 1.e-10
        xx =  self.xx - x_centre #should here be the x_centre
        yy =  self.yy - y_centre
        xx, yy = self.rotation(xx, yy, pa)
        commen_factor = jnp.sqrt(w*w + xx*xx + yy*yy)/(xx*xx+yy*yy)
        condition  = (xx*xx+yy*yy)/(w*w)
        alp_x = jnp.where(condition>=tol, theta_E*xx*((-w/(xx*xx+ yy*yy) + commen_factor)), 0.5*theta_E*(xx)/w)
        alp_y = jnp.where(condition>=tol, theta_E*yy*(-(w/(xx*xx + yy*yy))+ commen_factor),  0.5*theta_E*(yy)/w)
        alp_x, alp_y = self.rotation(alp_x, alp_y, -pa)
        return alp_x, alp_y

    @partial(jax.jit, static_argnums=0)
    def _deflection_angle(self, x_centre, y_centre, q, pa ,theta_E, w, flags): #flags should be an array here, using vmap to parallise the code
        # return deflection angle
        e = (1.-q)/(1.+q)
        theta_E = jnp.where(flags, theta_E**2/(jnp.sqrt(theta_E*theta_E + w * w) - w), theta_E)
        return jax.lax.cond(e!=0, self._def_q, self._def_spherical, x_centre, y_centre, e, pa, theta_E, w)

    @partial(jax.jit, static_argnums=0)
    def get_deflection_angle(self,  x_centre, y_centre, q, pa ,theta_E, w, flags):

        ax, ay = vmap(self._deflection_angle)(x_centre, y_centre, q, pa ,theta_E, w, flags)
        ax = jnp.sum(ax, axis = 0)
        ay = jnp.sum(ay, axis = 0)
        return jnp.array([ax, ay])

    @partial(jax.jit, static_argnums=0)
    def _hessian_spherical(self, x_centre, y_centre, _, pa ,theta_E, w):
        xx =  self.xx - x_centre #should here be the x_centre
        yy =  self.yy - y_centre
        dx, dy = self.rotation(xx, yy, pa)
        psi11 = (theta_E*(w*w*(-(dx)*(dx) + (dy)*(dy)) +(dy)*(dy)*((dx)*(dx) + (dy)*(dy)) +
		          w*((dx) - (dy))*((dx) + (dy))*jnp.sqrt(w*w + (dx)*(dx) + (dy)*(dy))))/(jnp.power((dx)*(dx) + (dy)*(dy),2)*jnp.sqrt(w*w + (dx)*(dx) + (dy)*(dy)))

        psi22 = (theta_E*(w*w*((dx) - (dy))*((dx) + (dy)) + (dx)*(dx)*((dx)*(dx) + (dy)*(dy)) + w*(-(dx)*(dx) + (dy)*(dy))* jnp.sqrt(w*w + (dx)*(dx) + (dy)*(dy))))/(jnp.power((dx)*(dx) + (dy)*(dy),2)*jnp.sqrt(w*w + (dx)*(dx) + (dy)*(dy)))
        psi12 = -((theta_E*(dx)*(dy)*((dx)*(dx) + (dy)*(dy) + 2*w*(w - jnp.sqrt(w*w + (dx)*(dx) + (dy)*(dy)))))/(jnp.power((dx)*(dx) + (dy)*(dy),2)*jnp.sqrt(w*w + (dx)*(dx) + (dy)*(dy))))
        return psi11, psi22, psi12

    @partial(jax.jit, static_argnums=0)
    def _hessian_q(self, x_centre, y_centre, e, pa ,theta_E, w):
        xx =  self.xx - x_centre #should here be the x_centre
        yy =  self.yy - y_centre
        dx, dy = self.rotation(xx, yy, pa)

        psi11= jnp.imag(((1 - (e*e))*theta_E*((2.j)*jnp.sqrt(e)*w + (dx) - (1.j)*(dy))*
		            (((1 - e)/(1 + e) + ((2.j)*jnp.sqrt(e)*(dx))/
		             (((1+e)*(1+e))*jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
		                    (dy)*(dy)/((1-e)*(1-e)))))/
		            ((2.j)*jnp.sqrt(e)*w + (dx) - (1.j)*(dy)) -
		             (((1 - e)*(dx))/(1 + e) - ((1.j)*(1 + e)*(dy))/(1 - e) +
		             (2.j)*jnp.sqrt(e)*
	               jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
		                  (dy)*(dy)/((1-e)*(1-e))))/
	              jnp.power((2.j)*jnp.sqrt(e)*w + (dx) - (1.j)*(dy),2)))/
		          (jnp.sqrt(e)*(((1 - e)*(dx))/(1 + e) -
		                    ((1.j)*(1 + e)*(dy))/(1 - e) +
		                    (2.j)*jnp.sqrt(e)*
		                  jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
	                          (dy)*(dy)/((1-e)*(1-e))))))/2.

        psi22=-jnp.real(((1 - (e*e))*theta_E*(2.j*jnp.sqrt(e)*w + (dx) - 1.j*(dy))*
		              ((((-1.j)*(1 + e))/(1 - e) +
		                (2.j*jnp.sqrt(e)*(dy))/
		               (((1-e)*(1-e))*
		                jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
		                      (dy)*(dy)/((1-e)*(1-e)))))/
		            (2.j*jnp.sqrt(e)*w + (dx) - 1.j*(dy)) +
	                 (1.j*(((1 - e)*(dx))/(1 + e) -
	                  (1.j*(1 + e)*(dy))/(1 - e) + 2.j*jnp.sqrt(e)*
		                   jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
	                        (dy)*(dy)/((1-e)*(1-e)))))/
	              jnp.power(2.j*jnp.sqrt(e)*w + (dx) - 1.j*(dy),2)))/
		          (jnp.sqrt(e)*(((1 - e)*(dx))/(1 + e) -
		                      (1.j*(1 + e)*(dy))/(1 - e) +
	                      2.j*jnp.sqrt(e)*jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
		                           (dy)*(dy)/((1-e)*(1-e))))))/2.

        psi12=-jnp.real(((1 - (e*e))*theta_E*(2.j*jnp.sqrt(e)*w + (dx) - 1.j*(dy))*
		             (((1 - e)/(1 + e) + (2.j*jnp.sqrt(e)*(dx))/
		                (((1+e)*(1+e))*jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
		                     (dy)*(dy)/((1-e)*(1-e)))))/
		              (2.j*jnp.sqrt(e)*w + (dx) - 1.j*(dy)) -
		              (((1 - e)*(dx))/(1 + e) - (1.j*(1 + e)*(dy))/(1 - e) +
	               2.j*jnp.sqrt(e)*jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
		                    (dy)*(dy)/((1-e)*(1-e))))/
		             jnp.power(2.j*jnp.sqrt(e)*w + (dx) - 1.j*(dy),2)))/
		 	   (jnp.sqrt(e)*(((1 - e)*(dx))/(1 + e) -
	                       (1.j*(1 + e)*(dy))/(1 - e) +
	                       2.j*jnp.sqrt(e)*jnp.sqrt(w*w + (dx)*(dx)/((1+e)*(1+e)) +
	                        (dy)*(dy)/((1-e)*(1-e))))))/2.
        return psi11, psi22, psi12

    @partial(jax.jit, static_argnums=0)
    def _hessian(self, x_centre, y_centre, q, pa ,theta_E, w, flags):
        e = (1.-q)/(1.+q)
        theta_E = jnp.where(flags, theta_E**2/(jnp.sqrt(theta_E*theta_E + w * w) - w), theta_E)
        return jax.lax.cond(e!=0, self._hessian_q, self. _hessian_spherical, x_centre, y_centre, e, pa, theta_E, w)

    @partial(jax.jit, static_argnums=0)
    def get_hessian(self, x_centre, y_centre, q, pa ,theta_E, w, flags):
        psi11, psi22, psi12 = vmap(self._hessian)( x_centre, y_centre, q, pa ,theta_E, w, flags)
        psi11 = jnp.sum(psi11, axis = 0)
        psi22 = jnp.sum(psi22, axis = 0)
        psi12 = jnp.sum(psi12, axis = 0)
        return psi11, psi22, psi12

   
