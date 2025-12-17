# Compton sampler convention (locked)

We will store Compton inverse-CDF tables as **cos(theta)**.

## `.h5` path
`/samplers/photon/compton/inv_cdf` with shape `[E, K]`

## Attribute
`/samplers/photon/compton` group attribute:
- `convention = "cos_theta"`

## Sampling
Given `u ~ U(0,1)`:
- `t = u*(K-1)`
- `i0 = floor(t)`, `f = t - i0`
- `cosθ = lerp(inv_cdf[Ebin, i0], inv_cdf[Ebin, i0+1], f)`

## Kinematics
\[
E' = \frac{E}{1 + (E/m_ec^2)(1-\cos\theta)}
\quad , \quad
T = E - E'
\]
Recoil electron kinetic energy `T` is spawned as an electron particle (or deposited locally if below cutoff).
Scattered photon continues with energy `E'` and rotated direction.