// Copyright 2026 The Wave Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "math"

// Spatial is a particle's position-space packet: a Gaussian envelope
// of width Sigma centered at (X, Y) with velocity (VX, VY).
type Spatial struct {
	X, Y   float64
	VX, VY float64
	Mass   float64
	Sigma  float64
}

// Frequency is the Fourier dual of a Gaussian packet. The spectrum is
// a Gaussian of width SigmaK centered at wavevector (KX, KY), with
// phase Phase. SyncFrequency sets K = V / ħ and SigmaK = 1 / (2σ).
type Frequency struct {
	KX, KY float64
	SigmaK float64
	Phase  float64
}

// Particle is one body in the simulation. Gravity acts on the packet
// center. The wave field used for entropy superposes the packets coherently,
// so the frequency-domain phase is visible in both domains.
type Particle struct {
	S Spatial
	F Frequency
}

// SyncFrequency refreshes the frequency representation from the spatial
// one. Phase is kept; it is the packet's internal phase, not a function
// of velocity.
func (p *Particle) SyncFrequency(hbar float64) {
	p.F.Phase = wrapPhase(p.F.Phase)
	p.F.KX = p.S.VX / hbar
	p.F.KY = p.S.VY / hbar
	p.F.SigmaK = 1 / (2 * p.S.Sigma)
}

func cloneParticles(ps []Particle) []Particle {
	out := make([]Particle, len(ps))
	copy(out, ps)
	return out
}

func wrap(x, L float64) float64 {
	x = math.Mod(x, L)
	if x < 0 {
		x += L
	}
	return x
}

func wrapPhase(p float64) float64 {
	const twoPi = 2 * math.Pi
	p = math.Mod(p, twoPi)
	if p < 0 {
		p += twoPi
	}
	return p
}

// minImage folds a separation into (-L/2, L/2] for a periodic box.
func minImage(d, L float64) float64 {
	if d > L/2 {
		d -= L
	} else if d < -L/2 {
		d += L
	}
	return d
}

func wrapIndex(i, n int) int {
	i %= n
	if i < 0 {
		i += n
	}
	return i
}

func momentum(ps []Particle) (px, py float64) {
	for i := range ps {
		px += ps[i].S.Mass * ps[i].S.VX
		py += ps[i].S.Mass * ps[i].S.VY
	}
	return px, py
}

// restoreMomentum shifts every velocity by the same amount so the total
// momentum equals (px, py). Equal-and-opposite gravity already conserves
// momentum; this puts mutations back on that constraint.
func restoreMomentum(ps []Particle, px, py float64) {
	var mx, my, mass float64
	for i := range ps {
		mx += ps[i].S.Mass * ps[i].S.VX
		my += ps[i].S.Mass * ps[i].S.VY
		mass += ps[i].S.Mass
	}
	if mass == 0 {
		return
	}
	dvx := (mx - px) / mass
	dvy := (my - py) / mass
	for i := range ps {
		ps[i].S.VX -= dvx
		ps[i].S.VY -= dvy
	}
}

// circularSpeed is the softened circular-orbit speed at radius r about
// an enclosed mass menc. The force is Plummer-softened 3D Newton gravity
// in the plane, a = G M r / (r²+ε²)^(3/2), so v² = G M r² / (r²+ε²)^(3/2).
func circularSpeed(cfg Config, r, menc float64) float64 {
	if r < 1e-8 || menc <= 0 || cfg.Gravity == 0 {
		return 0
	}
	soft := r*r + cfg.Softening*cfg.Softening
	return r * math.Sqrt(cfg.Gravity*menc) / math.Pow(soft, 0.75)
}

// Initialize places particles in a disk with tangential velocities and
// random phases, then removes the net momentum and builds each frequency
// representation.
func Initialize(cfg Config, rng *randRand) ([]Particle, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	ps := make([]Particle, cfg.N)
	cx, cy := cfg.L/2, cfg.L/2
	R := cfg.Radius
	for i := range ps {
		var x, y float64
		for {
			x = (rng.Float64()*2 - 1) * R
			y = (rng.Float64()*2 - 1) * R
			if x*x+y*y <= R*R {
				break
			}
		}
		r := math.Hypot(x, y)
		menc := float64(cfg.N) * cfg.Mass * (r * r) / (R * R)
		speed := circularSpeed(cfg, r, menc)
		tx, ty := 0.0, 0.0
		if r > 1e-8 {
			tx, ty = -y/r, x/r
		}
		ps[i].S = Spatial{
			X:     wrap(cx+x, cfg.L),
			Y:     wrap(cy+y, cfg.L),
			VX:    speed*tx + 0.05*speed*rng.NormFloat64(),
			VY:    speed*ty + 0.05*speed*rng.NormFloat64(),
			Mass:  cfg.Mass,
			Sigma: cfg.Sigma,
		}
		ps[i].F.Phase = rng.Float64() * 2 * math.Pi
	}
	restoreMomentum(ps, 0, 0)
	for i := range ps {
		ps[i].SyncFrequency(cfg.Hbar)
	}
	return ps, nil
}

// accelerations returns the softened gravitational acceleration of each
// packet center. Pairs use the periodic minimum image.
func accelerations(cfg Config, ps []Particle) (ax, ay []float64) {
	n := len(ps)
	ax = make([]float64, n)
	ay = make([]float64, n)
	if cfg.Gravity == 0 {
		return ax, ay
	}
	eps2 := cfg.Softening * cfg.Softening
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			dx := minImage(ps[j].S.X-ps[i].S.X, cfg.L)
			dy := minImage(ps[j].S.Y-ps[i].S.Y, cfg.L)
			r2 := dx*dx + dy*dy + eps2
			inv := cfg.Gravity / (r2 * math.Sqrt(r2))
			axij := dx * inv
			ayij := dy * inv
			ax[i] += axij * ps[j].S.Mass
			ay[i] += ayij * ps[j].S.Mass
			ax[j] -= axij * ps[i].S.Mass
			ay[j] -= ayij * ps[i].S.Mass
		}
	}
	return ax, ay
}

func energies(cfg Config, ps []Particle) (kinetic, potential float64) {
	for i := range ps {
		kinetic += 0.5 * ps[i].S.Mass * (ps[i].S.VX*ps[i].S.VX + ps[i].S.VY*ps[i].S.VY)
	}
	eps2 := cfg.Softening * cfg.Softening
	for i := 0; i < len(ps); i++ {
		for j := i + 1; j < len(ps); j++ {
			dx := minImage(ps[j].S.X-ps[i].S.X, cfg.L)
			dy := minImage(ps[j].S.Y-ps[i].S.Y, cfg.L)
			r := math.Sqrt(dx*dx + dy*dy + eps2)
			potential -= cfg.Gravity * ps[i].S.Mass * ps[j].S.Mass / r
		}
	}
	return kinetic, potential
}

func meanSeparation(ps []Particle, L float64) float64 {
	n := len(ps)
	if n < 2 {
		return 0
	}
	sum := 0.0
	pairs := 0.0
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			dx := minImage(ps[j].S.X-ps[i].S.X, L)
			dy := minImage(ps[j].S.Y-ps[i].S.Y, L)
			sum += math.Hypot(dx, dy)
			pairs++
		}
	}
	return sum / pairs
}

// propose advances packet centers with velocity Verlet and advances each
// packet's phase by the free-particle Schrödinger phase -E dt / ħ,
// with E = ħ² k² / 2.
func propose(cfg Config, ps []Particle) []Particle {
	next := cloneParticles(ps)
	ax, ay := accelerations(cfg, next)
	dt := cfg.DT
	for i := range next {
		next[i].S.VX += ax[i] * 0.5 * dt
		next[i].S.VY += ay[i] * 0.5 * dt
		next[i].S.X = wrap(next[i].S.X+next[i].S.VX*dt, cfg.L)
		next[i].S.Y = wrap(next[i].S.Y+next[i].S.VY*dt, cfg.L)
	}
	bx, by := accelerations(cfg, next)
	for i := range next {
		next[i].S.VX += bx[i] * 0.5 * dt
		next[i].S.VY += by[i] * 0.5 * dt
		next[i].SyncFrequency(cfg.Hbar)
		k2 := next[i].F.KX*next[i].F.KX + next[i].F.KY*next[i].F.KY
		next[i].F.Phase = wrapPhase(next[i].F.Phase - 0.5*cfg.Hbar*k2*dt)
	}
	return next
}
