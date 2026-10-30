// Copyright 2026 The Wave Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/cmplx"
)

const packetTail = 4.5

// evaluator rasterizes the coherent wave field and scores its quantum entropy.
type evaluator struct {
	cfg     Config
	fft     fft2
	field   []complex128
	scratch []complex128
	rho     []complex128
}

func newEvaluator(cfg Config) *evaluator {
	n := cfg.Grid * cfg.Grid
	return &evaluator{
		cfg:     cfg,
		fft:     newFFT2(cfg.Grid),
		field:   make([]complex128, n),
		scratch: make([]complex128, n),
		rho:     make([]complex128, n),
	}
}

// render writes Ψ, the coherent sum of Gaussian packets.
// The envelope comes from the spatial representation. The carrier
// wave number and phase come from the frequency representation:
//
//	ψ ∝ √m (2πσ²)^{-1/2} exp(-|r-r0|²/(4σ²)) exp(i k·(r-r0) + iφ)
func (e *evaluator) render(ps []Particle) {
	clear(e.field)
	g := e.cfg.Grid
	L := e.cfg.L
	cell := L / float64(g)
	maxRad := g/2 - 1
	for i := range ps {
		p := &ps[i]
		sigma := p.S.Sigma
		if sigma <= 0 || p.S.Mass <= 0 {
			continue
		}
		amp := math.Sqrt(p.S.Mass) / (sigma * math.Sqrt(2*math.Pi))
		invFourSigma2 := 1 / (4 * sigma * sigma)
		lim2 := (packetTail * sigma) * (packetTail * sigma)
		rad := int(math.Ceil(packetTail * sigma / cell))
		if rad > maxRad {
			rad = maxRad
		}
		cx := wrapIndex(int(math.Floor(p.S.X/cell)), g)
		cy := wrapIndex(int(math.Floor(p.S.Y/cell)), g)
		for dy := -rad; dy <= rad; dy++ {
			iy := wrapIndex(cy+dy, g)
			y := (float64(iy) + 0.5) * cell
			ddy := minImage(y-p.S.Y, L)
			for dx := -rad; dx <= rad; dx++ {
				ix := wrapIndex(cx+dx, g)
				x := (float64(ix) + 0.5) * cell
				ddx := minImage(x-p.S.X, L)
				r2 := ddx*ddx + ddy*ddy
				if r2 > lim2 {
					continue
				}
				env := amp * math.Exp(-r2*invFourSigma2)
				ang := p.F.KX*ddx + p.F.KY*ddy + p.F.Phase
				e.field[iy*g+ix] += complex(env*math.Cos(ang), env*math.Sin(ang))
			}
		}
	}
}

// score returns the normalized von Neumann entanglement entropies of Ψ and
// of its FFT. Each is the entropy of the state reduced onto one axis.
func (e *evaluator) score(ps []Particle) (hs, hf float64) {
	g := e.cfg.Grid
	e.render(ps)
	hs = vonNeumann(e.field, g, e.rho)
	copy(e.scratch, e.field)
	e.fft.transform(e.scratch, false)
	hf = vonNeumann(e.scratch, g, e.rho)
	return hs, hf
}

// densityAndSpectrum returns |Ψ|² and the fftshifted spectral power
// |Ψ̂|² for display. The buffers are owned by the caller.
func (e *evaluator) densityAndSpectrum(ps []Particle) (density, spectrum []float64) {
	e.render(ps)
	g := e.cfg.Grid
	density = make([]float64, len(e.field))
	for i, c := range e.field {
		density[i] = real(c)*real(c) + imag(c)*imag(c)
	}
	copy(e.scratch, e.field)
	e.fft.transform(e.scratch, false)
	power := make([]float64, len(e.scratch))
	for i, c := range e.scratch {
		power[i] = real(c)*real(c) + imag(c)*imag(c)
	}
	return density, fftShift(power, g)
}

// vonNeumann is the entanglement entropy between the two axes of psi,
// divided by log(n). psi is row-major n by n. A product wave scores 0 and
// a maximally entangled wave scores 1. rho is an n*n scratch buffer.
// The solve runs on the calling goroutine.
func vonNeumann(psi []complex128, n int, rho []complex128) float64 {
	if n <= 1 {
		return 0
	}
	var norm float64
	for _, z := range psi {
		norm += real(z)*real(z) + imag(z)*imag(z)
	}
	if norm == 0 || math.IsNaN(norm) || math.IsInf(norm, 0) {
		return 0
	}
	inv := 1 / norm
	for y1 := 0; y1 < n; y1++ {
		row1 := y1 * n
		for y2 := 0; y2 < n; y2++ {
			row2 := y2 * n
			var re, im float64
			for x := 0; x < n; x++ {
				re1, im1 := real(psi[row1+x]), imag(psi[row1+x])
				re2, im2 := real(psi[row2+x]), imag(psi[row2+x])
				re += re1*re2 + im1*im2
				im += im1*re2 - re1*im2
			}
			rho[y1*n+y2] = complex(re*inv, im*inv)
		}
	}
	h := 0.0
	for _, lam := range hermitianEigenvalues(rho, n) {
		if lam > 1e-15 {
			h -= lam * math.Log(lam)
		}
	}
	if math.IsNaN(h) || math.IsInf(h, 0) || h < 0 {
		return 0
	}
	return h / math.Log(float64(n))
}

// hermitianEigenvalues returns the eigenvalues of the n by n Hermitian
// matrix a, overwriting a. Jacobi rotations drive the off-diagonal entries
// to zero.
func hermitianEigenvalues(a []complex128, n int) []float64 {
	const tol = 1e-14
	for sweep := 0; sweep < 40; sweep++ {
		off := 0.0
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				off += cmplx.Abs(a[p*n+q])
			}
		}
		if off <= tol {
			break
		}
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				jacobiHermitian(a, n, p, q)
			}
		}
	}
	ev := make([]float64, n)
	for i := 0; i < n; i++ {
		ev[i] = real(a[i*n+i])
	}
	return ev
}

// jacobiHermitian zeros a[p,q] with a unitary rotation in the p-q plane.
func jacobiHermitian(a []complex128, n, p, q int) {
	apq := a[p*n+q]
	mag := cmplx.Abs(apq)
	if mag < 1e-15 {
		return
	}
	// Remove the phase of a[p,q] by a diagonal unitary on q.
	ph := apq / complex(mag, 0)
	conj := cmplx.Conj(ph)
	for k := 0; k < n; k++ {
		if k == q {
			continue
		}
		a[k*n+q] *= conj
		a[q*n+k] *= ph
	}
	a[q*n+q] = complex(real(a[q*n+q]), 0)
	app := real(a[p*n+p])
	aqq := real(a[q*n+q])
	tau := (aqq - app) / (2 * mag)
	var t float64
	if tau >= 0 {
		t = 1 / (tau + math.Hypot(1, tau))
	} else {
		t = -1 / (-tau + math.Hypot(1, tau))
	}
	c := 1 / math.Hypot(1, t)
	s := t * c
	for k := 0; k < n; k++ {
		if k == p || k == q {
			continue
		}
		kp := a[k*n+p]
		kq := a[k*n+q]
		a[k*n+p] = complex(c, 0)*kp - complex(s, 0)*kq
		a[k*n+q] = complex(s, 0)*kp + complex(c, 0)*kq
		a[p*n+k] = cmplx.Conj(a[k*n+p])
		a[q*n+k] = cmplx.Conj(a[k*n+q])
	}
	a[p*n+p] = complex(app-t*mag, 0)
	a[q*n+q] = complex(aqq+t*mag, 0)
	a[p*n+q] = 0
	a[q*n+p] = 0
}

func fftShift(v []float64, n int) []float64 {
	out := make([]float64, len(v))
	h := n / 2
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			out[((y+h)%n)*n+(x+h)%n] = v[y*n+x]
		}
	}
	return out
}

func measure(cfg Config, ps []Particle) (hs, hf float64) {
	return newEvaluator(cfg).score(ps)
}

// peakBin returns the (x, y) bin of the maximum spectral power of Ψ.
func peakBin(cfg Config, ps []Particle) (x, y int, power float64) {
	e := newEvaluator(cfg)
	e.render(ps)
	copy(e.scratch, e.field)
	e.fft.transform(e.scratch, false)
	g := cfg.Grid
	best := -1.0
	for iy := 0; iy < g; iy++ {
		for ix := 0; ix < g; ix++ {
			c := e.scratch[iy*g+ix]
			p := real(c)*real(c) + imag(c)*imag(c)
			if p > best {
				best = p
				x, y = ix, iy
			}
		}
	}
	return x, y, best
}
