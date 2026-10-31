// Copyright 2026 The Wave Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "math"

const packetTail = 4.5

// evaluator rasterizes the coherent wave field and scores the entropy of
// its quantum Fourier transform.
type evaluator struct {
	cfg     Config
	fft     fft2
	field   []complex128
	scratch []complex128
}

func newEvaluator(cfg Config) *evaluator {
	n := cfg.Grid * cfg.Grid
	return &evaluator{
		cfg:     cfg,
		fft:     newFFT2(cfg.Grid),
		field:   make([]complex128, n),
		scratch: make([]complex128, n),
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

// score returns the Shannon entropy of |Ψ|² and the Shannon entropy of the
// power after a quantum Fourier transform of the whole grid. hs is reported.
// hf is the search score. Both are divided by the log of the number of
// cells, so a single occupied bin scores 0 and a flat distribution scores 1.
func (e *evaluator) score(ps []Particle) (hs, hf float64) {
	e.render(ps)
	hs = shannon(e.field)
	copy(e.scratch, e.field)
	e.fft.transform(e.scratch, false)
	return hs, shannon(e.scratch)
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

// shannon is the Shannon entropy of the normalized power |z|², divided by
// log of the number of bins. One occupied bin scores 0 and a flat
// distribution scores 1. An empty field scores 0.
func shannon(a []complex128) float64 {
	if len(a) <= 1 {
		return 0
	}
	var norm float64
	for _, z := range a {
		norm += real(z)*real(z) + imag(z)*imag(z)
	}
	if norm == 0 || math.IsNaN(norm) || math.IsInf(norm, 0) {
		return 0
	}
	inv := 1 / norm
	h := 0.0
	for _, z := range a {
		p := (real(z)*real(z) + imag(z)*imag(z)) * inv
		if p > 0 {
			h -= p * math.Log(p)
		}
	}
	if math.IsNaN(h) || math.IsInf(h, 0) || h < 0 {
		return 0
	}
	return h / math.Log(float64(len(a)))
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
