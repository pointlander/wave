// Copyright 2026 The Wave Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "math"

const packetTail = 4.5

// entropyHistory is the number of earlier accepted steps mixed into a score.
const entropyHistory = 2

// entropyWindow holds the summed, per-frame normalized densities of up to
// entropyHistory earlier accepted states. Each frame sums to one, so adding
// the candidate's own normalized density and running Shannon entropy gives
// the equal-weight mixture of those steps.
type entropyWindow struct {
	spatial  []float64
	spectral []float64
	n        int
}

// evaluator rasterizes the coherent wave field and scores its entropy.
type evaluator struct {
	cfg     Config
	fft     fft2
	field   []complex128
	scratch []complex128
	values  []float64
	mix     []float64
}

func newEvaluator(cfg Config) *evaluator {
	n := cfg.Grid * cfg.Grid
	return &evaluator{
		cfg:     cfg,
		fft:     newFFT2(cfg.Grid),
		field:   make([]complex128, n),
		scratch: make([]complex128, n),
		values:  make([]float64, n),
		mix:     make([]float64, n),
	}
}

// newEntropyWindow precomputes the prior mixture from the most recent
// accepted states. history is oldest first. A nil window means the score
// uses the current frame alone.
func newEntropyWindow(cfg Config, history [][]Particle) *entropyWindow {
	if len(history) == 0 {
		return nil
	}
	if len(history) > entropyHistory {
		history = history[len(history)-entropyHistory:]
	}
	n := cfg.Grid * cfg.Grid
	w := &entropyWindow{
		spatial:  make([]float64, n),
		spectral: make([]float64, n),
	}
	ev := newEvaluator(cfg)
	for _, ps := range history {
		spatial, spectral, ok := ev.normalizedPowers(ps)
		if !ok {
			continue
		}
		for i := range spatial {
			w.spatial[i] += spatial[i]
			w.spectral[i] += spectral[i]
		}
		w.n++
	}
	if w.n == 0 {
		return nil
	}
	return w
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

// score returns the normalized Shannon entropies of |Ψ|² and |Ψ̂|².
// With a window, each entropy is the equal mixture of this state and the
// prior accepted states stored in the window.
func (e *evaluator) score(ps []Particle, w *entropyWindow) (hs, hf float64) {
	e.render(ps)
	writePower(e.values, e.field)
	n := 0
	var spatial, spectral []float64
	if w != nil {
		n = w.n
		spatial = w.spatial
		spectral = w.spectral
	}
	hs = e.mixEntropy(e.values, spatial, n)
	copy(e.scratch, e.field)
	e.fft.transform(e.scratch, false)
	writePower(e.values, e.scratch)
	hf = e.mixEntropy(e.values, spectral, n)
	return hs, hf
}

// mixEntropy is the Shannon entropy of current mixed equally with nPrior
// already-normalized frames summed in prior. current is normalized in place
// when priors are present.
func (e *evaluator) mixEntropy(current, prior []float64, nPrior int) float64 {
	if nPrior == 0 || prior == nil {
		return shannon(current)
	}
	if !normalizePositive(current) {
		return shannon(prior)
	}
	for i := range e.mix {
		e.mix[i] = prior[i] + current[i]
	}
	return shannon(e.mix)
}

func (e *evaluator) normalizedPowers(ps []Particle) (spatial, spectral []float64, ok bool) {
	e.render(ps)
	spatial = normalizedPower(e.field)
	if spatial == nil {
		return nil, nil, false
	}
	copy(e.scratch, e.field)
	e.fft.transform(e.scratch, false)
	spectral = normalizedPower(e.scratch)
	if spectral == nil {
		return nil, nil, false
	}
	return spatial, spectral, true
}

func writePower(dst []float64, field []complex128) {
	for i, c := range field {
		dst[i] = real(c)*real(c) + imag(c)*imag(c)
	}
}

func normalizedPower(field []complex128) []float64 {
	out := make([]float64, len(field))
	writePower(out, field)
	if !normalizePositive(out) {
		return nil
	}
	return out
}

// normalizePositive scales a non-negative vector to sum to one.
// Non-positive entries are cleared. It reports false when the mass is zero.
func normalizePositive(v []float64) bool {
	sum := 0.0
	for _, x := range v {
		if x > 0 {
			sum += x
		}
	}
	if sum == 0 || math.IsNaN(sum) || math.IsInf(sum, 0) {
		return false
	}
	inv := 1 / sum
	for i, x := range v {
		if x > 0 {
			v[i] = x * inv
		} else {
			v[i] = 0
		}
	}
	return true
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

// shannon is the Shannon entropy of a non-negative distribution,
// divided by log(N) so a spike scores 0 and a flat field scores 1.
func shannon(values []float64) float64 {
	if len(values) <= 1 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		if v > 0 {
			sum += v
		}
	}
	if sum == 0 || math.IsNaN(sum) {
		return 0
	}
	h := 0.0
	for _, v := range values {
		if v > 0 {
			p := v / sum
			h -= p * math.Log(p)
		}
	}
	return h / math.Log(float64(len(values)))
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
	return newEvaluator(cfg).score(ps, nil)
}

func measureWindow(cfg Config, ps []Particle, w *entropyWindow) (hs, hf float64) {
	return newEvaluator(cfg).score(ps, w)
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
