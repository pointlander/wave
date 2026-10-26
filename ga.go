// Copyright 2026 The Wave Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"runtime"
	"sort"
	"sync"
)

// genome is one candidate next state. fit = H_spatial + H_frequency,
// and lower is better. Crossover swaps whole spatial or phase genes; it
// does not average coordinates, so a lower entropy has to be selected
// rather than manufactured by blending particles together.
type genome struct {
	particles []Particle
	hs, hf    float64
	fit       float64
	evaluated bool
}

func (g genome) clone() genome {
	g.particles = cloneParticles(g.particles)
	return g
}

func mutate(cfg Config, rng *randRand, ps []Particle) {
	for i := range ps {
		ps[i].S.X += cfg.MutPos * rng.NormFloat64()
		ps[i].S.Y += cfg.MutPos * rng.NormFloat64()
		ps[i].S.VX += cfg.MutVel * rng.NormFloat64()
		ps[i].S.VY += cfg.MutVel * rng.NormFloat64()
		ps[i].F.Phase += cfg.MutPhase * rng.NormFloat64()
	}
}

func crossover(rng *randRand, a, b []Particle) []Particle {
	child := cloneParticles(a)
	for i := range child {
		if rng.Float64() < 0.5 {
			child[i].S.X = b[i].S.X
			child[i].S.Y = b[i].S.Y
		}
		if rng.Float64() < 0.5 {
			child[i].S.VX = b[i].S.VX
			child[i].S.VY = b[i].S.VY
		}
		if rng.Float64() < 0.5 {
			child[i].F.Phase = b[i].F.Phase
		}
	}
	return child
}

func finishParticles(cfg Config, ps []Particle, px, py float64) {
	for i := range ps {
		ps[i].S.X = wrap(ps[i].S.X, cfg.L)
		ps[i].S.Y = wrap(ps[i].S.Y, cfg.L)
		ps[i].F.Phase = wrapPhase(ps[i].F.Phase)
	}
	restoreMomentum(ps, px, py)
	for i := range ps {
		ps[i].SyncFrequency(cfg.Hbar)
	}
}

func tournament(rng *randRand, pop []genome, k int) int {
	if k > len(pop) {
		k = len(pop)
	}
	if k < 1 {
		k = 1
	}
	best := rng.IntN(len(pop))
	for i := 1; i < k; i++ {
		j := rng.IntN(len(pop))
		if pop[j].fit < pop[best].fit {
			best = j
		}
	}
	return best
}

func evaluatePop(cfg Config, pop []genome) {
	workers := runtime.GOMAXPROCS(0)
	if workers > len(pop) {
		workers = len(pop)
	}
	if workers < 1 {
		workers = 1
	}
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func(w int) {
			defer wg.Done()
			ev := newEvaluator(cfg)
			for i := w; i < len(pop); i += workers {
				if pop[i].evaluated {
					continue
				}
				hs, hf := ev.score(pop[i].particles)
				fit := hs + hf
				if math.IsNaN(fit) || math.IsInf(fit, 0) {
					fit = math.Inf(1)
				}
				pop[i].hs = hs
				pop[i].hf = hf
				pop[i].fit = fit
				pop[i].evaluated = true
			}
		}(w)
	}
	wg.Wait()
}

// optimize searches around base for the candidate with the lowest
// H_spatial + H_frequency. base itself is individual 0, so the result
// is never worse than that seed. Total momentum is restored to (px, py)
// on every candidate.
func optimize(cfg Config, rng *randRand, base []Particle, px, py float64) (best []Particle, hs, hf float64) {
	pop := make([]genome, cfg.PopSize)
	pop[0] = genome{particles: cloneParticles(base)}
	for i := 1; i < cfg.PopSize; i++ {
		child := cloneParticles(base)
		mutate(cfg, rng, child)
		finishParticles(cfg, child, px, py)
		pop[i] = genome{particles: child}
	}
	for gen := 0; gen < cfg.Gens; gen++ {
		evaluatePop(cfg, pop)
		sort.SliceStable(pop, func(i, j int) bool {
			return pop[i].fit < pop[j].fit
		})
		if gen == cfg.Gens-1 {
			break
		}
		next := make([]genome, cfg.PopSize)
		elite := cfg.Elite
		if elite > cfg.PopSize {
			elite = cfg.PopSize
		}
		for e := 0; e < elite; e++ {
			next[e] = pop[e].clone()
		}
		for i := elite; i < cfg.PopSize; i++ {
			a := tournament(rng, pop, cfg.Tournament)
			b := tournament(rng, pop, cfg.Tournament)
			child := crossover(rng, pop[a].particles, pop[b].particles)
			mutate(cfg, rng, child)
			finishParticles(cfg, child, px, py)
			next[i] = genome{particles: child}
		}
		pop = next
	}
	return pop[0].particles, pop[0].hs, pop[0].hf
}
