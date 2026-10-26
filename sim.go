// Copyright 2026 The Wave Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"time"

	randv2 "math/rand/v2"
)

// randRand is the random source the simulation uses. A local alias keeps
// call sites short and the generator swappable in tests.
type randRand = randv2.Rand

// Config holds the box, the n-body integrator, and the genetic search.
type Config struct {
	N          int
	Grid       int
	L          float64
	DT         float64
	Steps      int
	PopSize    int
	Gens       int
	Elite      int
	Tournament int
	Gravity    float64
	Softening  float64
	Hbar       float64
	Sigma      float64
	Radius     float64
	MutPos     float64
	MutVel     float64
	MutPhase   float64
	Seed       int64
}

// DefaultConfig returns a short run that keeps a rotating disk of wave
// packets inside a periodic unit box.
func DefaultConfig() Config {
	return Config{
		N:          12,
		Grid:       64,
		L:          1,
		DT:         0.02,
		Steps:      24,
		PopSize:    24,
		Gens:       5,
		Elite:      2,
		Tournament: 3,
		Gravity:    0.008,
		Softening:  0.02,
		Hbar:       0.01,
		Sigma:      0.04,
		Radius:     0.18,
		MutPos:     0.01,
		MutVel:     0.02,
		MutPhase:   0.2,
		Seed:       1,
	}
}

// Validate reports a configuration that cannot be simulated.
func (c Config) Validate() error {
	if c.N < 1 {
		return fmt.Errorf("n must be at least 1")
	}
	if c.Grid < 2 || c.Grid&(c.Grid-1) != 0 {
		return fmt.Errorf("grid must be a power of two and at least 2")
	}
	if c.L <= 0 {
		return fmt.Errorf("box length must be positive")
	}
	if c.DT <= 0 {
		return fmt.Errorf("dt must be positive")
	}
	if c.Steps < 0 {
		return fmt.Errorf("steps must be non-negative")
	}
	if c.PopSize < 2 {
		return fmt.Errorf("population must be at least 2")
	}
	if c.Gens < 1 {
		return fmt.Errorf("generations must be at least 1")
	}
	if c.Elite < 1 || c.Elite >= c.PopSize {
		return fmt.Errorf("elite must be in [1, pop)")
	}
	if c.Tournament < 1 {
		return fmt.Errorf("tournament size must be at least 1")
	}
	if c.Gravity < 0 {
		return fmt.Errorf("gravity must be non-negative")
	}
	if c.Softening <= 0 {
		return fmt.Errorf("softening must be positive")
	}
	if c.Hbar <= 0 {
		return fmt.Errorf("hbar must be positive")
	}
	if c.Sigma <= 0 {
		return fmt.Errorf("sigma must be positive")
	}
	if c.Radius <= 0 || c.Radius >= c.L/2 {
		return fmt.Errorf("radius must be in (0, L/2)")
	}
	if c.MutPos < 0 || c.MutVel < 0 || c.MutPhase < 0 {
		return fmt.Errorf("mutation scales must be non-negative")
	}
	return nil
}

// StepResult is one accepted state, plus the Newtonian proposal's joint
// entropy so the genetic improvement is visible.
type StepResult struct {
	HSpatial   float64
	HFrequency float64
	HJoint     float64
	HNewton    float64
	Separation float64
	Kinetic    float64
	Potential  float64
}

// Observe measures the current state. HNewton is left at 0; a step that
// has not yet searched has no proposal to compare.
func Observe(cfg Config, ps []Particle) (StepResult, error) {
	if err := cfg.Validate(); err != nil {
		return StepResult{}, err
	}
	hs, hf := measure(cfg, ps)
	kin, pot := energies(cfg, ps)
	return StepResult{
		HSpatial:   hs,
		HFrequency: hf,
		HJoint:     hs + hf,
		Separation: meanSeparation(ps, cfg.L),
		Kinetic:    kin,
		Potential:  pot,
	}, nil
}

// Step seeds a genetic search with a Newtonian proposal and accepts the
// candidate with the lowest sum of spatial and spectral entropies.
// Momentum is held at the value of ps.
func Step(cfg Config, rng *randRand, ps []Particle) ([]Particle, StepResult, error) {
	if err := cfg.Validate(); err != nil {
		return nil, StepResult{}, err
	}
	px, py := momentum(ps)
	proposal := propose(cfg, ps)
	restoreMomentum(proposal, px, py)
	for i := range proposal {
		proposal[i].SyncFrequency(cfg.Hbar)
	}
	nHs, nHf := measure(cfg, proposal)
	best, hs, hf := optimize(cfg, rng, proposal, px, py)
	kin, pot := energies(cfg, best)
	res := StepResult{
		HSpatial:   hs,
		HFrequency: hf,
		HJoint:     hs + hf,
		HNewton:    nHs + nHf,
		Separation: meanSeparation(best, cfg.L),
		Kinetic:    kin,
		Potential:  pot,
	}
	return best, res, nil
}

func newRNG(seed int64) *randRand {
	s := uint64(seed)
	return randv2.New(randv2.NewPCG(s, s^0x9e3779b97f4a7c15))
}

// simulate emits the initial state, then advances until duration of wall-clock
// time has elapsed. A zero duration uses cfg.Steps instead. The initial frame
// is step 0 and does not count toward the step total.
func simulate(cfg Config, rng *randRand, ps []Particle, duration time.Duration, emit func(step int, simTime float64, ps []Particle, res StepResult) error) (steps int, elapsed time.Duration, first, last StepResult, err error) {
	if err = cfg.Validate(); err != nil {
		return 0, 0, StepResult{}, StepResult{}, err
	}
	first, err = Observe(cfg, ps)
	if err != nil {
		return 0, 0, StepResult{}, StepResult{}, err
	}
	first.HNewton = first.HJoint
	last = first
	if emit != nil {
		if err = emit(0, 0, ps, first); err != nil {
			return 0, 0, StepResult{}, StepResult{}, err
		}
	}
	start := time.Now()
	simTime := 0.0
	for {
		if duration > 0 {
			if time.Since(start) >= duration {
				break
			}
		} else if steps >= cfg.Steps {
			break
		}
		var res StepResult
		ps, res, err = Step(cfg, rng, ps)
		if err != nil {
			return steps, time.Since(start), first, last, err
		}
		steps++
		simTime += cfg.DT
		last = res
		if emit != nil {
			if err = emit(steps, simTime, ps, res); err != nil {
				return steps, time.Since(start), first, last, err
			}
		}
	}
	return steps, time.Since(start), first, last, nil
}
