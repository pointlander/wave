// Copyright 2026 The Wave Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"time"

	randv2 "math/rand/v2"
)

// randRand is the random source the simulation uses. A local alias keeps
// call sites short and the generator swappable in tests.
type randRand = randv2.Rand

// Config holds the box, the n-body integrator, and the Monte Carlo search.
type Config struct {
	N          int
	Grid       int
	L          float64 // kpc
	Mass       float64 // M_sun
	DT         float64 // kpc/(km/s)
	Steps      int
	Iterations int     // MCTS iterations per step
	Widen      float64 // progressive-widening coefficient
	Explore    float64 // UCB exploration constant
	Rollout    int     // mutations evaluated past each new child
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

const (
	// newtonG is Newton's gravitational constant in kpc M_sun^-1 (km/s)^2.
	newtonG = 4.3009172706e-6
	// gyrPerKpcOverKms converts the integrator time unit, kpc/(km/s), to Gyr.
	gyrPerKpcOverKms = 0.9777922216809356

	galaxyMass = 1e12 // M_sun, Milky-Way-scale halo
	boxKpc     = 3000 // 3 Mpc periodic cell

	// The dimensionless model below was tuned for a stable search. Lengths
	// scale by boxKpc and velocities by velocityScale, which is the factor
	// that makes modelG correspond to newtonG at galaxyMass.
	modelG        = 0.008
	modelDT       = 0.02
	modelHbar     = 0.01
	modelSigma    = 0.04
	modelSoft     = 0.02
	modelRadius   = 0.18
	modelMutPos   = 0.01
	modelMutVel   = 0.02
	modelMutPhase = 0.2
)

// velocityScale is the factor from the dimensionless velocity to km/s.
func velocityScale() float64 {
	return math.Sqrt(newtonG * galaxyMass / (boxKpc * modelG))
}

// DefaultConfig returns a group of galaxies in a 3 Mpc box.
// Lengths are kpc, masses are solar masses, velocities are km/s, and the
// integrator time step is in kpc/(km/s). Displayed time is Gyr.
func DefaultConfig() Config {
	v := velocityScale()
	return Config{
		N:          12,
		Grid:       64,
		L:          boxKpc,
		Mass:       galaxyMass,
		DT:         modelDT * boxKpc / v,
		Steps:      24,
		Iterations: 48,
		Widen:      1,
		Explore:    0.15,
		Rollout:    1,
		Gravity:    newtonG,
		Softening:  modelSoft * boxKpc,
		Hbar:       modelHbar * boxKpc * v,
		Sigma:      modelSigma * boxKpc,
		Radius:     modelRadius * boxKpc,
		MutPos:     modelMutPos * boxKpc,
		MutVel:     modelMutVel * v,
		MutPhase:   modelMutPhase,
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
	if c.Mass <= 0 {
		return fmt.Errorf("mass must be positive")
	}
	if c.DT <= 0 {
		return fmt.Errorf("dt must be positive")
	}
	if c.Steps < 0 {
		return fmt.Errorf("steps must be non-negative")
	}
	if c.Iterations < 0 {
		return fmt.Errorf("iterations must be non-negative")
	}
	if c.Widen <= 0 {
		return fmt.Errorf("widen must be positive")
	}
	if c.Explore < 0 {
		return fmt.Errorf("explore must be non-negative")
	}
	if c.Rollout < 0 {
		return fmt.Errorf("rollout must be non-negative")
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

// StepResult is one accepted state, plus the Newtonian proposal's fitness.
// Fitness is the frequency entropy of the merged frame. Spatial entropy is
// recorded and is not part of the score. Agree is true when that frame
// matches the Newtonian proposal. AgreeCount and AgreeRate accumulate that
// comparison over the run.
type StepResult struct {
	HSpatial   float64
	HFrequency float64
	Fitness    float64
	Newton     float64
	Separation float64
	Kinetic    float64
	Potential  float64
	Agree      bool
	AgreeCount int
	AgreeRate  float64
}

// Observe measures the current state. Newton is left at 0; a step that
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
		Fitness:    fitness(hf),
		Separation: meanSeparation(ps, cfg.L),
		Kinetic:    kin,
		Potential:  pot,
	}, nil
}

// Step roots a Monte Carlo tree search at a Newtonian proposal and merges
// the frames with the lowest score. The score is the frequency entropy plus
// a charge for leaving the proposal, so a small entropy drop does not move
// the galaxies. A worse frame does not move the result. Reported entropy is
// measured on the current frame and does not include the charge. Spatial
// entropy is not part of the score. Momentum is held at the value of ps.
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
	_, nHf := measure(cfg, proposal)
	best, hs, hf := search(cfg, rng, proposal, px, py)
	kin, pot := energies(cfg, best)
	res := StepResult{
		HSpatial:   hs,
		HFrequency: hf,
		Fitness:    fitness(hf),
		Newton:     fitness(nHf),
		Separation: meanSeparation(best, cfg.L),
		Kinetic:    kin,
		Potential:  pot,
		Agree:      mechanicsAgree(best, proposal, cfg.L),
	}
	return best, res, nil
}

// mechanicsAgree reports whether two states describe the same mechanics.
// Positions are compared on the periodic box. Phase is compared on the circle.
func mechanicsAgree(a, b []Particle, L float64) bool {
	const tol = 1e-8
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(minImage(a[i].S.X-b[i].S.X, L)) > tol || math.Abs(minImage(a[i].S.Y-b[i].S.Y, L)) > tol {
			return false
		}
		if math.Abs(a[i].S.VX-b[i].S.VX) > tol || math.Abs(a[i].S.VY-b[i].S.VY) > tol {
			return false
		}
		if phaseDelta(a[i].F.Phase, b[i].F.Phase) > tol {
			return false
		}
	}
	return true
}

func phaseDelta(a, b float64) float64 {
	d := math.Mod(a-b, 2*math.Pi)
	if d < 0 {
		d += 2 * math.Pi
	}
	if d > math.Pi {
		d = 2*math.Pi - d
	}
	return d
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
	first.Newton = first.Fitness
	last = first
	if emit != nil {
		if err = emit(0, 0, ps, first); err != nil {
			return 0, 0, StepResult{}, StepResult{}, err
		}
	}
	start := time.Now()
	simTime := 0.0
	agreed := 0
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
		if res.Agree {
			agreed++
		}
		res.AgreeCount = agreed
		res.AgreeRate = float64(agreed) / float64(steps)
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
