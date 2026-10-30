// Copyright 2026 The Wave Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "math"

// rolloutBatch is how many mutations one rollout step scores. The step
// continues from the lowest score, and every sample remains a candidate.
// Tree expansion still adds one mutation.
const rolloutBatch = 4

// mctsNode is one mutated state in the search tree. mean is the average
// simulation return seen through this node. A simulation's return is the
// lowest fitness reached while evaluating the new child and its rollout,
// and lower is better.
type mctsNode struct {
	parent    *mctsNode
	particles []Particle
	children  []*mctsNode
	visits    int
	valueSum  float64
	mean      float64
	hs, hf    float64
	fit       float64
}

// departure is the mean squared distance from ref, in units of the mutation
// scales. The Newtonian proposal has departure 0 from itself.
func departure(cfg Config, ps, ref []Particle) float64 {
	if len(ps) == 0 || len(ps) != len(ref) {
		return 0
	}
	var sum, n float64
	for i := range ps {
		if cfg.MutPos > 0 {
			dx := minImage(ps[i].S.X-ref[i].S.X, cfg.L) / cfg.MutPos
			dy := minImage(ps[i].S.Y-ref[i].S.Y, cfg.L) / cfg.MutPos
			sum += dx*dx + dy*dy
			n += 2
		}
		if cfg.MutVel > 0 {
			dvx := (ps[i].S.VX - ref[i].S.VX) / cfg.MutVel
			dvy := (ps[i].S.VY - ref[i].S.VY) / cfg.MutVel
			sum += dvx*dvx + dvy*dvy
			n += 2
		}
		if cfg.MutPhase > 0 {
			dp := phaseDelta(ps[i].F.Phase, ref[i].F.Phase) / cfg.MutPhase
			sum += dp * dp
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / n
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

// maxChildren is progressive widening: a node may grow about
// widen * sqrt(visits) children. That keeps a continuous mutation
// space from adding a new child on every visit.
func maxChildren(widen float64, visits int) int {
	if visits < 1 {
		return 1
	}
	m := int(math.Ceil(widen * math.Sqrt(float64(visits))))
	if m < 1 {
		return 1
	}
	return m
}

// ucb is UCB1 arranged so a lower value is a better choice.
// Unvisited children score -Inf. A non-finite mean is never preferred
// over a finite one.
func ucb(cfg Config, parent, child *mctsNode) float64 {
	if child.visits < 1 {
		return math.Inf(-1)
	}
	if math.IsInf(child.mean, 0) || math.IsNaN(child.mean) {
		return math.Inf(1)
	}
	bonus := cfg.Explore * math.Sqrt(math.Log(float64(parent.visits))/float64(child.visits))
	return child.mean - bonus
}

func selectChild(cfg Config, n *mctsNode) *mctsNode {
	best := n.children[0]
	bestU := ucb(cfg, n, best)
	for _, c := range n.children[1:] {
		u := ucb(cfg, n, c)
		if u < bestU {
			best = c
			bestU = u
		}
	}
	return best
}

// scoredState is one frame the search evaluated, with its current-frame fitness.
type scoredState struct {
	particles []Particle
	fit       float64
}

// search runs Monte Carlo tree search rooted at the Newtonian proposal.
// The returned state is the single best-scoring frame. The score is the
// frequency entropy plus a charge for leaving the Newtonian proposal.
// Reported entropies are measured on that frame and do not include the
// charge. Spatial entropy is not part of the score.
func search(cfg Config, rng *randRand, base []Particle, px, py float64) (best []Particle, hs, hf float64) {
	_, states := runMCTS(cfg, rng, base, px, py)
	best = bestState(states)
	hs, hf = newEvaluator(cfg).score(best)
	return best, hs, hf
}

// runMCTS returns the tree and every state it scored, root first.
func runMCTS(cfg Config, rng *randRand, base []Particle, px, py float64) (root *mctsNode, states []scoredState) {
	ev := newEvaluator(cfg)
	root = &mctsNode{particles: cloneParticles(base)}
	root.hs, root.hf = ev.score(root.particles)
	ref := root.particles
	root.fit = stateScore(cfg, root.hf, root.particles, ref)
	root.visits = 1
	root.valueSum = root.fit
	root.mean = root.fit
	states = append(states, scoredState{particles: root.particles, fit: root.fit})

	expand := func(parent *mctsNode) *mctsNode {
		childP := cloneParticles(parent.particles)
		mutate(cfg, rng, childP)
		finishParticles(cfg, childP, px, py)
		chs, chf := ev.score(childP)
		cfit := stateScore(cfg, chf, childP, ref)
		child := &mctsNode{
			parent:    parent,
			particles: childP,
			hs:        chs,
			hf:        chf,
			fit:       cfit,
		}
		states = append(states, scoredState{particles: childP, fit: cfit})
		parent.children = append(parent.children, child)
		return child
	}

	for i := 0; i < cfg.Iterations; i++ {
		node := root
		for len(node.children) >= maxChildren(cfg.Widen, node.visits) && len(node.children) > 0 {
			node = selectChild(cfg, node)
		}
		// Expand the selected node, then one ply further, so the tree
		// keeps the next mutation instead of leaving it to the rollout.
		child := expand(node)
		deeper := expand(child)

		value := child.fit
		if deeper.fit < value {
			value = deeper.fit
		}
		roll := deeper.particles
		for d := 0; d < cfg.Rollout; d++ {
			samples, next := rolloutStep(cfg, rng, ev, roll, ref, px, py)
			for _, s := range samples {
				states = append(states, s)
				if s.fit < value {
					value = s.fit
				}
			}
			roll = next
		}
		for n := deeper; n != nil; n = n.parent {
			n.visits++
			n.valueSum += value
			n.mean = n.valueSum / float64(n.visits)
		}
	}
	return root, states
}

// rolloutStep scores rolloutBatch mutations of from and continues from the
// lowest score. The earliest sample wins a tie.
func rolloutStep(cfg Config, rng *randRand, ev *evaluator, from, ref []Particle, px, py float64) (samples []scoredState, next []Particle) {
	var bestFit float64
	for b := 0; b < rolloutBatch; b++ {
		sample := cloneParticles(from)
		mutate(cfg, rng, sample)
		finishParticles(cfg, sample, px, py)
		_, hf := ev.score(sample)
		fit := stateScore(cfg, hf, sample, ref)
		samples = append(samples, scoredState{particles: sample, fit: fit})
		if next == nil || fit < bestFit {
			next = sample
			bestFit = fit
		}
	}
	return samples, next
}

// bestState returns the evaluated frame with the lowest score. The earliest
// frame wins a tie, and the Newtonian proposal is evaluated first. A
// non-finite score is skipped. When every score is non-finite, the first
// frame is returned.
func bestState(states []scoredState) []Particle {
	if len(states) == 0 {
		return nil
	}
	best := -1
	bestFit := math.Inf(1)
	for i, s := range states {
		if math.IsNaN(s.fit) || math.IsInf(s.fit, 0) {
			continue
		}
		if best < 0 || s.fit < bestFit {
			best = i
			bestFit = s.fit
		}
	}
	if best < 0 {
		best = 0
	}
	return cloneParticles(states[best].particles)
}

// departureCost is the frequency entropy charged for a mean squared
// mutation of 1 away from the Newtonian proposal. A move is kept only when
// it lowers the frequency entropy by more than this charge.
const departureCost = 0.005

// fitness is the frequency-domain entropy. Lower is better. Spatial entropy
// is not part of the score. A non-finite frequency entropy scores +Inf.
func fitness(hf float64) float64 {
	if math.IsNaN(hf) || math.IsInf(hf, 0) || hf < 0 {
		return math.Inf(1)
	}
	return hf
}

// stateScore is the frequency entropy plus the cost of leaving ref.
// ref is the Newtonian proposal, whose own score is its frequency entropy.
func stateScore(cfg Config, hf float64, ps, ref []Particle) float64 {
	fit := fitness(hf)
	if math.IsInf(fit, 0) {
		return fit
	}
	return fit + departureCost*departure(cfg, ps, ref)
}
