// Copyright 2026 The Wave Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "math"

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
// The returned state merges the evaluated frames that share the lowest
// entropy ratio, weighted by the inverse of that ratio. A worse frame does
// not move the result. Entropies are measured on the merged frame.
func search(cfg Config, rng *randRand, base []Particle, px, py float64) (best []Particle, hs, hf float64) {
	_, states := runMCTS(cfg, rng, base, px, py)
	merged := mergeStates(cfg, states, px, py)
	hs, hf = newEvaluator(cfg).score(merged)
	return merged, hs, hf
}

// runMCTS returns the tree and every state it scored, root first.
func runMCTS(cfg Config, rng *randRand, base []Particle, px, py float64) (root *mctsNode, states []scoredState) {
	ev := newEvaluator(cfg)
	root = &mctsNode{particles: cloneParticles(base)}
	root.hs, root.hf = ev.score(root.particles)
	root.fit = fitness(root.hs, root.hf)
	root.visits = 1
	root.valueSum = root.fit
	root.mean = root.fit
	states = append(states, scoredState{particles: root.particles, fit: root.fit})

	expand := func(parent *mctsNode) *mctsNode {
		childP := cloneParticles(parent.particles)
		mutate(cfg, rng, childP)
		finishParticles(cfg, childP, px, py)
		chs, chf := ev.score(childP)
		cfit := fitness(chs, chf)
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
			roll = cloneParticles(roll)
			mutate(cfg, rng, roll)
			finishParticles(cfg, roll, px, py)
			rhs, rhf := ev.score(roll)
			rfit := fitness(rhs, rhf)
			states = append(states, scoredState{particles: roll, fit: rfit})
			if rfit < value {
				value = rfit
			}
		}
		for n := deeper; n != nil; n = n.parent {
			n.visits++
			n.valueSum += value
			n.mean = n.valueSum / float64(n.visits)
		}
	}
	return root, states
}

// ratioWeight is 1 / (H_frequency / H_spatial). A zero ratio takes the
// whole step. A non-finite ratio, including zero spatial entropy, has no weight.
func ratioWeight(fit float64) float64 {
	if fit == 0 {
		return math.Inf(1)
	}
	if fit < 0 || math.IsNaN(fit) || math.IsInf(fit, 0) {
		return 0
	}
	return 1 / fit
}

// mergeStates averages the evaluated frames that share the lowest ratio.
// A worse frame does not move the result, so a Newtonian proposal that
// has the best ratio is returned unchanged. Positions and phases use the
// circular mean. Velocities are linear. Total momentum is then restored
// to (px, py).
func mergeStates(cfg Config, states []scoredState, px, py float64) []Particle {
	if len(states) == 0 {
		return nil
	}
	bestFit := math.Inf(1)
	for _, s := range states {
		if !math.IsNaN(s.fit) && s.fit < bestFit {
			bestFit = s.fit
		}
	}
	chosen := make([]scoredState, 0, len(states))
	weights := make([]float64, 0, len(states))
	perfect := false
	for _, s := range states {
		if math.IsNaN(s.fit) || s.fit > bestFit {
			continue
		}
		w := ratioWeight(s.fit)
		if math.IsInf(w, 1) {
			if !perfect {
				chosen = chosen[:0]
				weights = weights[:0]
				perfect = true
			}
			chosen = append(chosen, s)
			weights = append(weights, 1)
			continue
		}
		if perfect || w <= 0 || math.IsNaN(w) {
			continue
		}
		chosen = append(chosen, s)
		weights = append(weights, w)
	}
	if len(chosen) == 0 {
		out := cloneParticles(states[0].particles)
		restoreMomentum(out, px, py)
		for i := range out {
			out[i].SyncFrequency(cfg.Hbar)
		}
		return out
	}
	out := cloneParticles(chosen[0].particles)
	n := len(out)
	xs := make([]float64, len(chosen))
	ys := make([]float64, len(chosen))
	ph := make([]float64, len(chosen))
	for i := 0; i < n; i++ {
		var vx, vy, wsum float64
		for k, s := range chosen {
			xs[k] = s.particles[i].S.X
			ys[k] = s.particles[i].S.Y
			ph[k] = s.particles[i].F.Phase
			vx += weights[k] * s.particles[i].S.VX
			vy += weights[k] * s.particles[i].S.VY
			wsum += weights[k]
		}
		out[i].S.X = circularMean(xs, weights, cfg.L)
		out[i].S.Y = circularMean(ys, weights, cfg.L)
		out[i].S.VX = vx / wsum
		out[i].S.VY = vy / wsum
		out[i].F.Phase = circularMean(ph, weights, 2*math.Pi)
	}
	restoreMomentum(out, px, py)
	for i := range out {
		out[i].SyncFrequency(cfg.Hbar)
	}
	return out
}

// circularMean is the weighted mean of values on a circle of the given period.
func circularMean(values, weights []float64, period float64) float64 {
	var s, c float64
	for i, v := range values {
		ang := 2 * math.Pi * v / period
		s += weights[i] * math.Sin(ang)
		c += weights[i] * math.Cos(ang)
	}
	if s*s+c*c < 1e-24 {
		best := 0
		for i := 1; i < len(weights); i++ {
			if weights[i] > weights[best] {
				best = i
			}
		}
		return values[best]
	}
	ang := math.Atan2(s, c)
	if ang < 0 {
		ang += 2 * math.Pi
	}
	return ang * period / (2 * math.Pi)
}

// fitness is H_frequency / H_spatial. Minimizing it lowers spectral
// entropy and raises spatial entropy. A non-positive spatial entropy
// cannot be improved further, so it scores +Inf.
func fitness(hs, hf float64) float64 {
	if hs <= 0 || math.IsNaN(hs) || math.IsNaN(hf) {
		return math.Inf(1)
	}
	fit := hf / hs
	if math.IsNaN(fit) || math.IsInf(fit, 0) {
		return math.Inf(1)
	}
	return fit
}
