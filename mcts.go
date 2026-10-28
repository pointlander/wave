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

// search runs Monte Carlo tree search rooted at the Newtonian proposal.
// The returned state is the lowest-fitness state evaluated, including
// the root, so the step does not score worse than that proposal.
func search(cfg Config, rng *randRand, base []Particle, px, py float64, window *entropyWindow) (best []Particle, hs, hf float64) {
	_, best, hs, hf = runMCTS(cfg, rng, base, px, py, window)
	return best, hs, hf
}

// runMCTS returns the tree and the lowest-fitness state it evaluated.
func runMCTS(cfg Config, rng *randRand, base []Particle, px, py float64, window *entropyWindow) (root *mctsNode, best []Particle, hs, hf float64) {
	ev := newEvaluator(cfg)
	root = &mctsNode{particles: cloneParticles(base)}
	root.hs, root.hf = ev.score(root.particles, window)
	root.fit = fitness(root.hs, root.hf)
	root.visits = 1
	root.valueSum = root.fit
	root.mean = root.fit

	bestP := root.particles
	bestHS, bestHF := root.hs, root.hf
	bestFit := root.fit
	consider := func(ps []Particle, shs, shf, fit float64) {
		if fit < bestFit {
			bestFit = fit
			bestHS, bestHF = shs, shf
			bestP = ps
		}
	}

	expand := func(parent *mctsNode) *mctsNode {
		childP := cloneParticles(parent.particles)
		mutate(cfg, rng, childP)
		finishParticles(cfg, childP, px, py)
		chs, chf := ev.score(childP, window)
		cfit := fitness(chs, chf)
		child := &mctsNode{
			parent:    parent,
			particles: childP,
			hs:        chs,
			hf:        chf,
			fit:       cfit,
		}
		consider(childP, chs, chf, cfit)
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
			rhs, rhf := ev.score(roll, window)
			rfit := fitness(rhs, rhf)
			consider(roll, rhs, rhf, rfit)
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
	return root, bestP, bestHS, bestHF
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
