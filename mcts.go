// Copyright 2026 The Wave Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"runtime"
	"sync"
)

// rolloutBatch is how many mutations one rollout step scores. The step
// continues from the lowest score, and that winner is added to the tree.
// The other samples remain candidates and are not expanded. Tree expansion
// still adds one mutation.
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

// scorePair is one frame's entropies. The Jacobi solve that produces it
// stays on a single goroutine; the pool only runs several frames at once.
type scorePair struct {
	hs, hf float64
}

type scoreJob struct {
	ps     []Particle
	hs, hf float64
	wg     *sync.WaitGroup
}

// scorePool scores independent frames concurrently. Each worker owns its
// evaluator. With one GOMAXPROCS the pool scores on the caller.
type scorePool struct {
	cfg  Config
	ev   *evaluator
	jobs chan *scoreJob
	quit chan struct{}
	wg   sync.WaitGroup
}

func newScorePool(cfg Config) *scorePool {
	p := &scorePool{cfg: cfg}
	n := runtime.GOMAXPROCS(0)
	if n < 2 {
		p.ev = newEvaluator(cfg)
		return p
	}
	p.jobs = make(chan *scoreJob)
	p.quit = make(chan struct{})
	for i := 0; i < n; i++ {
		ev := newEvaluator(cfg)
		p.wg.Add(1)
		go p.worker(ev)
	}
	return p
}

func (p *scorePool) stop() {
	if p.quit == nil {
		return
	}
	close(p.quit)
	p.wg.Wait()
}

func (p *scorePool) worker(ev *evaluator) {
	defer p.wg.Done()
	for {
		select {
		case <-p.quit:
			return
		case job := <-p.jobs:
			job.hs, job.hf = ev.score(job.ps)
			job.wg.Done()
		}
	}
}

func (p *scorePool) scoreAll(frames [][]Particle) []scorePair {
	out := make([]scorePair, len(frames))
	if p.jobs == nil {
		for i, ps := range frames {
			out[i].hs, out[i].hf = p.ev.score(ps)
		}
		return out
	}
	var wg sync.WaitGroup
	wg.Add(len(frames))
	jobs := make([]scoreJob, len(frames))
	for i, ps := range frames {
		jobs[i] = scoreJob{ps: ps, wg: &wg}
		p.jobs <- &jobs[i]
	}
	wg.Wait()
	for i := range jobs {
		out[i].hs = jobs[i].hs
		out[i].hf = jobs[i].hf
	}
	return out
}

func mutateFinished(cfg Config, rng *randRand, from []Particle, px, py float64) []Particle {
	sample := cloneParticles(from)
	mutate(cfg, rng, sample)
	finishParticles(cfg, sample, px, py)
	return sample
}

// search runs Monte Carlo tree search rooted at the Newtonian proposal.
// The returned state is the single best-scoring frame. The score is the
// block entropy plus a charge for leaving the Newtonian proposal.
// Reported entropies are measured on that frame and do not include the
// charge. Spatial entropy is not part of the score.
func search(cfg Config, rng *randRand, base []Particle, px, py float64) (best []Particle, hs, hf float64) {
	_, states := runMCTS(cfg, rng, base, px, py)
	best = bestState(states)
	hs, hf = newEvaluator(cfg).score(best)
	return best, hs, hf
}

// runMCTS returns the tree and every state it scored, root first.
// One iteration still expands one child, expands that child once more,
// then rolls out. The two new plies and the first rollout batch do not
// depend on each other's scores, so they are scored together.
func runMCTS(cfg Config, rng *randRand, base []Particle, px, py float64) (root *mctsNode, states []scoredState) {
	pool := newScorePool(cfg)
	defer pool.stop()

	root = &mctsNode{particles: cloneParticles(base)}
	rootPair := pool.scoreAll([][]Particle{root.particles})[0]
	root.hs, root.hf = rootPair.hs, rootPair.hf
	ref := root.particles
	root.fit = stateScore(cfg, root.hf, root.particles, ref)
	root.visits = 1
	root.valueSum = root.fit
	root.mean = root.fit
	states = append(states, scoredState{particles: root.particles, fit: root.fit})

	for i := 0; i < cfg.Iterations; i++ {
		node := root
		for len(node.children) >= maxChildren(cfg.Widen, node.visits) && len(node.children) > 0 {
			node = selectChild(cfg, node)
		}
		// Draw the next two plies, and the first rollout batch, before
		// scoring. None of those draws read a score, so the random
		// stream matches a serial iteration.
		childP := mutateFinished(cfg, rng, node.particles, px, py)
		deeperP := mutateFinished(cfg, rng, childP, px, py)
		frames := [][]Particle{childP, deeperP}
		var first [][]Particle
		if cfg.Rollout > 0 {
			first = make([][]Particle, rolloutBatch)
			for b := range first {
				first[b] = mutateFinished(cfg, rng, deeperP, px, py)
			}
			frames = append(frames, first...)
		}
		scored := pool.scoreAll(frames)

		cfit := stateScore(cfg, scored[0].hf, childP, ref)
		child := &mctsNode{
			parent:    node,
			particles: childP,
			hs:        scored[0].hs,
			hf:        scored[0].hf,
			fit:       cfit,
		}
		node.children = append(node.children, child)
		states = append(states, scoredState{particles: childP, fit: cfit})

		dfit := stateScore(cfg, scored[1].hf, deeperP, ref)
		deeper := &mctsNode{
			parent:    child,
			particles: deeperP,
			hs:        scored[1].hs,
			hf:        scored[1].hf,
			fit:       dfit,
		}
		child.children = append(child.children, deeper)
		states = append(states, scoredState{particles: deeperP, fit: dfit})

		value := cfit
		if dfit < value {
			value = dfit
		}
		rollParent := deeper
		if len(first) > 0 {
			next, nextFit, v := adoptBatch(first, scored[2:], ref, cfg, &states, value)
			value = v
			winner := &mctsNode{parent: rollParent, particles: next, fit: nextFit}
			rollParent.children = append(rollParent.children, winner)
			rollParent = winner
			for d := 1; d < cfg.Rollout; d++ {
				samples, nxt, nxtFit := rolloutStep(cfg, rng, pool, next, ref, px, py)
				for _, s := range samples {
					states = append(states, s)
					if s.fit < value {
						value = s.fit
					}
				}
				w := &mctsNode{parent: rollParent, particles: nxt, fit: nxtFit}
				rollParent.children = append(rollParent.children, w)
				rollParent = w
				next = nxt
			}
		}
		for n := rollParent; n != nil; n = n.parent {
			n.visits++
			n.valueSum += value
			n.mean = n.valueSum / float64(n.visits)
		}
	}
	return root, states
}

// adoptBatch records a rollout batch in draw order and returns the lowest
// score. The earliest sample wins a tie. states and value are updated.
func adoptBatch(frames [][]Particle, scored []scorePair, ref []Particle, cfg Config, states *[]scoredState, value float64) (next []Particle, nextFit, newValue float64) {
	newValue = value
	for i, ps := range frames {
		fit := stateScore(cfg, scored[i].hf, ps, ref)
		*states = append(*states, scoredState{particles: ps, fit: fit})
		if fit < newValue {
			newValue = fit
		}
		if next == nil || fit < nextFit {
			next = ps
			nextFit = fit
		}
	}
	return next, nextFit, newValue
}

// rolloutStep scores rolloutBatch mutations of from and continues from the
// lowest score. The earliest sample wins a tie. A nil pool scores on the
// caller; otherwise the batch is scored concurrently.
func rolloutStep(cfg Config, rng *randRand, pool *scorePool, from, ref []Particle, px, py float64) (samples []scoredState, next []Particle, nextFit float64) {
	frames := make([][]Particle, rolloutBatch)
	for b := range frames {
		frames[b] = mutateFinished(cfg, rng, from, px, py)
	}
	var scored []scorePair
	if pool == nil {
		ev := newEvaluator(cfg)
		scored = make([]scorePair, len(frames))
		for i, ps := range frames {
			scored[i].hs, scored[i].hf = ev.score(ps)
		}
	} else {
		scored = pool.scoreAll(frames)
	}
	for i, ps := range frames {
		fit := stateScore(cfg, scored[i].hf, ps, ref)
		samples = append(samples, scoredState{particles: ps, fit: fit})
		if next == nil || fit < nextFit {
			next = ps
			nextFit = fit
		}
	}
	return samples, next, nextFit
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

// departureCost is the block entropy charged for a mean squared
// mutation of 1 away from the Newtonian proposal. A move is kept only when
// it lowers the block entropy by more than this charge.
const departureCost = 0.03

// fitness is the block entropy. Lower is better. Spatial entropy
// is not part of the score. A non-finite block entropy scores +Inf.
func fitness(hf float64) float64 {
	if math.IsNaN(hf) || math.IsInf(hf, 0) || hf < 0 {
		return math.Inf(1)
	}
	return hf
}

// stateScore is the block entropy plus the cost of leaving ref.
// ref is the Newtonian proposal, whose own score is its block entropy.
func stateScore(cfg Config, hf float64, ps, ref []Particle) float64 {
	fit := fitness(hf)
	if math.IsInf(fit, 0) {
		return fit
	}
	return fit + departureCost*departure(cfg, ps, ref)
}
