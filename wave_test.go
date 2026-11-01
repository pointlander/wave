// Copyright 2026 The Wave Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"image/png"
	"math"
	"math/cmplx"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func testConfig() Config {
	return Config{
		N:          8,
		Grid:       32,
		L:          1,
		Mass:       1,
		DT:         0.02,
		Steps:      3,
		Iterations: 8,
		Widen:      1,
		Explore:    0.15,
		Rollout:    1,
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

func TestValidate(t *testing.T) {
	cfg := DefaultConfig()
	if err := cfg.Validate(); err != nil {
		t.Fatal(err)
	}
	cfg.Grid = 48
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected grid rejection")
	}
}

func TestMinImage(t *testing.T) {
	if d := minImage(0.99-0.01, 1); math.Abs(d-(-0.02)) > 1e-12 {
		t.Fatalf("min image = %v", d)
	}
	if d := minImage(0.01-0.99, 1); math.Abs(d-0.02) > 1e-12 {
		t.Fatalf("min image = %v", d)
	}
	if d := minImage(0.2, 1); math.Abs(d-0.2) > 1e-12 {
		t.Fatalf("min image = %v", d)
	}
}

func TestPageRankEntropy(t *testing.T) {
	if pageRankEntropy(nil, 1) != 0 {
		t.Fatal("empty rank entropy")
	}
	one := []Particle{{S: Spatial{X: 0.2, Y: 0.3}}}
	if pageRankEntropy(one, 1) != 0 {
		t.Fatal("single galaxy rank entropy")
	}
	pair := []Particle{
		{S: Spatial{X: 0.1, Y: 0.2}},
		{S: Spatial{X: 0.7, Y: 0.8}},
	}
	if math.Abs(pageRankEntropy(pair, 1)-1) > 1e-12 {
		t.Fatalf("pair rank entropy %v", pageRankEntropy(pair, 1))
	}
	cfg := testConfig()
	for i := range pair {
		pair[i].S.Mass = 1
		pair[i].S.Sigma = cfg.Sigma
		pair[i].SyncFrequency(cfg.Hbar)
	}
	res, err := Observe(cfg, pair)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(res.HRank-1) > 1e-12 {
		t.Fatalf("observed rank entropy %v", res.HRank)
	}
	s := 0.1
	tri := []Particle{
		{S: Spatial{X: 0.5, Y: 0.5}},
		{S: Spatial{X: 0.5 + s, Y: 0.5}},
		{S: Spatial{X: 0.5 + s/2, Y: 0.5 + s*math.Sqrt(3)/2}},
	}
	if math.Abs(pageRankEntropy(tri, 1)-1) > 1e-9 {
		t.Fatalf("triangle rank entropy %v", pageRankEntropy(tri, 1))
	}

	// A coincident pair and a distant third. At this separation the third
	// galaxy's links vanish, so its rank is the teleport mass (1-α)/3.
	far := []Particle{
		{S: Spatial{X: 0, Y: 0}},
		{S: Spatial{X: 0, Y: 0}},
		{S: Spatial{X: 100, Y: 0}},
	}
	c := (1 - pageRankDamping) / 3
	a := (1 - c) / 2
	want := -(2*a*math.Log(a) + c*math.Log(c)) / math.Log(3)
	if got := pageRankEntropy(far, 1000); math.Abs(got-want) > 1e-9 {
		t.Fatalf("cluster rank entropy %v, want %v", got, want)
	}

	// C sits across the periodic boundary from A and B.
	wrapped := []Particle{
		{S: Spatial{X: 0, Y: 0.5}},
		{S: Spatial{X: 0.10, Y: 0.5}},
		{S: Spatial{X: 0.95, Y: 0.5}},
	}
	got := pageRankEntropy(wrapped, 1)
	if math.Abs(got-rankEntropyOracle([]float64{0.10, 0.05, 0.15})) > 1e-12 {
		t.Fatalf("wrapped rank entropy %v", got)
	}
	unwrapped := rankEntropyOracle([]float64{0.10, 0.95, 0.85})
	if math.Abs(got-unwrapped) < 1e-6 {
		t.Fatalf("wrapped entropy %v matches the long-way distances %v", got, unwrapped)
	}
	if traceHeader()[4] != "h_rank" {
		t.Fatalf("trace header %v", traceHeader())
	}
}

// rankEntropyOracle is the normalized entropy of PageRank on a triangle
// whose pairwise separations are ab, ac, bc. Link weights are e^{-r}.
func rankEntropyOracle(sep []float64) float64 {
	r := [][]float64{
		{0, sep[0], sep[1]},
		{sep[0], 0, sep[2]},
		{sep[1], sep[2], 0},
	}
	n := 3
	p := make([][]float64, n)
	for i := 0; i < n; i++ {
		p[i] = make([]float64, n)
		var sum float64
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			p[i][j] = math.Exp(-r[i][j])
			sum += p[i][j]
		}
		for j := 0; j < n; j++ {
			p[i][j] /= sum
		}
	}
	a := make([][]float64, n)
	b := make([]float64, n)
	invN := 1 / float64(n)
	for i := 0; i < n; i++ {
		a[i] = make([]float64, n)
		b[i] = (1 - pageRankDamping) * invN
		for j := 0; j < n; j++ {
			a[i][j] = -pageRankDamping * p[j][i]
		}
		a[i][i] += 1
	}
	x := solveLinear(a, b)
	h := 0.0
	for _, v := range x {
		if v > 0 {
			h -= v * math.Log(v)
		}
	}
	return h / math.Log(float64(n))
}

func solveLinear(a [][]float64, b []float64) []float64 {
	n := len(b)
	m := make([][]float64, n)
	for i := 0; i < n; i++ {
		m[i] = append(append([]float64{}, a[i]...), b[i])
	}
	for col := 0; col < n; col++ {
		piv := col
		for row := col + 1; row < n; row++ {
			if math.Abs(m[row][col]) > math.Abs(m[piv][col]) {
				piv = row
			}
		}
		m[col], m[piv] = m[piv], m[col]
		div := m[col][col]
		for c := col; c <= n; c++ {
			m[col][c] /= div
		}
		for row := 0; row < n; row++ {
			if row == col {
				continue
			}
			f := m[row][col]
			for c := col; c <= n; c++ {
				m[row][c] -= f * m[col][c]
			}
		}
	}
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = m[i][n]
	}
	return x
}

func TestShannonPower(t *testing.T) {
	spike := make([]complex128, 4)
	spike[2] = 1
	if h := shannon(spike); h > 1e-12 {
		t.Fatalf("spike entropy %v", h)
	}
	if h := shannon(nil); h != 0 {
		t.Fatalf("empty entropy %v", h)
	}
	flat := []complex128{1, 1, 1, 1}
	if math.Abs(shannon(flat)-1) > 1e-12 {
		t.Fatalf("flat entropy %v", shannon(flat))
	}
	half := []complex128{1, 0, 1, 0}
	if math.Abs(shannon(half)-0.5) > 1e-12 {
		t.Fatalf("two-bin entropy %v", shannon(half))
	}
}

func TestFullQFTEntropy(t *testing.T) {
	cfg := testConfig()
	cfg.Grid = 8
	ev := newEvaluator(cfg)
	for i := range ev.field {
		ev.field[i] = 1
	}
	copy(ev.scratch, ev.field)
	ev.fft.transform(ev.scratch, false)
	// A constant field transforms to one DC bin.
	if h := shannon(ev.scratch); h > 1e-9 {
		t.Fatalf("constant field entropy %v", h)
	}
	for i := range ev.field {
		ev.field[i] = 0
	}
	ev.field[0] = 1
	copy(ev.scratch, ev.field)
	ev.fft.transform(ev.scratch, false)
	// A spike transforms to a flat spectrum.
	if h := shannon(ev.scratch); math.Abs(h-1) > 1e-9 {
		t.Fatalf("spike entropy %v", h)
	}
	hs, hf := ev.score(nil)
	if hf != 0 {
		t.Fatalf("empty score %v %v", hs, hf)
	}
}

func TestFFTRoundtripAndDelta(t *testing.T) {
	rng := newRNG(3)
	n := 16
	a := make([]complex128, n*n)
	orig := make([]complex128, n*n)
	for i := range a {
		a[i] = complex(rng.Float64(), rng.Float64())
		orig[i] = a[i]
	}
	f := newFFT2(n)
	f.transform(a, false)
	f.transform(a, true)
	for i := range a {
		if cmplx.Abs(a[i]-orig[i]) > 1e-9 {
			t.Fatalf("roundtrip error %v at %d", cmplx.Abs(a[i]-orig[i]), i)
		}
	}

	d := make([]complex128, 8*8)
	d[0] = 1
	newFFT2(8).transform(d, false)
	for i, v := range d {
		if cmplx.Abs(v-1) > 1e-9 {
			t.Fatalf("delta transform bin %d = %v", i, v)
		}
	}
}

func TestGalacticScale(t *testing.T) {
	cfg := DefaultConfig()
	if err := cfg.Validate(); err != nil {
		t.Fatal(err)
	}
	if cfg.Mass != 1e12 || cfg.L != 3000 {
		t.Fatalf("mass %v box %v", cfg.Mass, cfg.L)
	}
	if math.Abs(cfg.Gravity-newtonG) > 1e-18 {
		t.Fatalf("G = %v", cfg.Gravity)
	}
	v := circularSpeed(cfg, cfg.Radius, float64(cfg.N)*cfg.Mass)
	if v < 200 || v > 500 {
		t.Fatalf("group edge speed %v km/s", v)
	}
	if cfg.Sigma < 50 || cfg.Sigma > 300 {
		t.Fatalf("galaxy width %v kpc", cfg.Sigma)
	}
	dtGyr := cfg.DT * gyrPerKpcOverKms
	if dtGyr < 0.05 || dtGyr > 0.5 {
		t.Fatalf("dt %v Gyr", dtGyr)
	}
}

func TestGravityPullsAcrossPairAndBoundary(t *testing.T) {
	cfg := testConfig()
	ps := []Particle{
		{S: Spatial{X: 0.4, Y: 0.5, Mass: 1, Sigma: cfg.Sigma}},
		{S: Spatial{X: 0.6, Y: 0.5, Mass: 1, Sigma: cfg.Sigma}},
	}
	ax, ay := accelerations(cfg, ps)
	if ax[0] <= 0 || ax[1] >= 0 {
		t.Fatalf("pair acceleration = %v %v", ax[0], ax[1])
	}
	if math.Abs(ay[0]) > 1e-12 || math.Abs(ay[1]) > 1e-12 {
		t.Fatalf("transverse acceleration = %v %v", ay[0], ay[1])
	}
	if math.Abs(ax[0]+ax[1]) > 1e-12 {
		t.Fatalf("momentum rate %v", ax[0]+ax[1])
	}

	ps[0].S.X = 0.05
	ps[1].S.X = 0.95
	ax, _ = accelerations(cfg, ps)
	if ax[0] >= 0 || ax[1] <= 0 {
		t.Fatalf("boundary acceleration = %v %v", ax[0], ax[1])
	}
}

func TestMomentumOfForcesVanishes(t *testing.T) {
	cfg := testConfig()
	ps, err := Initialize(cfg, newRNG(9))
	if err != nil {
		t.Fatal(err)
	}
	ax, ay := accelerations(cfg, ps)
	var fx, fy float64
	for i := range ps {
		fx += ps[i].S.Mass * ax[i]
		fy += ps[i].S.Mass * ay[i]
	}
	if math.Hypot(fx, fy) > 1e-9 {
		t.Fatalf("net force %v %v", fx, fy)
	}
	px, py := momentum(ps)
	if math.Hypot(px, py) > 1e-9 {
		t.Fatalf("initial momentum %v %v", px, py)
	}
}

func TestSpectrumPeaksAtWavevector(t *testing.T) {
	cfg := testConfig()
	cfg.Grid = 64
	cfg.L = 1
	cfg.Hbar = 0.01
	binX, binY := 5, -3
	kx := 2 * math.Pi * float64(binX) / cfg.L
	ky := 2 * math.Pi * float64(binY) / cfg.L
	p := Particle{S: Spatial{
		X: 0.37, Y: 0.62, Mass: 1, Sigma: 0.08,
		VX: kx * cfg.Hbar, VY: ky * cfg.Hbar,
	}}
	p.F.Phase = 0.4
	p.SyncFrequency(cfg.Hbar)
	if math.Abs(p.F.SigmaK-1/(2*p.S.Sigma)) > 1e-12 {
		t.Fatalf("sigma_k = %v", p.F.SigmaK)
	}
	gotX, gotY, power := peakBin(cfg, []Particle{p})
	if power <= 0 {
		t.Fatal("empty spectrum")
	}
	wantX := freqBin(p.F.KX, cfg.L, cfg.Grid)
	wantY := freqBin(p.F.KY, cfg.L, cfg.Grid)
	if torusDist(gotX, wantX, cfg.Grid) > 1 || torusDist(gotY, wantY, cfg.Grid) > 1 {
		t.Fatalf("peak bin (%d,%d), want (%d,%d)", gotX, gotY, wantX, wantY)
	}
}

func torusDist(a, b, n int) int {
	d := a - b
	if d < 0 {
		d = -d
	}
	if d > n-d {
		d = n - d
	}
	return d
}

func TestCoherentClusterHasLowerEntropy(t *testing.T) {
	cfg := testConfig()
	cfg.Grid = 64
	cfg.L = 1
	cfg.Hbar = 0.01
	packet := func(x, y float64) Particle {
		p := Particle{S: Spatial{
			X: x, Y: y, Mass: 1, Sigma: 0.04, VX: 0.2, VY: 0,
		}}
		p.SyncFrequency(cfg.Hbar)
		return p
	}
	together := []Particle{packet(0.5, 0.5), packet(0.5, 0.5)}
	// Displace both axes so the separated packets raise the entanglement.
	apart := []Particle{packet(0.25, 0.3), packet(0.75, 0.7)}
	ths, thf := measure(cfg, together)
	ahs, ahf := measure(cfg, apart)
	if ahs < ths+0.05 {
		t.Fatalf("spatial entropy together %v, apart %v", ths, ahs)
	}
	if math.Abs(fitness(ahf)-ahf) > 1e-15 || math.Abs(fitness(thf)-thf) > 1e-15 {
		t.Fatalf("fitness %v %v, frequency entropy %v %v", fitness(ahf), fitness(thf), ahf, thf)
	}
}

func TestFitnessIsFrequencyEntropy(t *testing.T) {
	if got := fitness(0.2); math.Abs(got-0.2) > 1e-12 {
		t.Fatalf("fitness = %v", got)
	}
	if fitness(0) != 0 {
		t.Fatal("zero frequency entropy should be the best fitness")
	}
	if !math.IsInf(fitness(math.NaN()), 1) {
		t.Fatal("non-finite frequency entropy should be the worst fitness")
	}
	if !math.IsInf(fitness(-0.1), 1) {
		t.Fatal("negative frequency entropy should be the worst fitness")
	}
}

func TestScoreIsBlockSpectrumEntropy(t *testing.T) {
	if stateScore(0.5) != 0.5 {
		t.Fatalf("score %v", stateScore(0.5))
	}
	if !math.IsInf(stateScore(math.NaN()), 1) {
		t.Fatal("non-finite entropy should be the worst score")
	}
}

func TestMechanicsAgree(t *testing.T) {
	a := []Particle{{
		S: Spatial{X: 0.2, Y: 0.3, VX: 0.1, VY: -0.2},
		F: Frequency{Phase: 0.4},
	}}
	b := cloneParticles(a)
	if !mechanicsAgree(a, b, 1) {
		t.Fatal("identical states disagree")
	}
	b[0].S.X = 0.21
	if mechanicsAgree(a, b, 1) {
		t.Fatal("shifted state agrees")
	}
	b = cloneParticles(a)
	b[0].S.X = 0.2 + 1
	if !mechanicsAgree(a, b, 1) {
		t.Fatal("periodic copy disagrees")
	}
	b = cloneParticles(a)
	b[0].F.Phase = 0.4 + 2*math.Pi
	if !mechanicsAgree(a, b, 1) {
		t.Fatal("wrapped phase disagrees")
	}
}

func TestNewtonianAgreesWhenMutationsAreOff(t *testing.T) {
	cfg := testConfig()
	cfg.Steps = 2
	cfg.MutPos = 0
	cfg.MutVel = 0
	cfg.MutPhase = 0
	ps, err := Initialize(cfg, newRNG(1))
	if err != nil {
		t.Fatal(err)
	}
	steps, _, _, last, err := simulate(cfg, newRNG(1), ps, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if steps != 2 || last.AgreeCount != 2 || math.Abs(last.AgreeRate-1) > 1e-12 {
		t.Fatalf("steps %d agree %d rate %v", steps, last.AgreeCount, last.AgreeRate)
	}
	if last.RankPredCount != 2 || math.Abs(last.Accuracy-1) > 1e-12 {
		t.Fatalf("pagerank accuracy %d rate %v", last.RankPredCount, last.Accuracy)
	}
}

func TestMergedStepKeepsMomentumAndField(t *testing.T) {
	cfg := testConfig()
	rng := newRNG(cfg.Seed)
	ps, err := Initialize(cfg, rng)
	if err != nil {
		t.Fatal(err)
	}
	px0, py0 := momentum(ps)
	for step := 0; step < cfg.Steps; step++ {
		next, res, err := Step(cfg, rng, ps)
		if err != nil {
			t.Fatal(err)
		}
		if math.IsNaN(res.Fitness) || math.IsInf(res.Fitness, 0) || math.IsNaN(res.Kinetic) || math.IsNaN(res.Potential) {
			t.Fatalf("step %d produced a bad result: %+v", step, res)
		}
		px, py := momentum(next)
		if math.Abs(px-px0) > 1e-8 || math.Abs(py-py0) > 1e-8 {
			t.Fatalf("momentum drifted to %v %v", px, py)
		}
		againS, againF := measure(cfg, next)
		if math.Abs(againS-res.HSpatial) > 1e-9 || math.Abs(againF-res.HFrequency) > 1e-9 {
			t.Fatalf("rescored entropy %v %v, recorded %v %v", againS, againF, res.HSpatial, res.HFrequency)
		}
		ps = next
	}
}

func TestBestState(t *testing.T) {
	cfg := testConfig()
	cfg.L = 1
	body := func(x0, x1, vx float64) []Particle {
		return []Particle{
			{S: Spatial{X: x0, Y: 0.5, VX: vx, Mass: 1, Sigma: cfg.Sigma}},
			{S: Spatial{X: x1, Y: 0.5, VX: -vx, Mass: 1, Sigma: cfg.Sigma}},
		}
	}
	best := body(0.32, 0.50, 0.25)
	states := []scoredState{
		{particles: body(0.20, 0.40, 0.10), fit: 1},
		{particles: best, fit: 0.5},
		{particles: body(0.90, 0.90, 5), fit: math.Inf(1)},
	}
	got := bestState(states)
	if !mechanicsAgree(got, best, cfg.L) {
		t.Fatalf("kept %v %v instead of the best state", got[0].S.X, got[0].S.VX)
	}
	first := body(0.20, 0.40, 0.10)
	tied := []scoredState{
		{particles: first, fit: 0.5},
		{particles: body(0.32, 0.50, 0.25), fit: 0.5},
	}
	kept := bestState(tied)
	if !mechanicsAgree(kept, first, cfg.L) {
		t.Fatalf("tie kept %v %v", kept[0].S.X, kept[0].S.VX)
	}
	onlyInf := []scoredState{
		{particles: first, fit: math.Inf(1)},
		{particles: best, fit: math.NaN()},
	}
	fallback := bestState(onlyInf)
	if !mechanicsAgree(fallback, first, cfg.L) {
		t.Fatal("non-finite scores did not fall back to the first frame")
	}
	perfect := append(states, scoredState{particles: body(0.80, 0.70, 0.3), fit: 0})
	only := bestState(perfect)
	if math.Abs(only[0].S.X-0.80) > 1e-9 || math.Abs(only[1].S.X-0.70) > 1e-9 {
		t.Fatalf("zero score did not win: %v %v", only[0].S.X, only[1].S.X)
	}
}

func TestRankPredictsNewtonian(t *testing.T) {
	L := 1000.0
	side := 1.0
	spread := []Particle{
		{S: Spatial{X: 100, Y: 100}},
		{S: Spatial{X: 100 + side, Y: 100}},
		{S: Spatial{X: 100 + side/2, Y: 100 + side*math.Sqrt(3)/2}},
	}
	cluster := []Particle{
		{S: Spatial{X: 0, Y: 0}},
		{S: Spatial{X: 0, Y: 0}},
		{S: Spatial{X: 100, Y: 0}},
	}
	if pageRankEntropy(cluster, L) >= pageRankEntropy(spread, L) {
		t.Fatalf("cluster %v spread %v", pageRankEntropy(cluster, L), pageRankEntropy(spread, L))
	}
	miss := []scoredState{{particles: spread}, {particles: cluster}}
	if rankPredictsNewton(miss, spread, L) {
		t.Fatal("clustered mutant was not preferred")
	}
	hit := []scoredState{{particles: cluster}, {particles: spread}}
	if !rankPredictsNewton(hit, cluster, L) {
		t.Fatal("proposal lost to a higher rank entropy")
	}
	tie := []scoredState{{particles: spread}, {particles: cloneParticles(spread)}}
	if !rankPredictsNewton(tie, spread, L) {
		t.Fatal("tie did not keep the proposal")
	}
	header := traceHeader()
	if header[len(header)-2] != "pred" || header[len(header)-1] != "accuracy" {
		t.Fatalf("trace header %v", header)
	}
}

func TestMCTSSelection(t *testing.T) {
	if maxChildren(1, 1) != 1 || maxChildren(1, 4) != 2 || maxChildren(1, 9) != 3 {
		t.Fatalf("widening 1,4,9 -> %d %d %d", maxChildren(1, 1), maxChildren(1, 4), maxChildren(1, 9))
	}
	cfg := testConfig()
	parent := &mctsNode{visits: 20}
	good := &mctsNode{visits: 5, mean: 0.5}
	bad := &mctsNode{visits: 5, mean: 0.9}
	parent.children = []*mctsNode{bad, good}
	if selectChild(cfg, parent) != good {
		t.Fatal("ucb did not select the lower mean")
	}
	parent.children = []*mctsNode{{visits: 5, mean: math.Inf(1)}, good}
	if selectChild(cfg, parent) != good {
		t.Fatal("ucb preferred a non-finite mean")
	}
	cfg.Explore = 1
	parent.visits = 30
	rare := &mctsNode{visits: 1, mean: 0.55}
	common := &mctsNode{visits: 20, mean: 0.4}
	parent.children = []*mctsNode{common, rare}
	if selectChild(cfg, parent) != rare {
		t.Fatal("ucb did not explore the rare child")
	}
}

func TestRolloutContinuesFromBestSample(t *testing.T) {
	cfg := testConfig()
	cfg.Gravity = 0
	cfg.MutPos = 0.05
	cfg.MutVel = 0.05
	cfg.MutPhase = 0.3
	p := Particle{S: Spatial{X: 0.5, Y: 0.5, VX: 0.1, Mass: 1, Sigma: cfg.Sigma}, F: Frequency{Phase: 0.2}}
	p.SyncFrequency(cfg.Hbar)
	from := []Particle{p}
	pool := newScorePool(cfg)
	defer pool.stop()
	samples, next, _ := rolloutStep(cfg, newRNG(1), pool, from, 0, 0)
	if len(samples) != rolloutBatch {
		t.Fatalf("samples = %d", len(samples))
	}
	bestFit := samples[0].fit
	for _, s := range samples[1:] {
		if s.fit < bestFit {
			bestFit = s.fit
		}
	}
	continued := false
	for _, s := range samples {
		if s.fit == bestFit && mechanicsAgree(s.particles, next, cfg.L) {
			continued = true
			break
		}
	}
	if !continued {
		t.Fatal("rollout continued from a sample that was not the best")
	}
}

func TestParallelRolloutMatchesSerial(t *testing.T) {
	cfg := testConfig()
	cfg.Grid = 16
	cfg.N = 2
	cfg.Gravity = 0
	ps, err := Initialize(cfg, newRNG(1))
	if err != nil {
		t.Fatal(err)
	}
	px, py := momentum(ps)
	serial, sNext, sFit := rolloutStep(cfg, newRNG(5), nil, ps, px, py)
	pool := newScorePool(cfg)
	defer pool.stop()
	parallel, pNext, pFit := rolloutStep(cfg, newRNG(5), pool, ps, px, py)
	if sFit != pFit || !mechanicsAgree(sNext, pNext, cfg.L) {
		t.Fatalf("winner serial %v parallel %v", sFit, pFit)
	}
	if len(serial) != len(parallel) {
		t.Fatalf("samples %d %d", len(serial), len(parallel))
	}
	for i := range serial {
		if serial[i].fit != parallel[i].fit || !mechanicsAgree(serial[i].particles, parallel[i].particles, cfg.L) {
			t.Fatalf("sample %d serial %v parallel %v", i, serial[i].fit, parallel[i].fit)
		}
	}
}

func TestRolloutWinnerJoinsTheTree(t *testing.T) {
	cfg := testConfig()
	cfg.Iterations = 1
	cfg.Rollout = 2
	cfg.N = 1
	cfg.Gravity = 0
	p := Particle{S: Spatial{X: 0.5, Y: 0.5, Mass: 1, Sigma: cfg.Sigma}}
	p.SyncFrequency(cfg.Hbar)
	root, states := runMCTS(cfg, newRNG(1), []Particle{p}, 0, 0)
	if len(states) != 1+2+cfg.Rollout*rolloutBatch {
		t.Fatalf("states = %d", len(states))
	}
	deeper := root.children[0].children[0]
	if len(root.children) != 1 || len(root.children[0].children) != 1 || len(deeper.children) != 1 {
		t.Fatal("expansion added more than the rollout winner")
	}
	winner := deeper.children[0]
	if len(winner.children) != 1 || len(winner.children[0].children) != 0 {
		t.Fatal("second rollout winner was not the only grafted child")
	}
	firstBatch := states[3 : 3+rolloutBatch]
	secondBatch := states[3+rolloutBatch:]
	if !mechanicsAgree(winner.particles, lowestSample(firstBatch).particles, cfg.L) {
		t.Fatal("first grafted child is not the best rollout sample")
	}
	if !mechanicsAgree(winner.children[0].particles, lowestSample(secondBatch).particles, cfg.L) {
		t.Fatal("second grafted child is not the best rollout sample")
	}
	if winner.visits != 1 || winner.children[0].visits != 1 {
		t.Fatalf("grafted visits %d %d", winner.visits, winner.children[0].visits)
	}
}

func lowestSample(samples []scoredState) scoredState {
	best := samples[0]
	for _, s := range samples[1:] {
		if s.fit < best.fit {
			best = s
		}
	}
	return best
}

func TestSearchExpandsOneExtraPly(t *testing.T) {
	cfg := testConfig()
	cfg.Iterations = 1
	cfg.Rollout = 0
	cfg.N = 1
	cfg.Gravity = 0
	p := Particle{S: Spatial{X: 0.5, Y: 0.5, Mass: 1, Sigma: cfg.Sigma}}
	p.SyncFrequency(cfg.Hbar)
	root, _ := runMCTS(cfg, newRNG(1), []Particle{p}, 0, 0)
	if len(root.children) != 1 {
		t.Fatalf("root children = %d", len(root.children))
	}
	if len(root.children[0].children) != 1 {
		t.Fatalf("depth-1 children = %d", len(root.children[0].children))
	}
	if len(root.children[0].children[0].children) != 0 {
		t.Fatal("expanded past the extra ply")
	}
}

func TestWriteFrame(t *testing.T) {
	cfg := testConfig()
	ps, err := Initialize(cfg, newRNG(2))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "frame.png")
	if err := WriteFrame(path, cfg, ps); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	b := img.Bounds()
	gap := 2
	if b.Dx() != cfg.Grid*2+gap || b.Dy() != cfg.Grid {
		t.Fatalf("image size %v", b)
	}
	bright := false
	for y := b.Min.Y; y < b.Max.Y && !bright; y++ {
		for x := b.Min.X; x < b.Min.X+cfg.Grid; x++ {
			r, g, bl, _ := img.At(x, y).RGBA()
			if r|g|bl != 0 {
				bright = true
				break
			}
		}
	}
	if !bright {
		t.Fatal("spatial panel is empty")
	}
}

func TestDurationFlag(t *testing.T) {
	var d durationFlag
	if err := d.Set("30"); err != nil {
		t.Fatal(err)
	}
	if time.Duration(d) != 30*time.Second {
		t.Fatalf("bare seconds = %v", time.Duration(d))
	}
	if err := d.Set("30s"); err != nil {
		t.Fatal(err)
	}
	if time.Duration(d) != 30*time.Second {
		t.Fatalf("30s = %v", time.Duration(d))
	}
	if err := d.Set("-1s"); err == nil {
		t.Fatal("expected negative duration to fail")
	}
}

func TestSimulateHonorsStepCount(t *testing.T) {
	cfg := testConfig()
	cfg.Steps = 2
	ps, err := Initialize(cfg, newRNG(1))
	if err != nil {
		t.Fatal(err)
	}
	steps, _, _, _, err := simulate(cfg, newRNG(1), ps, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if steps != 2 {
		t.Fatalf("steps = %d", steps)
	}
}

func TestSimulateHonorsDuration(t *testing.T) {
	cfg := testConfig()
	cfg.N = 4
	cfg.Grid = 16
	cfg.Iterations = 2
	cfg.Rollout = 0
	cfg.Steps = 1_000_000
	ps, err := Initialize(cfg, newRNG(1))
	if err != nil {
		t.Fatal(err)
	}
	const limit = 200 * time.Millisecond
	steps, elapsed, _, _, err := simulate(cfg, newRNG(1), ps, limit, nil)
	if err != nil {
		t.Fatal(err)
	}
	if steps < 1 {
		t.Fatalf("steps = %d", steps)
	}
	if elapsed < limit {
		t.Fatalf("elapsed %v shorter than %v", elapsed, limit)
	}
	if elapsed > limit+2*time.Second {
		t.Fatalf("elapsed %v ran well past %v after %d steps", elapsed, limit, steps)
	}
}

func TestFFmpegArgsUseVP9(t *testing.T) {
	args := ffmpegArgs("out.webm", 12)
	joined := strings.Join(args, " ")
	for _, want := range []string{"image2pipe", "pipe:0", "libvpx-vp9", "yuv420p", "-f webm", "out.webm"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("args %q missing %q", joined, want)
		}
	}
}

func TestWebMPipe(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip(err)
	}
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip(err)
	}
	cfg := testConfig()
	ps, err := Initialize(cfg, newRNG(4))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "sim.webm")
	w, err := newWebMWriter(path, 15)
	if err != nil {
		t.Fatal(err)
	}
	const frames = 3
	for i := 0; i < frames; i++ {
		if err := w.writeFrame(cfg, ps); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(
		"ffprobe", "-v", "error", "-count_frames",
		"-select_streams", "v:0",
		"-show_entries", "stream=codec_name,width,height,nb_read_frames",
		"-show_entries", "format=format_name",
		"-of", "default=noprint_wrappers=1",
		path,
	).Output()
	if err != nil {
		t.Fatal(err)
	}
	text := string(out)
	wants := []string{
		"codec_name=vp9",
		"width=" + strconv.Itoa(cfg.Grid*2+2),
		"height=" + strconv.Itoa(cfg.Grid),
		"nb_read_frames=" + strconv.Itoa(frames),
		"webm",
	}
	for _, want := range wants {
		if !strings.Contains(text, want) {
			t.Fatalf("probe %q missing %q", text, want)
		}
	}
}
