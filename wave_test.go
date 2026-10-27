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
		PopSize:    10,
		Gens:       3,
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

func TestShannonBounds(t *testing.T) {
	if h := shannon([]float64{0, 1, 0, 0}); h != 0 {
		t.Fatalf("spike entropy = %v", h)
	}
	h := shannon([]float64{1, 1, 1, 1})
	if math.Abs(h-1) > 1e-12 {
		t.Fatalf("uniform entropy = %v", h)
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
	apart := []Particle{packet(0.25, 0.5), packet(0.75, 0.5)}
	ths, thf := measure(cfg, together)
	ahs, ahf := measure(cfg, apart)
	if ths >= ahs {
		t.Fatalf("spatial entropy together %v, apart %v", ths, ahs)
	}
	if fitness(ahs, ahf) >= fitness(ths, thf) {
		t.Fatalf("spread fitness %v, coincident fitness %v", fitness(ahs, ahf), fitness(ths, thf))
	}
}

func TestFitnessRatio(t *testing.T) {
	if got := fitness(0.4, 0.2); math.Abs(got-0.5) > 1e-12 {
		t.Fatalf("fitness = %v", got)
	}
	if !math.IsInf(fitness(0, 0.3), 1) {
		t.Fatal("zero spatial entropy should be the worst fitness")
	}
	if !math.IsInf(fitness(0, 0), 1) {
		t.Fatal("zero entropies should be the worst fitness")
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
}

func TestSelectionDoesNotExceedNewton(t *testing.T) {
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
		if res.Fitness > res.Newton+1e-9 {
			t.Fatalf("step %d fitness %v exceeds newton %v", step, res.Fitness, res.Newton)
		}
		if math.IsNaN(res.Fitness) || math.IsNaN(res.Kinetic) || math.IsNaN(res.Potential) {
			t.Fatalf("step %d produced NaN: %+v", step, res)
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
	cfg.PopSize = 4
	cfg.Gens = 1
	cfg.Elite = 1
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
