// Copyright 2026 The Wave Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command wave simulates a two-dimensional n-body system of Gaussian
// wave packets. Each packet has a spatial representation and its Fourier
// dual. A Newtonian step proposes the next state. A genetic algorithm
// then searches around that proposal and keeps the candidate with the
// lowest H_frequency / H_spatial.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "wave: %v\n", err)
		os.Exit(1)
	}
}

func run() (err error) {
	cfg := DefaultConfig()
	outDir := ""
	webmPath := ""
	fps := 12
	var runFor durationFlag
	flag.IntVar(&cfg.N, "n", cfg.N, "number of particles")
	flag.IntVar(&cfg.Steps, "steps", cfg.Steps, "number of time steps after the initial state, used when -duration is 0")
	flag.Float64Var(&cfg.DT, "dt", cfg.DT, "Newtonian time step")
	flag.IntVar(&cfg.Grid, "grid", cfg.Grid, "field grid size, a power of two")
	flag.IntVar(&cfg.PopSize, "pop", cfg.PopSize, "genetic population size")
	flag.IntVar(&cfg.Gens, "gens", cfg.Gens, "generations per time step")
	flag.IntVar(&cfg.Elite, "elite", cfg.Elite, "elite genomes kept each generation")
	flag.Int64Var(&cfg.Seed, "seed", cfg.Seed, "random seed")
	flag.Float64Var(&cfg.Gravity, "gravity", cfg.Gravity, "gravitational constant")
	flag.Float64Var(&cfg.Softening, "softening", cfg.Softening, "Plummer softening length")
	flag.Float64Var(&cfg.Hbar, "hbar", cfg.Hbar, "de Broglie scale, k = v / hbar")
	flag.Float64Var(&cfg.Sigma, "sigma", cfg.Sigma, "Gaussian packet width")
	flag.Float64Var(&cfg.Radius, "radius", cfg.Radius, "initial disk radius")
	flag.Float64Var(&cfg.MutPos, "mutpos", cfg.MutPos, "position mutation scale")
	flag.Float64Var(&cfg.MutVel, "mutvel", cfg.MutVel, "velocity mutation scale")
	flag.Float64Var(&cfg.MutPhase, "mutphase", cfg.MutPhase, "phase mutation scale")
	flag.StringVar(&outDir, "out", "", "directory for PNG frames and trace.csv")
	flag.StringVar(&webmPath, "webm", "", "pipe frames to ffmpeg and write a VP9 WebM")
	flag.IntVar(&fps, "fps", fps, "frame rate of the WebM")
	flag.Var(&runFor, "duration", "wall-clock limit, such as 30s or 30; overrides -steps")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: wave [flags]\n\n")
		fmt.Fprintf(os.Stderr, "Run a 2D n-body simulation of wave packets. Each step is a\n")
		fmt.Fprintf(os.Stderr, "genetic search for the lowest H_frequency / H_spatial\n")
		fmt.Fprintf(os.Stderr, "in a neighborhood of the Newtonian proposal.\n\n")
		fmt.Fprintf(os.Stderr, "Frames show |Ψ|² on the left and |Ψ̂|² on the right.\n")
		fmt.Fprintf(os.Stderr, "-out writes PNGs. -webm pipes those frames to ffmpeg as VP9.\n")
		fmt.Fprintf(os.Stderr, "-duration 30s keeps stepping for 30 seconds of real time.\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if err := cfg.Validate(); err != nil {
		return err
	}

	var (
		csvW *csv.Writer
		file *os.File
		enc  *webmWriter
	)
	if webmPath != "" {
		enc, err = newWebMWriter(webmPath, fps)
		if err != nil {
			return err
		}
		defer func() {
			if cerr := enc.Close(); cerr != nil && err == nil {
				err = cerr
			}
		}()
	}
	if outDir != "" {
		if err := os.MkdirAll(outDir, 0o755); err != nil {
			return err
		}
		file, err = os.Create(filepath.Join(outDir, "trace.csv"))
		if err != nil {
			return err
		}
		defer file.Close()
		csvW = csv.NewWriter(file)
		if err := csvW.Write(traceHeader()); err != nil {
			return err
		}
	}

	rng := newRNG(cfg.Seed)
	ps, err := Initialize(cfg, rng)
	if err != nil {
		return err
	}
	printHeader()
	var gain float64
	steps, elapsed, first, last, err := simulate(cfg, rng, ps, time.Duration(runFor), func(step int, simTime float64, ps []Particle, res StepResult) error {
		if step > 0 {
			gain += res.Newton - res.Fitness
		}
		printRow(step, simTime, res)
		return writeOutputs(outDir, csvW, enc, cfg, ps, step, simTime, res)
	})
	if err != nil {
		return err
	}
	if csvW != nil {
		csvW.Flush()
		if err := csvW.Error(); err != nil {
			return err
		}
	}
	if enc != nil {
		if err := enc.Close(); err != nil {
			return err
		}
	}
	fmt.Printf("\nfitness %.4f -> %.4f\n", first.Fitness, last.Fitness)
	fmt.Printf("spatial %.4f -> %.4f, frequency %.4f -> %.4f\n",
		first.HSpatial, last.HSpatial, first.HFrequency, last.HFrequency)
	fmt.Printf("mean separation %.4f -> %.4f\n", first.Separation, last.Separation)
	if steps > 0 {
		fmt.Printf("mean fitness improvement over the Newtonian proposal %.4f\n", gain/float64(steps))
		fmt.Printf("newtonian agreement %d/%d (%.3f)\n", last.AgreeCount, steps, last.AgreeRate)
	}
	if runFor > 0 {
		fmt.Printf("ran %d steps in %.3fs\n", steps, elapsed.Seconds())
	}
	if webmPath != "" {
		fmt.Printf("wrote %s\n", webmPath)
	}
	return nil
}

// durationFlag accepts a Go duration (30s, 1m) or a bare number of seconds.
type durationFlag time.Duration

func (d *durationFlag) String() string {
	if d == nil || *d == 0 {
		return "0"
	}
	return time.Duration(*d).String()
}

func (d *durationFlag) Set(s string) error {
	if v, err := time.ParseDuration(s); err == nil {
		if v < 0 {
			return fmt.Errorf("duration must be non-negative")
		}
		*d = durationFlag(v)
		return nil
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return fmt.Errorf("duration %q is not a number of seconds or a Go duration", s)
	}
	if v < 0 {
		return fmt.Errorf("duration must be non-negative")
	}
	*d = durationFlag(time.Duration(v * float64(time.Second)))
	return nil
}

func traceHeader() []string {
	return []string{
		"step", "time", "h_spatial", "h_frequency", "fitness", "newton",
		"separation", "kinetic", "potential", "agree", "agree_rate",
	}
}

func printHeader() {
	fmt.Printf("%4s %8s %9s %9s %9s %9s %9s %9s %10s %5s %10s\n",
		"step", "time", "H_space", "H_freq", "Hf/Hs", "newton", "sep", "kinetic", "potential", "agree", "agree_rate")
}

func printRow(step int, time float64, r StepResult) {
	agree := 0
	if r.Agree {
		agree = 1
	}
	fmt.Printf("%4d %8.3f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %10.4f %5d %10.4f\n",
		step, time, r.HSpatial, r.HFrequency, r.Fitness, r.Newton, r.Separation, r.Kinetic, r.Potential, agree, r.AgreeRate)
}

func boolInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

func writeOutputs(dir string, csvW *csv.Writer, enc *webmWriter, cfg Config, ps []Particle, step int, time float64, r StepResult) error {
	if dir == "" && enc == nil {
		return nil
	}
	if csvW != nil {
		row := []string{
			strconv.Itoa(step),
			strconv.FormatFloat(time, 'f', 6, 64),
			strconv.FormatFloat(r.HSpatial, 'f', 6, 64),
			strconv.FormatFloat(r.HFrequency, 'f', 6, 64),
			strconv.FormatFloat(r.Fitness, 'f', 6, 64),
			strconv.FormatFloat(r.Newton, 'f', 6, 64),
			strconv.FormatFloat(r.Separation, 'f', 6, 64),
			strconv.FormatFloat(r.Kinetic, 'f', 6, 64),
			strconv.FormatFloat(r.Potential, 'f', 6, 64),
			strconv.Itoa(boolInt(r.Agree)),
			strconv.FormatFloat(r.AgreeRate, 'f', 6, 64),
		}
		if err := csvW.Write(row); err != nil {
			return err
		}
		path := filepath.Join(dir, fmt.Sprintf("frame_%04d.png", step))
		if err := WriteFrame(path, cfg, ps); err != nil {
			return err
		}
	}
	if enc != nil {
		return enc.writeFrame(cfg, ps)
	}
	return nil
}
