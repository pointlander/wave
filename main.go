// Copyright 2026 The Wave Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command wave simulates a two-dimensional n-body system of Gaussian
// wave packets. Each packet has a spatial representation and its Fourier
// dual. A Newtonian step proposes the next state. A genetic algorithm
// then searches around that proposal and keeps the candidate with the
// lowest sum of Shannon entropies of the position density |Ψ|² and the
// spectral power |Ψ̂|².
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

func main() {
	cfg := DefaultConfig()
	outDir := ""
	flag.IntVar(&cfg.N, "n", cfg.N, "number of particles")
	flag.IntVar(&cfg.Steps, "steps", cfg.Steps, "number of time steps after the initial state")
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
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: wave [flags]\n\n")
		fmt.Fprintf(os.Stderr, "Run a 2D n-body simulation of wave packets. Each step is a\n")
		fmt.Fprintf(os.Stderr, "genetic search for the lowest spatial plus spectral entropy\n")
		fmt.Fprintf(os.Stderr, "in a neighborhood of the Newtonian proposal.\n\n")
		fmt.Fprintf(os.Stderr, "With -out, each frame is |Ψ|² on the left and |Ψ̂|² on the right.\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "wave: %v\n", err)
		os.Exit(1)
	}

	var (
		csvW *csv.Writer
		file *os.File
	)
	if outDir != "" {
		if err := os.MkdirAll(outDir, 0o755); err != nil {
			fmt.Fprintf(os.Stderr, "wave: %v\n", err)
			os.Exit(1)
		}
		var err error
		file, err = os.Create(filepath.Join(outDir, "trace.csv"))
		if err != nil {
			fmt.Fprintf(os.Stderr, "wave: %v\n", err)
			os.Exit(1)
		}
		defer file.Close()
		csvW = csv.NewWriter(file)
		if err := csvW.Write(traceHeader()); err != nil {
			fmt.Fprintf(os.Stderr, "wave: %v\n", err)
			os.Exit(1)
		}
	}

	rng := newRNG(cfg.Seed)
	ps, err := Initialize(cfg, rng)
	if err != nil {
		fmt.Fprintf(os.Stderr, "wave: %v\n", err)
		os.Exit(1)
	}
	first, err := Observe(cfg, ps)
	if err != nil {
		fmt.Fprintf(os.Stderr, "wave: %v\n", err)
		os.Exit(1)
	}
	first.HNewton = first.HJoint
	printHeader()
	printRow(0, 0, first)
	if err := writeOutputs(outDir, csvW, cfg, ps, 0, 0, first); err != nil {
		fmt.Fprintf(os.Stderr, "wave: %v\n", err)
		os.Exit(1)
	}

	last := first
	t := 0.0
	gain := 0.0
	for step := 1; step <= cfg.Steps; step++ {
		var res StepResult
		ps, res, err = Step(cfg, rng, ps)
		if err != nil {
			fmt.Fprintf(os.Stderr, "wave: %v\n", err)
			os.Exit(1)
		}
		t += cfg.DT
		gain += res.HNewton - res.HJoint
		printRow(step, t, res)
		if err := writeOutputs(outDir, csvW, cfg, ps, step, t, res); err != nil {
			fmt.Fprintf(os.Stderr, "wave: %v\n", err)
			os.Exit(1)
		}
		last = res
	}
	if csvW != nil {
		csvW.Flush()
		if err := csvW.Error(); err != nil {
			fmt.Fprintf(os.Stderr, "wave: %v\n", err)
			os.Exit(1)
		}
	}
	fmt.Printf("\njoint entropy %.4f -> %.4f\n", first.HJoint, last.HJoint)
	fmt.Printf("spatial %.4f -> %.4f, frequency %.4f -> %.4f\n",
		first.HSpatial, last.HSpatial, first.HFrequency, last.HFrequency)
	fmt.Printf("mean separation %.4f -> %.4f\n", first.Separation, last.Separation)
	if cfg.Steps > 0 {
		fmt.Printf("mean entropy improvement over the Newtonian proposal %.4f\n", gain/float64(cfg.Steps))
	}
}

func traceHeader() []string {
	return []string{
		"step", "time", "h_spatial", "h_frequency", "h_joint", "h_newton",
		"separation", "kinetic", "potential",
	}
}

func printHeader() {
	fmt.Printf("%4s %8s %9s %9s %9s %9s %9s %9s %10s\n",
		"step", "time", "H_space", "H_freq", "H_joint", "H_newton", "sep", "kinetic", "potential")
}

func printRow(step int, time float64, r StepResult) {
	fmt.Printf("%4d %8.3f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %10.4f\n",
		step, time, r.HSpatial, r.HFrequency, r.HJoint, r.HNewton, r.Separation, r.Kinetic, r.Potential)
}

func writeOutputs(dir string, csvW *csv.Writer, cfg Config, ps []Particle, step int, time float64, r StepResult) error {
	if dir == "" {
		return nil
	}
	row := []string{
		strconv.Itoa(step),
		strconv.FormatFloat(time, 'f', 6, 64),
		strconv.FormatFloat(r.HSpatial, 'f', 6, 64),
		strconv.FormatFloat(r.HFrequency, 'f', 6, 64),
		strconv.FormatFloat(r.HJoint, 'f', 6, 64),
		strconv.FormatFloat(r.HNewton, 'f', 6, 64),
		strconv.FormatFloat(r.Separation, 'f', 6, 64),
		strconv.FormatFloat(r.Kinetic, 'f', 6, 64),
		strconv.FormatFloat(r.Potential, 'f', 6, 64),
	}
	if err := csvW.Write(row); err != nil {
		return err
	}
	path := filepath.Join(dir, fmt.Sprintf("frame_%04d.png", step))
	return WriteFrame(path, cfg, ps)
}
