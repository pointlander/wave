// Copyright 2026 The Wave Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
)

// renderImage builds a side-by-side frame: spatial density |Ψ|² on the
// left and the fftshifted spectral power |Ψ̂|² on the right.
func renderImage(cfg Config, ps []Particle) (*image.RGBA, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	ev := newEvaluator(cfg)
	density, spectrum := ev.densityAndSpectrum(ps)
	g := cfg.Grid
	gap := 2
	img := image.NewRGBA(image.Rect(0, 0, g*2+gap, g))
	paint(img, 0, normalizeLog(density), g)
	paint(img, g+gap, normalizeLog(spectrum), g)
	return img, nil
}

// WriteFrame writes a PNG of the spatial density and the spectrum.
func WriteFrame(path string, cfg Config, ps []Particle) error {
	img, err := renderImage(cfg, ps)
	if err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func paint(img *image.RGBA, x0 int, values []float64, g int) {
	for y := 0; y < g; y++ {
		for x := 0; x < g; x++ {
			img.SetRGBA(x0+x, y, heat(values[y*g+x]))
		}
	}
}

func normalizeLog(v []float64) []float64 {
	max := 0.0
	for _, x := range v {
		if x > max {
			max = x
		}
	}
	out := make([]float64, len(v))
	if max == 0 {
		return out
	}
	denom := math.Log1p(max)
	for i, x := range v {
		if x < 0 {
			x = 0
		}
		out[i] = math.Log1p(x) / denom
	}
	return out
}

func heat(t float64) color.RGBA {
	if t < 0 {
		t = 0
	}
	if t > 1 {
		t = 1
	}
	stops := [...][3]float64{
		{0, 0, 0},
		{70, 0, 110},
		{190, 30, 50},
		{240, 160, 40},
		{255, 245, 220},
	}
	x := t * float64(len(stops)-1)
	i := int(x)
	if i >= len(stops)-1 {
		s := stops[len(stops)-1]
		return color.RGBA{R: uint8(s[0]), G: uint8(s[1]), B: uint8(s[2]), A: 255}
	}
	f := x - float64(i)
	a, b := stops[i], stops[i+1]
	return color.RGBA{
		R: uint8(a[0] + (b[0]-a[0])*f),
		G: uint8(a[1] + (b[1]-a[1])*f),
		B: uint8(a[2] + (b[2]-a[2])*f),
		A: 255,
	}
}
