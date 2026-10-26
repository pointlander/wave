// Copyright 2026 The Wave Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "math"

// fft2 is an in-place radix-2 complex DFT. The forward transform uses
// e^{-2πi}; the inverse scales by 1/n per axis.
type fft2 struct {
	n   int
	buf []complex128
}

func newFFT2(n int) fft2 {
	if n < 2 || n&(n-1) != 0 {
		panic("fft2: n must be a power of two and at least 2")
	}
	return fft2{n: n, buf: make([]complex128, 2*n)}
}

func (f fft2) transform(a []complex128, invert bool) {
	n := f.n
	if len(a) != n*n {
		panic("fft2: bad length")
	}
	row := f.buf[:n]
	for y := 0; y < n; y++ {
		copy(row, a[y*n:(y+1)*n])
		fft1D(row, invert)
		copy(a[y*n:(y+1)*n], row)
	}
	col := f.buf[n : 2*n]
	for x := 0; x < n; x++ {
		for y := 0; y < n; y++ {
			col[y] = a[y*n+x]
		}
		fft1D(col, invert)
		for y := 0; y < n; y++ {
			a[y*n+x] = col[y]
		}
	}
}

func fft1D(a []complex128, invert bool) {
	n := len(a)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j &^= bit
		}
		j |= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	for length := 2; length <= n; length <<= 1 {
		ang := 2 * math.Pi / float64(length)
		if !invert {
			ang = -ang
		}
		wlen := complex(math.Cos(ang), math.Sin(ang))
		half := length / 2
		for i := 0; i < n; i += length {
			w := 1 + 0i
			for j := 0; j < half; j++ {
				u := a[i+j]
				v := a[i+j+half] * w
				a[i+j] = u + v
				a[i+j+half] = u - v
				w *= wlen
			}
		}
	}
	if invert {
		scale := 1 / float64(n)
		for i := range a {
			a[i] *= complex(scale, 0)
		}
	}
}

// freqBin maps an angular wave number to a DFT bin on a periodic
// interval of length L.
func freqBin(k, L float64, n int) int {
	m := int(math.Round(k * L / (2 * math.Pi)))
	return wrapIndex(m, n)
}
