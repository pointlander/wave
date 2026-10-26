// Copyright 2026 The Wave Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// webmWriter pipes PNG frames to ffmpeg and encodes them as VP9 in a WebM.
type webmWriter struct {
	path   string
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stderr bytes.Buffer
	done   bool
	err    error
}

func newWebMWriter(path string, fps int) (*webmWriter, error) {
	if fps < 1 {
		return nil, fmt.Errorf("fps must be at least 1")
	}
	if path == "" {
		return nil, fmt.Errorf("webm path is empty")
	}
	bin, err := exec.LookPath("ffmpeg")
	if err != nil {
		return nil, fmt.Errorf("ffmpeg not found: %w", err)
	}
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
	}
	cmd := exec.Command(bin, ffmpegArgs(path, fps)...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	w := &webmWriter{path: path, cmd: cmd, stdin: stdin}
	cmd.Stdout = io.Discard
	cmd.Stderr = &w.stderr
	if err := cmd.Start(); err != nil {
		stdin.Close()
		return nil, err
	}
	return w, nil
}

// ffmpegArgs reads a concatenated PNG stream on stdin and writes VP9 WebM.
func ffmpegArgs(path string, fps int) []string {
	return []string{
		"-y",
		"-hide_banner",
		"-loglevel", "error",
		"-framerate", strconv.Itoa(fps),
		"-f", "image2pipe",
		"-vcodec", "png",
		"-i", "pipe:0",
		"-c:v", "libvpx-vp9",
		"-pix_fmt", "yuv420p",
		"-b:v", "0",
		"-crf", "18",
		"-an",
		"-f", "webm",
		path,
	}
}

func (w *webmWriter) writeFrame(cfg Config, ps []Particle) error {
	img, err := renderImage(cfg, ps)
	if err != nil {
		return err
	}
	return w.writeImage(img)
}

func (w *webmWriter) writeImage(img image.Image) error {
	if err := png.Encode(w.stdin, img); err != nil {
		return fmt.Errorf("pipe frame to ffmpeg: %w", err)
	}
	return nil
}

// Close finishes the PNG stream and waits for ffmpeg to exit.
// Further calls return the same result.
func (w *webmWriter) Close() error {
	if w == nil || w.done {
		if w == nil {
			return nil
		}
		return w.err
	}
	w.done = true
	cerr := w.stdin.Close()
	werr := w.cmd.Wait()
	if werr != nil {
		msg := strings.TrimSpace(w.stderr.String())
		if msg != "" {
			w.err = fmt.Errorf("ffmpeg: %w: %s", werr, msg)
		} else {
			w.err = fmt.Errorf("ffmpeg: %w", werr)
		}
		return w.err
	}
	w.err = cerr
	return w.err
}
