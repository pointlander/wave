# wave

A two-dimensional n-body simulation of Gaussian wave packets. Each particle has a spatial representation and its Fourier dual. Gravity proposes the next state. A genetic search around that proposal keeps the candidate with the lowest frequency entropy divided by spatial entropy.

## Model

Particles live in a periodic unit box. Gravity is Plummer-softened Newtonian gravity in the plane, integrated with velocity Verlet. Total momentum is held fixed.

Each packet is a Gaussian envelope of width `σ` centered at `(X, Y)`, moving with velocity `V`. Its frequency representation is the matching Gaussian in wavevector space:

- carrier `K = V / ħ`
- spectral width `σ_k = 1 / (2σ)`
- an internal phase `φ`

The field `Ψ` is the coherent sum of the packets. Two entropies are measured on the grid, each normalized so a spike scores 0 and a flat field scores 1:

- spatial entropy of the density `|Ψ|²`
- frequency entropy of the power spectrum `|Ψ̂|²`

The fitness of a candidate is `H_frequency / H_spatial`. Lower is better, so the search minimizes spectral entropy and maximizes spatial entropy. A state with zero spatial entropy scores as infinitely bad. The Newtonian step is always in the population, so the accepted state is never worse than that proposal. Crossover swaps position, velocity, or phase genes. It does not average coordinates.

The initial condition is a rotating disk of equal masses with random phases.

## Build and run

Requires Go 1.27. [ffmpeg](https://ffmpeg.org/) with `libvpx-vp9` is required only for `-webm`.

```
go test ./...
go run .
```

A default run prints one row per step: spatial entropy, frequency entropy, their ratio, the Newtonian proposal's ratio, mean separation, kinetic energy, and potential energy. `agree` is 1 when the accepted state matches the Newtonian proposal, and `agree_rate` is how often that has happened so far. The summary reports the same fraction for the whole run.

Run for 30 seconds of real time and write a VP9 WebM. Frames show `|Ψ|²` on the left and the fftshifted spectrum `|Ψ̂|²` on the right. `-duration` overrides `-steps`. Playback length is `frames / fps`, which is not the same as the wall-clock limit.

```
go run . -duration 30s -webm wave.webm
```

Write a PNG sequence and `trace.csv` as well:

```
go run . -duration 30s -webm wave.webm -out frames
```

`-duration` accepts a Go duration (`30s`, `1m`) or a bare number of seconds (`30`).

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `-n` | 12 | particle count |
| `-steps` | 24 | steps after the initial state, used when `-duration` is 0 |
| `-duration` | 0 | wall-clock limit; overrides `-steps` |
| `-dt` | 0.02 | Newtonian time step |
| `-grid` | 64 | field size, a power of two |
| `-pop` | 24 | genetic population size |
| `-gens` | 5 | generations per step |
| `-elite` | 2 | elite genomes kept each generation |
| `-seed` | 1 | random seed |
| `-gravity` | 0.008 | gravitational constant |
| `-softening` | 0.02 | Plummer softening length |
| `-hbar` | 0.01 | de Broglie scale, `K = V / ħ` |
| `-sigma` | 0.04 | packet width |
| `-radius` | 0.18 | initial disk radius |
| `-mutpos` | 0.01 | position mutation scale |
| `-mutvel` | 0.02 | velocity mutation scale |
| `-mutphase` | 0.2 | phase mutation scale |
| `-out` | | directory for PNG frames and `trace.csv` |
| `-webm` | | path of a VP9 WebM written by ffmpeg |
| `-fps` | 12 | WebM frame rate |

Set the mutation scales to 0 to take the Newtonian step unchanged.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
